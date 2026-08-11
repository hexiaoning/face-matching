from types import SimpleNamespace

import numpy as np
import pytest

from face_matching.config import AppConfig
from face_matching.engine import FaceEngine, reference_alignment_variants
from face_matching.matching import TARGET_PERSON_ID, MatchResult, TargetMatchResult
from face_matching.vision.detector import Detection, distance_to_bbox, distance_to_landmarks, nms
from face_matching.vision.recognizer import FaceEmbedder, l2_normalize
from face_matching.vision.tracker import FaceTracker, Observation, bbox_iou, robust_aggregate


class FakeSession:
    def __init__(self, batch):
        self.batch = batch
        self.calls = []

    def get_inputs(self):
        return [SimpleNamespace(name="input", shape=[self.batch, 3, 112, 112])]

    def get_outputs(self):
        return [SimpleNamespace(name="output")]

    def run(self, output_names, inputs):
        tensor = inputs["input"]
        self.calls.append(tensor.copy())
        # Deterministic non-zero embeddings, one row per input image.
        means = tensor.mean(axis=(1, 2, 3))
        return [np.column_stack((np.ones(len(tensor)), means + 0.25)).astype(np.float32)]


def test_embedder_batches_dynamic_model_and_mirror_tta():
    session = FakeSession("batch")
    embedder = FaceEmbedder(session)
    faces = [np.full((112, 112, 3), 40, np.uint8), np.full((112, 112, 3), 180, np.uint8)]

    embeddings = embedder.embed_many(faces, mirror_augmentation=True)

    assert len(session.calls) == 1
    assert session.calls[0].shape == (4, 3, 112, 112)
    assert len(embeddings) == 2
    assert all(np.linalg.norm(item) == pytest.approx(1.0) for item in embeddings)


def test_embedder_honors_fixed_batch_without_changing_result_count():
    session = FakeSession(2)
    embedder = FaceEmbedder(session)
    faces = [np.full((112, 112, 3), value, np.uint8) for value in (20, 80, 140)]

    embeddings = embedder.embed_many(faces, mirror_augmentation=False)

    assert [call.shape[0] for call in session.calls] == [2, 2]
    assert len(embeddings) == 3


def test_normalize_rejects_zero_vector():
    with pytest.raises(ValueError):
        l2_normalize(np.zeros(3, dtype=np.float32))


def test_detector_geometry_and_nms():
    points = np.asarray([[10, 10]], dtype=np.float32)
    assert np.allclose(distance_to_bbox(points, np.asarray([[1, 2, 3, 4]])), [[9, 8, 13, 14]])
    landmarks = distance_to_landmarks(
        points, np.asarray([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=np.float32)
    )
    assert np.allclose(landmarks, [[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
    boxes = np.asarray(
        [[0, 0, 10, 10, 0.9], [1, 1, 11, 11, 0.8], [30, 30, 40, 40, 0.7]],
        dtype=np.float32,
    )
    assert nms(boxes, 0.4) == [0, 2]


def test_tracker_aggregates_quality_and_requires_match_consensus():
    tracker = FaceTracker(max_misses=2)
    embedding = l2_normalize(np.asarray([1.0, 0.1], dtype=np.float32))
    for frame in range(1, 4):
        tracks = tracker.update(
            [Observation(np.asarray([0, 0, 20, 20]), 0.9, embedding, 0.4 + frame / 10)],
            frame,
        )
    track = tracks[0]
    aggregate = track.aggregate(top_k=2)
    assert aggregate is not None
    assert np.linalg.norm(aggregate) == pytest.approx(1.0)
    assert track.quality > 0.5

    match = MatchResult(True, "person", "Alice", "ID", 0.8, 0.2, 0.6)
    track.apply_match(match)
    assert track.person_id is None
    # Reusing the same aggregate must not manufacture a second confirmation.
    track.apply_match(match)
    assert track.person_id is None
    tracker.update(
        [Observation(np.asarray([1, 0, 21, 20]), 0.9, embedding, 0.8)],
        4,
    )
    track.apply_match(match)
    assert track.person_id == "person"
    assert track.name == "Alice"


def test_bbox_iou_handles_disjoint_and_equal_boxes():
    box = np.asarray([0, 0, 10, 10], dtype=np.float32)
    assert bbox_iou(box, box) == pytest.approx(1.0)
    assert bbox_iou(box, np.asarray([20, 20, 30, 30])) == 0.0


def test_robust_aggregate_rejects_identity_switch_outlier():
    inlier_a = l2_normalize(np.asarray([1.0, 0.0], dtype=np.float32))
    inlier_b = l2_normalize(np.asarray([0.98, 0.10], dtype=np.float32))
    outlier = l2_normalize(np.asarray([0.0, 1.0], dtype=np.float32))

    aggregate, quality = robust_aggregate(
        [(inlier_a, 0.8), (outlier, 0.95), (inlier_b, 0.75)],
        top_k=3,
        min_similarity=0.20,
    )

    assert aggregate @ inlier_a > 0.99
    assert aggregate @ outlier < 0.20
    assert 0.7 < quality < 0.9


def test_tracker_motion_prediction_preserves_fast_moving_track():
    tracker = FaceTracker(max_misses=2)
    embedding = l2_normalize(np.asarray([1.0, 0.0], dtype=np.float32))
    ids = []
    for frame, x in enumerate((0, 9, 18), 1):
        tracks = tracker.update(
            [Observation(np.asarray([x, 0, x + 10, 10]), 0.9, embedding, 0.8)],
            frame,
        )
        ids.append(tracks[-1].id)
    assert ids == [1, 1, 1]


def test_reference_alignment_variants_are_distinct_model_sized_images():
    face = np.arange(112 * 112 * 3, dtype=np.uint8).reshape(112, 112, 3)
    variants = reference_alignment_variants(face)

    assert len(variants) == 4
    assert all(item.shape == (112, 112, 3) for item in variants)
    assert any(not np.array_equal(face, item) for item in variants[1:])


def test_tightly_cropped_enrollment_retries_with_neutral_context():
    class ContextOnlyDetector:
        def __init__(self):
            self.widths = []

        def detect(self, image):
            self.widths.append(image.shape[1])
            if image.shape[1] == 100:
                return []
            offset = 35.0
            landmarks = np.asarray(
                [[32, 38], [68, 38], [50, 55], [36, 73], [64, 73]], dtype=np.float32
            ) + offset
            return [
                Detection(
                    np.asarray([20, 18, 80, 85], dtype=np.float32) + offset,
                    0.9,
                    landmarks,
                )
            ]

    engine = FaceEngine.__new__(FaceEngine)
    engine.config = AppConfig(enrollment_min_quality=0.0)
    engine.detector = ContextOnlyDetector()
    image = np.tile(np.arange(100, dtype=np.uint8), (100, 1))
    image = np.repeat(image[:, :, None], 3, axis=2)

    aligned, _, detection = engine._prepare_enrollment_face(image)

    assert engine.detector.widths == [100, 170]
    assert aligned.shape == (112, 112, 3)
    assert np.allclose(detection.bbox, [20, 18, 80, 85])


def test_target_confirmation_is_sticky_for_the_track():
    tracker = FaceTracker(max_misses=2)
    embedding = l2_normalize(np.asarray([1.0, 0.0], dtype=np.float32))
    track = tracker.update(
        [Observation(np.asarray([0, 0, 20, 20]), 0.9, embedding, 0.8)], 1
    )[0]
    track.apply_target_match(TargetMatchResult("confirmed", 0.22, 0.23, 4, 4), "person.jpg")
    assert track.person_id == TARGET_PERSON_ID
    assert track.name == "目标：person.jpg"

    tracker.update(
        [Observation(np.asarray([1, 0, 21, 20]), 0.9, embedding, 0.8)], 2
    )
    track.apply_target_match(TargetMatchResult("rejected", 0.05, 0.06, 0, 5), "person.jpg")
    assert track.person_id == TARGET_PERSON_ID
    assert track.score == pytest.approx(0.22)
