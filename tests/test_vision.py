from types import SimpleNamespace

import numpy as np
import pytest

from face_matching.matching import MatchResult
from face_matching.vision.detector import distance_to_bbox, distance_to_landmarks, nms
from face_matching.vision.recognizer import FaceEmbedder, l2_normalize
from face_matching.vision.tracker import FaceTracker, Observation, bbox_iou


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
    track.apply_match(match)
    assert track.person_id == "person"
    assert track.name == "Alice"


def test_bbox_iou_handles_disjoint_and_equal_boxes():
    box = np.asarray([0, 0, 10, 10], dtype=np.float32)
    assert bbox_iou(box, box) == pytest.approx(1.0)
    assert bbox_iou(box, np.asarray([20, 20, 30, 30])) == 0.0
