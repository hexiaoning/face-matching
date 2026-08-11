from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from face_matching.detector import FaceDetection
from face_matching.config import EngineConfig
from face_matching.enrollment import _select_enrollment_face
from face_matching.errors import EnrollmentError
from face_matching.paths import model_dir
from face_matching.recognizer import LVFaceRecognizer
from face_matching.tracker import FaceTrack


LANDMARKS = np.array(
    [[4.0, 5.0], [8.0, 5.0], [6.0, 7.0], [4.5, 9.0], [7.5, 9.0]],
    dtype=np.float32,
)


def test_robust_track_aggregate_rejects_identity_outlier() -> None:
    track = FaceTrack(1, np.array([0, 0, 10, 10]), LANDMARKS, max_embeddings=8)
    track.add_embedding(np.array([1.0, 0.0]), 0.8, 1)
    track.add_embedding(np.array([0.98, 0.10]), 0.9, 2)
    track.add_embedding(np.array([-1.0, 0.0]), 1.0, 3)

    aggregate = track.aggregate_embedding(top_k=8, similarity_floor=0.12)

    assert aggregate is not None
    assert aggregate[0] > 0.98
    assert abs(float(aggregate[1])) < 0.1


def test_enrollment_ignores_tiny_background_face_but_rejects_two_people() -> None:
    primary = FaceDetection(np.array([0, 0, 100, 100]), LANDMARKS, 0.95)
    tiny = FaceDetection(np.array([120, 0, 135, 15]), LANDMARKS, 0.8)
    second_person = FaceDetection(np.array([110, 0, 180, 80]), LANDMARKS, 0.9)

    assert _select_enrollment_face([tiny, primary], 32, "photo.jpg") is primary
    with pytest.raises(EnrollmentError, match="多张明显人脸"):
        _select_enrollment_face([primary, second_person], 32, "photo.jpg")


def test_recognizer_batches_mirror_augmentation(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSession:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def get_inputs(self):
            return [SimpleNamespace(name="input", shape=[2, 3, 112, 112])]

        def get_outputs(self):
            return [SimpleNamespace(name="output", shape=[2, 2])]

        def run(self, names, inputs):
            tensor = inputs["input"]
            self.batch_sizes.append(len(tensor))
            mean = tensor.mean(axis=(1, 2, 3))
            return [np.stack([mean, np.ones_like(mean)], axis=1).astype(np.float32)]

    session = FakeSession()
    monkeypatch.setattr(
        "face_matching.recognizer.create_gpu_session", lambda *args, **kwargs: session
    )
    recognizer = LVFaceRecognizer("recognizer.onnx")
    faces = [
        np.zeros((112, 112, 3), dtype=np.uint8),
        np.full((112, 112, 3), 255, dtype=np.uint8),
    ]

    embeddings = recognizer.embed_batch(faces, mirror_augmentation=True)

    assert embeddings.shape == (2, 2)
    assert session.batch_sizes == [2, 2]
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-6)


def test_frozen_bundle_prefers_embedded_models(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundled_models = tmp_path / "models"
    bundled_models.mkdir()
    monkeypatch.delenv("FACE_MATCHING_MODEL_DIR", raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)

    assert model_dir() == bundled_models


def test_accuracy_profile_defaults_and_tta_feature_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FACE_MATCHING_DETECTOR_SIZE", raising=False)
    monkeypatch.delenv("FACE_MATCHING_MODEL_ID", raising=False)
    monkeypatch.delenv("FACE_MATCHING_MIRROR_TTA", raising=False)
    config = EngineConfig()
    assert config.detector_size == (960, 960)
    assert config.mirror_augmentation
    assert "tta" in config.model_id

    monkeypatch.setenv("FACE_MATCHING_MIRROR_TTA", "0")
    speed_config = EngineConfig()
    assert not speed_config.mirror_augmentation
    assert "single" in speed_config.model_id
