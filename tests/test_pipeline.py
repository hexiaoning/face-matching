from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from face_matching.config import EngineConfig
from face_matching.detector import FaceDetection
from face_matching.matcher import MatchResult
from face_matching.pipeline import VideoFacePipeline


class FakeEngine:
    def __init__(self) -> None:
        self.config = EngineConfig(
            detector_model=Path("detector.onnx"),
            recognizer_model=Path("recognizer.onnx"),
        )
        self.config.min_face_size = 1
        self.config.min_quality = 0.0
        self.config.min_track_embeddings = 1
        self.config.recognition_interval = 3
        self.config.confirmation_frames = 2

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        detection = FaceDetection(
            bbox=np.array([0.0, 0.0, 100.0, 100.0], dtype=np.float32),
            landmarks=np.zeros((5, 2), dtype=np.float32),
            score=0.9,
            aligned=np.zeros((112, 112, 3), dtype=np.uint8),
            quality=SimpleNamespace(overall=0.9),
        )
        return [detection]

    def embed(self, faces: list[np.ndarray]) -> np.ndarray:
        return np.repeat(np.array([[1.0, 0.0]], dtype=np.float32), len(faces), axis=0)


class FakeMatcher:
    def __init__(self) -> None:
        self.revision = 1
        self.calls = 0

    def match(self, embedding: np.ndarray, threshold: float, min_margin: float) -> MatchResult:
        self.calls += 1
        return MatchResult("person", "张三", "110101199001011234", 0.82, 0.2, True)


def test_pipeline_requires_two_fresh_embedding_decisions() -> None:
    engine = FakeEngine()
    matcher = FakeMatcher()
    pipeline = VideoFacePipeline(engine, matcher, engine.config)
    frame = np.zeros((120, 120, 3), dtype=np.uint8)

    first = pipeline.process(frame)
    assert first.faces[0].state == "待确认"
    assert matcher.calls == 1

    pipeline.process(frame)
    pipeline.process(frame)
    assert matcher.calls == 1

    fourth = pipeline.process(frame)
    assert matcher.calls == 2
    assert fourth.faces[0].accepted
    assert len(fourth.events) == 1
    assert fourth.events[0].name == "张三"


def test_gallery_revision_invalidates_stable_identity() -> None:
    engine = FakeEngine()
    engine.config.confirmation_frames = 1
    matcher = FakeMatcher()
    pipeline = VideoFacePipeline(engine, matcher, engine.config)
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    assert pipeline.process(frame).faces[0].accepted

    matcher.revision += 1
    refreshed = pipeline.process(frame)
    assert refreshed.faces[0].accepted
    assert matcher.calls == 2
