from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .paths import default_model_path


@dataclass(slots=True)
class EngineConfig:
    detector_model: Path = field(default_factory=lambda: Path(
        os.environ.get(
            "FACE_MATCHING_DETECTOR_MODEL", default_model_path("scrfd_10g_bnkps.onnx")
        )
    ))
    recognizer_model: Path = field(default_factory=lambda: Path(
        os.environ.get(
            "FACE_MATCHING_RECOGNIZER_MODEL", default_model_path("LVFace-B_Glint360K.onnx")
        )
    ))
    model_id: str = field(default_factory=lambda: os.environ.get(
        "FACE_MATCHING_MODEL_ID", "lvface-b-glint360k-v2-tta"
    ))
    detector_size: tuple[int, int] = (640, 640)
    detector_threshold: float = 0.45
    nms_threshold: float = 0.40
    match_threshold: float = 0.50
    min_margin: float = 0.035
    min_face_size: int = 36
    min_quality: float = 0.18
    enrollment_min_quality: float = 0.30
    recognition_interval: int = 3
    min_track_embeddings: int = 2
    confirmation_frames: int = 2
    max_track_age: int = 18
    max_track_embeddings: int = 24
    mirror_augmentation: bool = True
    gpu_backend: str = field(default_factory=lambda: os.environ.get(
        "FACE_MATCHING_BACKEND", "auto"
    ).strip().lower())
    prefer_tensorrt: bool = field(default_factory=lambda: os.environ.get(
        "FACE_MATCHING_TENSORRT", "0"
    ).strip().lower() in {"1", "true", "yes"})
