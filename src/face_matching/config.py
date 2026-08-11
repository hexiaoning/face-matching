from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .paths import model_dir


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _detector_size() -> tuple[int, int]:
    value = os.environ.get("FACE_MATCHING_DETECTOR_SIZE", "960").strip()
    try:
        size = int(value)
    except ValueError:
        size = 960
    if size not in {640, 960, 1280}:
        size = 960
    return size, size


def _gpu_device_id() -> int:
    try:
        return max(0, int(os.environ.get("FACE_MATCHING_GPU", "0")))
    except ValueError:
        return 0


def _model_id() -> str:
    explicit = os.environ.get("FACE_MATCHING_MODEL_ID")
    if explicit:
        return explicit
    suffix = "tta" if _env_bool("FACE_MATCHING_MIRROR_TTA", True) else "single"
    return f"lvface-b-glint360k-{suffix}-v2"


@dataclass(slots=True)
class EngineConfig:
    detector_model: Path = field(default_factory=lambda: Path(
        os.environ.get("FACE_MATCHING_DETECTOR_MODEL", model_dir() / "scrfd_10g_bnkps.onnx")
    ))
    recognizer_model: Path = field(default_factory=lambda: Path(
        os.environ.get("FACE_MATCHING_RECOGNIZER_MODEL", model_dir() / "LVFace-B_Glint360K.onnx")
    ))
    model_id: str = field(default_factory=_model_id)
    gpu_device_id: int = field(default_factory=_gpu_device_id)
    detector_size: tuple[int, int] = field(default_factory=_detector_size)
    detector_threshold: float = 0.40
    nms_threshold: float = 0.40
    match_threshold: float = 0.50
    min_margin: float = 0.035
    min_face_size: int = 32
    min_quality: float = 0.18
    enrollment_min_quality: float = 0.30
    recognition_interval: int = 3
    min_track_embeddings: int = 2
    confirmation_frames: int = 2
    max_track_age: int = 18
    max_track_embeddings: int = 18
    track_top_k: int = 10
    track_similarity_floor: float = 0.12
    mirror_augmentation: bool = field(default_factory=lambda: _env_bool(
        "FACE_MATCHING_MIRROR_TTA", True
    ))
    prefer_tensorrt: bool = field(default_factory=lambda: _env_bool(
        "FACE_MATCHING_TENSORRT"
    ))
