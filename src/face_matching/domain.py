from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class Person:
    id: int
    name: str
    id_number: str
    created_at: str
    updated_at: str
    photo_count: int = 0


@dataclass(frozen=True, slots=True)
class PhotoRecord:
    id: int
    person_id: int
    path: str
    source_name: str
    quality: float
    embedding: np.ndarray
    created_at: str


@dataclass(frozen=True, slots=True)
class PreparedPhoto:
    path: str
    source_name: str
    quality: float
    embedding: np.ndarray


@dataclass(slots=True)
class FaceObservation:
    bbox: np.ndarray
    landmarks: np.ndarray
    detection_score: float
    quality: float = 0.0
    embedding: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class MatchResult:
    person_id: int
    name: str
    id_number: str
    score: float
    best_photo_id: int | None = None


@dataclass(frozen=True, slots=True)
class TrackView:
    track_id: int
    bbox: tuple[float, float, float, float]
    label: str
    score: float
    quality: float
    confirmed: bool
    person_id: int | None
    id_number: str = ""


@dataclass(frozen=True, slots=True)
class RecognitionEvent:
    occurred_at: datetime
    source: str
    track_id: int
    match: MatchResult
    quality: float


@dataclass(frozen=True, slots=True)
class ModelPaths:
    detector: Path
    recognizer: Path
    model_name: str


@dataclass(frozen=True, slots=True)
class GPUInfo:
    name: str
    driver_version: str
    memory_mb: int | None
    provider: str = "CUDAExecutionProvider"


@dataclass(slots=True)
class EnrollmentReport:
    person_id: int
    warnings: list[str] = field(default_factory=list)


def normalize_embedding(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("人脸特征向量无效")
    return np.ascontiguousarray(vector / norm, dtype=np.float32)


def mask_id_number(value: str) -> str:
    value = value.strip()
    if len(value) <= 7:
        return value[:1] + "*" * max(0, len(value) - 2) + value[-1:]
    return f"{value[:3]}{'*' * (len(value) - 7)}{value[-4:]}"

