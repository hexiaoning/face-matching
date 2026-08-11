from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Person:
    id: int
    name: str
    id_number: str
    created_at: str
    updated_at: str
    photo_count: int = 0


@dataclass(frozen=True)
class PersonPhoto:
    id: int
    person_id: int
    path: Path
    original_name: str
    quality: float
    model_version: str


@dataclass(frozen=True)
class EmbeddingRecord:
    photo_id: int
    person_id: int
    person_name: str
    id_number: str
    embedding: np.ndarray
    quality: float


@dataclass(frozen=True)
class FaceDetection:
    bbox: np.ndarray
    score: float
    landmarks: np.ndarray


@dataclass(frozen=True)
class FaceQuality:
    overall: float
    sharpness: float
    pose: float
    resolution: float
    illumination: float


@dataclass(frozen=True)
class MatchResult:
    accepted: bool
    person_id: int | None
    name: str
    id_number: str
    score: float
    second_score: float
    reason: str

    @classmethod
    def unknown(cls, reason: str, score: float = 0.0, second_score: float = 0.0) -> MatchResult:
        return cls(False, None, "未知人员", "", score, second_score, reason)


@dataclass
class FaceObservation:
    detection: FaceDetection
    quality: FaceQuality
    embedding: np.ndarray | None
    aligned_face: np.ndarray | None = None


@dataclass
class TrackView:
    track_id: int
    bbox: np.ndarray
    quality: FaceQuality
    sample_count: int
    match: MatchResult
    is_new_event: bool = False


@dataclass
class TrackState:
    track_id: int
    bbox: np.ndarray
    landmarks: np.ndarray
    quality: FaceQuality
    last_frame: int
    missed: int = 0
    embeddings: list[tuple[float, np.ndarray]] = field(default_factory=list)
    aggregate: np.ndarray | None = None
    candidate_person_id: int | None = None
    candidate_hits: int = 0
    last_reported_person_id: int | None = None
