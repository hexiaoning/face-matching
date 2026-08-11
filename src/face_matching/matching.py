from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from face_matching.database import GalleryEntry
from face_matching.inference import EmbeddedFace


@dataclass(frozen=True, slots=True)
class MatchResult:
    person_id: int | None
    name: str
    score: float
    second_score: float
    accepted: bool


class IdentityMatcher:
    def __init__(self, threshold: float, min_margin: float):
        self.threshold = threshold
        self.min_margin = min_margin
        self._names: dict[int, str] = {}
        self._vectors: dict[int, np.ndarray] = {}

    def replace_gallery(self, entries: list[GalleryEntry]) -> None:
        grouped: dict[int, list[np.ndarray]] = {}
        self._names = {}
        for entry in entries:
            grouped.setdefault(entry.person_id, []).append(entry.embedding)
            self._names[entry.person_id] = entry.name
        self._vectors = {person_id: np.stack(vectors) for person_id, vectors in grouped.items()}

    @property
    def identity_count(self) -> int:
        return len(self._vectors)

    def match(self, embedding: np.ndarray) -> MatchResult:
        if not self._vectors:
            return MatchResult(None, "未知", -1.0, -1.0, False)
        query = np.asarray(embedding, dtype=np.float32).reshape(-1)
        query /= max(float(np.linalg.norm(query)), 1e-8)
        scores = sorted(
            (
                (float(np.max(vectors @ query)), person_id)
                for person_id, vectors in self._vectors.items()
            ),
            reverse=True,
        )
        best_score, best_id = scores[0]
        second_score = scores[1][0] if len(scores) > 1 else -1.0
        accepted = best_score >= self.threshold and (
            len(scores) == 1 or best_score - second_score >= self.min_margin
        )
        return MatchResult(
            best_id if accepted else None,
            self._names[best_id] if accepted else "未知",
            best_score,
            second_score,
            accepted,
        )


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
    second_area = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
    return intersection / max(first_area + second_area - intersection, 1e-8)


@dataclass(slots=True)
class Track:
    id: int
    bbox: np.ndarray
    embedding: np.ndarray
    weight: float
    last_frame: int
    hits: int = 1
    quality: float = 0.0
    candidate_person_id: int | None = None
    candidate_hits: int = 0
    decision: str = "画质不足"
    match: MatchResult = field(default_factory=lambda: MatchResult(None, "未知", -1.0, -1.0, False))

    def update(self, face: EmbeddedFace, frame_index: int) -> None:
        decay = 0.88
        contribution = max(0.05, face.quality)
        aggregate = self.embedding * self.weight * decay + face.embedding * contribution
        norm = max(float(np.linalg.norm(aggregate)), 1e-8)
        self.embedding = aggregate / norm
        self.weight = min(8.0, self.weight * decay + contribution)
        self.bbox = face.bbox.copy()
        self.last_frame = frame_index
        self.hits += 1
        self.quality = max(self.quality * 0.95, face.quality)

    def update_decision(
        self, matcher: IdentityMatcher, min_confirmations: int, min_quality: float
    ) -> None:
        if self.quality < min_quality:
            self.candidate_person_id = None
            self.candidate_hits = 0
            self.match = MatchResult(None, "未知", -1.0, -1.0, False)
            self.decision = "画质不足"
            return
        result = matcher.match(self.embedding)
        if not result.accepted:
            self.candidate_person_id = None
            self.candidate_hits = 0
            self.match = result
            self.decision = "待人工复核"
            return
        if result.person_id == self.candidate_person_id:
            self.candidate_hits += 1
        else:
            self.candidate_person_id = result.person_id
            self.candidate_hits = 1
        if self.candidate_hits >= min_confirmations:
            self.match = result
            self.decision = "已匹配"
        else:
            self.match = MatchResult(
                None,
                result.name,
                result.score,
                result.second_score,
                False,
            )
            self.decision = f"稳定确认 {self.candidate_hits}/{min_confirmations}"


class TemporalTracker:
    def __init__(
        self,
        max_age: int = 12,
        min_confirmations: int = 3,
        min_quality: float = 0.35,
    ):
        self.max_age = max_age
        self.min_confirmations = min_confirmations
        self.min_quality = min_quality
        self._tracks: dict[int, Track] = {}
        self._next_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def update(
        self, faces: list[EmbeddedFace], frame_index: int, matcher: IdentityMatcher
    ) -> list[Track]:
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if frame_index - track.last_frame <= self.max_age
        }
        candidates: list[tuple[float, int, int]] = []
        for face_index, face in enumerate(faces):
            for track_id, track in self._tracks.items():
                overlap = _iou(face.bbox, track.bbox)
                similarity = float(face.embedding @ track.embedding)
                if overlap >= 0.12 or similarity >= 0.50:
                    affinity = 0.6 * overlap + 0.4 * max(0.0, similarity)
                    candidates.append((affinity, face_index, track_id))
        assigned_faces: set[int] = set()
        assigned_tracks: set[int] = set()
        for _, face_index, track_id in sorted(candidates, reverse=True):
            if face_index in assigned_faces or track_id in assigned_tracks:
                continue
            self._tracks[track_id].update(faces[face_index], frame_index)
            assigned_faces.add(face_index)
            assigned_tracks.add(track_id)
        for face_index, face in enumerate(faces):
            if face_index in assigned_faces:
                continue
            track = Track(
                id=self._next_id,
                bbox=face.bbox.copy(),
                embedding=face.embedding.copy(),
                weight=max(0.05, face.quality),
                last_frame=frame_index,
                quality=face.quality,
            )
            self._tracks[track.id] = track
            self._next_id += 1
            assigned_tracks.add(track.id)
        visible: list[Track] = []
        for track_id in assigned_tracks:
            track = self._tracks[track_id]
            track.update_decision(matcher, self.min_confirmations, self.min_quality)
            visible.append(track)
        return sorted(visible, key=lambda item: item.id)
