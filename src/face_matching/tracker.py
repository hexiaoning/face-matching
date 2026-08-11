from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .detector import FaceDetection
from .matcher import MatchResult
from .recognizer import l2_normalize


def bbox_iou(first: np.ndarray, second: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(intersection / max(area_a + area_b - intersection, 1e-8))


def robust_aggregate(
    embeddings: list[np.ndarray] | deque[np.ndarray],
    qualities: list[float] | deque[float],
) -> np.ndarray | None:
    """Quality-weighted track template with medoid outlier rejection."""
    if not embeddings:
        return None
    matrix = l2_normalize(np.stack(embeddings))
    quality = np.maximum(np.asarray(qualities, dtype=np.float32), 0.05)
    recency = np.linspace(0.75, 1.0, len(quality), dtype=np.float32)

    if len(matrix) >= 3:
        cosine = matrix @ matrix.T
        medoid = int(np.argmax(cosine.mean(axis=1)))
        consistent = cosine[medoid] >= 0.20
        if np.count_nonzero(consistent) >= 2:
            matrix = matrix[consistent]
            quality = quality[consistent]
            recency = recency[consistent]

    weights = quality**1.5 * recency
    aggregate = np.average(matrix, axis=0, weights=weights)
    return l2_normalize(aggregate.reshape(1, -1))[0]


@dataclass(slots=True)
class FaceTrack:
    id: int
    bbox: np.ndarray
    landmarks: np.ndarray
    max_embeddings: int
    missed: int = 0
    hits: int = 1
    age: int = 1
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    last_embedding_frame: int = -10_000
    embeddings: deque[np.ndarray] = field(init=False)
    qualities: deque[float] = field(init=False)
    candidate_id: str | None = None
    candidate_count: int = 0
    stable_match: MatchResult | None = None
    last_match: MatchResult | None = None
    embedding_version: int = 0
    matched_embedding_version: int = 0
    unknown_count: int = 0

    def __post_init__(self) -> None:
        self.bbox = np.asarray(self.bbox, dtype=np.float32).copy()
        self.landmarks = np.asarray(self.landmarks, dtype=np.float32).copy()
        self.embeddings = deque(maxlen=self.max_embeddings)
        self.qualities = deque(maxlen=self.max_embeddings)

    def update(self, detection: FaceDetection) -> None:
        new_bbox = np.asarray(detection.bbox, dtype=np.float32)
        delta = new_bbox - self.bbox
        self.velocity = self.velocity * 0.65 + delta * 0.35
        self.bbox = new_bbox.copy()
        self.landmarks = np.asarray(detection.landmarks, dtype=np.float32).copy()
        self.missed = 0
        self.hits += 1

    def predict(self) -> np.ndarray:
        return self.bbox + self.velocity

    def add_embedding(self, embedding: np.ndarray, quality: float, frame_index: int) -> None:
        self.embeddings.append(np.asarray(embedding, dtype=np.float32).reshape(-1).copy())
        self.qualities.append(max(float(quality), 0.01))
        self.last_embedding_frame = frame_index
        self.embedding_version += 1

    def aggregate_embedding(self) -> np.ndarray | None:
        return robust_aggregate(self.embeddings, self.qualities)

    def update_identity(self, match: MatchResult, confirmations: int) -> bool:
        """Consume one fresh embedding decision and report a newly stable identity."""
        previous_id = self.stable_match.person_id if self.stable_match else None
        self.last_match = match
        self.matched_embedding_version = self.embedding_version
        if not match.accepted or match.person_id is None:
            self.unknown_count += 1
            if self.unknown_count >= 5:
                self.stable_match = None
                self.candidate_id = None
                self.candidate_count = 0
            return False
        self.unknown_count = 0
        if self.candidate_id == match.person_id:
            self.candidate_count += 1
        else:
            self.candidate_id = match.person_id
            self.candidate_count = 1
        if self.candidate_count >= confirmations:
            self.stable_match = match
        return self.stable_match is not None and self.stable_match.person_id != previous_id

    def invalidate_identity(self) -> None:
        """Force re-matching when the gallery changes without discarding track evidence."""
        self.candidate_id = None
        self.candidate_count = 0
        self.stable_match = None
        self.last_match = None
        self.matched_embedding_version = -1
        self.unknown_count = 0


class MultiFaceTracker:
    def __init__(self, max_age: int = 18, max_embeddings: int = 24) -> None:
        self.max_age = max_age
        self.max_embeddings = max_embeddings
        self.tracks: dict[int, FaceTrack] = {}
        self._next_id = 1

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1

    def update(self, detections: list[FaceDetection]) -> list[tuple[FaceTrack, FaceDetection]]:
        for track in self.tracks.values():
            track.age += 1
            track.missed += 1

        candidates: list[tuple[float, int, int]] = []
        for track_id, track in self.tracks.items():
            predicted = track.predict()
            pcx = (predicted[0] + predicted[2]) * 0.5
            pcy = (predicted[1] + predicted[3]) * 0.5
            diagonal = max(float(np.hypot(predicted[2] - predicted[0], predicted[3] - predicted[1])), 1.0)
            for detection_index, detection in enumerate(detections):
                iou = bbox_iou(predicted, detection.bbox)
                dcx = (detection.bbox[0] + detection.bbox[2]) * 0.5
                dcy = (detection.bbox[1] + detection.bbox[3]) * 0.5
                distance = float(np.hypot(dcx - pcx, dcy - pcy)) / diagonal
                score = iou + 0.20 * max(0.0, 1.0 - distance)
                if iou >= 0.16 or distance <= 0.45:
                    candidates.append((score, track_id, detection_index))
        candidates.sort(reverse=True)
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        assignments: list[tuple[FaceTrack, FaceDetection]] = []
        for _, track_id, detection_index in candidates:
            if track_id in used_tracks or detection_index in used_detections:
                continue
            track = self.tracks[track_id]
            detection = detections[detection_index]
            track.update(detection)
            used_tracks.add(track_id)
            used_detections.add(detection_index)
            assignments.append((track, detection))

        for detection_index, detection in enumerate(detections):
            if detection_index in used_detections:
                continue
            track = FaceTrack(
                id=self._next_id,
                bbox=detection.bbox,
                landmarks=detection.landmarks,
                max_embeddings=self.max_embeddings,
            )
            self.tracks[track.id] = track
            self._next_id += 1
            assignments.append((track, detection))

        expired = [track_id for track_id, track in self.tracks.items() if track.missed > self.max_age]
        for track_id in expired:
            del self.tracks[track_id]
        return assignments
