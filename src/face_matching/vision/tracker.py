from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..matching import MatchResult, TargetMatchResult, TargetObservation
from .recognizer import l2_normalize


def bbox_iou(first: np.ndarray, second: np.ndarray) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
    second_area = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
    return intersection / max(first_area + second_area - intersection, 1e-9)


def robust_aggregate(
    observations: list[tuple[np.ndarray, float]],
    top_k: int = 8,
    min_similarity: float = 0.20,
) -> tuple[np.ndarray, float] | None:
    """Fuse high-quality samples while rejecting a likely identity-switch outlier."""
    if not observations:
        return None
    selected = sorted(observations, key=lambda item: item[1], reverse=True)[:top_k]
    matrix = np.vstack([l2_normalize(item[0]) for item in selected]).astype(np.float32)
    qualities = np.asarray([max(float(item[1]), 0.05) for item in selected], dtype=np.float32)
    if len(matrix) >= 3:
        cosine = matrix @ matrix.T
        medoid = int(np.argmax(np.average(cosine, axis=1, weights=qualities)))
        consistent = cosine[medoid] >= float(min_similarity)
        if np.count_nonzero(consistent) >= 2:
            matrix = matrix[consistent]
            qualities = qualities[consistent]
    weights = qualities**2
    aggregate = l2_normalize(np.average(matrix, axis=0, weights=weights))
    quality = float(np.average(qualities, weights=weights))
    return aggregate, quality


@dataclass(slots=True)
class Observation:
    bbox: np.ndarray
    detection_score: float
    embedding: np.ndarray | None
    quality: float
    alternate_embeddings: tuple[np.ndarray, ...] = ()
    timestamp: float = 0.0
    pose: float = 0.0


@dataclass(slots=True)
class Track:
    id: int
    bbox: np.ndarray
    last_frame: int
    misses: int = 0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    observations: list[tuple[np.ndarray, float]] = field(default_factory=list)
    target_observations: list[TargetObservation] = field(default_factory=list)
    last_embedding: np.ndarray | None = None
    name: str = "采集中"
    person_id: str | None = None
    id_card: str = ""
    score: float = 0.0
    quality: float = 0.0
    match_history: list[str] = field(default_factory=list)
    embedding_version: int = 0
    matched_embedding_version: int = 0
    candidate_id: str | None = None
    candidate_count: int = 0
    unknown_count: int = 0
    decision: str = "low"
    support: int = 0
    best_score: float = 0.0
    evidence: int = 0

    def add(self, observation: Observation, frame_index: int) -> None:
        new_bbox = np.asarray(observation.bbox, dtype=np.float32)
        delta = new_bbox - self.bbox
        self.velocity = self.velocity * 0.65 + delta * 0.35
        self.bbox = new_bbox
        self.last_frame = frame_index
        self.misses = 0
        if observation.embedding is not None:
            embedding = l2_normalize(observation.embedding)
            self.observations.append((embedding, float(observation.quality)))
            variants = (embedding,) + tuple(
                l2_normalize(item) for item in observation.alternate_embeddings
            )
            self.target_observations.append(
                TargetObservation(
                    variants,
                    float(observation.quality),
                    float(observation.timestamp),
                    float(observation.pose),
                )
            )
            self.last_embedding = embedding
            self.quality = max(self.quality, float(observation.quality))
            self.embedding_version += 1
            if len(self.observations) > 64:
                ranked = sorted(
                    range(len(self.observations)),
                    key=lambda index: self.observations[index][1],
                    reverse=True,
                )
                keep = set(ranked[:32])
                keep.update(range(max(0, len(self.observations) - 16), len(self.observations)))
                ordered = sorted(keep)
                self.observations = [self.observations[index] for index in ordered]
                self.target_observations = [
                    self.target_observations[index] for index in ordered
                ]

    def predict(self) -> np.ndarray:
        horizon = 1.0 + 0.25 * min(self.misses, 2)
        return self.bbox + self.velocity * horizon

    def aggregate(self, top_k: int = 8, min_similarity: float = 0.20) -> np.ndarray | None:
        result = robust_aggregate(self.observations, top_k, min_similarity)
        if result is None:
            return None
        aggregate, self.quality = result
        return aggregate

    def apply_match(self, match: MatchResult, consensus: int = 2) -> None:
        if self.matched_embedding_version == self.embedding_version:
            return
        self.matched_embedding_version = self.embedding_version
        self.score = match.score
        self.match_history.append(match.person_id or "")
        self.match_history = self.match_history[-5:]
        if match.accepted and match.person_id:
            self.unknown_count = 0
            if self.person_id is not None and self.person_id != match.person_id:
                self.name = "确认中"
                self.person_id = None
                self.id_card = ""
            if self.candidate_id == match.person_id:
                self.candidate_count += 1
            else:
                self.candidate_id = match.person_id
                self.candidate_count = 1
            if self.candidate_count >= consensus:
                self.name = match.name
                self.person_id = match.person_id
                self.id_card = match.id_card
            elif self.person_id is None:
                self.name = "确认中"
        else:
            self.unknown_count += 1
            self.candidate_id = None
            self.candidate_count = 0
            if self.unknown_count >= 3 or self.person_id is None:
                self.name = "陌生人"
                self.person_id = None
                self.id_card = ""

    def apply_target_match(self, match: TargetMatchResult, target_name: str) -> None:
        if self.matched_embedding_version == self.embedding_version:
            return
        first_result = self.matched_embedding_version <= 0
        self.matched_embedding_version = self.embedding_version
        self.score = match.score if first_result else max(self.score, match.score)
        self.best_score = max(self.best_score, match.best_score)
        self.support = max(self.support, match.support)
        self.evidence = max(self.evidence, match.evidence)
        priority = {"low": 0, "review": 1, "confirmed": 2}
        if priority.get(match.decision, 0) > priority.get(self.decision, 0):
            self.decision = match.decision
        if self.decision == "confirmed":
            self.name = target_name
            self.person_id = "__search_target__"
            self.id_card = ""
        elif self.decision == "review":
            self.name = "待复核"
            self.person_id = None
            self.id_card = ""
        else:
            self.name = "低置信度"
            self.person_id = None
            self.id_card = ""

    def invalidate_identity(self) -> None:
        self.name = "采集中"
        self.person_id = None
        self.id_card = ""
        self.score = 0.0
        self.match_history.clear()
        self.candidate_id = None
        self.candidate_count = 0
        self.unknown_count = 0
        self.decision = "low"
        self.support = 0
        self.best_score = 0.0
        self.evidence = 0
        self.matched_embedding_version = -1


class FaceTracker:
    def __init__(self, max_misses: int = 5) -> None:
        self.max_misses = int(max_misses)
        self.tracks: dict[int, Track] = {}
        self._next_id = 1

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1

    def invalidate_identities(self) -> None:
        for track in self.tracks.values():
            track.invalidate_identity()

    def _association_score(self, track: Track, observation: Observation) -> float:
        predicted = track.predict()
        overlap = bbox_iou(predicted, observation.bbox)
        similarity = -1.0
        if track.last_embedding is not None and observation.embedding is not None:
            similarity = float(track.last_embedding @ l2_normalize(observation.embedding))
        pcx = (predicted[0] + predicted[2]) * 0.5
        pcy = (predicted[1] + predicted[3]) * 0.5
        ocx = (observation.bbox[0] + observation.bbox[2]) * 0.5
        ocy = (observation.bbox[1] + observation.bbox[3]) * 0.5
        diagonal = max(float(np.hypot(predicted[2] - predicted[0], predicted[3] - predicted[1])), 1.0)
        distance = float(np.hypot(ocx - pcx, ocy - pcy)) / diagonal
        if overlap < 0.06 and similarity < 0.35 and distance > 0.55:
            return -1.0
        proximity = max(0.0, 1.0 - distance)
        return 0.55 * overlap + 0.25 * max(similarity, 0.0) + 0.20 * proximity

    def update(self, observations: list[Observation], frame_index: int) -> list[Track]:
        for track in self.tracks.values():
            track.misses += 1
        candidates: list[tuple[float, int, int]] = []
        for track_id, track in self.tracks.items():
            for observation_index, observation in enumerate(observations):
                score = self._association_score(track, observation)
                if score >= 0.10:
                    candidates.append((score, track_id, observation_index))
        candidates.sort(reverse=True)
        used_tracks: set[int] = set()
        used_observations: set[int] = set()
        for _, track_id, observation_index in candidates:
            if track_id in used_tracks or observation_index in used_observations:
                continue
            self.tracks[track_id].add(observations[observation_index], frame_index)
            used_tracks.add(track_id)
            used_observations.add(observation_index)
        for index, observation in enumerate(observations):
            if index in used_observations:
                continue
            track = Track(self._next_id, np.asarray(observation.bbox, dtype=np.float32), frame_index)
            track.add(observation, frame_index)
            self.tracks[track.id] = track
            self._next_id += 1
        expired = [track_id for track_id, track in self.tracks.items() if track.misses > self.max_misses]
        for track_id in expired:
            del self.tracks[track_id]
        return sorted(self.tracks.values(), key=lambda item: item.id)
