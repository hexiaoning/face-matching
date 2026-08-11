from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..matching import MatchResult
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


@dataclass(slots=True)
class Observation:
    bbox: np.ndarray
    detection_score: float
    embedding: np.ndarray | None
    quality: float


@dataclass(slots=True)
class Track:
    id: int
    bbox: np.ndarray
    last_frame: int
    misses: int = 0
    observations: list[tuple[np.ndarray, float]] = field(default_factory=list)
    last_embedding: np.ndarray | None = None
    name: str = "采集中"
    person_id: str | None = None
    id_card: str = ""
    score: float = 0.0
    quality: float = 0.0
    match_history: list[str] = field(default_factory=list)

    def add(self, observation: Observation, frame_index: int) -> None:
        self.bbox = np.asarray(observation.bbox, dtype=np.float32)
        self.last_frame = frame_index
        self.misses = 0
        if observation.embedding is not None:
            embedding = l2_normalize(observation.embedding)
            self.observations.append((embedding, float(observation.quality)))
            self.last_embedding = embedding
            if len(self.observations) > 32:
                self.observations = sorted(self.observations, key=lambda item: item[1], reverse=True)[:24]

    def aggregate(self, top_k: int = 8) -> np.ndarray | None:
        if not self.observations:
            return None
        selected = sorted(self.observations, key=lambda item: item[1], reverse=True)[:top_k]
        matrix = np.vstack([item[0] for item in selected])
        weights = np.square(np.maximum([item[1] for item in selected], 0.05))
        self.quality = float(np.average([item[1] for item in selected], weights=weights))
        return l2_normalize(np.average(matrix, axis=0, weights=weights))

    def apply_match(self, match: MatchResult, consensus: int = 2) -> None:
        self.score = match.score
        self.match_history.append(match.person_id or "")
        self.match_history = self.match_history[-5:]
        if match.accepted:
            votes = self.match_history[-3:].count(match.person_id or "")
            if votes >= consensus:
                self.name = match.name
                self.person_id = match.person_id
                self.id_card = match.id_card
            elif self.person_id is None:
                self.name = "确认中"
        elif self.person_id is None:
            self.name = "陌生人"
            self.id_card = ""


class FaceTracker:
    def __init__(self, max_misses: int = 5) -> None:
        self.max_misses = int(max_misses)
        self.tracks: dict[int, Track] = {}
        self._next_id = 1

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1

    def _association_score(self, track: Track, observation: Observation) -> float:
        overlap = bbox_iou(track.bbox, observation.bbox)
        similarity = -1.0
        if track.last_embedding is not None and observation.embedding is not None:
            similarity = float(track.last_embedding @ l2_normalize(observation.embedding))
        if overlap < 0.08 and similarity < 0.35:
            return -1.0
        return 0.65 * overlap + 0.35 * max(similarity, 0.0)

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
