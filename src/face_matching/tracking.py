from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .domain import FaceObservation, MatchResult, TrackView, normalize_embedding
from .gallery import GalleryIndex


def bbox_iou(first: np.ndarray, second: np.ndarray) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2] - first[0])) * max(
        0.0, float(first[3] - first[1])
    )
    second_area = max(0.0, float(second[2] - second[0])) * max(
        0.0, float(second[3] - second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


@dataclass(slots=True)
class FaceTrack:
    id: int
    bbox: np.ndarray
    last_frame: int
    max_embeddings: int
    misses: int = 0
    quality: float = 0.0
    samples: deque[tuple[np.ndarray, float]] = field(default_factory=deque)
    candidate: MatchResult | None = None
    candidate_hits: int = 0
    confirmed: MatchResult | None = None
    event_emitted: bool = False

    def add_sample(self, embedding: np.ndarray, quality: float) -> None:
        self.samples.append((normalize_embedding(embedding), max(0.05, float(quality))))
        while len(self.samples) > self.max_embeddings:
            self.samples.popleft()

    def aggregate(self) -> np.ndarray | None:
        if not self.samples:
            return None
        vectors = np.stack([item[0] for item in self.samples])
        # Squaring gives the sharp, near-frontal frames more influence without discarding others.
        weights = np.asarray([item[1] ** 2 for item in self.samples], dtype=np.float32)
        return normalize_embedding(np.average(vectors, axis=0, weights=weights))


class MultiFaceTracker:
    def __init__(self, max_embeddings: int = 12, ttl_frames: int = 18) -> None:
        self.max_embeddings = max_embeddings
        self.ttl_frames = ttl_frames
        self._tracks: dict[int, FaceTrack] = {}
        self._next_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def update(
        self,
        observations: list[FaceObservation],
        gallery: GalleryIndex,
        frame_index: int,
        threshold: float,
        minimum_quality: float,
        confirmation_hits: int,
    ) -> tuple[list[TrackView], list[FaceTrack]]:
        assignments = self._assign(observations)
        matched_tracks = set(assignments.values())
        for track_id, track in list(self._tracks.items()):
            if track_id not in matched_tracks:
                track.misses += 1
                if track.misses > self.ttl_frames:
                    del self._tracks[track_id]

        for observation_index, observation in enumerate(observations):
            track_id = assignments.get(observation_index)
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = FaceTrack(
                    id=track_id,
                    bbox=observation.bbox.copy(),
                    last_frame=frame_index,
                    max_embeddings=self.max_embeddings,
                )
            track = self._tracks[track_id]
            track.bbox = 0.65 * observation.bbox + 0.35 * track.bbox
            track.last_frame = frame_index
            track.misses = 0
            track.quality = observation.quality
            if observation.embedding is not None and observation.quality >= minimum_quality:
                track.add_sample(observation.embedding, observation.quality)
                aggregate = track.aggregate()
                result = gallery.match(aggregate) if aggregate is not None else None
                if result is not None and result.score >= threshold:
                    if track.candidate and track.candidate.person_id == result.person_id:
                        track.candidate_hits += 1
                    else:
                        track.candidate = result
                        track.candidate_hits = 1
                    # Retain the newest score while counting consistent identities.
                    track.candidate = result
                    if track.candidate_hits >= confirmation_hits:
                        if track.confirmed is None or track.confirmed.person_id != result.person_id:
                            track.event_emitted = False
                        track.confirmed = result
                else:
                    track.candidate = None
                    track.candidate_hits = 0

        visible_ids = set(assignments.values())
        # Include tracks created during this update.
        visible_ids.update(
            track.id for track in self._tracks.values() if track.last_frame == frame_index
        )
        views: list[TrackView] = []
        events: list[FaceTrack] = []
        for track_id in sorted(visible_ids):
            track = self._tracks.get(track_id)
            if track is None:
                continue
            result = track.confirmed
            if result:
                label = result.name
                score = result.score
                person_id = result.person_id
            elif track.candidate:
                label = f"确认中 {track.candidate_hits}/{confirmation_hits}"
                score = track.candidate.score
                person_id = None
            elif track.quality < minimum_quality:
                label = "质量不足"
                score = 0.0
                person_id = None
            else:
                label = "未识别"
                score = 0.0
                person_id = None
            views.append(
                TrackView(
                    track_id=track.id,
                    bbox=tuple(float(value) for value in track.bbox),
                    label=label,
                    score=float(score),
                    quality=float(track.quality),
                    confirmed=result is not None,
                    person_id=person_id,
                    id_number=result.id_number if result else "",
                )
            )
            if result is not None and not track.event_emitted:
                track.event_emitted = True
                events.append(track)
        return views, events

    def _assign(self, observations: list[FaceObservation]) -> dict[int, int]:
        candidates: list[tuple[float, int, int]] = []
        for observation_index, observation in enumerate(observations):
            for track_id, track in self._tracks.items():
                overlap = bbox_iou(observation.bbox, track.bbox)
                if overlap >= 0.20:
                    candidates.append((overlap, observation_index, track_id))
        candidates.sort(reverse=True)
        assigned_observations: set[int] = set()
        assigned_tracks: set[int] = set()
        result: dict[int, int] = {}
        for _, observation_index, track_id in candidates:
            if observation_index in assigned_observations or track_id in assigned_tracks:
                continue
            result[observation_index] = track_id
            assigned_observations.add(observation_index)
            assigned_tracks.add(track_id)
        return result
