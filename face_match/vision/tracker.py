from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from face_match.domain import FaceObservation, TrackState


def intersection_over_union(first: np.ndarray, second: np.ndarray) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
    second_area = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
    return intersection / max(first_area + second_area - intersection, 1e-8)


def robust_aggregate(samples: Sequence[tuple[float, np.ndarray]]) -> np.ndarray | None:
    if not samples:
        return None
    matrix = np.stack([np.asarray(vector, dtype=np.float32).reshape(-1) for _, vector in samples])
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)
    qualities = np.array([max(0.05, quality) for quality, _ in samples], dtype=np.float32)
    if len(samples) >= 3:
        cosine = matrix @ matrix.T
        medoid = int(np.argmax(cosine.mean(axis=1)))
        consistent = cosine[medoid] >= 0.20
        if np.count_nonzero(consistent) >= 2:
            matrix = matrix[consistent]
            qualities = qualities[consistent]
    weights = qualities**2
    aggregate = np.average(matrix, axis=0, weights=weights)
    norm = float(np.linalg.norm(aggregate))
    return (aggregate / norm).astype(np.float32) if norm > 1e-8 else None


class FaceTracker:
    def __init__(self, maximum_samples: int = 12, maximum_missed: int = 12) -> None:
        self.maximum_samples = maximum_samples
        self.maximum_missed = maximum_missed
        self._tracks: dict[int, TrackState] = {}
        self._next_id = 1
        self._frame_index = 0

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self._frame_index = 0

    def update(self, observations: Sequence[FaceObservation]) -> list[TrackState]:
        self._frame_index += 1
        track_ids = list(self._tracks)
        candidates: list[tuple[float, int, int]] = []
        for track_id in track_ids:
            track = self._tracks[track_id]
            for observation_index, observation in enumerate(observations):
                iou = intersection_over_union(track.bbox, observation.detection.bbox)
                similarity = -1.0
                if track.aggregate is not None and observation.embedding is not None:
                    similarity = float(track.aggregate @ observation.embedding.reshape(-1))
                if iou >= 0.10 or similarity >= 0.52:
                    association = 0.78 * iou + 0.22 * max(0.0, similarity)
                    candidates.append((association, track_id, observation_index))
        candidates.sort(reverse=True)
        assigned_tracks: set[int] = set()
        assigned_observations: set[int] = set()
        current: list[TrackState] = []
        for _, track_id, observation_index in candidates:
            if track_id in assigned_tracks or observation_index in assigned_observations:
                continue
            track = self._tracks[track_id]
            self._apply_observation(track, observations[observation_index])
            assigned_tracks.add(track_id)
            assigned_observations.add(observation_index)
            current.append(track)

        for track_id in track_ids:
            if track_id not in assigned_tracks:
                self._tracks[track_id].missed += 1
        for observation_index, observation in enumerate(observations):
            if observation_index in assigned_observations:
                continue
            track = TrackState(
                track_id=self._next_id,
                bbox=observation.detection.bbox.copy(),
                landmarks=observation.detection.landmarks.copy(),
                quality=observation.quality,
                last_frame=self._frame_index,
            )
            self._next_id += 1
            self._tracks[track.track_id] = track
            self._apply_observation(track, observation)
            current.append(track)

        stale = [
            track_id
            for track_id, track in self._tracks.items()
            if track.missed > self.maximum_missed
        ]
        for track_id in stale:
            del self._tracks[track_id]
        return sorted(current, key=lambda item: item.track_id)

    def _apply_observation(self, track: TrackState, observation: FaceObservation) -> None:
        track.bbox = observation.detection.bbox.copy()
        track.landmarks = observation.detection.landmarks.copy()
        track.quality = observation.quality
        track.last_frame = self._frame_index
        track.missed = 0
        if observation.embedding is not None:
            track.embeddings.append((observation.quality.overall, observation.embedding.copy()))
            track.embeddings.sort(key=lambda item: item[0], reverse=True)
            del track.embeddings[self.maximum_samples :]
            track.aggregate = robust_aggregate(track.embeddings)
