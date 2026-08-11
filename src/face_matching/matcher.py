from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from .database import FaceDatabase, GallerySample
from .recognizer import l2_normalize


@dataclass(frozen=True, slots=True)
class MatchResult:
    person_id: str | None
    name: str
    id_card: str
    score: float
    margin: float
    accepted: bool

    @property
    def masked_id_card(self) -> str:
        if not self.id_card:
            return ""
        if len(self.id_card) <= 7:
            return self.id_card[:2] + "***"
        return f"{self.id_card[:3]}********{self.id_card[-4:]}"


class GalleryMatcher:
    def __init__(self, database: FaceDatabase, model_id: str) -> None:
        self.database = database
        self.model_id = model_id
        self._lock = threading.RLock()
        self._samples: list[GallerySample] = []
        self._matrix = np.empty((0, 0), dtype=np.float32)
        self._person_indices: dict[str, list[int]] = {}
        self._person_centroids: dict[str, np.ndarray] = {}
        self._revision = 0
        self.reload()

    @property
    def sample_count(self) -> int:
        with self._lock:
            return len(self._samples)

    @property
    def person_count(self) -> int:
        with self._lock:
            return len(self._person_indices)

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    def reload(self) -> None:
        samples = self.database.gallery(self.model_id)
        if samples:
            dimensions = {sample.embedding.size for sample in samples}
            if len(dimensions) != 1:
                samples = []
        matrix = (
            np.stack([sample.embedding for sample in samples]).astype(np.float32)
            if samples else np.empty((0, 0), dtype=np.float32)
        )
        people: dict[str, list[int]] = {}
        for index, sample in enumerate(samples):
            people.setdefault(sample.person_id, []).append(index)
        centroids: dict[str, np.ndarray] = {}
        for person_id, indices in people.items():
            rows = matrix[indices]
            weights = np.asarray(
                [max(samples[index].quality, 0.05) for index in indices],
                dtype=np.float32,
            )
            centroid = np.average(rows, axis=0, weights=weights)
            if float(np.linalg.norm(centroid)) <= 1e-12:
                centroid = rows[int(np.argmax(weights))]
            centroids[person_id] = l2_normalize(centroid.reshape(1, -1))[0]
        with self._lock:
            self._samples = samples
            self._matrix = matrix
            self._person_indices = people
            self._person_centroids = centroids
            self._revision += 1

    def match(self, embedding: np.ndarray, threshold: float, min_margin: float) -> MatchResult:
        query = l2_normalize(np.asarray(embedding, dtype=np.float32).reshape(1, -1))[0]
        with self._lock:
            if not self._samples or self._matrix.shape[1] != query.size:
                return MatchResult(None, "未知", "", 0.0, 0.0, False)
            similarities = self._matrix @ query
            ranked: list[tuple[float, str, int]] = []
            for person_id, indices in self._person_indices.items():
                order = np.asarray(indices)[np.argsort(similarities[indices])[::-1]]
                top_indices = order[: min(3, len(order))]
                top_scores = similarities[top_indices]
                quality_weights = np.asarray(
                    [max(self._samples[int(index)].quality, 0.05) for index in top_indices],
                    dtype=np.float32,
                )
                weighted_mean = float(np.average(top_scores, weights=quality_weights))
                centroid_score = float(self._person_centroids[person_id] @ query)
                # A strong individual pose remains dominant, while the
                # quality-weighted person centroid rewards agreement across
                # several enrollment photos.
                score = float(
                    0.62 * top_scores[0]
                    + 0.23 * weighted_mean
                    + 0.15 * centroid_score
                )
                ranked.append((score, person_id, indices[0]))
            ranked.sort(reverse=True, key=lambda item: item[0])
            best_score, person_id, sample_index = ranked[0]
            second_score = ranked[1][0] if len(ranked) > 1 else -1.0
            margin = best_score - second_score
            sample = self._samples[sample_index]
            accepted = best_score >= threshold and margin >= min_margin
            return MatchResult(
                person_id if accepted else None,
                sample.name if accepted else "未知",
                sample.id_card if accepted else "",
                best_score,
                margin,
                accepted,
            )
