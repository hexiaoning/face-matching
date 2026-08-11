from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from .database import FaceDatabase, GallerySample
from .privacy import mask_id_card
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
        return mask_id_card(self.id_card)


class GalleryMatcher:
    def __init__(self, database: FaceDatabase, model_id: str) -> None:
        self.database = database
        self.model_id = model_id
        self._lock = threading.RLock()
        self._samples: list[GallerySample] = []
        self._matrix = np.empty((0, 0), dtype=np.float32)
        self._person_indices: dict[str, list[int]] = {}
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
        with self._lock:
            self._samples = samples
            self._matrix = matrix
            self._person_indices = people
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
                score = float(0.72 * top_scores[0] + 0.28 * weighted_mean)
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
