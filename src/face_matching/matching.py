from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .database import FaceDatabase, GallerySample
from .vision.recognizer import l2_normalize


@dataclass(frozen=True, slots=True)
class MatchResult:
    accepted: bool
    person_id: str | None
    name: str
    id_card: str
    score: float
    second_score: float
    margin: float


@dataclass(slots=True)
class _Identity:
    person_id: str
    name: str
    id_card: str


class GalleryMatcher:
    def __init__(
        self,
        database: FaceDatabase,
        model_id: str,
        threshold: float = 0.45,
        min_margin: float = 0.06,
    ) -> None:
        self.database = database
        self.model_id = model_id
        self.threshold = float(threshold)
        self.min_margin = float(min_margin)
        self.identities: list[_Identity] = []
        self.sample_matrix = np.empty((0, 0), dtype=np.float32)
        self.sample_owners = np.empty(0, dtype=np.intp)
        self.centroid_matrix = np.empty((0, 0), dtype=np.float32)
        self.refresh()

    def refresh(self) -> None:
        grouped: dict[str, list[GallerySample]] = {}
        for sample in self.database.list_gallery(self.model_id):
            grouped.setdefault(sample.person_id, []).append(sample)
        identities: list[_Identity] = []
        sample_rows: list[np.ndarray] = []
        sample_owners: list[int] = []
        centroid_rows: list[np.ndarray] = []
        for samples in grouped.values():
            first = samples[0]
            matrix = np.vstack([l2_normalize(item.embedding) for item in samples]).astype(np.float32)
            centroid = l2_normalize(np.average(
                matrix,
                axis=0,
                weights=np.maximum([item.quality for item in samples], 0.1),
            ))
            identity_index = len(identities)
            identities.append(_Identity(first.person_id, first.name, first.id_card))
            sample_rows.extend(matrix)
            sample_owners.extend([identity_index] * len(matrix))
            centroid_rows.append(centroid)
        self.identities = identities
        if identities:
            self.sample_matrix = np.vstack(sample_rows).astype(np.float32, copy=False)
            self.sample_owners = np.asarray(sample_owners, dtype=np.intp)
            self.centroid_matrix = np.vstack(centroid_rows).astype(np.float32, copy=False)
        else:
            self.sample_matrix = np.empty((0, 0), dtype=np.float32)
            self.sample_owners = np.empty(0, dtype=np.intp)
            self.centroid_matrix = np.empty((0, 0), dtype=np.float32)

    def match(self, embedding: np.ndarray) -> MatchResult:
        if not self.identities:
            return MatchResult(False, None, "未录入人员", "", 0.0, 0.0, 0.0)
        query = l2_normalize(embedding)
        if self.sample_matrix.shape[1] != query.size:
            raise ValueError("查询特征与当前人员库模型维度不一致，请重建特征")
        sample_scores = self.sample_matrix @ query
        best_samples = np.full(len(self.identities), -1.0, dtype=np.float32)
        np.maximum.at(best_samples, self.sample_owners, sample_scores)
        centroid_scores = self.centroid_matrix @ query
        scores = 0.65 * best_samples + 0.35 * centroid_scores
        order = np.argsort(scores)[::-1]
        best_index = int(order[0])
        best_score = float(scores[best_index])
        best_identity = self.identities[best_index]
        second_score = float(scores[int(order[1])]) if len(order) > 1 else -1.0
        margin = best_score - second_score if len(order) > 1 else 1.0
        accepted = best_score >= self.threshold and margin >= self.min_margin
        if not accepted:
            return MatchResult(False, None, "陌生人", "", best_score, second_score, margin)
        return MatchResult(
            True,
            best_identity.person_id,
            best_identity.name,
            best_identity.id_card,
            best_score,
            second_score,
            margin,
        )
