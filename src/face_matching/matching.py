from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .database import FaceDatabase, GallerySample
from .vision.recognizer import l2_normalize

TARGET_PERSON_ID = "__target_search__"


@dataclass(frozen=True, slots=True)
class MatchResult:
    accepted: bool
    person_id: str | None
    name: str
    id_card: str
    score: float
    second_score: float
    margin: float


@dataclass(frozen=True, slots=True)
class TargetMatchResult:
    """A verification decision for one requested person and one video track."""

    decision: str
    score: float
    best_score: float
    support: int
    observations: int

    @property
    def accepted(self) -> bool:
        return self.decision == "confirmed"

    @property
    def review(self) -> bool:
        return self.decision == "review"


class TargetMatcher:
    """Verify one target from repeated frames instead of doing 1:N identification."""

    def __init__(
        self,
        references: list[np.ndarray],
        threshold: float = 0.18,
        review_threshold: float = 0.12,
        min_support: int = 4,
        top_k: int = 5,
    ) -> None:
        if not references:
            raise ValueError("目标人物至少需要一个有效人脸特征")
        if not 0.0 <= review_threshold < threshold <= 1.0:
            raise ValueError("目标检索阈值无效")
        self.reference_matrix = np.vstack(
            [l2_normalize(item) for item in references]
        ).astype(np.float32)
        self.threshold = float(threshold)
        self.review_threshold = float(review_threshold)
        self.min_support = max(2, int(min_support))
        self.top_k = max(1, int(top_k))

    def match(self, observations: list[tuple[np.ndarray, float]]) -> TargetMatchResult:
        if not observations:
            return TargetMatchResult("collecting", 0.0, 0.0, 0, 0)
        matrix = np.vstack([l2_normalize(item[0]) for item in observations]).astype(
            np.float32
        )
        if matrix.shape[1] != self.reference_matrix.shape[1]:
            raise ValueError("目标照片与视频人脸的特征维度不一致")
        qualities = np.asarray(
            [max(float(item[1]), 0.05) for item in observations], dtype=np.float32
        )
        # Keep alternate reference alignments separate. Taking the best view
        # avoids averaging away a useful pose while repeated video support
        # prevents one weak coincidental frame from becoming a hit.
        per_observation = np.max(matrix @ self.reference_matrix.T, axis=1)
        count = min(self.top_k, len(per_observation))
        selected = np.argsort(per_observation)[::-1][:count]
        best_score = float(per_observation[selected[0]])
        supported_mean = float(
            np.average(per_observation[selected], weights=qualities[selected] ** 2)
        )
        score = 0.60 * best_score + 0.40 * supported_mean
        support = int(np.count_nonzero(per_observation >= self.threshold))
        if score >= self.threshold and support >= self.min_support:
            decision = "confirmed"
        elif best_score >= self.review_threshold:
            decision = "review"
        elif len(observations) >= self.min_support:
            decision = "rejected"
        else:
            decision = "collecting"
        return TargetMatchResult(decision, score, best_score, support, len(observations))


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
        self.sample_qualities = np.empty(0, dtype=np.float32)
        self.identity_sample_indices: list[np.ndarray] = []
        self.centroid_matrix = np.empty((0, 0), dtype=np.float32)
        self.refresh()

    @property
    def person_count(self) -> int:
        return len(self.identities)

    def refresh(self) -> None:
        grouped: dict[str, list[GallerySample]] = {}
        for sample in self.database.list_gallery(self.model_id):
            grouped.setdefault(sample.person_id, []).append(sample)
        identities: list[_Identity] = []
        sample_rows: list[np.ndarray] = []
        sample_owners: list[int] = []
        sample_qualities: list[float] = []
        identity_sample_indices: list[np.ndarray] = []
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
            start = len(sample_rows)
            sample_rows.extend(matrix)
            sample_owners.extend([identity_index] * len(matrix))
            sample_qualities.extend(max(float(item.quality), 0.05) for item in samples)
            identity_sample_indices.append(np.arange(start, start + len(matrix), dtype=np.intp))
            centroid_rows.append(centroid)
        self.identities = identities
        self.identity_sample_indices = identity_sample_indices
        if identities:
            self.sample_matrix = np.vstack(sample_rows).astype(np.float32, copy=False)
            self.sample_owners = np.asarray(sample_owners, dtype=np.intp)
            self.sample_qualities = np.asarray(sample_qualities, dtype=np.float32)
            self.centroid_matrix = np.vstack(centroid_rows).astype(np.float32, copy=False)
        else:
            self.sample_matrix = np.empty((0, 0), dtype=np.float32)
            self.sample_owners = np.empty(0, dtype=np.intp)
            self.sample_qualities = np.empty(0, dtype=np.float32)
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
        top_means = np.empty(len(self.identities), dtype=np.float32)
        for identity_index, indices in enumerate(self.identity_sample_indices):
            values = sample_scores[indices]
            count = min(3, len(indices))
            top_local = np.argpartition(values, -count)[-count:]
            selected = indices[top_local]
            top_means[identity_index] = np.average(
                sample_scores[selected], weights=self.sample_qualities[selected]
            )
        # Best-view matching preserves recall for pose changes; the top-three
        # mean and centroid make a single accidental enrollment less decisive.
        scores = 0.70 * best_samples + 0.25 * top_means + 0.05 * centroid_scores
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
