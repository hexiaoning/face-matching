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
    DIFFICULT_LVFACE_THRESHOLD = 0.19
    DIFFICULT_MIN_SUPPORT = 4

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

    @staticmethod
    def _empty_match(name: str = "陌生人") -> MatchResult:
        return MatchResult(False, None, name, "", 0.0, 0.0, 0.0)

    def _identity_scores(self, embedding: np.ndarray) -> np.ndarray:
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
        return 0.70 * best_samples + 0.25 * top_means + 0.05 * centroid_scores

    def _result(
        self,
        scores: np.ndarray,
        threshold: float,
        min_margin: float,
    ) -> MatchResult:
        order = np.argsort(scores)[::-1]
        best_index = int(order[0])
        best_score = float(scores[best_index])
        best_identity = self.identities[best_index]
        second_score = float(scores[int(order[1])]) if len(order) > 1 else -1.0
        margin = best_score - second_score if len(order) > 1 else 1.0
        accepted = best_score >= threshold and margin >= min_margin
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

    def match(self, embedding: np.ndarray) -> MatchResult:
        if not self.identities:
            return self._empty_match("未录入人员")
        return self._result(
            self._identity_scores(embedding),
            self.threshold,
            self.min_margin,
        )

    def match_track(
        self,
        aggregate: np.ndarray,
        observations: list[tuple[tuple[np.ndarray, ...], float]],
    ) -> MatchResult:
        """Match the gallery, with a conservative multi-frame LVFace fallback.

        The normal configured threshold remains the primary open-set decision.
        A difficult surveillance track below that threshold is accepted only
        when at least four independently processed frames select the same
        gallery identity and preserve the first/second-person margin. Alignment
        variants from one frame can improve its score but never its support.
        """
        normal = self.match(aggregate)
        if (
            normal.accepted
            or not self.identities
            or not self.model_id.startswith("lvface-b-")
            or len(observations) < self.DIFFICULT_MIN_SUPPORT
        ):
            return normal

        frame_rows: list[np.ndarray] = []
        qualities: list[float] = []
        for variants, quality in observations:
            if not variants:
                continue
            variant_rows = np.vstack(
                [self._identity_scores(embedding) for embedding in variants]
            )
            frame_rows.append(np.max(variant_rows, axis=0))
            qualities.append(max(float(quality), 0.05))
        if len(frame_rows) < self.DIFFICULT_MIN_SUPPORT:
            return normal

        frame_scores = np.vstack(frame_rows).astype(np.float32, copy=False)
        quality_weights = np.asarray(qualities, dtype=np.float32) ** 2
        evidence_scores = np.empty(len(self.identities), dtype=np.float32)
        top_k = min(5, len(frame_scores))
        for identity_index in range(len(self.identities)):
            values = frame_scores[:, identity_index]
            selected = np.argsort(values)[::-1][:top_k]
            supported_mean = float(
                np.average(values[selected], weights=quality_weights[selected])
            )
            evidence_scores[identity_index] = 0.60 * float(values[selected[0]]) + 0.40 * supported_mean

        order = np.argsort(evidence_scores)[::-1]
        best_index = int(order[0])
        per_frame_winner = np.argmax(frame_scores, axis=1)
        best_per_frame = frame_scores[:, best_index]
        if len(self.identities) > 1:
            competitor_per_frame = np.max(
                np.delete(frame_scores, best_index, axis=1), axis=1
            )
            frame_margin = best_per_frame - competitor_per_frame
        else:
            frame_margin = np.ones(len(frame_scores), dtype=np.float32)
        difficult_margin = max(self.min_margin, 0.08)
        support = int(
            np.count_nonzero(
                (per_frame_winner == best_index)
                & (best_per_frame >= self.DIFFICULT_LVFACE_THRESHOLD)
                & (frame_margin >= difficult_margin)
            )
        )
        result = self._result(
            evidence_scores,
            self.DIFFICULT_LVFACE_THRESHOLD,
            difficult_margin,
        )
        if result.accepted and support >= self.DIFFICULT_MIN_SUPPORT:
            return result
        return MatchResult(
            False,
            None,
            "陌生人",
            "",
            result.score,
            result.second_score,
            result.margin,
        )
