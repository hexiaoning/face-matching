from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from face_match.domain import EmbeddingRecord, MatchResult


class MultiTemplateMatcher:
    """Open-set 1:N matcher that keeps every enrollment photo as a template."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._matrix = np.empty((0, 0), dtype=np.float32)
        self._records: list[EmbeddingRecord] = []

    def refresh(self, records: Sequence[EmbeddingRecord]) -> None:
        with self._lock:
            self._records = list(records)
            if records:
                dimensions = {record.embedding.size for record in records}
                if len(dimensions) != 1:
                    raise ValueError("数据库中存在维度不一致的人脸特征")
                self._matrix = np.ascontiguousarray(
                    np.stack([record.embedding for record in records]), dtype=np.float32
                )
            else:
                self._matrix = np.empty((0, 0), dtype=np.float32)

    @property
    def template_count(self) -> int:
        with self._lock:
            return len(self._records)

    @property
    def person_count(self) -> int:
        with self._lock:
            return len({record.person_id for record in self._records})

    def match(self, query: np.ndarray, threshold: float, ambiguity_margin: float) -> MatchResult:
        vector = np.asarray(query, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-8 or not np.isfinite(vector).all():
            return MatchResult.unknown("查询特征无效")
        vector = vector / norm
        with self._lock:
            if not self._records:
                return MatchResult.unknown("人员库为空")
            if self._matrix.shape[1] != vector.size:
                return MatchResult.unknown("人员库特征版本不兼容")
            similarities = self._matrix @ vector
            grouped: dict[int, list[tuple[float, EmbeddingRecord]]] = defaultdict(list)
            for score, record in zip(similarities.tolist(), self._records):
                grouped[record.person_id].append((float(score), record))

        candidates: list[tuple[float, EmbeddingRecord]] = []
        for templates in grouped.values():
            templates.sort(key=lambda item: item[0], reverse=True)
            best_score, best_record = templates[0]
            top = templates[: min(3, len(templates))]
            quality_weights = np.array(
                [max(0.2, item[1].quality) for item in top], dtype=np.float32
            )
            top_scores = np.array([item[0] for item in top], dtype=np.float32)
            weighted_mean = float(np.average(top_scores, weights=quality_weights))
            fused_score = 0.72 * best_score + 0.28 * weighted_mean
            candidates.append((fused_score, best_record))
        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_record = candidates[0]
        second_score = candidates[1][0] if len(candidates) > 1 else -1.0
        if best_score < threshold:
            return MatchResult.unknown("低于相似度阈值", best_score, second_score)
        if len(candidates) > 1 and best_score - second_score < ambiguity_margin:
            return MatchResult.unknown("前两名过于接近", best_score, second_score)
        return MatchResult(
            accepted=True,
            person_id=best_record.person_id,
            name=best_record.person_name,
            id_number=best_record.id_number,
            score=best_score,
            second_score=second_score,
            reason="匹配通过",
        )
