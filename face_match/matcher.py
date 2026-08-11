"""相似度匹配：probe embedding vs 人员库（每人多张照片取最大相似度）。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .database import Person


@dataclass
class MatchResult:
    person: Person
    score: float          # 该人的最高相似度
    photo_scores: list[float]  # 该人各照片的相似度（调试用）


class GalleryIndex:
    """人员库 embedding 的内存索引，库变更后 rebuild。"""

    def __init__(self) -> None:
        self.persons: list[Person] = []
        self.person_ids: list[int] = []     # 每条 embedding 对应的 person 下标
        self.embeddings: np.ndarray | None = None  # (N, 512) 已 L2 归一化

    def rebuild(self, gallery: list[tuple[Person, np.ndarray]]) -> None:
        persons: dict[int, Person] = {}
        pids: list[int] = []
        embs: list[np.ndarray] = []
        for person, emb in gallery:
            if person.id not in persons:
                persons[person.id] = person
            pids.append(person.id)
            embs.append(emb)
        self.persons = list(persons.values())
        self.person_ids = pids
        self.embeddings = np.stack(embs).astype(np.float32) if embs else None

    def __len__(self) -> int:
        return 0 if self.embeddings is None else len(self.embeddings)

    def match(self, probe: np.ndarray, threshold: float, top_k: int = 5) -> list[MatchResult]:
        """余弦相似度（embedding 均已归一化，点积即相似度）。"""
        if self.embeddings is None or len(self.embeddings) == 0:
            return []
        sims = self.embeddings @ probe.astype(np.float32)
        by_person: dict[int, list[float]] = {}
        for i, pid in enumerate(self.person_ids):
            by_person.setdefault(pid, []).append(float(sims[i]))
        pmap = {p.id: p for p in self.persons}
        results = [
            MatchResult(person=pmap[pid], score=max(scores), photo_scores=sorted(scores, reverse=True))
            for pid, scores in by_person.items() if max(scores) >= threshold
        ]
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def best_match(self, probe: np.ndarray, threshold: float) -> MatchResult | None:
        r = self.match(probe, threshold, top_k=1)
        return r[0] if r else None
