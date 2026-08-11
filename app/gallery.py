"""人员特征库：内存中的余弦相似度比对。

登记人员规模通常在数百~数万级，归一化矩阵乘法（numpy BLAS）足够快，
无需引入 FAISS。比对粒度为"人"：对一个人的多张照片取最大相似度。
"""
from __future__ import annotations

import threading

import numpy as np

from .db import Database


class Gallery:
    def __init__(self, db: Database):
        self._db = db
        self._lock = threading.Lock()
        self._matrix = np.zeros((0, 512), dtype=np.float32)
        self._persons: list[dict] = []
        self.reload()

    def reload(self) -> None:
        matrix, persons = self._db.load_gallery()
        with self._lock:
            self._matrix = matrix
            self._persons = persons

    @property
    def size(self) -> int:
        return len(self._persons)

    def match(self, embedding: np.ndarray, threshold: float) -> tuple[dict | None, float]:
        """返回 (最佳人员 dict 或 None, 相似度分数)。

        embedding 需已归一化。一人多张照片时取最大相似度。
        """
        with self._lock:
            matrix, persons = self._matrix, self._persons
        if matrix.shape[0] == 0:
            return None, 0.0

        sims = matrix @ embedding.astype(np.float32)
        # 按人聚合：取该人所有照片的最大相似度
        best_by_person: dict[int, tuple[float, dict]] = {}
        for sim, person in zip(sims, persons):
            pid = person["id"]
            if pid not in best_by_person or sim > best_by_person[pid][0]:
                best_by_person[pid] = (float(sim), person)
        score, person = max(best_by_person.values(), key=lambda t: t[0])
        if score >= threshold:
            return person, score
        return None, score
