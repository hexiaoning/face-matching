"""轻量 IoU tracker + track 级 embedding 质量加权融合。

监控视频中同一个人会出现很多帧：单帧可能模糊/侧脸，但一个 track 里通常
存在若干质量较好的帧。取质量 Top-K 帧的 embedding 加权平均，比任何单帧
都稳。
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from . import config


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    tid: int
    box: np.ndarray
    last_frame: int
    first_frame: int
    samples: list[tuple[np.ndarray, float]] = field(default_factory=list)  # (embedding, quality)
    missed: int = 0

    def add_sample(self, embedding: np.ndarray, quality: float) -> None:
        self.samples.append((embedding, quality))
        if len(self.samples) > config.TRACK_TOP_K * 3:
            self.samples.sort(key=lambda s: s[1], reverse=True)
            del self.samples[config.TRACK_TOP_K * 2:]

    def fused_embedding(self) -> np.ndarray | None:
        """质量 Top-K 帧加权平均，返回归一化后的融合 embedding。"""
        if len(self.samples) < config.TRACK_MIN_SAMPLES:
            return None
        top = sorted(self.samples, key=lambda s: s[1], reverse=True)[:config.TRACK_TOP_K]
        embs = np.stack([e for e, _ in top])
        weights = np.array([q ** 2 for _, q in top], dtype=np.float32)
        weights /= weights.sum() + 1e-9
        fused = (embs * weights[:, None]).sum(axis=0)
        n = np.linalg.norm(fused)
        return (fused / n).astype(np.float32) if n > 0 else None


class IoUTracker:
    def __init__(self, iou_thresh: float = 0.3, max_age: int = config.TRACK_MAX_AGE):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.tracks: dict[int, Track] = {}
        self._ids = itertools.count(1)

    def update(self, frame_idx: int, dets: list[tuple[np.ndarray, np.ndarray | None, float]]
               ) -> dict[int, Track]:
        """dets: [(box, embedding or None, quality)]。返回活跃 track。"""
        boxes = [d[0] for d in dets]
        assigned: set[int] = set()
        det_used: set[int] = set()

        # 贪心匹配：按 IoU 从大到小配对
        pairs = []
        for tid, trk in self.tracks.items():
            for di, box in enumerate(boxes):
                v = iou(trk.box, box)
                if v >= self.iou_thresh:
                    pairs.append((v, tid, di))
        pairs.sort(reverse=True)
        for v, tid, di in pairs:
            if tid in assigned or di in det_used:
                continue
            trk = self.tracks[tid]
            trk.box = boxes[di]
            trk.last_frame = frame_idx
            trk.missed = 0
            emb, q = dets[di][1], dets[di][2]
            if emb is not None:
                trk.add_sample(emb, q)
            assigned.add(tid)
            det_used.add(di)

        # 未匹配的旧 track
        for tid, trk in list(self.tracks.items()):
            if tid not in assigned:
                trk.missed = frame_idx - trk.last_frame
                if trk.missed > self.max_age:
                    del self.tracks[tid]

        # 新 track
        for di, (box, emb, q) in enumerate(dets):
            if di in det_used:
                continue
            tid = next(self._ids)
            trk = Track(tid=tid, box=box, last_frame=frame_idx, first_frame=frame_idx)
            if emb is not None:
                trk.add_sample(emb, q)
            self.tracks[tid] = trk

        return self.tracks

    def active(self) -> list[Track]:
        return list(self.tracks.values())
