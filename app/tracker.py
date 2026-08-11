"""轻量 IoU 多目标跟踪器 + 多帧质量加权特征融合。

视频中同一个人连续出现多帧，对每一帧的人脸提取特征并按质量加权平均，
比单帧比对在模糊/侧脸监控场景下稳定得多（视频监控识别的标准做法）。
"""
from __future__ import annotations

import numpy as np

from . import config


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class Track:
    __slots__ = (
        "id", "bbox", "missed", "samples", "best_crop", "best_quality",
        "identity", "score", "last_match_frame", "agg_cache", "_face",
    )

    def __init__(self, tid: int, bbox: np.ndarray):
        self.id = tid
        self.bbox = bbox.copy()
        self.missed = 0
        self.samples: list[tuple[np.ndarray, float]] = []  # (特征, 质量)
        self.best_crop: np.ndarray | None = None
        self.best_quality = 0.0
        self.identity: dict | None = None  # 命中的人员信息
        self.score = 0.0
        self.last_match_frame = -10**9  # 上次执行比对的帧号
        self.agg_cache: np.ndarray | None = None

    def add_sample(self, embedding: np.ndarray, quality: float, crop: np.ndarray) -> None:
        self.samples.append((embedding, quality))
        # 只保留质量最高的 TOP_K 个样本
        self.samples.sort(key=lambda s: s[1], reverse=True)
        del self.samples[config.TOP_K_SAMPLES:]
        if quality > self.best_quality:
            self.best_quality = quality
            self.best_crop = crop
        self.agg_cache = None

    def aggregated_embedding(self) -> np.ndarray | None:
        """质量加权融合后的归一化特征。"""
        if self.agg_cache is not None:
            return self.agg_cache
        if not self.samples:
            return None
        embs = np.stack([s[0] for s in self.samples])
        weights = np.array([s[1] for s in self.samples], dtype=np.float32)
        weights = weights / (weights.sum() + 1e-9)
        agg = (embs * weights[:, None]).sum(axis=0)
        agg /= np.linalg.norm(agg) + 1e-9
        self.agg_cache = agg.astype(np.float32)
        return self.agg_cache

    @property
    def sample_count(self) -> int:
        return len(self.samples)


class IoUTracker:
    def __init__(self):
        self._tracks: list[Track] = []
        self._next_id = 1

    @property
    def tracks(self) -> list[Track]:
        return self._tracks

    def update(self, faces: list) -> list[Track]:
        """用新一帧的检测结果更新跟踪，返回当前存活（本帧可见）的跟踪列表。"""
        det_bboxes = [f.bbox for f in faces]
        assigned_tracks: set[int] = set()
        assigned_dets: set[int] = set()

        # 贪心 IoU 匹配
        pairs = []
        for ti, track in enumerate(self._tracks):
            for di, db in enumerate(det_bboxes):
                pairs.append((_iou(track.bbox, db), ti, di))
        pairs.sort(reverse=True)
        for iou, ti, di in pairs:
            if iou < config.IOU_THRESHOLD:
                break
            if ti in assigned_tracks or di in assigned_dets:
                continue
            assigned_tracks.add(ti)
            assigned_dets.add(di)
            track = self._tracks[ti]
            track.bbox = det_bboxes[di].copy()
            track.missed = 0
            track._face = faces[di]  # 本帧对应的检测对象（供 worker 取特征）

        # 未匹配的检测 -> 新目标
        for di, face in enumerate(faces):
            if di not in assigned_dets:
                track = Track(self._next_id, face.bbox)
                track._face = face
                self._next_id += 1
                self._tracks.append(track)
                assigned_tracks.add(len(self._tracks) - 1)

        # 未匹配的目标 -> 丢失计数
        visible = []
        alive = []
        for ti, track in enumerate(self._tracks):
            if ti in assigned_tracks:
                visible.append(track)
            else:
                track.missed += 1
            if track.missed <= config.MAX_MISSED:
                alive.append(track)
        self._tracks = alive
        return visible
