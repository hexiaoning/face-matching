"""核心 pipeline：视频帧 → 检测 → 质量过滤 → 识别 → 跟踪 → 多帧融合 → 匹配。"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import config, quality as quality_mod
from .detector import FaceDet, SCRFDDetector
from .matcher import GalleryIndex, MatchResult
from .recognizer import FaceRecognizer
from .tracker import IoUTracker, Track


@dataclass
class TrackResult:
    tid: int
    box: np.ndarray
    quality: float            # 本帧质量分
    fused_ready: bool         # 是否已积累足够样本
    match: MatchResult | None
    best_score: float = 0.0   # 该 track 历史最高分


@dataclass
class FrameResult:
    frame_idx: int
    timestamp: float
    tracks: list[TrackResult] = field(default_factory=list)


class FacePipeline:
    def __init__(self, detector: SCRFDDetector, recognizer: FaceRecognizer,
                 gallery: GalleryIndex, threshold: float = config.MATCH_THRESHOLD):
        self.detector = detector
        self.recognizer = recognizer
        self.gallery = gallery
        self.threshold = threshold
        self.tracker = IoUTracker()
        self._best_scores: dict[int, float] = {}

    def reset(self) -> None:
        self.tracker = IoUTracker()
        self._best_scores.clear()

    def process_frame(self, frame: np.ndarray, frame_idx: int,
                      timestamp: float = 0.0) -> FrameResult:
        dets = self.detector.detect(frame)
        tracker_dets: list[tuple[np.ndarray, np.ndarray | None, float]] = []
        det_quality: dict[int, float] = {}

        for det in dets:
            q = quality_mod.quality_score(det, None)
            if det.size < config.MIN_FACE_SIZE:
                continue
            emb: np.ndarray | None = None
            aligned = None
            if quality_mod.usable(det, 0.0):
                aligned = None
                try:
                    from .recognizer import align_face
                    aligned = align_face(frame, det.kps, self.recognizer.input_size)
                    if quality_mod.usable(det, 0.2, aligned):
                        emb = self.recognizer.embed_aligned(aligned)
                        q = quality_mod.quality_score(det, aligned)
                except cv2.error:
                    emb = None
            tracker_dets.append((det.box, emb, q))

        tracks = self.tracker.update(frame_idx, tracker_dets)

        results: list[TrackResult] = []
        # 按当前帧的检测顺序输出
        box_to_tid: dict[int, int] = {}
        for trk in tracks.values():
            if trk.last_frame == frame_idx:
                box_to_tid[id(trk.box)] = trk.tid

        for box, emb, q in tracker_dets:
            trk = self._find_track(tracks, box, frame_idx)
            if trk is None:
                continue
            match: MatchResult | None = None
            fused = trk.fused_embedding()
            ready = fused is not None
            if ready:
                match = self.gallery.best_match(fused, self.threshold)
            score = match.score if match else 0.0
            prev = self._best_scores.get(trk.tid, 0.0)
            self._best_scores[trk.tid] = max(prev, score)
            results.append(TrackResult(tid=trk.tid, box=box, quality=q,
                                       fused_ready=ready, match=match,
                                       best_score=self._best_scores[trk.tid]))
        return FrameResult(frame_idx=frame_idx, timestamp=timestamp, tracks=results)

    @staticmethod
    def _find_track(tracks: dict[int, Track], box: np.ndarray, frame_idx: int) -> Track | None:
        for trk in tracks.values():
            if trk.last_frame == frame_idx and np.array_equal(trk.box, box):
                return trk
        return None


def embed_gallery_photo(detector: SCRFDDetector, recognizer: FaceRecognizer,
                        img: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """为人员库照片提取 embedding。

    取图中最大且质量合格的人脸。返回 (embedding, quality, aligned_face)。
    找不到合格人脸时抛 ValueError。
    """
    dets = detector.detect(img, score_thresh=0.3)
    if not dets:
        raise ValueError("照片中未检测到人脸")
    dets.sort(key=lambda d: d.size, reverse=True)
    best_err = "照片中的人脸质量不达标（过小/过糊/过侧），请换一张清晰正脸照"
    for det in dets[:3]:
        if det.size < config.MIN_FACE_SIZE:
            continue
        aligned = None
        try:
            from .recognizer import align_face
            aligned = align_face(img, det.kps, recognizer.input_size)
        except cv2.error:
            continue
        q = quality_mod.quality_score(det, aligned)
        if quality_mod.usable(det, 0.15, aligned):
            return recognizer.embed_aligned(aligned), q, aligned
        best_err = f"人脸质量分过低({q:.2f})，请换一张更清晰、更正脸的照片"
    raise ValueError(best_err)
