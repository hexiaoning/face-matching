"""视频处理工作线程：读取视频源 -> GPU 检测 -> 跟踪 -> 质量过滤 ->
多帧特征融合 -> 与人员库比对 -> 输出标注画面与命中事件。"""
from __future__ import annotations

import time

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from . import config
from .db import Database
from .draw import draw_labels
from .engine import FaceEngine
from .gallery import Gallery
from .quality import face_quality, passes_gate
from .tracker import IoUTracker, Track

# 标注颜色 (BGR)
COLOR_KNOWN = (60, 200, 60)      # 命中人员：绿
COLOR_UNKNOWN = (60, 60, 220)    # 未知人员：红
COLOR_PENDING = (200, 170, 60)   # 样本积累中：蓝黄


class VideoWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray)   # 标注后的 BGR 帧
    match_event = pyqtSignal(dict)         # 新命中/更新事件
    status = pyqtSignal(str)               # 状态文本
    error = pyqtSignal(str)                # 致命错误
    finished_clean = pyqtSignal()          # 正常结束（视频播完/停止）

    def __init__(
        self,
        source: str,
        engine: FaceEngine,
        db: Database,
        gallery: Gallery,
        parent=None,
    ):
        super().__init__(parent)
        self.source = source
        self.engine = engine
        self.db = db
        self.gallery = gallery
        self.threshold = config.DEFAULT_THRESHOLD
        self._running = False

    # ---- 对外控制（GUI 线程调用）----

    def stop(self) -> None:
        self._running = False

    def set_threshold(self, value: float) -> None:
        self.threshold = value

    def reload_gallery(self) -> None:
        self.gallery.reload()

    # ---- 主循环 ----

    def run(self) -> None:
        src = self.source.strip()
        cap_source = int(src) if src.isdigit() else src
        cap = cv2.VideoCapture(cap_source)
        if not cap.isOpened():
            self.error.emit(f"无法打开视频源: {self.source}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = 1.0 / fps if fps and fps > 1 else 0.0
        tracker = IoUTracker()
        frame_idx = 0
        reported: dict[int, tuple] = {}  # track_id -> (person_id, score) 已上报的状态
        self._running = True
        self.status.emit(f"已打开视频源: {self.source}")

        try:
            while self._running:
                t0 = time.monotonic()
                ok, frame = cap.read()
                if not ok:
                    self.status.emit("视频播放结束")
                    break
                frame_idx += 1

                if frame_idx % config.DET_STRIDE == 0:
                    faces = self.engine.detect(frame)
                    visible = tracker.update(faces)
                    for track in visible:
                        face = getattr(track, "_face", None)
                        if face is None:
                            continue
                        self._process_face(frame, track, face, frame_idx, reported)
                else:
                    visible = tracker.tracks

                annotated = self._annotate(frame, tracker.tracks)
                self.frame_ready.emit(annotated)

                # 按源帧率节奏播放，避免界面被刷爆
                if frame_interval > 0:
                    elapsed = time.monotonic() - t0
                    if elapsed < frame_interval:
                        self.msleep(int((frame_interval - elapsed) * 1000))
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"视频处理出错: {e}")
        finally:
            cap.release()
            self.finished_clean.emit()

    # ---- 内部 ----

    def _process_face(
        self,
        frame: np.ndarray,
        track: Track,
        face,
        frame_idx: int,
        reported: dict,
    ) -> None:
        q = face_quality(frame, face)
        if passes_gate(q):
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            h, w = frame.shape[:2]
            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            track.add_sample(face.normed_embedding, q["quality"], crop)

        # 样本不足或未到期，跳过比对
        if track.sample_count < config.MIN_SAMPLES_TO_MATCH:
            return
        if frame_idx - track.last_match_frame < config.REMATCH_INTERVAL:
            return
        track.last_match_frame = frame_idx

        agg = track.aggregated_embedding()
        if agg is None:
            return
        person, score = self.gallery.match(agg, self.threshold)
        prev = reported.get(track.id)
        track.identity = person
        track.score = score

        # 首次命中、或命中人员变化、或分数明显提升时上报事件
        changed = (
            person is not None
            and (
                prev is None
                or prev[0] != person["id"]
                or score - prev[1] > 0.03
            )
        )
        if changed:
            reported[track.id] = (person["id"], score)
            snapshot_path = self._save_snapshot(track)
            self.db.log_match(person, score, self.source, snapshot_path)
            self.match_event.emit(
                {
                    "name": person["name"],
                    "id_card": person["id_card"],
                    "score": score,
                    "snapshot": snapshot_path,
                    "source": self.source,
                    "time": time.strftime("%H:%M:%S"),
                }
            )
        elif person is None and prev is not None:
            reported.pop(track.id, None)

    def _save_snapshot(self, track: Track) -> str | None:
        if track.best_crop is None:
            return None
        path = config.SNAPSHOT_DIR / f"track{track.id}_{int(time.time()*1000)}.jpg"
        cv2.imwrite(str(path), track.best_crop)
        return str(path)

    def _annotate(self, frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
        out = frame.copy()
        labels = []
        for track in tracks:
            if track.missed > config.DET_STRIDE:
                continue  # 已短暂消失的目标不再绘制
            x1, y1, x2, y2 = [int(v) for v in track.bbox]
            if track.identity is not None:
                color = COLOR_KNOWN
                label = f"{track.identity['name']} {track.score:.2f}"
            elif track.sample_count >= config.MIN_SAMPLES_TO_MATCH:
                color = COLOR_UNKNOWN
                label = f"Unknown {track.score:.2f}" if track.score else "Unknown"
            else:
                color = COLOR_PENDING
                label = f"#{track.id} sampling"
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            labels.append((label, x1, y1 - 28 if y1 > 32 else y2 + 6, color))
        draw_labels(out, labels)
        return out
