from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from face_matching.inference import FaceEngine
from face_matching.matching import IdentityMatcher, TemporalTracker, Track


@dataclass(frozen=True, slots=True)
class TrackView:
    id: int
    bbox: tuple[int, int, int, int]
    name: str
    score: float
    accepted: bool
    quality: float
    decision: str
    confirmation_hits: int


@dataclass(frozen=True, slots=True)
class FramePayload:
    rgb: np.ndarray
    tracks: tuple[TrackView, ...]
    fps: float
    frame_index: int


class VideoWorker(QThread):
    frame_ready = Signal(object)
    tracks_ready = Signal(object)
    status_changed = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        engine: FaceEngine,
        matcher: IdentityMatcher,
        source: str | int,
        frame_stride: int,
        max_track_age: int,
        min_track_hits: int,
        min_recognition_quality: float,
        parent=None,
    ):
        super().__init__(parent)
        self.engine = engine
        self.matcher = matcher
        self.source = source
        self.frame_stride = frame_stride
        self.tracker = TemporalTracker(
            max_track_age,
            min_confirmations=min_track_hits,
            min_quality=min_recognition_quality,
        )
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    @staticmethod
    def _track_views(tracks: list[Track]) -> tuple[TrackView, ...]:
        return tuple(
            TrackView(
                id=track.id,
                bbox=tuple(int(round(value)) for value in track.bbox),
                name=track.match.name,
                score=track.match.score,
                accepted=track.match.accepted,
                quality=track.quality,
                decision=track.decision,
                confirmation_hits=track.candidate_hits,
            )
            for track in tracks
        )

    def run(self) -> None:
        capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            if isinstance(self.source, int):
                source_label = f"本机摄像头 {self.source}"
            elif self.source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://")):
                source_label = "网络视频流（地址已隐藏）"
            else:
                source_label = self.source
            self.failed.emit(f"无法打开视频源：{source_label}")
            return
        is_file = isinstance(self.source, str) and not self.source.lower().startswith(
            ("rtsp://", "rtmp://", "http://", "https://")
        )
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not 1.0 <= source_fps <= 240.0:
            source_fps = 25.0
        frame_interval = 1.0 / source_fps
        frame_index = 0
        last_views: tuple[TrackView, ...] = ()
        last_tick = time.perf_counter()
        display_fps = 0.0
        self.status_changed.emit("识别运行中（CUDA）")
        try:
            while not self._stop_event.is_set():
                tick = time.perf_counter()
                ok, frame = capture.read()
                if not ok:
                    if is_file:
                        self.status_changed.emit("视频播放完成")
                    else:
                        self.failed.emit("视频流读取失败或连接已断开。")
                    break
                frame_index += 1
                if frame_index == 1 or frame_index % self.frame_stride == 0:
                    faces = self.engine.analyze(frame)
                    tracks = self.tracker.update(faces, frame_index, self.matcher)
                    last_views = self._track_views(tracks)
                    self.tracks_ready.emit(last_views)
                now = time.perf_counter()
                instantaneous = 1.0 / max(now - last_tick, 1e-6)
                display_fps = (
                    instantaneous if display_fps == 0 else 0.9 * display_fps + 0.1 * instantaneous
                )
                last_tick = now
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_ready.emit(
                    FramePayload(rgb.copy(), last_views, display_fps, frame_index)
                )
                if is_file:
                    remaining = frame_interval - (time.perf_counter() - tick)
                    if remaining > 0:
                        self.msleep(max(1, int(remaining * 1000)))
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            capture.release()
            self.tracker.reset()
            self.status_changed.emit("已停止")
