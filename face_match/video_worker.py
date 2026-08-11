from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Literal

from PySide6.QtCore import QThread, Signal

from face_match.errors import VideoSourceError
from face_match.vision.pipeline import VideoFacePipeline, annotate_frame

SourceKind = Literal["file", "camera", "rtsp"]


@dataclass(frozen=True)
class VideoSource:
    kind: SourceKind
    value: str | int
    display_name: str


@dataclass(frozen=True)
class VideoStats:
    processed_fps: float
    face_count: int
    frame_index: int
    position_seconds: float


class VideoWorker(QThread):
    frame_ready = Signal(object, object)
    event_ready = Signal(object)
    status_changed = Signal(str)
    failed = Signal(str)
    stream_finished = Signal()

    def __init__(
        self,
        source: VideoSource,
        pipeline: VideoFacePipeline,
        target_fps: float,
        parent: Any | None = None,
    ) -> None:
        super().__init__(parent)
        self.source = source
        self.pipeline = pipeline
        self.target_fps = target_fps
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()

    def request_stop(self) -> None:
        self._stop_event.set()
        self._pause_event.clear()

    def set_paused(self, paused: bool) -> None:
        if paused:
            self._pause_event.set()
        else:
            self._pause_event.clear()

    def run(self) -> None:
        capture = None
        try:
            import cv2

            self.pipeline.reset()
            capture = self._open_capture(cv2)
            source_fps = float(capture.get(cv2.CAP_PROP_FPS))
            fps_is_valid = 1.0 <= source_fps <= 240.0
            if not fps_is_valid:
                source_fps = self.target_fps
            stride = max(1, round(source_fps / self.target_fps))
            frame_index = 0
            processed = 0
            fps_window_start = time.perf_counter()
            measured_fps = 0.0
            last_process_time = 0.0
            consecutive_failures = 0
            self.status_changed.emit(f"正在识别：{self.source.display_name}")
            while not self._stop_event.is_set():
                if self._pause_event.is_set():
                    self.msleep(50)
                    continue
                ok, frame = capture.read()
                if not ok or frame is None:
                    consecutive_failures += 1
                    if self.source.kind == "file" or consecutive_failures >= 8:
                        break
                    self.msleep(100)
                    continue
                consecutive_failures = 0
                frame_index += 1
                if frame_index % stride:
                    continue
                if self.source.kind != "file" and not fps_is_valid:
                    now = time.perf_counter()
                    if now - last_process_time < 1.0 / self.target_fps:
                        continue
                    last_process_time = now
                started = time.perf_counter()
                views = self.pipeline.process(frame)
                rendered = annotate_frame(frame, views)
                processed += 1
                window_elapsed = time.perf_counter() - fps_window_start
                if window_elapsed >= 1.0:
                    measured_fps = processed / window_elapsed
                    processed = 0
                    fps_window_start = time.perf_counter()
                position_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                stats = VideoStats(
                    processed_fps=measured_fps,
                    face_count=len(views),
                    frame_index=frame_index,
                    position_seconds=max(0.0, position_ms / 1000.0),
                )
                self.frame_ready.emit(rendered, stats)
                for view in views:
                    if view.is_new_event:
                        self.event_ready.emit(view)
                if self.source.kind == "file":
                    remaining = (1.0 / self.target_fps) - (time.perf_counter() - started)
                    if remaining > 0:
                        self.msleep(int(remaining * 1000))
            if (
                not self._stop_event.is_set()
                and self.source.kind != "file"
                and consecutive_failures
            ):
                raise VideoSourceError("视频流连续读取失败，请检查摄像头、网络或 RTSP 地址")
        except Exception as exc:
            if not self._stop_event.is_set():
                self.failed.emit(str(exc))
        finally:
            if capture is not None:
                capture.release()
            self.stream_finished.emit()

    def _open_capture(self, cv2: Any) -> Any:
        if self.source.kind == "camera":
            backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
            capture = cv2.VideoCapture(int(self.source.value), backend)
        else:
            capture = cv2.VideoCapture()
            if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8_000)
            if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5_000)
            opened = capture.open(str(self.source.value), cv2.CAP_FFMPEG)
            if not opened:
                capture.release()
                capture = cv2.VideoCapture(str(self.source.value))
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        if not capture.isOpened():
            capture.release()
            raise VideoSourceError(f"无法打开视频源：{self.source.display_name}")
        return capture
