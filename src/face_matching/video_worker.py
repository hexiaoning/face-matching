from __future__ import annotations

import threading
import time

import cv2
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from .engine import FaceEngine
from .matcher import GalleryMatcher
from .pipeline import VideoFacePipeline


class VideoWorker(QThread):
    frame_ready = Signal(QImage, object, float)
    recognition = Signal(object)
    stream_error = Signal(str)
    stream_ended = Signal()

    def __init__(
        self,
        engine: FaceEngine,
        matcher: GalleryMatcher,
        source: str | int,
        threshold: float,
        processing_stride: int = 1,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.engine = engine
        self.matcher = matcher
        self.source = source
        self._threshold = threshold
        self._processing_stride = max(1, int(processing_stride))
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._settings_lock = threading.Lock()

    def stop(self) -> None:
        self._stop_event.set()
        self._pause_event.clear()

    def set_paused(self, paused: bool) -> None:
        if paused:
            self._pause_event.set()
        else:
            self._pause_event.clear()

    def set_threshold(self, threshold: float) -> None:
        with self._settings_lock:
            self._threshold = float(threshold)

    def set_processing_stride(self, stride: int) -> None:
        with self._settings_lock:
            self._processing_stride = max(1, int(stride))

    def run(self) -> None:
        pipeline = VideoFacePipeline(self.engine, self.matcher)
        capture = None
        try:
            capture = self._open_capture()
            if not capture.isOpened():
                self.stream_error.emit(f"无法打开视频源：{self._source_description()}")
                return
            reported_fps = capture.get(cv2.CAP_PROP_FPS)
            is_file = isinstance(self.source, str) and not self.source.lower().startswith(
                ("rtsp://", "http://", "https://", "rtmp://")
            )
            target_interval = (
                1.0 / reported_fps if is_file and 1.0 <= reported_fps <= 120 else 0.0
            )
            fps = 0.0
            previous = time.perf_counter()
            frame_index = 0
            consecutive_failures = 0
            last_result = None
            while not self._stop_event.is_set():
                if self._pause_event.is_set():
                    self.msleep(50)
                    continue
                loop_start = time.perf_counter()
                ok, frame = capture.read()
                if not ok:
                    consecutive_failures += 1
                    if is_file or consecutive_failures >= 8:
                        if not self._stop_event.is_set() and not is_file:
                            self.stream_error.emit("视频流连续读取失败或连接已断开")
                        break
                    self.msleep(100)
                    continue
                consecutive_failures = 0
                frame_index += 1
                with self._settings_lock:
                    pipeline.match_threshold = self._threshold
                    processing_stride = self._processing_stride
                processed_frame = last_result is None or frame_index % processing_stride == 0
                if processed_frame:
                    last_result = pipeline.process(frame)
                result = last_result
                now = time.perf_counter()
                instantaneous = 1.0 / max(now - previous, 1e-6)
                fps = instantaneous if fps == 0.0 else 0.88 * fps + 0.12 * instantaneous
                previous = now
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channels = rgb.shape
                image = QImage(
                    rgb.data, width, height, channels * width, QImage.Format.Format_RGB888
                ).copy()
                self.frame_ready.emit(image, result, fps)
                if processed_frame:
                    for event in result.events:
                        self.recognition.emit(event)
                if target_interval:
                    remaining = target_interval - (time.perf_counter() - loop_start)
                    if remaining > 0:
                        self._stop_event.wait(remaining)
        except Exception as exc:
            self.stream_error.emit(f"视频识别失败：{exc}")
        finally:
            if capture is not None:
                capture.release()
            self.stream_ended.emit()

    def _open_capture(self):
        if isinstance(self.source, int):
            backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
            capture = cv2.VideoCapture(self.source, backend)
        else:
            capture = cv2.VideoCapture()
            if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8_000)
            if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5_000)
            capture.open(self.source)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        return capture

    def _source_description(self) -> str:
        if isinstance(self.source, int):
            return f"摄像头 {self.source}"
        return self.source
