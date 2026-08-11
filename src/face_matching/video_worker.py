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
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.engine = engine
        self.matcher = matcher
        self.source = source
        self._threshold = threshold
        self._stop_event = threading.Event()
        self._settings_lock = threading.Lock()

    def stop(self) -> None:
        self._stop_event.set()

    def set_threshold(self, threshold: float) -> None:
        with self._settings_lock:
            self._threshold = float(threshold)

    def run(self) -> None:
        pipeline = VideoFacePipeline(self.engine, self.matcher)
        capture = cv2.VideoCapture(self.source)
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not capture.isOpened():
                self.stream_error.emit(f"无法打开视频源：{self.source}")
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
            while not self._stop_event.is_set():
                loop_start = time.perf_counter()
                ok, frame = capture.read()
                if not ok:
                    if not self._stop_event.is_set() and not is_file:
                        self.stream_error.emit("视频流读取中断")
                    break
                with self._settings_lock:
                    pipeline.match_threshold = self._threshold
                result = pipeline.process(frame)
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
                for event in result.events:
                    self.recognition.emit(event)
                if target_interval:
                    remaining = target_interval - (time.perf_counter() - loop_start)
                    if remaining > 0:
                        self._stop_event.wait(remaining)
        except Exception as exc:
            self.stream_error.emit(f"视频识别失败：{exc}")
        finally:
            capture.release()
            self.stream_ended.emit()
