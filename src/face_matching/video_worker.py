from __future__ import annotations

import os
import threading
import time
from datetime import datetime

import cv2
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from .config import RecognitionSettings
from .domain import RecognitionEvent
from .face_engine import FaceEngine
from .gallery import GalleryIndex
from .tracking import MultiFaceTracker


class VideoWorker(QThread):
    frame_ready = Signal(object, object)
    recognition = Signal(object)
    statistics = Signal(float, int)
    status_changed = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        source: int | str,
        source_name: str,
        engine: FaceEngine,
        gallery: GalleryIndex,
        settings: RecognitionSettings,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.source = source
        self.source_name = source_name
        self.engine = engine
        self.gallery = gallery
        self.settings = settings
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._tracker = MultiFaceTracker(
            max_embeddings=settings.max_track_embeddings,
            ttl_frames=settings.track_ttl_frames,
        )

    def stop(self) -> None:
        self._stop_event.set()
        self._pause_event.clear()

    def set_paused(self, paused: bool) -> None:
        if paused:
            self._pause_event.set()
            self.status_changed.emit("已暂停")
        else:
            self._pause_event.clear()
            self.status_changed.emit("识别中")

    @property
    def paused(self) -> bool:
        return self._pause_event.is_set()

    def _open_capture(self):
        if isinstance(self.source, int) and os.name == "nt":
            capture = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        elif isinstance(self.source, str) and self.source.lower().startswith(
            ("rtsp://", "rtmp://", "http://", "https://")
        ):
            capture = cv2.VideoCapture()
            parameters: list[int] = []
            if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                parameters.extend([cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000])
            if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                parameters.extend([cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000])
            capture.open(self.source, cv2.CAP_FFMPEG, parameters)
        else:
            capture = cv2.VideoCapture(self.source)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        return capture

    def run(self) -> None:
        capture = self._open_capture()
        if not capture.isOpened():
            self.failed.emit(f"无法打开视频源：{self.source_name}")
            capture.release()
            return
        self.status_changed.emit("识别中")
        frame_index = 0
        measured_frames = 0
        measurement_start = time.perf_counter()
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        is_file = isinstance(self.source, str) and not self.source.lower().startswith(
            ("rtsp://", "rtmp://", "http://", "https://")
        )
        try:
            while not self._stop_event.is_set():
                if self._pause_event.is_set():
                    self.msleep(50)
                    continue
                iteration_start = time.perf_counter()
                ok, frame = capture.read()
                if not ok:
                    if is_file:
                        self.status_changed.emit("视频播放完毕")
                    else:
                        self.failed.emit("视频源读取失败或连接已断开")
                    break
                frame_index += 1
                observations = self.engine.detect(
                    frame, threshold=self.settings.detection_threshold
                )
                if (
                    self.gallery.person_count > 0
                    and frame_index % self.settings.recognition_interval == 0
                ):
                    eligible = [
                        observation
                        for observation in observations
                        if observation.quality >= self.settings.minimum_quality
                    ]
                    self.engine.add_embeddings(frame, eligible)
                views, event_tracks = self._tracker.update(
                    observations=observations,
                    gallery=self.gallery,
                    frame_index=frame_index,
                    threshold=self.settings.similarity_threshold,
                    minimum_quality=self.settings.minimum_quality,
                    confirmation_hits=self.settings.confirmation_hits,
                )
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channels = rgb.shape
                image = QImage(
                    rgb.data,
                    width,
                    height,
                    channels * width,
                    QImage.Format.Format_RGB888,
                ).copy()
                self.frame_ready.emit(image, views)
                for track in event_tracks:
                    if track.confirmed is None:
                        continue
                    self.recognition.emit(
                        RecognitionEvent(
                            occurred_at=datetime.now(),
                            source=self.source_name,
                            track_id=track.id,
                            match=track.confirmed,
                            quality=track.quality,
                        )
                    )
                measured_frames += 1
                elapsed = time.perf_counter() - measurement_start
                if elapsed >= 1.0:
                    self.statistics.emit(measured_frames / elapsed, len(views))
                    measurement_start = time.perf_counter()
                    measured_frames = 0
                if is_file and 1.0 <= source_fps <= 120.0:
                    remaining = 1.0 / source_fps - (time.perf_counter() - iteration_start)
                    if remaining > 0:
                        self.msleep(int(remaining * 1000))
        except Exception as exc:
            self.failed.emit(f"视频处理失败：{exc}")
        finally:
            capture.release()
