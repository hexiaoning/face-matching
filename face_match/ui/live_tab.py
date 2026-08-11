from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from face_match.domain import TrackView
from face_match.identity import mask_id_number
from face_match.services import ApplicationServices
from face_match.video_worker import VideoSource, VideoStats, VideoWorker
from face_match.vision.pipeline import VideoFacePipeline


class LiveTab(QWidget):
    running_changed = Signal(bool)

    def __init__(self, services: ApplicationServices, parent: Any | None = None) -> None:
        super().__init__(parent)
        self.services = services
        self.worker: VideoWorker | None = None
        self._last_image: QImage | None = None
        self._enrollment_busy = False

        self.file_button = QPushButton("打开视频…")
        self.camera_button = QPushButton("本地摄像头…")
        self.rtsp_button = QPushButton("RTSP 网络流…")
        self.pause_button = QPushButton("暂停")
        self.stop_button = QPushButton("停止")
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.file_button.clicked.connect(self.open_file)
        self.camera_button.clicked.connect(self.open_camera)
        self.rtsp_button.clicked.connect(self.open_rtsp)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.stop_button.clicked.connect(self.stop)
        self._paused = False

        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.0, 1.0)
        self.threshold.setDecimals(3)
        self.threshold.setSingleStep(0.01)
        self.threshold.setValue(services.settings.similarity_threshold)
        self.threshold.setToolTip("余弦相似度，不是概率；部署前应使用现场数据标定")
        self.threshold.valueChanged.connect(self._threshold_changed)
        toolbar = QHBoxLayout()
        toolbar.addWidget(self.file_button)
        toolbar.addWidget(self.camera_button)
        toolbar.addWidget(self.rtsp_button)
        toolbar.addWidget(self.pause_button)
        toolbar.addWidget(self.stop_button)
        toolbar.addStretch()
        toolbar.addWidget(QLabel("相似度阈值："))
        toolbar.addWidget(self.threshold)

        self.video_label = QLabel("请选择视频、摄像头或 RTSP 网络流")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(720, 405)
        self.video_label.setObjectName("videoSurface")

        self.events = QTableWidget(0, 6)
        self.events.setHorizontalHeaderLabels(
            ["时间", "轨迹", "姓名", "身份证号", "相似度", "画面质量"]
        )
        self.events.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.events.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.events.verticalHeader().setVisible(False)
        self.events.horizontalHeader().setStretchLastSection(True)
        self.events.setMinimumWidth(470)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.video_label)
        splitter.addWidget(self.events)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self.status = QLabel("就绪 · 推理强制使用 CUDA GPU")
        self.status.setObjectName("statusLabel")
        layout = QVBoxLayout(self)
        layout.addLayout(toolbar)
        layout.addWidget(splitter, 1)
        layout.addWidget(self.status)

    def open_file(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频",
            "",
            "视频 (*.mp4 *.avi *.mkv *.mov *.wmv *.m4v);;所有文件 (*)",
        )
        if filename:
            self.start_source(VideoSource("file", filename, Path(filename).name))

    def open_camera(self) -> None:
        index, accepted = QInputDialog.getInt(self, "本地摄像头", "摄像头编号：", 0, 0, 32, 1)
        if accepted:
            self.start_source(VideoSource("camera", index, f"本地摄像头 {index}"))

    def open_rtsp(self) -> None:
        url, accepted = QInputDialog.getText(
            self,
            "RTSP 网络流",
            "RTSP 地址（含用户名密码时不会保存）：",
            text="rtsp://",
        )
        url = url.strip()
        if accepted and url:
            if not url.lower().startswith(("rtsp://", "rtsps://", "http://", "https://")):
                QMessageBox.warning(
                    self, "地址不正确", "请输入 RTSP、RTSPS、HTTP 或 HTTPS 视频流地址。"
                )
                return
            safe_name = url.split("@")[-1] if "@" in url else url
            self.start_source(VideoSource("rtsp", url, safe_name))

    def start_source(self, source: VideoSource) -> None:
        if self._enrollment_busy:
            return
        if self.worker is not None:
            self.stop(wait=True)
        if self.services.matcher.template_count == 0:
            QMessageBox.information(
                self,
                "人员库为空",
                "当前没有可匹配的人员照片。视频仍可运行，但所有人都会显示为未知人员。",
            )
        pipeline = VideoFacePipeline(
            self.services.detector,
            self.services.embedder,
            self.services.matcher,
            self.services.settings,
        )
        worker = VideoWorker(source, pipeline, self.services.settings.target_fps, self)
        worker.frame_ready.connect(self._frame_ready)
        worker.event_ready.connect(self._event_ready)
        worker.status_changed.connect(self.status.setText)
        worker.failed.connect(self._failed)
        worker.stream_finished.connect(self._stream_finished)
        worker.finished.connect(worker.deleteLater)
        self.worker = worker
        self._paused = False
        self.pause_button.setText("暂停")
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self._set_source_buttons(False)
        self.running_changed.emit(True)
        worker.start()

    def toggle_pause(self) -> None:
        if not self.worker:
            return
        self._paused = not self._paused
        self.worker.set_paused(self._paused)
        self.pause_button.setText("继续" if self._paused else "暂停")
        self.status.setText("已暂停" if self._paused else "继续识别")

    def stop(self, wait: bool = False) -> None:
        worker = self.worker
        if worker is None:
            return
        worker.request_stop()
        self.status.setText("正在停止视频源……")
        if wait and not worker.wait(8_000):
            QMessageBox.warning(self, "视频源未响应", "视频读取仍在退出，请稍后再试。")

    def shutdown(self) -> None:
        if self.worker:
            self.worker.request_stop()
            self.worker.wait(8_000)

    def _frame_ready(self, frame: object, stats: VideoStats) -> None:
        array = frame
        height, width, channels = array.shape
        image = QImage(
            array.data, width, height, channels * width, QImage.Format.Format_BGR888
        ).copy()
        self._last_image = image
        self._update_pixmap()
        self.status.setText(
            f"CUDA 处理中 · {stats.processed_fps:.1f} FPS · 当前 {stats.face_count} 张人脸 · "
            f"位置 {stats.position_seconds:.1f}s"
        )

    def _event_ready(self, view: TrackView) -> None:
        row = 0
        self.events.insertRow(row)
        values = [
            datetime.now().astimezone().strftime("%H:%M:%S"),
            f"#{view.track_id}",
            view.match.name,
            mask_id_number(view.match.id_number),
            f"{view.match.score:.3f}",
            f"{view.quality.overall:.2f}",
        ]
        for column, value in enumerate(values):
            self.events.setItem(row, column, QTableWidgetItem(value))
        while self.events.rowCount() > 500:
            self.events.removeRow(self.events.rowCount() - 1)
        self.events.resizeColumnsToContents()

    def _failed(self, message: str) -> None:
        QMessageBox.critical(self, "视频识别失败", message)

    def _stream_finished(self) -> None:
        sender = self.sender()
        if self.worker is not None and sender is not self.worker:
            return
        self.worker = None
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self._set_source_buttons(True)
        self.status.setText("视频已停止 · 推理强制使用 CUDA GPU")
        self.running_changed.emit(False)

    def _set_source_buttons(self, enabled: bool) -> None:
        for button in (self.file_button, self.camera_button, self.rtsp_button):
            button.setEnabled(enabled and not self._enrollment_busy)

    def set_enrollment_busy(self, busy: bool) -> None:
        self._enrollment_busy = busy
        self._set_source_buttons(self.worker is None)

    def _threshold_changed(self, value: float) -> None:
        self.services.settings.similarity_threshold = value
        try:
            self.services.settings.save(self.services.paths.settings)
        except OSError as exc:
            self.status.setText(f"设置保存失败：{exc}")

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)
        self._update_pixmap()

    def _update_pixmap(self) -> None:
        if self._last_image is None:
            return
        pixmap = QPixmap.fromImage(self._last_image).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(pixmap)
