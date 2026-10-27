from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QRectF, Qt, QThread, Signal
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..detection_video import VideoControl, analyze_video
from ..image_enhancement import save_new_png
from ..person_vehicle import MODEL_TITLE, load_person_vehicle_detector


class VideoCanvas(QLabel):
    def __init__(self) -> None:
        super().__init__("打开本地视频，点击“开始检测”")
        self.image = QImage()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(480, 300)
        self.setStyleSheet("background:#111820;color:#b4c4d4;")

    def set_frame(self, pixels: np.ndarray) -> None:
        height, width = pixels.shape[:2]
        self.image = QImage(pixels.data, width, height, pixels.strides[0], QImage.Format_BGR888).copy()
        self.setText("")
        self.update()

    def paintEvent(self, event: object) -> None:
        super().paintEvent(event)
        if self.image.isNull():
            return
        size = self.image.size().scaled(self.size(), Qt.KeepAspectRatio)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.drawImage(QRectF((self.width() - size.width()) / 2,
                                (self.height() - size.height()) / 2,
                                size.width(), size.height()), self.image)
        painter.end()


class VideoDetectionWorker(QThread):
    frame_ready = Signal(object, object, int, int, float)
    failed = Signal(str)
    backend_ready = Signal(str)

    def __init__(self, source: Path, backend: str, device_id: int, threshold: float,
                 people: bool, vehicles: bool, output: Path | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.source, self.backend, self.device_id = source, backend, device_id
        self.threshold, self.people, self.vehicles = threshold, people, vehicles
        self.output = output
        self.control = VideoControl()
        self.report = None

    def stop(self) -> None:
        self.control.stopped.set()

    def toggle_pause(self) -> bool:
        if self.control.paused.is_set():
            self.control.paused.clear()
        else:
            self.control.paused.set()
        return self.control.paused.is_set()

    def run(self) -> None:
        try:
            if self.control.stopped.is_set():
                return
            detector = load_person_vehicle_detector(
                self.backend, self.device_id, self.threshold, self.people, self.vehicles,
            )
            self.backend_ready.emit(" / ".join(detector.session.get_providers()))
            self.report = analyze_video(
                self.source, detector, output=self.output, on_frame=self.frame_ready.emit,
                control=self.control, realtime=self.output is None,
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class VideoDetectionWidget(QWidget):
    def __init__(self, backend: str = "auto", device_id: int = 0,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.device_id = device_id
        self.source: Path | None = None
        self.worker: VideoDetectionWorker | None = None
        self.last_frame: np.ndarray | None = None
        self.source_edit = QLineEdit()
        self.source_edit.setReadOnly(True)
        self.source_edit.setPlaceholderText("选择待检测的视频")
        self.open_button = QPushButton("打开视频…")
        self.start_button = QPushButton("开始检测")
        self.pause_button = QPushButton("暂停")
        self.stop_button = QPushButton("停止")
        self.export_button = QPushButton("导出完整标注视频（无音轨）…")
        self.snapshot_button = QPushButton("保存当前帧…")
        self.open_button.clicked.connect(self._open)
        self.start_button.clicked.connect(lambda: self.start())
        self.pause_button.clicked.connect(self._pause)
        self.stop_button.clicked.connect(self.stop)
        self.export_button.clicked.connect(self._export)
        self.snapshot_button.clicked.connect(self._snapshot)
        source_row = QHBoxLayout()
        source_row.addWidget(self.source_edit, 1)
        source_row.addWidget(self.open_button)
        self.people = QCheckBox("人（红框）")
        self.vehicles = QCheckBox("车辆（黄框）")
        self.people.setChecked(True)
        self.vehicles.setChecked(True)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.05, 0.99)
        self.threshold.setSingleStep(0.05)
        self.threshold.setValue(0.5)
        self.backend = QComboBox()
        for title, value in (("自动选择 GPU", "auto"), ("NVIDIA CUDA", "cuda"), ("Intel OpenVINO GPU", "openvino")):
            self.backend.addItem(title, value)
        self.backend.setCurrentIndex(max(0, self.backend.findData(backend)))
        controls = QHBoxLayout()
        for item in (self.people, self.vehicles, QLabel("置信度门槛："), self.threshold,
                     self.backend,
                     self.start_button, self.pause_button, self.stop_button):
            controls.addWidget(item)
        controls.addStretch()
        self.preview = VideoCanvas()
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["类别", "置信分值", "位置 x1,y1,x2,y2"])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMaximumWidth(400)
        body = QHBoxLayout()
        body.addWidget(self.preview, 1)
        body.addWidget(self.table)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.status = QLabel("打开视频开始。检测模型随离线包携带，运行时不下载。")
        self.status.setWordWrap(True)
        self.counts = QLabel("当前帧：人 0 · 车 0")
        footer = QHBoxLayout()
        footer.addWidget(self.counts)
        footer.addStretch()
        footer.addWidget(self.snapshot_button)
        footer.addWidget(self.export_button)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"{MODEL_TITLE} · 人为红框，车辆为黄框；框左上角显示类别和 0～1 置信分值。"))
        layout.addLayout(source_row)
        layout.addLayout(controls)
        layout.addLayout(body, 1)
        layout.addWidget(self.progress)
        layout.addLayout(footer)
        layout.addWidget(self.status)
        self._update_buttons()

    def is_busy(self) -> bool:
        return self.worker is not None

    def _update_buttons(self) -> None:
        busy = self.is_busy()
        self.open_button.setEnabled(not busy)
        self.start_button.setEnabled(self.source is not None and not busy)
        self.export_button.setEnabled(self.source is not None and not busy)
        self.snapshot_button.setEnabled(self.last_frame is not None and not busy)
        self.pause_button.setEnabled(busy)
        self.stop_button.setEnabled(busy)
        for control in (self.threshold, self.people, self.vehicles, self.backend):
            control.setEnabled(not busy)

    def _open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "打开视频", "", "视频 (*.mp4 *.avi *.mkv *.mov *.wmv *.m4v)")
        if path:
            self.load_video(Path(path))

    def load_video(self, path: Path) -> bool:
        if self.is_busy():
            return False
        capture = cv2.VideoCapture(str(path))
        try:
            ok, frame = capture.read()
            if not ok:
                QMessageBox.warning(self, "无法打开视频", "无法解码视频首帧，请检查文件。")
                return False
        finally:
            capture.release()
        self.source = path
        self.source_edit.setText(str(path))
        self.last_frame = frame
        self.preview.set_frame(frame)
        self.table.setRowCount(0)
        self.counts.setText("当前帧：尚未检测")
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.status.setText(f"已打开 {path.name} · 点击开始检测；导出会从头逐帧处理完整视频。")
        self._update_buttons()
        return True

    def start(self, output: Path | None = None) -> None:
        if self.source is None or self.is_busy():
            return
        if not self.people.isChecked() and not self.vehicles.isChecked():
            QMessageBox.warning(self, "请选择检测类别", "请至少勾选人或车辆。")
            return
        if output is not None and output.exists():
            QMessageBox.warning(self, "文件已存在", "请换一个新文件名，不会覆盖原视频或已有结果。")
            return
        self.worker = VideoDetectionWorker(
            self.source, self.backend.currentData(), self.device_id, self.threshold.value(),
            self.people.isChecked(), self.vehicles.isChecked(), output, self,
        )
        self.worker.frame_ready.connect(self._frame)
        self.worker.backend_ready.connect(lambda value: self.status.setText(f"GPU：{value} · 正在逐帧检测…"))
        self.worker.failed.connect(self._failed)
        self.worker.finished.connect(self._finished)
        self.pause_button.setText("暂停")
        self.table.setRowCount(0)
        self.progress.setRange(0, 0)
        self.status.setText("正在离线校验模型并初始化 GPU…")
        self.worker.start()
        self._update_buttons()

    def _frame(self, frame: np.ndarray, detections: list, index: int, total: int, fps: float) -> None:
        self.last_frame = frame
        self.preview.set_frame(frame)
        self.table.setRowCount(len(detections))
        for row, detection in enumerate(detections):
            values = (detection.label, f"{detection.score:.2f}",
                      ", ".join(str(round(value)) for value in detection.bbox))
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value))
        people = sum(detection.is_person for detection in detections)
        self.counts.setText(f"当前帧：人 {people} · 车 {len(detections) - people} · {(index - 1) / fps:.2f} 秒 · 第 {index} 帧")
        if total > 0:
            self.progress.setRange(0, 1000)
            self.progress.setValue(min(1000, int(index * 1000 / total)))

    def _pause(self) -> None:
        if self.worker:
            paused = self.worker.toggle_pause()
            self.pause_button.setText("继续" if paused else "暂停")
            self.status.setText("已请求暂停，可继续或停止。" if paused else "继续逐帧检测…")

    def stop(self) -> None:
        if self.worker:
            self.worker.stop()
            self.pause_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.status.setText("正在停止；未完成的导出不会保留。")

    def _failed(self, message: str) -> None:
        self.status.setText(f"检测失败：{message}")

    def _finished(self) -> None:
        worker, self.worker = self.worker, None
        report = worker.report
        if report is None and worker.control.stopped.is_set():
            self.status.setText("已停止。")
        if report is not None:
            if report.cancelled:
                self.status.setText(f"已停止 · 已处理 {report.frames} 帧 · 未保留未完成的导出。")
            elif report.output:
                self.status.setText(f"已导出完整标注视频（无音轨）：{report.output} · {report.frames} 帧 · {report.fps:g} FPS")
            else:
                self.status.setText(f"检测完成 · 共 {report.frames} 帧 · {report.frames / report.fps:.2f} 秒")
        self.progress.setRange(0, 1000)
        self._update_buttons()
        worker.deleteLater()

    def _export(self) -> None:
        if self.source is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出完整标注视频（无音轨）",
                                             f"{self.source.stem}-人车检测.mp4", "MP4 (*.mp4)")
        if path:
            output = Path(path)
            if output.suffix.lower() != ".mp4":
                output = output.with_name(output.name + ".mp4")
            self.start(output)

    def _snapshot(self) -> None:
        if self.last_frame is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存当前显示帧", "人车检测帧.png", "PNG (*.png)")
        if not path:
            return
        destination = Path(path)
        if destination.suffix.lower() != ".png":
            destination = destination.with_name(destination.name + ".png")
        try:
            save_new_png(destination, self.last_frame)
            self.status.setText(f"已保存：{destination}")
        except OSError as exc:
            QMessageBox.warning(self, "保存失败", str(exc))


class VideoDetectionDialog(QDialog):
    def __init__(self, backend: str = "auto", device_id: int = 0) -> None:
        super().__init__()
        self.setWindowTitle("离线人车视频检测")
        self.resize(1340, 840)
        self.tools = VideoDetectionWidget(backend, device_id, self)
        QVBoxLayout(self).addWidget(self.tools)

    def reject(self) -> None:
        if self.tools.is_busy():
            self.tools.stop()
        else:
            super().reject()

    def closeEvent(self, event: object) -> None:
        if self.tools.is_busy():
            self.tools.stop()
            event.ignore()
        else:
            event.accept()
