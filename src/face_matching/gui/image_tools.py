from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QRectF, Qt, QThread, Signal
from PySide6.QtGui import QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout, QGraphicsScene,
    QGraphicsView, QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from ..image_enhancement import (
    PRESETS, EnhancementSettings, enhance_image, export_enhancement, save_new_png,
)


class ImagePane(QGraphicsView):
    region_selected = Signal(object)

    def __init__(self, selectable: bool = False) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setMinimumSize(280, 280)
        self.setBackgroundBrush(Qt.black)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.RubberBandDrag if selectable else QGraphicsView.NoDrag)
        self.selectable = selectable
        self.image = QImage()
        self.region: tuple[int, int, int, int] | None = None
        self.region_item = None
        self.origin = None
        self.fit_mode = True

    def set_image(self, pixels: np.ndarray) -> None:
        height, width = pixels.shape[:2]
        self.image = QImage(pixels.data, width, height, pixels.strides[0], QImage.Format_BGR888).copy()
        self.scene().clear()
        self.region_item = None
        self.scene().addPixmap(QPixmap.fromImage(self.image))
        self.scene().setSceneRect(QRectF(0, 0, width, height))
        self.set_region(self.region)
        self.fit_image()

    def set_region(self, region: tuple[int, int, int, int] | None) -> None:
        self.region = region
        if self.region_item is not None:
            self.scene().removeItem(self.region_item)
            self.region_item = None
        if region is not None:
            x1, y1, x2, y2 = region
            pen = QPen(Qt.cyan, 2)
            pen.setCosmetic(True)
            self.region_item = self.scene().addRect(QRectF(x1, y1, x2 - x1, y2 - y1), pen)

    def fit_image(self) -> None:
        self.fit_mode = True
        if not self.image.isNull():
            rect = QRectF(*self.region[:2], self.region[2] - self.region[0],
                          self.region[3] - self.region[1]) if self.region else self.sceneRect()
            self.fitInView(rect, Qt.KeepAspectRatio)

    def actual_size(self) -> None:
        self.fit_mode = False
        self.resetTransform()
        if self.region:
            self.centerOn((self.region[0] + self.region[2]) / 2,
                          (self.region[1] + self.region[3]) / 2)
        else:
            self.centerOn(self.sceneRect().center())

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)
        if self.fit_mode:
            self.fit_image()

    def mousePressEvent(self, event: object) -> None:
        if self.selectable and event.button() == Qt.LeftButton and not self.image.isNull():
            self.origin = self.mapToScene(event.position().toPoint())
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: object) -> None:
        super().mouseReleaseEvent(event)
        if self.origin is None or event.button() != Qt.LeftButton:
            return
        point = self.mapToScene(event.position().toPoint())
        rect = QRectF(self.origin, point).normalized().intersected(self.sceneRect())
        self.origin = None
        x1, y1 = max(0, int(rect.left())), max(0, int(rect.top()))
        x2, y2 = min(self.image.width(), int(rect.right())), min(self.image.height(), int(rect.bottom()))
        if x2 - x1 >= 4 and y2 - y1 >= 4:
            self.region_selected.emit((x1, y1, x2, y2))


class EnhancementWorker(QThread):
    completed = Signal(object, object)
    failed = Signal(str)

    def __init__(self, pixels: np.ndarray, settings: EnhancementSettings, parent: QWidget) -> None:
        super().__init__(parent)
        self.pixels = pixels
        self.settings = settings

    def run(self) -> None:
        try:
            self.completed.emit(enhance_image(self.pixels, self.settings), self.settings)
        except Exception as exc:
            self.failed.emit(str(exc))


class EnhancementExportWorker(QThread):
    failed = Signal(str)

    def __init__(self, destination: Path, png_only: bool, tool: ImageToolsWidget) -> None:
        super().__init__(tool)
        self.destination = destination
        self.png_only = png_only
        self.source_name = tool.source_name
        self.source_bytes = tool.source_bytes
        self.original = tool.original
        self.processed = tool.processed
        self.settings = tool.applied_settings
        self.region = tool.region

    def run(self) -> None:
        try:
            if self.png_only:
                save_new_png(self.destination, self.processed)
            else:
                export_enhancement(
                    self.destination, self.source_name, self.source_bytes, self.original,
                    self.processed, self.settings, self.region,
                )
        except FileExistsError:
            self.failed.emit("文件已存在。为保护原图和已有结果，请换一个新文件名。")
        except Exception as exc:
            self.failed.emit(str(exc))


class ImageToolsWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.source_name = ""
        self.source_bytes = b""
        self.original: np.ndarray | None = None
        self.processed: np.ndarray | None = None
        self.applied_settings = EnhancementSettings()
        self.region = None
        self.worker: EnhancementWorker | None = None
        self.export_worker: EnhancementExportWorker | None = None
        self.export_failed = False
        self.controls: dict[str, QDoubleSpinBox] = {}
        self.original_view = ImagePane(selectable=True)
        self.processed_view = ImagePane()
        self.original_view.region_selected.connect(self._select_region)
        self.open_button = QPushButton("打开图片…")
        self.open_button.clicked.connect(self._open)
        self.apply_button = QPushButton("应用参数")
        self.apply_button.clicked.connect(self._apply)
        self.png_button = QPushButton("另存处理图 PNG…")
        self.png_button.clicked.connect(lambda: self._export(True))
        self.zip_button = QPushButton("导出原图与处理记录 ZIP…")
        self.zip_button.clicked.connect(lambda: self._export(False))
        self.preset = QComboBox()
        self.preset.addItems(PRESETS)
        self.preset.currentTextChanged.connect(self._preset_changed)
        top = QHBoxLayout()
        top.addWidget(self.open_button)
        top.addWidget(QLabel("参数预设："))
        top.addWidget(self.preset)
        top.addWidget(self.apply_button)
        top.addStretch()
        top.addWidget(self.png_button)
        top.addWidget(self.zip_button)
        form = QFormLayout()
        for name, label, lower, upper, step in (
            ("texture", "屏幕纹理抑制", 0, 2, 0.1),
            ("denoise", "降噪强度", 0, 1, 0.1),
            ("gamma", "暗部提亮（1 为原始）", 0.5, 2, 0.05),
            ("contrast", "局部对比度", 0, 3, 0.1),
            ("sharpen", "锐化强度", 0, 1, 0.1),
        ):
            control = QDoubleSpinBox()
            control.setRange(lower, upper)
            control.setSingleStep(step)
            control.setDecimals(2)
            control.setValue(getattr(self.applied_settings, name))
            control.valueChanged.connect(self._parameters_changed)
            self.controls[name] = control
            form.addRow(label, control)
        note = QLabel("纹理抑制也会柔化真实细节；提亮无法恢复过曝或缺失的信息。\n"
                      "参数修改后点击“应用参数”。所有处理从原图开始，不累计叠加。")
        note.setWordWrap(True)
        settings = QVBoxLayout()
        settings.addLayout(form)
        settings.addWidget(note)
        settings.addStretch()
        settings_widget = QWidget()
        settings_widget.setLayout(settings)
        settings_widget.setMaximumWidth(285)
        panes = QHBoxLayout()
        for label, view in (("原图 · 拖动框选局部", self.original_view),
                            ("处理结果", self.processed_view)):
            column = QVBoxLayout()
            column.addWidget(QLabel(label))
            column.addWidget(view, 1)
            panes.addLayout(column, 1)
        panes.addWidget(settings_widget)
        navigation = QHBoxLayout()
        for label, callback in (("适应窗口 / 放大选区", self._fit),
                                ("100% 像素查看", self._actual_size),
                                ("清除选区 / 查看全图", lambda: self._select_region(None))):
            button = QPushButton(label)
            button.clicked.connect(callback)
            navigation.addWidget(button)
        self.region_label = QLabel("未选择局部")
        navigation.addWidget(self.region_label)
        navigation.addStretch()
        self.status = QLabel("打开本地图片开始。此工具可离线使用，无需模型或 GPU。")
        self.status.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("图像增强 · 保留原图，结果仅供辅助观察，无法确认缺失的人脸细节或车牌字符。"))
        layout.addLayout(top)
        layout.addLayout(panes, 1)
        layout.addLayout(navigation)
        layout.addWidget(self.status)
        self._update_buttons()

    def is_busy(self) -> bool:
        return bool((self.worker and self.worker.isRunning())
                    or (self.export_worker and self.export_worker.isRunning()))

    def settings(self) -> EnhancementSettings:
        return EnhancementSettings(**{name: control.value() for name, control in self.controls.items()})

    def _update_buttons(self) -> None:
        busy = self.is_busy()
        self.open_button.setEnabled(not busy)
        self.apply_button.setEnabled(self.original is not None and not busy)
        current = self.processed is not None and self.settings() == self.applied_settings
        self.png_button.setEnabled(current and not busy)
        self.zip_button.setEnabled(current and not busy)
        self.preset.setEnabled(not busy)
        for control in self.controls.values():
            control.setEnabled(not busy)

    def _parameters_changed(self) -> None:
        self._update_buttons()
        if self.original is not None and self.settings() != self.applied_settings:
            self.status.setText("参数已修改，点击“应用参数”更新结果后再导出。")

    def _preset_changed(self, name: str) -> None:
        for field, value in asdict(PRESETS[name]).items():
            self.controls[field].setValue(value)

    def _open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "打开图片", "", "图片 (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)"
        )
        if path:
            self.load_image(Path(path))

    def load_image(self, path: Path) -> bool:
        if self.is_busy():
            return False
        try:
            # Keep bytes from the time of opening, even if the source is later moved.
            source_bytes = path.read_bytes()
            image = cv2.imdecode(np.frombuffer(source_bytes, np.uint8), cv2.IMREAD_COLOR)
            if image is None or not image.size:
                raise ValueError("无法解码图片，请检查文件格式")
            if image.shape[0] * image.shape[1] > 40_000_000:
                raise ValueError("图片超过 4000 万像素，请先裁剪或缩小后打开")
        except (OSError, ValueError, cv2.error) as exc:
            QMessageBox.warning(self, "无法打开图片", str(exc))
            return False
        self.source_name = path.name
        self.source_bytes = source_bytes
        self.original = image
        self.processed = image.copy()
        self.applied_settings = EnhancementSettings()
        self.region = None
        for view in (self.original_view, self.processed_view):
            view.set_region(None)
        self.original_view.set_image(image)
        self.processed_view.set_image(self.processed)
        self.region_label.setText("未选择局部")
        self.status.setText(f"{path.name} · {image.shape[1]} × {image.shape[0]} · 当前为原图，选择预设后应用。")
        self._update_buttons()
        return True

    def _apply(self) -> None:
        if self.original is None or self.is_busy():
            return
        self.worker = EnhancementWorker(self.original, self.settings(), self)
        self.worker.completed.connect(self._processed)
        self.worker.failed.connect(self._processing_failed)
        self.worker.finished.connect(self._processing_finished)
        self.status.setText("正在处理完整分辨率图片…")
        self.worker.start()
        self._update_buttons()

    def _processed(self, image: np.ndarray, settings: EnhancementSettings) -> None:
        self.processed = image
        self.applied_settings = settings
        self.processed_view.set_image(image)
        self._fit()
        self.status.setText(f"处理完成 · {self.source_name} · 可框选局部比较或导出。")

    def _processing_failed(self, message: str) -> None:
        self.status.setText(f"处理失败：{message}")

    def _processing_finished(self) -> None:
        worker, self.worker = self.worker, None
        self._update_buttons()
        worker.deleteLater()

    def _select_region(self, region: tuple[int, int, int, int] | None) -> None:
        self.region = region
        for view in (self.original_view, self.processed_view):
            view.set_region(region)
        self.region_label.setText(f"原图坐标：{region}" if region else "未选择局部")
        self._fit()

    def _fit(self) -> None:
        for view in (self.original_view, self.processed_view):
            view.fit_image()

    def _actual_size(self) -> None:
        for view in (self.original_view, self.processed_view):
            view.actual_size()

    def _export(self, png_only: bool) -> None:
        if self.processed is None or self.is_busy() or self.settings() != self.applied_settings:
            return
        extension = ".png" if png_only else ".zip"
        path, _ = QFileDialog.getSaveFileName(
            self, "另存处理结果", f"{Path(self.source_name).stem}-增强{extension}",
            "PNG (*.png)" if png_only else "ZIP (*.zip)",
        )
        if not path:
            return
        destination = Path(path)
        if destination.suffix.lower() != extension:
            destination = destination.with_name(destination.name + extension)
        self.export_failed = False
        self.export_worker = EnhancementExportWorker(destination, png_only, self)
        self.export_worker.failed.connect(self._export_failed)
        self.export_worker.finished.connect(self._export_finished)
        self.status.setText("正在导出…")
        self.export_worker.start()
        self._update_buttons()

    def _export_failed(self, message: str) -> None:
        self.export_failed = True
        self.status.setText(f"导出失败：{message}")
        QMessageBox.warning(self, "导出失败", message)

    def _export_finished(self) -> None:
        worker, self.export_worker = self.export_worker, None
        if not self.export_failed:
            self.status.setText(f"已导出：{worker.destination}")
        self._update_buttons()
        worker.deleteLater()


class ImageToolsDialog(QDialog):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("离线图像增强")
        self.resize(1340, 820)
        self.tools = ImageToolsWidget(self)
        layout = QVBoxLayout(self)
        layout.addWidget(self.tools)

    def closeEvent(self, event: object) -> None:
        if self.tools.is_busy():
            self.tools.status.setText("请等待处理或导出完成后再关闭。")
            event.ignore()
        else:
            event.accept()

    def reject(self) -> None:
        if self.tools.is_busy():
            self.tools.status.setText("请等待处理或导出完成后再关闭。")
            return
        super().reject()
