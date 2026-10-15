from __future__ import annotations

import json
from pathlib import Path

import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QPushButton,
    QVBoxLayout, QWidget,
)

from ..database import PersonRecord, StoredSample
from ..vision.io import is_image_source, read_image


class ImagePane(QLabel):
    def __init__(self, empty_text: str) -> None:
        super().__init__()
        self.empty_text = empty_text
        self.image = QImage()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(280, 260)
        self.setStyleSheet("background:#111820;color:#c5d1de;")
        self.set_image(self.image)

    def set_image(self, image: QImage) -> None:
        self.image = image
        self._render()

    def _render(self) -> None:
        if self.image.isNull():
            self.clear()
            self.setText(self.empty_text)
        else:
            self.setPixmap(QPixmap.fromImage(self.image).scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)
        self._render()


def source_frame(record: dict) -> QImage:
    """Fallback for older records without a saved frame; never open live streams."""
    source = Path(record["source"])
    if not source.is_file():
        return QImage()
    try:
        if is_image_source(str(source)):
            frame = read_image(source)
        else:
            capture = cv2.VideoCapture(str(source))
            try:
                anchor = max(0, record["frame_index"] - 1)
                if not capture.isOpened() or (anchor and not capture.set(cv2.CAP_PROP_POS_FRAMES, anchor)):
                    return QImage()
                position = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
                if position > anchor:
                    return QImage()
                for _ in range(anchor - position):
                    if not capture.grab():
                        return QImage()
                ok, frame = capture.read()
                if not ok:
                    return QImage()
            finally:
                capture.release()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888).copy()
    except (OSError, ValueError, cv2.error):
        return QImage()


class AnnotationCompareDialog(QDialog):
    def __init__(self, record: dict, person: PersonRecord | None,
                 samples: list[StoredSample], parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"标注样本对比 · #{record['id']}")
        self.resize(1100, 700)
        self.samples = samples
        self.sample_preview = ImagePane("标注样本文件不存在或无法读取")
        self.portrait_preview = ImagePane("人员照片文件不存在或无法读取")
        self.sample_choices = QComboBox()
        self.portrait_choices = QComboBox()
        self.sample_images = []
        frame = QImage(record["snapshot_path"]) if record.get("snapshot_path") else QImage()
        if frame.isNull():
            frame = source_frame(record)
        if not frame.isNull():
            bbox = json.loads(record["bbox"]) if record.get("bbox") else None
            if bbox:
                painter = QPainter(frame)
                painter.setPen(QPen(QColor("#ffca49"), max(2, frame.width() // 400)))
                x1, y1, x2, y2 = (int(value) for value in bbox)
                painter.drawRect(x1, y1, x2 - x1, y2 - y1)
                painter.end()
            self.sample_images.append(frame)
            self.sample_choices.addItem("标注原图 / 视频帧（框出对应人脸）")
        face = QImage(record["face_path"]) if record.get("face_path") else QImage()
        if not face.isNull():
            self.sample_images.append(face)
            self.sample_choices.addItem("标注人脸截图")
        if not self.sample_images:
            self.sample_images.append(QImage())
            self.sample_choices.addItem("样本不可用")
        for index, sample in enumerate(samples, 1):
            self.portrait_choices.addItem(f"照片 {index} · {Path(sample.image_path).name}")
        previous = QPushButton("上一张")
        next_button = QPushButton("下一张")
        previous.clicked.connect(lambda: self.portrait_choices.setCurrentIndex(self.portrait_choices.currentIndex() - 1))
        next_button.clicked.connect(lambda: self.portrait_choices.setCurrentIndex(self.portrait_choices.currentIndex() + 1))

        def select_portrait(row: int) -> None:
            self.portrait_preview.set_image(QImage(samples[row].image_path) if 0 <= row < len(samples) else QImage())
            previous.setEnabled(row > 0)
            next_button.setEnabled(0 <= row < len(samples) - 1)

        self.sample_choices.currentIndexChanged.connect(
            lambda row: self.sample_preview.set_image(self.sample_images[row])
        )
        self.portrait_choices.currentIndexChanged.connect(select_portrait)
        left = QVBoxLayout()
        left.addWidget(QLabel("标注样本"))
        left.addWidget(self.sample_preview, 1)
        left.addWidget(self.sample_choices)
        right = QVBoxLayout()
        right.addWidget(QLabel(f"人员库 · {person.name}" if person else "人员已删除"))
        right.addWidget(self.portrait_preview, 1)
        nav = QHBoxLayout()
        nav.addWidget(previous)
        nav.addWidget(self.portrait_choices, 1)
        nav.addWidget(next_button)
        right.addLayout(nav)
        body = QHBoxLayout()
        body.addLayout(left, 1)
        body.addLayout(right, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.button(QDialogButtonBox.Close).setText("关闭")
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        details = QLabel(f"轨迹 #{record['track_id']} · 帧 {record['frame_index']} · 来源：{record['source']}")
        details.setWordWrap(True)
        layout.addWidget(details)
        layout.addLayout(body, 1)
        layout.addWidget(buttons)
        self.sample_preview.set_image(self.sample_images[0])
        select_portrait(0 if samples else -1)
        if not samples:
            self.portrait_preview.empty_text = "该人员没有可用的录入照片" if person else "人员已删除，无法查看录入照片"
            self.portrait_preview.set_image(QImage())
