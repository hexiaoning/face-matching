from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QVBoxLayout, QWidget,
)

from ..database import PersonRecord, StoredSample


class PersonPhotosDialog(QDialog):
    def __init__(self, person: PersonRecord, samples: list[StoredSample], parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{person.name} · 已录入照片")
        self.resize(900, 650)
        self.samples = samples
        self.image = QImage()
        self.photos = QListWidget()
        self.photos.setIconSize(QSize(100, 100))
        self.photos.setMaximumWidth(210)
        self.preview = QLabel("没有录入照片")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(400, 360)
        self.preview.setStyleSheet("background:#111820;color:#c5d1de;")
        self.caption = QLabel()
        self.caption.setAlignment(Qt.AlignCenter)
        self.caption.setWordWrap(True)
        self.previous = QPushButton("上一张")
        self.next = QPushButton("下一张")
        self.previous.clicked.connect(lambda: self.photos.setCurrentRow(self.photos.currentRow() - 1))
        self.next.clicked.connect(lambda: self.photos.setCurrentRow(self.photos.currentRow() + 1))
        nav = QHBoxLayout()
        nav.addWidget(self.previous)
        nav.addWidget(self.caption, 1)
        nav.addWidget(self.next)
        pane = QVBoxLayout()
        pane.addWidget(self.preview, 1)
        pane.addLayout(nav)
        body = QHBoxLayout()
        body.addWidget(self.photos)
        body.addLayout(pane, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.button(QDialogButtonBox.Close).setText("关闭")
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"{person.name} · 身份证号：{person.id_card} · {len(samples)} 张照片"))
        layout.addLayout(body, 1)
        layout.addWidget(buttons)
        for index, sample in enumerate(samples, 1):
            image = QImage(sample.image_path)
            item = QListWidgetItem(f"照片 {index}\n{Path(sample.image_path).name}")
            item.setToolTip(sample.image_path)
            if not image.isNull():
                item.setIcon(QIcon(QPixmap.fromImage(image).scaled(
                    100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )))
            self.photos.addItem(item)
        self.photos.currentRowChanged.connect(self._select)
        if samples:
            self.photos.setCurrentRow(0)
        else:
            self.previous.setEnabled(False)
            self.next.setEnabled(False)

    def _select(self, row: int) -> None:
        if not 0 <= row < len(self.samples):
            return
        self.image = QImage(self.samples[row].image_path)
        self.caption.setText(f"{row + 1} / {len(self.samples)} · {Path(self.samples[row].image_path).name}")
        self.previous.setEnabled(row > 0)
        self.next.setEnabled(row + 1 < len(self.samples))
        self._render()

    def _render(self) -> None:
        if self.image.isNull():
            self.preview.clear()
            self.preview.setText("照片文件不存在或无法读取")
            return
        self.preview.setPixmap(QPixmap.fromImage(self.image).scaled(
            self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)
        if hasattr(self, "image") and not self.image.isNull():
            self._render()
