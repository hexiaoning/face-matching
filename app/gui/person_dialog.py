"""添加人员对话框：姓名 + 身份证号 + 1~N 张照片。"""
from __future__ import annotations

import re

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from PyQt6.QtCore import QSize
from PyQt6.QtGui import QIcon


class PersonDialog(QDialog):
    """收集新人员信息。accept 后通过 name / id_card / photo_paths 取值。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("添加人员")
        self.setMinimumWidth(480)
        self.photo_paths: list[str] = []

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("必填")
        self.id_edit = QLineEdit()
        self.id_edit.setPlaceholderText("必填，18 位身份证号")

        form = QFormLayout()
        form.addRow("姓名：", self.name_edit)
        form.addRow("身份证号：", self.id_edit)

        self.photo_list = QListWidget()
        self.photo_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.photo_list.setIconSize(QSize(96, 96))
        self.photo_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.photo_list.setMinimumHeight(140)

        add_btn = QPushButton("选择照片…")
        add_btn.clicked.connect(self._pick_photos)
        remove_btn = QPushButton("移除选中照片")
        remove_btn.clicked.connect(self._remove_selected)
        btn_row = QHBoxLayout()
        btn_row.addWidget(add_btn)
        btn_row.addWidget(remove_btn)
        btn_row.addStretch(1)

        hint = QLabel("每人至少 1 张照片，建议提供清晰的正面登记照；可提供多张不同角度照片提升识别率。")
        hint.setWordWrap(True)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(QLabel("照片："))
        layout.addWidget(self.photo_list)
        layout.addLayout(btn_row)
        layout.addWidget(hint)
        layout.addWidget(buttons)

    def _pick_photos(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择人员照片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp *.webp)"
        )
        for p in paths:
            if p not in self.photo_paths:
                self.photo_paths.append(p)
                item = QListWidgetItem(QIcon(QPixmap(p)), p.split("/")[-1])
                item.setToolTip(p)
                item.setData(Qt.ItemDataRole.UserRole, p)
                self.photo_list.addItem(item)

    def _remove_selected(self) -> None:
        for item in self.photo_list.selectedItems():
            self.photo_paths.remove(item.data(Qt.ItemDataRole.UserRole))
            self.photo_list.takeItem(self.photo_list.row(item))

    def _on_accept(self) -> None:
        name = self.name_edit.text().strip()
        id_card = self.id_edit.text().strip().upper()
        if not name:
            QMessageBox.warning(self, "信息不完整", "请填写姓名。")
            return
        if not re.fullmatch(r"\d{17}[\dX]", id_card):
            QMessageBox.warning(self, "信息不完整", "请填写正确的 18 位身份证号。")
            return
        if not self.photo_paths:
            QMessageBox.warning(self, "信息不完整", "请至少选择 1 张照片。")
            return
        self.name = name
        self.id_card = id_card
        self.accept()
