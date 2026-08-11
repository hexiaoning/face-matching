from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from ..database import Person


IMAGE_FILTER = "图片 (*.jpg *.jpeg *.png *.bmp *.webp)"


class EnrollmentDialog(QDialog):
    def __init__(self, person: Person | None = None, parent=None) -> None:
        super().__init__(parent)
        self.person = person
        self.setWindowTitle("修改人员" if person else "录入人员")
        self.setMinimumWidth(560)
        self.name_edit = QLineEdit(person.name if person else "")
        self.id_card_edit = QLineEdit(person.id_card if person else "")
        self.id_card_edit.setMaxLength(64)
        form = QFormLayout()
        form.addRow("姓名：", self.name_edit)
        form.addRow("身份证号：", self.id_card_edit)

        self.photo_list = QListWidget()
        self.photo_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        add_button = QPushButton("选择照片…")
        remove_button = QPushButton("移除所选")
        add_button.clicked.connect(self._add_photos)
        remove_button.clicked.connect(self._remove_photos)
        photo_buttons = QHBoxLayout()
        photo_buttons.addWidget(add_button)
        photo_buttons.addWidget(remove_button)
        photo_buttons.addStretch()

        existing = person.photo_count if person else 0
        help_label = QLabel(
            f"已有 {existing} 张照片。" if person else
            "请选择 1～多张只包含该人员的清晰照片；建议包含正面和轻微侧脸。"
        )
        help_label.setWordWrap(True)
        help_label.setProperty("class", "hint")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(QLabel("新增照片："))
        layout.addWidget(self.photo_list, 1)
        layout.addLayout(photo_buttons)
        layout.addWidget(help_label)
        layout.addWidget(buttons)

    @property
    def selected_photos(self) -> list[Path]:
        return [Path(self.photo_list.item(index).data(Qt.ItemDataRole.UserRole))
                for index in range(self.photo_list.count())]

    def _add_photos(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "选择人员照片", "", IMAGE_FILTER)
        existing = {str(path) for path in self.selected_photos}
        for value in paths:
            if value in existing:
                continue
            self.photo_list.addItem(Path(value).name)
            item = self.photo_list.item(self.photo_list.count() - 1)
            item.setData(Qt.ItemDataRole.UserRole, value)
            item.setToolTip(value)
            existing.add(value)

    def _remove_photos(self) -> None:
        for item in self.photo_list.selectedItems():
            self.photo_list.takeItem(self.photo_list.row(item))

    def _validate_and_accept(self) -> None:
        if not self.name_edit.text().strip() or not self.id_card_edit.text().strip():
            QMessageBox.warning(self, "信息不完整", "姓名和身份证号都不能为空。")
            return
        existing = self.person.photo_count if self.person else 0
        if existing + self.photo_list.count() < 1:
            QMessageBox.warning(self, "缺少照片", "每个人必须至少有一张照片。")
            return
        self.accept()
