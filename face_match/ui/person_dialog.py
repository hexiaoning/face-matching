from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
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

from face_match.domain import Person, PersonPhoto
from face_match.identity import validate_identity

PATH_ROLE = int(Qt.ItemDataRole.UserRole) + 1
PHOTO_ID_ROLE = int(Qt.ItemDataRole.UserRole) + 2


@dataclass(frozen=True)
class PersonFormData:
    name: str
    id_number: str
    new_photo_paths: list[Path]
    remove_photo_ids: list[int]


class PersonDialog(QDialog):
    def __init__(
        self,
        person: Person | None = None,
        photos: list[PersonPhoto] | None = None,
        parent: Any | None = None,
    ) -> None:
        super().__init__(parent)
        self.person = person
        self.original_photo_ids = {photo.id for photo in photos or []}
        self.setWindowTitle("编辑人员" if person else "新增人员")
        self.resize(720, 520)
        self.name_edit = QLineEdit(person.name if person else "")
        self.name_edit.setPlaceholderText("请输入姓名")
        self.id_edit = QLineEdit(person.id_number if person else "")
        self.id_edit.setPlaceholderText("居民身份证、护照或其他证件号码")
        form = QFormLayout()
        form.addRow("姓名：", self.name_edit)
        form.addRow("身份证号：", self.id_edit)

        guidance = QLabel(
            "每人至少 1 张照片。为了监控画面识别更稳，建议录入清晰正面、左侧和右侧照片；"
            "每张照片只能包含一张人脸。"
        )
        guidance.setWordWrap(True)
        guidance.setObjectName("hintLabel")
        self.photo_list = QListWidget()
        self.photo_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.photo_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.photo_list.setIconSize(QSize(112, 112))
        self.photo_list.setSpacing(10)
        self.photo_list.setMinimumHeight(260)
        for photo in photos or []:
            self._append_photo(photo.path, photo.id, photo.original_name)

        add_button = QPushButton("添加照片…")
        remove_button = QPushButton("移除选中")
        add_button.clicked.connect(self._choose_photos)
        remove_button.clicked.connect(self._remove_selected)
        photo_buttons = QHBoxLayout()
        photo_buttons.addWidget(add_button)
        photo_buttons.addWidget(remove_button)
        photo_buttons.addStretch()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("保存")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(guidance)
        layout.addWidget(self.photo_list)
        layout.addLayout(photo_buttons)
        layout.addWidget(buttons)

    def _append_photo(self, path: Path, photo_id: int | None, label: str | None = None) -> None:
        item = QListWidgetItem(QIcon(str(path)), label or path.name)
        item.setData(PATH_ROLE, str(path))
        item.setData(PHOTO_ID_ROLE, photo_id)
        item.setToolTip(str(path))
        self.photo_list.addItem(item)

    def _choose_photos(self) -> None:
        filenames, _ = QFileDialog.getOpenFileNames(
            self,
            "选择人员照片",
            "",
            "图片 (*.jpg *.jpeg *.png *.bmp *.webp);;所有文件 (*)",
        )
        existing = {
            str(Path(self.photo_list.item(index).data(PATH_ROLE)).resolve())
            for index in range(self.photo_list.count())
        }
        for filename in filenames:
            path = Path(filename).resolve()
            if str(path) not in existing:
                self._append_photo(path, None)
                existing.add(str(path))

    def _remove_selected(self) -> None:
        for item in self.photo_list.selectedItems():
            self.photo_list.takeItem(self.photo_list.row(item))

    def accept(self) -> None:
        try:
            validate_identity(self.name_edit.text(), self.id_edit.text())
        except ValueError as exc:
            QMessageBox.warning(self, "信息不完整", str(exc))
            return
        if self.photo_list.count() < 1:
            QMessageBox.warning(self, "缺少照片", "每个人至少需要一张照片。")
            return
        super().accept()

    def data(self) -> PersonFormData:
        kept_ids: set[int] = set()
        new_paths: list[Path] = []
        for index in range(self.photo_list.count()):
            item = self.photo_list.item(index)
            photo_id = item.data(PHOTO_ID_ROLE)
            if photo_id is None:
                new_paths.append(Path(item.data(PATH_ROLE)))
            else:
                kept_ids.add(int(photo_id))
        return PersonFormData(
            name=self.name_edit.text(),
            id_number=self.id_edit.text(),
            new_photo_paths=new_paths,
            remove_photo_ids=sorted(self.original_photo_ids - kept_ids),
        )
