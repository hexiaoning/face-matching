from __future__ import annotations

from pathlib import Path

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

from ..domain import Person, PhotoRecord
from ..enrollment import EnrollmentService


class PersonDialog(QDialog):
    def __init__(
        self,
        enrollment: EnrollmentService,
        person: Person | None = None,
        photos: list[PhotoRecord] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.enrollment = enrollment
        self.person = person
        self._removed_photo_ids: set[int] = set()
        self._new_paths: list[Path] = []
        self.setWindowTitle("编辑人员" if person else "录入人员")
        self.setMinimumSize(680, 520)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.name_edit = QLineEdit(person.name if person else "")
        self.name_edit.setPlaceholderText("必填")
        self.id_edit = QLineEdit(person.id_number if person else "")
        self.id_edit.setPlaceholderText("必填，作为唯一标识")
        form.addRow("姓名", self.name_edit)
        form.addRow("身份证号", self.id_edit)
        layout.addLayout(form)

        hint = QLabel("建议录入 3–5 张清晰照片，包含正脸及左右侧脸；每张照片只能有一个人。")
        hint.setProperty("muted", True)
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.photo_list = QListWidget()
        self.photo_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.photo_list.setIconSize(QSize(118, 118))
        self.photo_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.photo_list.setMovement(QListWidget.Movement.Static)
        self.photo_list.setSpacing(8)
        layout.addWidget(self.photo_list, 1)

        for photo in photos or []:
            absolute = enrollment.absolute_photo_path(photo.path)
            item = QListWidgetItem(QIcon(str(absolute)), photo.source_name)
            item.setData(Qt.ItemDataRole.UserRole, ("existing", photo.id, str(absolute)))
            item.setToolTip(f"质量：{photo.quality:.2f}\n{absolute}")
            self.photo_list.addItem(item)

        photo_buttons = QHBoxLayout()
        add_button = QPushButton("添加照片…")
        remove_button = QPushButton("移除选中")
        add_button.clicked.connect(self._add_photos)
        remove_button.clicked.connect(self._remove_selected)
        photo_buttons.addWidget(add_button)
        photo_buttons.addWidget(remove_button)
        photo_buttons.addStretch()
        layout.addLayout(photo_buttons)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("保存")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def name(self) -> str:
        return self.name_edit.text().strip()

    @property
    def id_number(self) -> str:
        return self.id_edit.text().strip()

    @property
    def new_photo_paths(self) -> list[Path]:
        return list(self._new_paths)

    @property
    def removed_photo_ids(self) -> list[int]:
        return sorted(self._removed_photo_ids)

    def _add_photos(self) -> None:
        filenames, _ = QFileDialog.getOpenFileNames(
            self,
            "选择人员照片",
            "",
            "图像文件 (*.jpg *.jpeg *.png *.bmp *.webp);;所有文件 (*)",
        )
        existing = {str(path.resolve()).casefold() for path in self._new_paths}
        for filename in filenames:
            path = Path(filename).resolve()
            if str(path).casefold() in existing:
                continue
            existing.add(str(path).casefold())
            self._new_paths.append(path)
            item = QListWidgetItem(QIcon(str(path)), path.name)
            item.setData(Qt.ItemDataRole.UserRole, ("new", str(path)))
            item.setToolTip(str(path))
            self.photo_list.addItem(item)

    def _remove_selected(self) -> None:
        for item in self.photo_list.selectedItems():
            data = item.data(Qt.ItemDataRole.UserRole)
            if data and data[0] == "existing":
                self._removed_photo_ids.add(int(data[1]))
            elif data and data[0] == "new":
                path = Path(data[1])
                self._new_paths = [value for value in self._new_paths if value != path]
            self.photo_list.takeItem(self.photo_list.row(item))

    def _validate_and_accept(self) -> None:
        if not self.name:
            QMessageBox.warning(self, "信息不完整", "请输入姓名。")
            self.name_edit.setFocus()
            return
        if not self.id_number:
            QMessageBox.warning(self, "信息不完整", "请输入身份证号。")
            self.id_edit.setFocus()
            return
        if self.photo_list.count() < 1:
            QMessageBox.warning(self, "信息不完整", "请至少保留或添加一张照片。")
            return
        self.accept()

