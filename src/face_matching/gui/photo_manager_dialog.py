from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from ..database import FaceDatabase, FaceSample, Person


class PhotoManagerDialog(QDialog):
    def __init__(self, database: FaceDatabase, person: Person, parent=None) -> None:
        super().__init__(parent)
        self.database = database
        self.person = person
        self.changed = False
        self.samples: list[FaceSample] = []
        self.setWindowTitle(f"照片管理 · {person.name}")
        self.resize(720, 520)

        self.list_widget = QListWidget()
        self.list_widget.setViewMode(QListWidget.ViewMode.IconMode)
        self.list_widget.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.list_widget.setIconSize(QSize(150, 150))
        self.list_widget.setGridSize(QSize(190, 205))
        self.list_widget.setResizeMode(QListWidget.ResizeMode.Adjust)

        delete_button = QPushButton("删除所选照片")
        delete_button.clicked.connect(self._delete_selected)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)

        hint = QLabel("可删除质量较差或重复照片；每个人必须至少保留一张。")
        hint.setWordWrap(True)
        hint.setProperty("class", "hint")
        layout = QVBoxLayout(self)
        layout.addWidget(hint)
        layout.addWidget(self.list_widget, 1)
        layout.addWidget(delete_button)
        layout.addWidget(buttons)
        self._reload()

    def _reload(self) -> None:
        self.samples = self.database.list_face_samples(self.person.id)
        self.list_widget.clear()
        for sample in self.samples:
            item = QListWidgetItem(QIcon(str(sample.image_path)), f"质量 {sample.quality:.2f}")
            item.setData(Qt.ItemDataRole.UserRole, sample.id)
            item.setToolTip(str(sample.image_path))
            self.list_widget.addItem(item)

    def _delete_selected(self) -> None:
        selected = self.list_widget.selectedItems()
        if not selected:
            QMessageBox.information(self, "请选择照片", "请先选择要删除的照片。")
            return
        if len(selected) >= len(self.samples):
            QMessageBox.warning(self, "至少保留一张", "不能删除该人员的全部照片。")
            return
        answer = QMessageBox.question(
            self,
            "确认删除",
            f"确定删除所选 {len(selected)} 张照片及其人脸特征吗？",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            for item in selected:
                self.database.delete_face_sample(str(item.data(Qt.ItemDataRole.UserRole)))
        except Exception as exc:
            QMessageBox.critical(self, "删除失败", str(exc))
            return
        self.changed = True
        self._reload()
