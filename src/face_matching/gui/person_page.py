from __future__ import annotations

import sqlite3

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..database import FaceDatabase, Person
from ..engine import FaceEngine
from ..enrollment import prepare_enrollment_samples
from ..errors import EnrollmentError
from ..matcher import GalleryMatcher
from .enrollment_dialog import EnrollmentDialog


class PersonPage(QWidget):
    persons_changed = Signal()

    def __init__(
        self,
        database: FaceDatabase,
        engine: FaceEngine,
        matcher: GalleryMatcher,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.database = database
        self.engine = engine
        self.matcher = matcher
        self.people: list[Person] = []
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["姓名", "身份证号", "照片数", "录入时间"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.doubleClicked.connect(self.edit_person)

        add_button = QPushButton("＋ 录入人员")
        edit_button = QPushButton("修改 / 补照片")
        delete_button = QPushButton("删除人员")
        refresh_button = QPushButton("刷新")
        add_button.setProperty("class", "primary")
        add_button.clicked.connect(self.add_person)
        edit_button.clicked.connect(self.edit_person)
        delete_button.clicked.connect(self.delete_person)
        refresh_button.clicked.connect(self.refresh)
        controls = QHBoxLayout()
        controls.addWidget(add_button)
        controls.addWidget(edit_button)
        controls.addWidget(delete_button)
        controls.addStretch()
        controls.addWidget(refresh_button)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.table, 1)
        self.refresh()

    def refresh(self) -> None:
        self.people = self.database.list_people()
        self.table.setRowCount(len(self.people))
        for row, person in enumerate(self.people):
            values = (person.name, person.id_card, str(person.photo_count), person.created_at[:19])
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 2:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, column, item)
        self.matcher.reload()

    def selected_person(self) -> Person | None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        row = rows[0].row()
        return self.people[row] if 0 <= row < len(self.people) else None

    def add_person(self) -> None:
        dialog = EnrollmentDialog(parent=self)
        if dialog.exec() != EnrollmentDialog.DialogCode.Accepted:
            return
        samples = self._prepare(dialog.selected_photos)
        if samples is None:
            return
        try:
            self.database.add_person(
                dialog.name_edit.text(), dialog.id_card_edit.text(), samples, self.engine.config.model_id
            )
        except sqlite3.IntegrityError:
            QMessageBox.warning(self, "身份证号重复", "该身份证号已经存在于人员库中。")
            return
        except Exception as exc:
            QMessageBox.critical(self, "录入失败", str(exc))
            return
        self.refresh()
        self.persons_changed.emit()

    def edit_person(self) -> None:
        person = self.selected_person()
        if person is None:
            QMessageBox.information(self, "请选择人员", "请先在表格中选择一名人员。")
            return
        dialog = EnrollmentDialog(person, self)
        if dialog.exec() != EnrollmentDialog.DialogCode.Accepted:
            return
        samples = self._prepare(dialog.selected_photos)
        if samples is None:
            return
        try:
            self.database.update_person(
                person.id,
                dialog.name_edit.text(),
                dialog.id_card_edit.text(),
                samples,
                self.engine.config.model_id,
            )
        except sqlite3.IntegrityError:
            QMessageBox.warning(self, "身份证号重复", "该身份证号已经属于其他人员。")
            return
        except Exception as exc:
            QMessageBox.critical(self, "修改失败", str(exc))
            return
        self.refresh()
        self.persons_changed.emit()

    def delete_person(self) -> None:
        person = self.selected_person()
        if person is None:
            QMessageBox.information(self, "请选择人员", "请先在表格中选择一名人员。")
            return
        answer = QMessageBox.question(
            self,
            "确认删除",
            f"确定删除“{person.name}”（{person.id_card}）及其 {person.photo_count} 张照片吗？\n此操作不可撤销。",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.database.delete_person(person.id)
        except Exception as exc:
            QMessageBox.critical(self, "删除失败", str(exc))
            return
        self.refresh()
        self.persons_changed.emit()

    def _prepare(self, paths):
        if not paths:
            return []
        progress = QProgressDialog("正在分析照片…", "取消", 0, len(paths), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)

        def update(current: int, total: int, name: str) -> None:
            progress.setMaximum(total)
            progress.setValue(current)
            progress.setLabelText(f"正在分析：{name}")
            from PySide6.QtWidgets import QApplication
            QApplication.processEvents()
            if progress.wasCanceled():
                raise EnrollmentError("操作已取消")

        try:
            return prepare_enrollment_samples(paths, self.engine, update)
        except EnrollmentError as exc:
            if str(exc) != "操作已取消":
                QMessageBox.warning(self, "照片不符合要求", str(exc))
            return None
        except Exception as exc:
            QMessageBox.critical(self, "照片处理失败", str(exc))
            return None
        finally:
            progress.close()
