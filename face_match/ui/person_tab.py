from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from face_match.identity import mask_id_number
from face_match.services import ApplicationServices
from face_match.ui.person_dialog import PersonDialog

PERSON_ID_ROLE = int(Qt.ItemDataRole.UserRole) + 20


class _EnrollmentWorker(QThread):
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, action: Callable[[], object], parent: Any | None = None) -> None:
        super().__init__(parent)
        self.action = action

    def run(self) -> None:
        try:
            result = self.action()
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.succeeded.emit(result)


class PersonTab(QWidget):
    busy_changed = Signal(bool)

    def __init__(self, services: ApplicationServices, parent: Any | None = None) -> None:
        super().__init__(parent)
        self.services = services
        self.worker: _EnrollmentWorker | None = None
        self.progress: QProgressDialog | None = None
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["姓名", "身份证号", "照片数", "更新时间"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.doubleClicked.connect(self.edit_selected)

        self.add_button = QPushButton("新增人员")
        self.edit_button = QPushButton("编辑")
        self.delete_button = QPushButton("删除")
        refresh_button = QPushButton("刷新")
        self.mutation_buttons = [self.add_button, self.edit_button, self.delete_button]
        self.add_button.clicked.connect(self.add_person)
        self.edit_button.clicked.connect(self.edit_selected)
        self.delete_button.clicked.connect(self.delete_selected)
        refresh_button.clicked.connect(self.refresh)
        toolbar = QHBoxLayout()
        toolbar.addWidget(self.add_button)
        toolbar.addWidget(self.edit_button)
        toolbar.addWidget(self.delete_button)
        toolbar.addStretch()
        toolbar.addWidget(refresh_button)
        layout = QVBoxLayout(self)
        layout.addLayout(toolbar)
        layout.addWidget(self.table)
        self.refresh()

    def set_video_running(self, running: bool) -> None:
        for button in self.mutation_buttons:
            button.setEnabled(not running and self.worker is None)
            button.setToolTip("请先停止视频识别再修改人员库" if running else "")

    def refresh(self) -> None:
        people = self.services.database.list_persons()
        self.table.setRowCount(len(people))
        for row, person in enumerate(people):
            name_item = QTableWidgetItem(person.name)
            name_item.setData(PERSON_ID_ROLE, person.id)
            self.table.setItem(row, 0, name_item)
            self.table.setItem(row, 1, QTableWidgetItem(mask_id_number(person.id_number)))
            count_item = QTableWidgetItem(str(person.photo_count))
            count_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 2, count_item)
            self.table.setItem(row, 3, QTableWidgetItem(person.updated_at.replace("T", " ")[:19]))
        self.table.resizeColumnsToContents()

    def _selected_person_id(self) -> int | None:
        row = self.table.currentRow()
        if row < 0 or not self.table.item(row, 0):
            return None
        return int(self.table.item(row, 0).data(PERSON_ID_ROLE))

    def add_person(self) -> None:
        dialog = PersonDialog(parent=self)
        if dialog.exec() != PersonDialog.DialogCode.Accepted:
            return
        data = dialog.data()
        self._run(
            lambda: self.services.enrollment.add_person(
                data.name, data.id_number, data.new_photo_paths
            ),
            "正在检测、对齐并提取人员照片特征……",
        )

    def edit_selected(self, *_: object) -> None:
        person_id = self._selected_person_id()
        if person_id is None:
            QMessageBox.information(self, "请选择人员", "请先在表格中选择一名人员。")
            return
        person = self.services.database.get_person(person_id)
        if person is None:
            self.refresh()
            return
        photos = self.services.database.list_photos(person_id)
        dialog = PersonDialog(person, photos, self)
        if dialog.exec() != PersonDialog.DialogCode.Accepted:
            return
        data = dialog.data()
        self._run(
            lambda: self.services.enrollment.update_person(
                person_id,
                data.name,
                data.id_number,
                data.new_photo_paths,
                data.remove_photo_ids,
            ),
            "正在更新人员照片特征……",
        )

    def delete_selected(self) -> None:
        person_id = self._selected_person_id()
        if person_id is None:
            QMessageBox.information(self, "请选择人员", "请先在表格中选择一名人员。")
            return
        person = self.services.database.get_person(person_id)
        if person is None:
            self.refresh()
            return
        answer = QMessageBox.question(
            self,
            "确认删除",
            f"确定删除“{person.name}”及其 {person.photo_count} 张照片和全部特征吗？\n此操作不可撤销。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._run(lambda: self.services.enrollment.delete_person(person_id), "正在删除人员……")

    def _run(self, action: Callable[[], object], message: str) -> None:
        if self.worker is not None:
            return
        self.busy_changed.emit(True)
        for button in self.mutation_buttons:
            button.setEnabled(False)
        self.progress = QProgressDialog(message, "", 0, 0, self)
        self.progress.setWindowTitle("请稍候")
        self.progress.setCancelButton(None)
        self.progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress.show()
        self.worker = _EnrollmentWorker(action, self)
        self.worker.succeeded.connect(self._completed)
        self.worker.failed.connect(self._failed)
        self.worker.finished.connect(self._cleanup_worker)
        self.worker.start()

    def _completed(self, _: object) -> None:
        self.services.refresh_matcher()
        self.refresh()

    def _failed(self, message: str) -> None:
        QMessageBox.critical(self, "人员库操作失败", message)

    def _cleanup_worker(self) -> None:
        if self.progress:
            self.progress.close()
            self.progress = None
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
        for button in self.mutation_buttons:
            button.setEnabled(True)
        self.busy_changed.emit(False)
