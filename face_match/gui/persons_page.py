"""人员库管理页：增删人员、编辑姓名/身份证号、管理多张照片。"""
from __future__ import annotations

import cv2
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMessageBox, QPushButton, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget, QAbstractItemView,
)

from ..database import PersonDB, Person
from .widgets import ModelHub


class _EmbedWorker(QThread):
    """后台算照片 embedding，避免卡界面。"""
    done = Signal(int, bool, str)  # photo_id, ok, message

    def __init__(self, hub: ModelHub, db: PersonDB, photo_id: int, path: str):
        super().__init__()
        self.hub, self.db, self.photo_id, self.path = hub, db, photo_id, path

    def run(self) -> None:
        img = cv2.imread(self.path)
        if img is None:
            self.done.emit(self.photo_id, False, "无法读取图片文件")
            return
        try:
            emb, q, _ = self.hub.embed_photo(img)
        except ValueError as e:
            self.done.emit(self.photo_id, False, str(e))
            return
        except Exception as e:  # noqa: BLE001
            self.done.emit(self.photo_id, False, f"识别模型出错: {e}")
            return
        self.db.set_photo_embedding(self.photo_id, emb, q)
        self.done.emit(self.photo_id, True, f"录入成功 (质量分 {q:.2f})")


class PersonsPage(QWidget):
    def __init__(self, db: PersonDB, hub: ModelHub, on_gallery_changed):
        super().__init__()
        self.db = db
        self.hub = hub
        self.on_gallery_changed = on_gallery_changed
        self._workers: list[_EmbedWorker] = []
        self._current_id: int | None = None

        root = QHBoxLayout(self)
        split = QSplitter()
        root.addWidget(split)

        # 左：人员列表
        left = QWidget()
        ll = QVBoxLayout(left)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["姓名", "身份证号", "照片数"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._on_select)
        ll.addWidget(self.table)
        btns = QHBoxLayout()
        self.btn_add = QPushButton("新增人员")
        self.btn_del = QPushButton("删除人员")
        self.btn_add.clicked.connect(self._add_person)
        self.btn_del.clicked.connect(self._delete_person)
        btns.addWidget(self.btn_add)
        btns.addWidget(self.btn_del)
        ll.addLayout(btns)
        split.addWidget(left)

        # 右：人员详情
        right = QWidget()
        rl = QVBoxLayout(right)
        form = QFormLayout()
        self.edit_name = QLineEdit()
        self.edit_idn = QLineEdit()
        self.btn_save = QPushButton("保存信息")
        self.btn_save.clicked.connect(self._save_info)
        form.addRow("姓名", self.edit_name)
        form.addRow("身份证号", self.edit_idn)
        form.addRow("", self.btn_save)
        rl.addLayout(form)

        self.photos = QListWidget()
        self.photos.setViewMode(QListWidget.ViewMode.IconMode)
        from PySide6.QtCore import QSize
        self.photos.setIconSize(QSize(120, 120))
        self.photos.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.photos.setSpacing(8)
        rl.addWidget(QLabel("照片（清晰正脸照效果最好，可添加多张）"))
        rl.addWidget(self.photos, 1)
        pbtns = QHBoxLayout()
        self.btn_add_photo = QPushButton("添加照片…")
        self.btn_del_photo = QPushButton("删除选中照片")
        self.btn_add_photo.clicked.connect(self._add_photos)
        self.btn_del_photo.clicked.connect(self._delete_photo)
        pbtns.addWidget(self.btn_add_photo)
        pbtns.addWidget(self.btn_del_photo)
        rl.addLayout(pbtns)
        split.addWidget(right)
        split.setSizes([420, 560])

        self.reload()

    # ---- 数据 ----
    def reload(self) -> None:
        self._persons = self.db.list_persons()
        self.table.setRowCount(len(self._persons))
        for r, p in enumerate(self._persons):
            self.table.setItem(r, 0, QTableWidgetItem(p.name))
            self.table.setItem(r, 1, QTableWidgetItem(p.id_number))
            ok = sum(1 for ph in p.photos if ph.embedding is not None)
            self.table.setItem(r, 2, QTableWidgetItem(f"{ok}/{len(p.photos)}"))
        self._show_detail(self._selected_person())

    def _on_select(self) -> None:
        self._show_detail(self._selected_person())

    def _selected_person(self) -> Person | None:
        rows = {i.row() for i in self.table.selectedIndexes()}
        if not rows:
            return None
        r = sorted(rows)[0]
        return self._persons[r] if 0 <= r < len(self._persons) else None

    def _show_detail(self, p: Person | None) -> None:
        self._current_id = p.id if p else None
        self.edit_name.setText(p.name if p else "")
        self.edit_idn.setText(p.id_number if p else "")
        self.photos.clear()
        if not p:
            return
        for ph in p.photos:
            img = cv2.imread(ph.path)
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, ph.id)
            if img is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qi = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
                item.setIcon(QPixmap.fromImage(qi))
            mark = "✓" if ph.embedding is not None else "✗"
            item.setText(f"{mark} q={ph.quality:.2f}")
            item.setToolTip(ph.path)
            self.photos.addItem(item)

    # ---- 操作 ----
    def _add_person(self) -> None:
        pid = self.db.add_person("未命名", "")
        self.reload()
        # 选中新行
        for r, p in enumerate(self._persons):
            if p.id == pid:
                self.table.selectRow(r)
                self.edit_name.setFocus()
                self.edit_name.selectAll()
                break

    def _save_info(self) -> None:
        if self._current_id is None:
            return
        name = self.edit_name.text().strip()
        if not name:
            QMessageBox.warning(self, "提示", "姓名不能为空")
            return
        self.db.update_person(self._current_id, name, self.edit_idn.text().strip())
        self.reload()

    def _delete_person(self) -> None:
        p = self._selected_person()
        if not p:
            return
        if QMessageBox.question(self, "确认", f"删除人员「{p.name}」及其全部照片？") \
                == QMessageBox.StandardButton.Yes:
            self.db.delete_person(p.id)
            self.on_gallery_changed()
            self.reload()

    def _add_photos(self) -> None:
        if self._current_id is None:
            QMessageBox.information(self, "提示", "请先选择或新增人员")
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择照片（可多选）", "",
            "图片 (*.jpg *.jpeg *.png *.bmp *.webp);;所有文件 (*)")
        if not paths:
            return
        for path in paths:
            photo_id = self.db.add_photo(self._current_id, path)
            w = _EmbedWorker(self.hub, self.db, photo_id, path)
            w.done.connect(self._on_embed_done)
            self._workers.append(w)
            w.start()

    def _on_embed_done(self, photo_id: int, ok: bool, msg: str) -> None:
        if not ok:
            self.db.delete_photo(photo_id)
            QMessageBox.warning(self, "照片录入失败", msg)
        self.on_gallery_changed()
        self.reload()

    def _delete_photo(self) -> None:
        items = self.photos.selectedItems()
        if not items:
            return
        for it in items:
            self.db.delete_photo(int(it.data(Qt.ItemDataRole.UserRole)))
        self.on_gallery_changed()
        self.reload()
