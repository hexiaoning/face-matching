"""主窗口：左侧人员库管理，右侧视频源控制 + 实时画面 + 命中记录。"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import config
from ..db import Database
from ..engine import FaceEngine
from ..gallery import Gallery
from ..gpu import gpu_summary
from ..worker import VideoWorker
from .person_dialog import PersonDialog


class MainWindow(QMainWindow):
    def __init__(self, engine: FaceEngine, db: Database, gallery: Gallery):
        super().__init__()
        self.engine = engine
        self.db = db
        self.gallery = gallery
        self.worker: VideoWorker | None = None

        self.setWindowTitle("视频人脸比对系统")
        self.resize(1280, 800)
        self.statusBar().showMessage(
            f"就绪 | 模型: {engine.model_name} | 后端: {engine.backend} | {gpu_summary()}"
        )

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_db_panel())
        splitter.addWidget(self._build_video_panel())
        splitter.setSizes([380, 900])
        self.setCentralWidget(splitter)
        self._refresh_person_table()

    # ---- 左侧：人员库 ----

    def _build_db_panel(self) -> QWidget:
        box = QGroupBox("人员库（姓名 / 身份证号 / 照片）")
        layout = QVBoxLayout(box)

        self.person_table = QTableWidget(0, 3)
        self.person_table.setHorizontalHeaderLabels(["姓名", "身份证号", "照片数"])
        self.person_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.person_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.person_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.person_table)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("添加人员…")
        add_btn.clicked.connect(self._add_person)
        del_btn = QPushButton("删除选中人员")
        del_btn.clicked.connect(self._delete_person)
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self._refresh_person_table)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(del_btn)
        btn_row.addWidget(refresh_btn)
        layout.addLayout(btn_row)
        return box

    def _refresh_person_table(self) -> None:
        persons = self.db.list_persons()
        self.person_table.setRowCount(len(persons))
        for row, p in enumerate(persons):
            for col, key in enumerate(("name", "id_card", "photo_count")):
                item = QTableWidgetItem(str(p[key]))
                if col == 0:
                    item.setData(Qt.ItemDataRole.UserRole, p["id"])
                self.person_table.setItem(row, col, item)

    def _selected_person_id(self) -> int | None:
        row = self.person_table.currentRow()
        if row < 0:
            return None
        return self.person_table.item(row, 0).data(Qt.ItemDataRole.UserRole)

    def _add_person(self) -> None:
        dlg = PersonDialog(self)
        if not dlg.exec():
            return
        try:
            person_id = self.db.add_person(dlg.name, dlg.id_card)
        except Exception as e:  # noqa: BLE001 — sqlite UNIQUE 冲突等
            QMessageBox.warning(self, "添加失败", f"无法保存人员（身份证号可能已存在）：\n{e}")
            return

        person_dir = config.PHOTO_DIR / str(person_id)
        person_dir.mkdir(parents=True, exist_ok=True)
        saved, failed = 0, []
        for src in dlg.photo_paths:
            try:
                img = cv2.imdecode(np.fromfile(src, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("无法读取图片文件")
                embedding, _ = self.engine.embed_photo(img)
                dst = person_dir / Path(src).name
                shutil.copy2(src, dst)
                self.db.add_photo(person_id, str(dst), embedding)
                saved += 1
            except ValueError as e:
                failed.append(f"{Path(src).name}：{e}")

        self.gallery.reload()
        if self.worker is not None:
            self.worker.reload_gallery()
        self._refresh_person_table()

        if saved == 0:
            self.db.delete_person(person_id)
            shutil.rmtree(person_dir, ignore_errors=True)
            QMessageBox.warning(
                self, "添加失败",
                "所有照片都无法提取人脸特征，人员未保存。\n\n" + "\n".join(failed),
            )
        elif failed:
            QMessageBox.information(
                self, "部分照片未入库",
                f"已保存 {saved} 张照片。以下照片被跳过：\n\n" + "\n".join(failed),
            )

    def _delete_person(self) -> None:
        person_id = self._selected_person_id()
        if person_id is None:
            QMessageBox.information(self, "提示", "请先在表格中选择要删除的人员。")
            return
        ret = QMessageBox.question(
            self, "确认删除", "删除该人员及其全部照片和特征？"
        )
        if ret != QMessageBox.StandardButton.Yes:
            return
        self.db.delete_person(person_id)
        shutil.rmtree(config.PHOTO_DIR / str(person_id), ignore_errors=True)
        self.gallery.reload()
        if self.worker is not None:
            self.worker.reload_gallery()
        self._refresh_person_table()

    # ---- 右侧：视频 ----

    def _build_video_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 视频源控制行
        src_row = QHBoxLayout()
        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText(
            "视频文件路径 / 摄像头编号（如 0） / RTSP 地址（如 rtsp://…）"
        )
        browse_btn = QPushButton("浏览…")
        browse_btn.clicked.connect(self._browse_source)
        self.start_btn = QPushButton("开始")
        self.start_btn.clicked.connect(self._start)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        src_row.addWidget(QLabel("视频源："))
        src_row.addWidget(self.source_edit, 1)
        src_row.addWidget(browse_btn)
        src_row.addWidget(self.start_btn)
        src_row.addWidget(self.stop_btn)
        layout.addLayout(src_row)

        # 阈值滑块
        thr_row = QHBoxLayout()
        self.thr_slider = QSlider(Qt.Orientation.Horizontal)
        self.thr_slider.setRange(25, 75)
        self.thr_slider.setValue(int(config.DEFAULT_THRESHOLD * 100))
        self.thr_label = QLabel(f"相似度阈值：{config.DEFAULT_THRESHOLD:.2f}")
        self.thr_slider.valueChanged.connect(self._threshold_changed)
        thr_row.addWidget(self.thr_label)
        thr_row.addWidget(self.thr_slider, 1)
        layout.addLayout(thr_row)

        # 视频画面
        self.video_label = QLabel("未开始")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumHeight(400)
        self.video_label.setStyleSheet("background:#202020; color:#888;")
        layout.addWidget(self.video_label, 1)

        # 命中记录
        hit_box = QGroupBox("命中记录（双击表格可放大查看抓拍图所在目录）")
        hit_layout = QVBoxLayout(hit_box)
        self.hit_table = QTableWidget(0, 5)
        self.hit_table.setHorizontalHeaderLabels(["时间", "姓名", "身份证号", "分数", "抓拍图"])
        self.hit_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.hit_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.hit_table.setMaximumHeight(180)
        self.hit_table.doubleClicked.connect(self._open_snapshot_dir)
        hit_layout.addWidget(self.hit_table)
        layout.addWidget(hit_box)
        return panel

    def _browse_source(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mkv *.mov *.flv *.ts);;所有文件 (*)"
        )
        if path:
            self.source_edit.setText(path)

    def _threshold_changed(self, v: int) -> None:
        self.thr_label.setText(f"相似度阈值：{v / 100:.2f}")
        if self.worker is not None:
            self.worker.set_threshold(v / 100)

    def _start(self) -> None:
        source = self.source_edit.text().strip()
        if not source:
            QMessageBox.information(
                self, "提示", "请先选择视频文件，或输入摄像头编号 / RTSP 地址。"
            )
            return
        if self.gallery.size == 0:
            ret = QMessageBox.question(
                self, "人员库为空",
                "人员库中还没有任何人员，视频中所有人都会被标记为 Unknown。\n是否继续？",
            )
            if ret != QMessageBox.StandardButton.Yes:
                return

        self.worker = VideoWorker(source, self.engine, self.db, self.gallery)
        self.worker.set_threshold(self.thr_slider.value() / 100)
        self.worker.frame_ready.connect(self._show_frame)
        self.worker.match_event.connect(self._add_hit)
        self.worker.status.connect(self.statusBar().showMessage)
        self.worker.error.connect(self._worker_error)
        self.worker.finished_clean.connect(self._worker_finished)
        self.worker.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _stop(self) -> None:
        if self.worker is not None:
            self.worker.stop()

    def _worker_finished(self) -> None:
        if self.worker is not None:
            self.worker.wait(3000)
            self.worker = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _worker_error(self, msg: str) -> None:
        self.statusBar().showMessage(msg)
        QMessageBox.critical(self, "错误", msg)

    def _show_frame(self, frame_bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(img).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(pix)

    def _add_hit(self, event: dict) -> None:
        row = 0
        self.hit_table.insertRow(row)
        values = [event["time"], event["name"], event["id_card"], f"{event['score']:.3f}"]
        for col, v in enumerate(values):
            self.hit_table.setItem(row, col, QTableWidgetItem(str(v)))
        snap_item = QTableWidgetItem(event["snapshot"] or "")
        if event["snapshot"] and os.path.exists(event["snapshot"]):
            from PyQt6.QtCore import QSize
            from PyQt6.QtGui import QIcon
            snap_item = QTableWidgetItem(QIcon(QPixmap(event["snapshot"])), "")
            snap_item.setData(Qt.ItemDataRole.UserRole, event["snapshot"])
            snap_item.setToolTip(event["snapshot"])
            self.hit_table.setIconSize(QSize(72, 72))
            self.hit_table.setRowHeight(row, 76)
        self.hit_table.setItem(row, 4, snap_item)

    def _open_snapshot_dir(self) -> None:
        item = self.hit_table.currentItem()
        if item is None:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            from PyQt6.QtGui import QDesktopServices
            from PyQt6.QtCore import QUrl
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).parent)))

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(3000)
        super().closeEvent(event)
