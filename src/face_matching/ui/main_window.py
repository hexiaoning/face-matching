from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..config import AppPaths, RecognitionSettings
from ..database import Database, DuplicateIdNumberError
from ..domain import GPUInfo, Person, RecognitionEvent, mask_id_number
from ..enrollment import EnrollmentError, EnrollmentService
from ..face_engine import FaceEngine
from ..gallery import GalleryIndex
from ..video_worker import VideoWorker
from .dialogs import PersonDialog
from .widgets import VideoWidget


class MainWindow(QMainWindow):
    def __init__(
        self,
        engine: FaceEngine,
        database: Database,
        paths: AppPaths,
        gpu_info: GPUInfo,
        settings: RecognitionSettings,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.database = database
        self.paths = paths
        self.gpu_info = gpu_info
        self.settings = settings
        self.gallery = GalleryIndex()
        self.gallery.reload(database)
        self.enrollment = EnrollmentService(database, engine, paths.gallery)
        self.worker: VideoWorker | None = None
        self.current_source_name = ""
        self._person_rows: list[Person] = []
        self._gallery_buttons: list[QPushButton] = []

        self.setWindowTitle("监控视频人脸比对")
        self.resize(1440, 900)
        self.setMinimumSize(1080, 700)
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        heading = QLabel("监控视频人脸比对")
        heading_font = heading.font()
        heading_font.setPointSize(20)
        heading_font.setBold(True)
        heading.setFont(heading_font)
        subtitle = QLabel("GPU 强制推理 · 多照片底库 · 质量感知多帧确认")
        subtitle.setProperty("muted", True)
        root_layout.addWidget(heading)
        root_layout.addWidget(subtitle)
        self.tabs = QTabWidget()
        root_layout.addWidget(self.tabs, 1)
        self.tabs.addTab(self._build_monitor_page(), "实时识别")
        self.tabs.addTab(self._build_people_page(), "人员库")
        self.tabs.addTab(self._build_settings_page(), "设置与状态")
        self._refresh_people()
        memory = f" / {gpu_info.memory_mb} MB" if gpu_info.memory_mb else ""
        gpu_label = QLabel(f"GPU: {gpu_info.name}{memory} · Driver {gpu_info.driver_version}")
        self.statusBar().addPermanentWidget(gpu_label)
        self.statusBar().showMessage(
            f"就绪 · {self.gallery.person_count} 人 / {self.gallery.photo_count} 张照片"
        )

    def _build_monitor_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        toolbar = QHBoxLayout()
        self.video_button = QPushButton("打开视频…")
        self.camera_button = QPushButton("连接摄像头…")
        self.rtsp_button = QPushButton("连接 RTSP…")
        self.pause_button = QPushButton("暂停")
        self.stop_button = QPushButton("停止")
        self.video_button.setProperty("primary", True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.video_button.clicked.connect(self._choose_video)
        self.camera_button.clicked.connect(self._choose_camera)
        self.rtsp_button.clicked.connect(self._choose_rtsp)
        self.pause_button.clicked.connect(self._toggle_pause)
        self.stop_button.clicked.connect(self._stop_video)
        for button in (
            self.video_button,
            self.camera_button,
            self.rtsp_button,
            self.pause_button,
            self.stop_button,
        ):
            toolbar.addWidget(button)
        toolbar.addStretch()
        self.source_label = QLabel("未连接视频源")
        self.source_label.setProperty("muted", True)
        toolbar.addWidget(self.source_label)
        layout.addLayout(toolbar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.video_widget = VideoWidget()
        splitter.addWidget(self.video_widget)
        side = QWidget()
        side.setMinimumWidth(350)
        side.setMaximumWidth(470)
        side_layout = QVBoxLayout(side)
        title = QLabel("最近确认")
        title_font = title.font()
        title_font.setBold(True)
        title.setFont(title_font)
        side_layout.addWidget(title)
        self.events_table = QTableWidget(0, 4)
        self.events_table.setHorizontalHeaderLabels(["时间", "姓名", "身份证号", "分数"])
        self.events_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.events_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.events_table.verticalHeader().setVisible(False)
        header = self.events_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        side_layout.addWidget(self.events_table, 1)
        self.stats_label = QLabel("FPS: — · 当前人脸: 0")
        self.stats_label.setProperty("muted", True)
        side_layout.addWidget(self.stats_label)
        splitter.addWidget(side)
        splitter.setStretchFactor(0, 1)
        layout.addWidget(splitter, 1)
        return page

    def _build_people_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        toolbar = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("按姓名或身份证号搜索")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.textChanged.connect(self._refresh_people)
        add_button = QPushButton("录入人员…")
        edit_button = QPushButton("编辑…")
        delete_button = QPushButton("删除")
        refresh_button = QPushButton("刷新")
        add_button.setProperty("primary", True)
        delete_button.setProperty("danger", True)
        add_button.clicked.connect(self._add_person)
        edit_button.clicked.connect(self._edit_person)
        delete_button.clicked.connect(self._delete_person)
        refresh_button.clicked.connect(self._refresh_people)
        self._gallery_buttons = [add_button, edit_button, delete_button]
        toolbar.addWidget(self.search_edit, 1)
        toolbar.addWidget(add_button)
        toolbar.addWidget(edit_button)
        toolbar.addWidget(delete_button)
        toolbar.addWidget(refresh_button)
        layout.addLayout(toolbar)
        self.people_table = QTableWidget(0, 4)
        self.people_table.setHorizontalHeaderLabels(["姓名", "身份证号", "照片数", "更新时间"])
        self.people_table.setAlternatingRowColors(True)
        self.people_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.people_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.people_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.people_table.doubleClicked.connect(self._edit_person)
        self.people_table.verticalHeader().setVisible(False)
        header = self.people_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.people_table, 1)
        privacy = QLabel("身份证号在列表和识别结果中默认脱敏；编辑人员时可查看完整号码。所有数据仅保存在本机 data 目录。")
        privacy.setProperty("muted", True)
        privacy.setWordWrap(True)
        layout.addWidget(privacy)
        return page

    def _build_settings_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        recognition_group = QGroupBox("识别参数")
        form = QFormLayout(recognition_group)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.20, 0.90)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setValue(self.settings.similarity_threshold)
        self.quality_spin = QDoubleSpinBox()
        self.quality_spin.setRange(0.0, 0.9)
        self.quality_spin.setSingleStep(0.01)
        self.quality_spin.setDecimals(2)
        self.quality_spin.setValue(self.settings.minimum_quality)
        self.detector_combo = QComboBox()
        for value in (640, 960, 1280):
            self.detector_combo.addItem(f"{value} × {value}", value)
        self.detector_combo.setCurrentIndex(
            max(0, self.detector_combo.findData(self.settings.detector_size))
        )
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 20)
        self.interval_spin.setValue(self.settings.recognition_interval)
        self.confirmation_spin = QSpinBox()
        self.confirmation_spin.setRange(1, 10)
        self.confirmation_spin.setValue(self.settings.confirmation_hits)
        form.addRow("相似度阈值", self.threshold_spin)
        form.addRow("最低帧质量", self.quality_spin)
        form.addRow("检测分辨率", self.detector_combo)
        form.addRow("每 N 帧提取特征", self.interval_spin)
        form.addRow("连续命中次数", self.confirmation_spin)
        layout.addWidget(recognition_group)
        note = QLabel(
            "阈值越高误报越少、漏报越多。0.50 是未经现场标定的保守起点；正式部署应使用同一摄像头采集验证集，按目标 FAR 标定。"
        )
        note.setWordWrap(True)
        note.setProperty("muted", True)
        layout.addWidget(note)
        save_button = QPushButton("保存设置")
        save_button.setProperty("primary", True)
        save_button.clicked.connect(self._save_settings)
        layout.addWidget(save_button, alignment=Qt.AlignmentFlag.AlignLeft)

        status_group = QGroupBox("运行状态")
        status_form = QFormLayout(status_group)
        memory = f"{self.gpu_info.memory_mb} MB" if self.gpu_info.memory_mb else "未知"
        status_form.addRow("GPU", QLabel(self.gpu_info.name))
        status_form.addRow("显存", QLabel(memory))
        status_form.addRow("驱动", QLabel(self.gpu_info.driver_version))
        status_form.addRow("推理后端", QLabel("ONNX Runtime CUDA（CPU 回退已禁用）"))
        status_form.addRow("模型", QLabel(self.engine.model_name))
        status_form.addRow("人员库", QLabel(str(self.paths.database)))
        layout.addWidget(status_group)
        warning = QLabel(
            "模型许可提醒：项目代码可自由使用；默认 InsightFace 预训练权重仅限非商业研究。商业部署前必须取得模型授权。"
        )
        warning.setWordWrap(True)
        warning.setStyleSheet("color: #ffcb70;")
        layout.addWidget(warning)
        layout.addStretch()
        return page

    def _choose_video(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频",
            "",
            "视频文件 (*.mp4 *.avi *.mkv *.mov *.wmv *.m4v);;所有文件 (*)",
        )
        if filename:
            self._start_video(filename, Path(filename).name)

    def _choose_camera(self) -> None:
        index, accepted = QInputDialog.getInt(self, "连接摄像头", "摄像头编号", 0, 0, 32)
        if accepted:
            self._start_video(index, f"本机摄像头 {index}")

    def _choose_rtsp(self) -> None:
        url, accepted = QInputDialog.getText(
            self,
            "连接 RTSP",
            "RTSP 地址（例如 rtsp://user:password@192.168.1.10/stream）",
        )
        url = url.strip()
        if accepted and url:
            if not url.lower().startswith(("rtsp://", "rtmp://", "http://", "https://")):
                QMessageBox.warning(self, "地址无效", "请输入 RTSP、RTMP、HTTP 或 HTTPS 视频流地址。")
                return
            self._start_video(url, self._safe_source_name(url))

    @staticmethod
    def _safe_source_name(url: str) -> str:
        try:
            parsed = urlsplit(url)
            host = parsed.hostname or "网络视频流"
            port = f":{parsed.port}" if parsed.port else ""
            return urlunsplit((parsed.scheme, f"{host}{port}", parsed.path, "", ""))
        except ValueError:
            return "网络视频流"

    def _start_video(self, source: int | str, source_name: str) -> None:
        self._stop_video()
        if self.worker is not None:
            return
        try:
            self.engine.set_detector_size(self.settings.detector_size)
        except ValueError as exc:
            QMessageBox.warning(self, "检测参数无效", str(exc))
            return
        self.gallery.reload(self.database)
        self.current_source_name = source_name
        self.source_label.setText(source_name)
        self.worker = VideoWorker(
            source, source_name, self.engine, self.gallery, self.settings, self
        )
        self.worker.frame_ready.connect(self.video_widget.set_frame)
        self.worker.recognition.connect(self._on_recognition)
        self.worker.statistics.connect(self._on_statistics)
        self.worker.status_changed.connect(self.statusBar().showMessage)
        self.worker.failed.connect(self._on_video_error)
        self.worker.finished.connect(self._on_worker_finished)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.pause_button.setText("暂停")
        self._set_gallery_editing_enabled(False)
        self.worker.start()

    def _toggle_pause(self) -> None:
        if not self.worker:
            return
        paused = not self.worker.paused
        self.worker.set_paused(paused)
        self.pause_button.setText("继续" if paused else "暂停")

    def _stop_video(self) -> None:
        worker = self.worker
        if worker is None:
            return
        worker.stop()
        if not worker.wait(5000):
            QMessageBox.warning(self, "正在停止", "GPU 正在完成当前帧，视频线程稍后会停止。")
            return
        self._on_worker_finished()

    def _on_worker_finished(self) -> None:
        worker = self.worker
        if worker is not None and worker.isRunning():
            return
        if worker is not None:
            worker.deleteLater()
        self.worker = None
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.pause_button.setText("暂停")
        self.source_label.setText("未连接视频源")
        self._set_gallery_editing_enabled(True)

    def _on_video_error(self, message: str) -> None:
        QMessageBox.critical(self, "视频错误", message)
        self.statusBar().showMessage(message)

    def _on_statistics(self, fps: float, faces: int) -> None:
        self.stats_label.setText(f"FPS: {fps:.1f} · 当前人脸: {faces}")

    def _on_recognition(self, event: RecognitionEvent) -> None:
        try:
            self.database.log_recognition(
                event.match.person_id,
                event.source,
                event.track_id,
                event.match.score,
                event.quality,
            )
        except Exception as exc:
            self.statusBar().showMessage(f"识别成功，但记录事件失败：{exc}")
        self.events_table.insertRow(0)
        values = (
            event.occurred_at.strftime("%H:%M:%S"),
            event.match.name,
            mask_id_number(event.match.id_number),
            f"{event.match.score:.3f}",
        )
        for column, value in enumerate(values):
            self.events_table.setItem(0, column, QTableWidgetItem(value))
        while self.events_table.rowCount() > 100:
            self.events_table.removeRow(self.events_table.rowCount() - 1)

    def _refresh_people(self, *_args) -> None:
        search = self.search_edit.text() if hasattr(self, "search_edit") else ""
        self._person_rows = self.database.list_persons(search)
        self.people_table.setRowCount(len(self._person_rows))
        for row, person in enumerate(self._person_rows):
            values = (
                person.name,
                mask_id_number(person.id_number),
                str(person.photo_count),
                person.updated_at.replace("T", " ")[:19],
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, person.id)
                self.people_table.setItem(row, column, item)

    def _selected_person(self) -> Person | None:
        row = self.people_table.currentRow()
        if row < 0 or row >= len(self._person_rows):
            QMessageBox.information(self, "未选择人员", "请先在表格中选择一个人员。")
            return None
        return self._person_rows[row]

    def _add_person(self) -> None:
        dialog = PersonDialog(self.enrollment, parent=self)
        if dialog.exec() != PersonDialog.DialogCode.Accepted:
            return
        self._with_busy_enrollment(
            lambda: self.enrollment.create_person(
                dialog.name, dialog.id_number, dialog.new_photo_paths
            )
        )

    def _edit_person(self, *_args) -> None:
        if self.worker is not None:
            QMessageBox.information(self, "视频正在运行", "请先停止视频，再修改人员库。")
            return
        person = self._selected_person()
        if not person:
            return
        dialog = PersonDialog(
            self.enrollment,
            person=person,
            photos=self.database.list_photos(person.id),
            parent=self,
        )
        if dialog.exec() != PersonDialog.DialogCode.Accepted:
            return
        self._with_busy_enrollment(
            lambda: self.enrollment.update_person(
                person.id,
                dialog.name,
                dialog.id_number,
                dialog.new_photo_paths,
                dialog.removed_photo_ids,
            )
        )

    def _with_busy_enrollment(self, operation) -> None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            report = operation()
        except (EnrollmentError, DuplicateIdNumberError, ValueError) as exc:
            QMessageBox.warning(self, "无法保存人员", str(exc))
            return
        except Exception as exc:
            QMessageBox.critical(self, "保存失败", f"保存人员时发生错误：{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.gallery.reload(self.database)
        self._refresh_people()
        self.statusBar().showMessage(
            f"人员库已更新 · {self.gallery.person_count} 人 / {self.gallery.photo_count} 张照片",
            6000,
        )
        if report.warnings:
            QMessageBox.information(self, "已保存，建议改善照片", "\n".join(report.warnings))

    def _delete_person(self) -> None:
        person = self._selected_person()
        if not person:
            return
        answer = QMessageBox.question(
            self,
            "确认删除",
            f"确定删除“{person.name}”及其 {person.photo_count} 张底库照片吗？此操作不可撤销。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.enrollment.delete_person(person.id)
            self.gallery.reload(self.database)
            self._refresh_people()
        except Exception as exc:
            QMessageBox.critical(self, "删除失败", str(exc))

    def _set_gallery_editing_enabled(self, enabled: bool) -> None:
        for button in self._gallery_buttons:
            button.setEnabled(enabled)

    def _save_settings(self) -> None:
        updated = RecognitionSettings(
            similarity_threshold=self.threshold_spin.value(),
            detection_threshold=self.settings.detection_threshold,
            minimum_quality=self.quality_spin.value(),
            detector_size=int(self.detector_combo.currentData()),
            recognition_interval=self.interval_spin.value(),
            confirmation_hits=self.confirmation_spin.value(),
            max_track_embeddings=self.settings.max_track_embeddings,
            track_ttl_frames=self.settings.track_ttl_frames,
        )
        try:
            updated.save(self.paths.settings)
            if self.worker is None:
                self.engine.set_detector_size(updated.detector_size)
        except Exception as exc:
            QMessageBox.warning(self, "设置无效", str(exc))
            return
        self.settings = updated
        if self.worker:
            QMessageBox.information(self, "设置已保存", "新参数会在下一次打开视频源时生效。")
        else:
            self.statusBar().showMessage("设置已保存", 4000)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        if self.worker:
            self.worker.stop()
            if not self.worker.wait(5000):
                event.ignore()
                QMessageBox.warning(self, "无法退出", "视频线程尚未停止，请稍后重试。")
                return
        self.database.close()
        event.accept()
