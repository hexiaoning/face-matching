from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from face_matching.config import AppConfig
from face_matching.database import PeopleDatabase
from face_matching.errors import FaceMatchingError, GpuUnavailableError, ModelError
from face_matching.gpu import load_onnxruntime_gpu
from face_matching.inference import FaceEngine
from face_matching.matching import IdentityMatcher
from face_matching.model_manager import download_research_models, models_ready
from face_matching.security import LocalVault
from face_matching.video_worker import FramePayload, TrackView, VideoWorker


def _read_image(path: Path) -> np.ndarray | None:
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


class ModelDownloadThread(QThread):
    progress = Signal(int, int)
    succeeded = Signal()
    failed = Signal(str)

    def __init__(self, model_dir: Path, parent=None):
        super().__init__(parent)
        self.model_dir = model_dir

    def run(self) -> None:
        try:
            download_research_models(
                self.model_dir,
                lambda current, total: self.progress.emit(current, total),
                accept_research_license=True,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.succeeded.emit()


class AddPersonDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("录入人员")
        self.resize(560, 240)
        self.name_edit = QLineEdit()
        self.id_edit = QLineEdit()
        self.id_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.id_visible = QCheckBox("显示身份证号")
        self.id_visible.toggled.connect(
            lambda checked: self.id_edit.setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )
        self.photos_edit = QLineEdit()
        self.photos_edit.setReadOnly(True)
        self.photo_paths: list[Path] = []
        choose_button = QPushButton("选择 1～多张照片…")
        choose_button.clicked.connect(self._choose_photos)
        photo_row = QHBoxLayout()
        photo_row.addWidget(self.photos_edit, 1)
        photo_row.addWidget(choose_button)
        form = QFormLayout()
        form.addRow("姓名", self.name_edit)
        id_row = QHBoxLayout()
        id_row.addWidget(self.id_edit, 1)
        id_row.addWidget(self.id_visible)
        form.addRow("身份证号", id_row)
        form.addRow("注册照片", photo_row)
        hint = QLabel("建议同时提供正面、左右侧脸和不同光照照片；每张照片只应有一个主要人脸。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #607080")
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._validate)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(hint)
        layout.addStretch()
        layout.addWidget(buttons)

    def _choose_photos(self) -> None:
        names, _ = QFileDialog.getOpenFileNames(
            self,
            "选择注册照片",
            "",
            "图片 (*.jpg *.jpeg *.png *.bmp *.webp)",
        )
        if names:
            self.photo_paths = [Path(name) for name in names]
            self.photos_edit.setText(f"已选择 {len(names)} 张")

    def _validate(self) -> None:
        if not self.name_edit.text().strip():
            QMessageBox.warning(self, "信息不完整", "请输入姓名。")
            return
        if not self.id_edit.text().strip():
            QMessageBox.warning(self, "信息不完整", "请输入身份证号。")
            return
        if not self.photo_paths:
            QMessageBox.warning(self, "信息不完整", "请至少选择一张照片。")
            return
        self.accept()


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.vault = LocalVault(config.root / "master.key")
        self.database = PeopleDatabase(config.database_path, config.enrollment_dir, self.vault)
        self.engine: FaceEngine | None = None
        self.matcher = IdentityMatcher(config.match_threshold, config.min_top2_margin)
        self.video_worker: VideoWorker | None = None
        self.download_thread: ModelDownloadThread | None = None
        self.progress_dialog: QProgressDialog | None = None
        self._current_frame: QImage | None = None
        self._current_tracks: tuple[TrackView, ...] = ()
        self.setWindowTitle("监控视频人脸比对（GPU）")
        self.resize(1280, 820)
        self.setMinimumSize(980, 680)
        self._build_ui()
        self._refresh_people()
        QTimer.singleShot(0, self._ensure_models)

    def _build_ui(self) -> None:
        tabs = QTabWidget()
        tabs.addTab(self._build_recognition_tab(), "视频识别")
        tabs.addTab(self._build_people_tab(), "人员库")
        tabs.addTab(self._build_settings_tab(), "设置与模型")
        self.setCentralWidget(tabs)
        self.statusBar().showMessage("正在检查模型和 CUDA 环境…")

    def _build_recognition_tab(self) -> QWidget:
        page = QWidget()
        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("视频文件、摄像头编号或 RTSP 地址")
        file_button = QPushButton("视频文件…")
        file_button.clicked.connect(self._choose_video)
        camera_button = QPushButton("本机摄像头")
        camera_button.clicked.connect(lambda: self.source_edit.setText("0"))
        rtsp_button = QPushButton("RTSP…")
        rtsp_button.clicked.connect(self._enter_rtsp)
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("视频源"))
        source_layout.addWidget(self.source_edit, 1)
        source_layout.addWidget(file_button)
        source_layout.addWidget(camera_button)
        source_layout.addWidget(rtsp_button)
        self.start_button = QPushButton("开始识别")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self._start_video)
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop_video)
        controls = QHBoxLayout()
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        controls.addStretch()
        self.gpu_label = QLabel("GPU：尚未初始化")
        controls.addWidget(self.gpu_label)
        self.video_label = QLabel("选择视频源后开始识别")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 420)
        self.video_label.setStyleSheet("background: #101820; color: #b9c5ce; border-radius: 6px")
        self.track_table = QTableWidget(0, 6)
        self.track_table.setHorizontalHeaderLabels(
            ["轨迹", "判断", "姓名/候选", "相似度", "画质", "连续确认"]
        )
        self.track_table.verticalHeader().setVisible(False)
        self.track_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.track_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.track_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.track_table.setMaximumHeight(210)
        notice = QLabel(
            "提示：人脸识别结果属于概率判断，应由人工复核；模糊、遮挡、极端侧脸会提高误识风险。"
        )
        notice.setStyleSheet("color: #9a5a00; background: #fff7e6; padding: 8px")
        notice.setWordWrap(True)
        layout = QVBoxLayout(page)
        layout.addLayout(source_layout)
        layout.addLayout(controls)
        layout.addWidget(self.video_label, 1)
        layout.addWidget(self.track_table)
        layout.addWidget(notice)
        return page

    def _build_people_tab(self) -> QWidget:
        page = QWidget()
        self.people_table = QTableWidget(0, 4)
        self.people_table.setHorizontalHeaderLabels(
            ["内部编号", "姓名", "身份证号（脱敏）", "照片数"]
        )
        self.people_table.verticalHeader().setVisible(False)
        self.people_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.people_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.people_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.people_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.add_person_button = QPushButton("录入人员…")
        self.add_person_button.setEnabled(False)
        self.add_person_button.clicked.connect(self._add_person)
        self.delete_person_button = QPushButton("删除所选")
        self.delete_person_button.clicked.connect(self._delete_person)
        refresh_button = QPushButton("刷新")
        refresh_button.clicked.connect(self._refresh_people)
        controls = QHBoxLayout()
        controls.addWidget(self.add_person_button)
        controls.addWidget(self.delete_person_button)
        controls.addWidget(refresh_button)
        controls.addStretch()
        privacy = QLabel(
            "身份证号在数据库中加密保存，此处仅显示脱敏值；识别画面和日志不显示身份证号。"
        )
        privacy.setStyleSheet("color: #607080")
        layout = QVBoxLayout(page)
        layout.addLayout(controls)
        layout.addWidget(self.people_table, 1)
        layout.addWidget(privacy)
        return page

    def _build_settings_tab(self) -> QWidget:
        page = QWidget()
        model_group = QGroupBox("ONNX 模型")
        self.model_dir_edit = QLineEdit(self.config.model_dir)
        self.model_dir_edit.setReadOnly(True)
        choose_model_button = QPushButton("选择自有模型目录…")
        choose_model_button.clicked.connect(self._choose_model_dir)
        research_button = QPushButton("下载非商业研究模型…")
        research_button.clicked.connect(self._confirm_research_download)
        model_layout = QGridLayout(model_group)
        model_layout.addWidget(QLabel("模型目录"), 0, 0)
        model_layout.addWidget(self.model_dir_edit, 0, 1)
        model_layout.addWidget(choose_model_button, 0, 2)
        model_layout.addWidget(research_button, 1, 2)
        license_label = QLabel(
            "程序代码可开源使用；InsightFace 官方预训练权重仅限非商业研究。商用请取得权重授权，"
            "或放入 detector.onnx 与 recognizer.onnx。"
        )
        license_label.setWordWrap(True)
        license_label.setStyleSheet("color: #9a5a00")
        model_layout.addWidget(license_label, 1, 0, 1, 2)
        tuning_group = QGroupBox("识别参数")
        self.match_spin = QDoubleSpinBox()
        self.match_spin.setRange(-0.99, 0.99)
        self.match_spin.setDecimals(2)
        self.match_spin.setSingleStep(0.01)
        self.match_spin.setValue(self.config.match_threshold)
        self.margin_spin = QDoubleSpinBox()
        self.margin_spin.setRange(0, 0.99)
        self.margin_spin.setDecimals(2)
        self.margin_spin.setSingleStep(0.01)
        self.margin_spin.setValue(self.config.min_top2_margin)
        self.det_spin = QDoubleSpinBox()
        self.det_spin.setRange(0.1, 0.99)
        self.det_spin.setDecimals(2)
        self.det_spin.setValue(self.config.detection_threshold)
        self.stride_spin = QSpinBox()
        self.stride_spin.setRange(1, 10)
        self.stride_spin.setValue(self.config.frame_stride)
        self.face_size_spin = QSpinBox()
        self.face_size_spin.setRange(16, 300)
        self.face_size_spin.setValue(self.config.min_face_size)
        self.detection_width_spin = QSpinBox()
        self.detection_width_spin.setRange(320, 1920)
        self.detection_width_spin.setSingleStep(32)
        self.detection_width_spin.setValue(self.config.detection_width)
        self.detection_height_spin = QSpinBox()
        self.detection_height_spin.setRange(320, 1920)
        self.detection_height_spin.setSingleStep(32)
        self.detection_height_spin.setValue(self.config.detection_height)
        self.min_hits_spin = QSpinBox()
        self.min_hits_spin.setRange(1, 20)
        self.min_hits_spin.setValue(self.config.min_track_hits)
        self.quality_spin = QDoubleSpinBox()
        self.quality_spin.setRange(0, 1)
        self.quality_spin.setDecimals(2)
        self.quality_spin.setSingleStep(0.05)
        self.quality_spin.setValue(self.config.min_recognition_quality)
        self.flip_tta_check = QCheckBox("启用（精度更高，识别网络计算量约翻倍）")
        self.flip_tta_check.setChecked(self.config.recognition_flip_tta)
        form = QFormLayout(tuning_group)
        form.addRow("身份相似度阈值", self.match_spin)
        form.addRow("第一/第二候选最小差值", self.margin_spin)
        form.addRow("人脸检测阈值", self.det_spin)
        form.addRow("每隔 N 帧推理", self.stride_spin)
        form.addRow("最小人脸边长（像素）", self.face_size_spin)
        form.addRow("检测输入宽度", self.detection_width_spin)
        form.addRow("检测输入高度", self.detection_height_spin)
        form.addRow("稳定确认帧数", self.min_hits_spin)
        form.addRow("自动确认最低画质", self.quality_spin)
        form.addRow("水平翻转 TTA", self.flip_tta_check)
        save_button = QPushButton("保存并重新加载模型")
        save_button.clicked.connect(self._save_settings)
        layout = QVBoxLayout(page)
        layout.addWidget(model_group)
        layout.addWidget(tuning_group)
        layout.addWidget(save_button, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addStretch()
        return page

    def _ensure_models(self) -> None:
        try:
            load_onnxruntime_gpu()
        except GpuUnavailableError as exc:
            self.gpu_label.setText("GPU：不可用（程序拒绝 CPU）")
            self.statusBar().showMessage("CUDA 不可用；CPU 回退已禁用。")
            QMessageBox.critical(self, "CUDA GPU 不可用", str(exc))
            return
        if models_ready(Path(self.config.model_dir)):
            self._initialize_engine()
            return
        response = QMessageBox.question(
            self,
            "需要人脸模型",
            "尚未找到人脸模型。是否下载 InsightFace buffalo_l？\n\n"
            "该预训练权重仅限非商业研究使用；商用需另行取得授权。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if response == QMessageBox.StandardButton.Yes:
            self._confirm_research_download(skip_confirmation=True)
        else:
            self.statusBar().showMessage("缺少模型：可在“设置与模型”中选择或下载。")

    def _confirm_research_download(
        self, checked: bool = False, *, skip_confirmation: bool = False
    ) -> None:
        del checked
        if not skip_confirmation:
            response = QMessageBox.question(
                self,
                "确认模型许可证",
                "我确认本次使用属于非商业研究，并接受 InsightFace 预训练模型的许可证限制。\n\n"
                "继续下载约 275 MiB？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if response != QMessageBox.StandardButton.Yes:
                return
        self._download_models()

    def _download_models(self) -> None:
        if self.download_thread and self.download_thread.isRunning():
            return
        self.progress_dialog = QProgressDialog("正在下载并校验模型…", "隐藏", 0, 100, self)
        self.progress_dialog.setWindowModality(Qt.WindowModality.NonModal)
        self.progress_dialog.setAutoClose(False)
        self.progress_dialog.show()
        self.download_thread = ModelDownloadThread(Path(self.config.model_dir), self)
        self.download_thread.progress.connect(self._update_download_progress)
        self.download_thread.succeeded.connect(self._download_finished)
        self.download_thread.failed.connect(self._download_failed)
        self.download_thread.start()

    def _update_download_progress(self, current: int, total: int) -> None:
        if not self.progress_dialog:
            return
        percent = int(current * 100 / total) if total else 0
        self.progress_dialog.setValue(percent)
        self.progress_dialog.setLabelText(
            f"正在下载并校验模型… {current / 1024 / 1024:.1f} / {total / 1024 / 1024:.1f} MiB"
            if total
            else "正在下载并校验模型…"
        )

    def _download_finished(self) -> None:
        if self.progress_dialog:
            self.progress_dialog.close()
        self.statusBar().showMessage("模型下载完成，正在初始化 CUDA…")
        self._initialize_engine()

    def _download_failed(self, message: str) -> None:
        if self.progress_dialog:
            self.progress_dialog.close()
        QMessageBox.critical(self, "模型下载失败", message)

    def _initialize_engine(self) -> None:
        if self.video_worker and self.video_worker.isRunning():
            QMessageBox.warning(self, "请先停止", "请先停止视频识别，再重新加载模型。")
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            engine = FaceEngine(
                Path(self.config.model_dir),
                self.config.gpu_device_id,
                (self.config.detection_width, self.config.detection_height),
                self.config.detection_threshold,
                self.config.min_face_size,
                self.config.recognition_flip_tta,
            )
            self.engine = engine
            self.matcher.replace_gallery(self.database.load_gallery(engine.model_id))
        except GpuUnavailableError as exc:
            self.engine = None
            self.start_button.setEnabled(False)
            self.add_person_button.setEnabled(False)
            self.gpu_label.setText("GPU：不可用（程序拒绝 CPU）")
            QMessageBox.critical(self, "CUDA GPU 不可用", str(exc))
            self.statusBar().showMessage("CUDA 不可用；CPU 回退已禁用。")
        except FaceMatchingError as exc:
            self.engine = None
            QMessageBox.critical(self, "模型初始化失败", str(exc))
        except Exception as exc:
            self.engine = None
            QMessageBox.critical(self, "初始化失败", str(exc))
        else:
            self.start_button.setEnabled(True)
            self.add_person_button.setEnabled(True)
            self.gpu_label.setText(f"GPU：CUDA:{self.config.gpu_device_id}（CPU 回退禁用）")
            self.statusBar().showMessage(
                f"模型就绪；图库 {self.matcher.identity_count} 人，CUDA GPU 推理。"
            )
        finally:
            QApplication.restoreOverrideCursor()

    def _choose_video(self) -> None:
        name, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "视频 (*.mp4 *.avi *.mkv *.mov *.m4v *.wmv);;所有文件 (*)"
        )
        if name:
            self.source_edit.setText(name)

    def _enter_rtsp(self) -> None:
        value, ok = QInputDialog.getText(self, "RTSP 视频流", "RTSP 地址")
        if ok and value.strip():
            self.source_edit.setText(value.strip())

    def _parse_source(self) -> str | int | None:
        value = self.source_edit.text().strip()
        if not value:
            return None
        if value.isdigit() and len(value) <= 2:
            return int(value)
        return value

    def _start_video(self) -> None:
        if not self.engine:
            QMessageBox.warning(self, "模型未就绪", "请先加载模型并通过 CUDA 检查。")
            return
        source = self._parse_source()
        if source is None:
            QMessageBox.warning(self, "缺少视频源", "请选择视频文件、摄像头或输入 RTSP 地址。")
            return
        self.matcher.replace_gallery(self.database.load_gallery(self.engine.model_id))
        self.video_worker = VideoWorker(
            self.engine,
            self.matcher,
            source,
            self.config.frame_stride,
            self.config.max_track_age,
            self.config.min_track_hits,
            self.config.min_recognition_quality,
            self,
        )
        self.video_worker.frame_ready.connect(self._show_frame)
        self.video_worker.tracks_ready.connect(self._show_tracks)
        self.video_worker.status_changed.connect(self.statusBar().showMessage)
        self.video_worker.failed.connect(self._video_failed)
        self.video_worker.finished.connect(self._video_finished)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.add_person_button.setEnabled(False)
        self.delete_person_button.setEnabled(False)
        self.video_worker.start()

    def _stop_video(self) -> None:
        if self.video_worker and self.video_worker.isRunning():
            self.video_worker.stop()
            self.video_worker.wait(5000)

    def _video_finished(self) -> None:
        self.start_button.setEnabled(self.engine is not None)
        self.stop_button.setEnabled(False)
        self.add_person_button.setEnabled(self.engine is not None)
        self.delete_person_button.setEnabled(True)

    def _video_failed(self, message: str) -> None:
        QMessageBox.critical(self, "视频处理失败", message)

    def _show_frame(self, payload: FramePayload) -> None:
        height, width = payload.rgb.shape[:2]
        image = QImage(
            payload.rgb.data,
            width,
            height,
            int(payload.rgb.strides[0]),
            QImage.Format.Format_RGB888,
        ).copy()
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = QFont("Microsoft YaHei UI", max(10, int(height / 48)))
        font.setBold(True)
        painter.setFont(font)
        for track in payload.tracks:
            if track.accepted:
                color = QColor("#23c55e")
            elif track.decision.startswith("稳定确认"):
                color = QColor("#38bdf8")
            else:
                color = QColor("#f59e0b")
            painter.setPen(QPen(color, max(2, int(height / 360))))
            x1, y1, x2, y2 = track.bbox
            painter.drawRect(x1, y1, max(1, x2 - x1), max(1, y2 - y1))
            score_text = f" {track.score:.3f}" if track.score > -0.999 else ""
            if track.accepted:
                label = f"#{track.id} {track.name}{score_text}"
            elif track.decision.startswith("稳定确认"):
                label = f"#{track.id} {track.decision}{score_text}"
            else:
                label = f"#{track.id} {track.decision}{score_text}"
            metrics = painter.fontMetrics()
            label_width = metrics.horizontalAdvance(label) + 12
            label_height = metrics.height() + 6
            top = max(0, y1 - label_height)
            painter.fillRect(x1, top, label_width, label_height, color)
            painter.setPen(QColor("#081018"))
            painter.drawText(x1 + 6, top + metrics.ascent() + 3, label)
        painter.setPen(QColor("#ffffff"))
        painter.drawText(12, 24, f"{payload.fps:.1f} FPS | CUDA")
        painter.end()
        self._current_frame = image
        self._render_scaled_frame()

    def _render_scaled_frame(self) -> None:
        if self._current_frame is None:
            return
        pixmap = QPixmap.fromImage(self._current_frame).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(pixmap)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render_scaled_frame()

    def _show_tracks(self, tracks: tuple[TrackView, ...]) -> None:
        self._current_tracks = tracks
        self.track_table.setRowCount(len(tracks))
        for row, track in enumerate(tracks):
            values = [
                f"#{track.id}",
                track.decision,
                track.name,
                f"{track.score:.3f}" if track.score > -0.999 else "—",
                f"{track.quality:.2f}",
                str(track.confirmation_hits),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.track_table.setItem(row, column, item)

    def _add_person(self) -> None:
        if not self.engine:
            return
        dialog = AddPersonDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        valid: list[tuple[Path, np.ndarray, float]] = []
        failures: list[str] = []
        try:
            for path in dialog.photo_paths:
                image = _read_image(path)
                if image is None:
                    failures.append(f"{path.name}：无法读取")
                    continue
                try:
                    face = self.engine.enroll_image(image)
                except Exception as exc:
                    failures.append(f"{path.name}：{exc}")
                    continue
                if face.quality < self.config.min_recognition_quality:
                    failures.append(
                        f"{path.name}：画质 {face.quality:.2f} 低于录入下限 "
                        f"{self.config.min_recognition_quality:.2f}"
                    )
                    continue
                valid.append((path, face.embedding, face.quality))
            if not valid:
                raise ModelError("所选照片均未提取到有效人脸。\n" + "\n".join(failures))
            self.database.add_person(
                dialog.name_edit.text(),
                dialog.id_edit.text(),
                valid,
                self.engine.model_id,
            )
            self.matcher.replace_gallery(self.database.load_gallery(self.engine.model_id))
        except Exception as exc:
            QMessageBox.critical(self, "录入失败", str(exc))
        else:
            message = f"已录入 {dialog.name_edit.text().strip()}，有效照片 {len(valid)} 张。"
            if failures:
                message += "\n\n以下照片已跳过：\n" + "\n".join(failures)
            QMessageBox.information(self, "录入完成", message)
            self._refresh_people()
        finally:
            QApplication.restoreOverrideCursor()

    def _refresh_people(self) -> None:
        people = self.database.list_people()
        self.people_table.setRowCount(len(people))
        for row, person in enumerate(people):
            values = [person.id, person.name, person.masked_government_id, person.photo_count]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column in {0, 3}:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.people_table.setItem(row, column, item)

    def _delete_person(self) -> None:
        selected = self.people_table.selectionModel().selectedRows()
        if not selected:
            QMessageBox.information(self, "请选择人员", "请先选中要删除的人员。")
            return
        row = selected[0].row()
        person_id = int(self.people_table.item(row, 0).text())
        name = self.people_table.item(row, 1).text()
        response = QMessageBox.warning(
            self,
            "确认删除",
            f"确定删除“{name}”及其全部注册照片和人脸向量吗？此操作不可恢复。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if response != QMessageBox.StandardButton.Yes:
            return
        self.database.delete_person(person_id)
        if self.engine:
            self.matcher.replace_gallery(self.database.load_gallery(self.engine.model_id))
        self._refresh_people()

    def _choose_model_dir(self) -> None:
        if self.video_worker and self.video_worker.isRunning():
            QMessageBox.warning(self, "请先停止", "请先停止视频识别，再切换模型。")
            return
        directory = QFileDialog.getExistingDirectory(
            self, "选择 ONNX 模型目录", self.config.model_dir
        )
        if not directory:
            return
        self.config.model_dir = str(Path(directory).resolve())
        self.model_dir_edit.setText(self.config.model_dir)
        self.config.save()
        self._initialize_engine()

    def _save_settings(self) -> None:
        if self.video_worker and self.video_worker.isRunning():
            QMessageBox.warning(self, "请先停止", "请先停止视频识别，再保存并重载参数。")
            return
        try:
            self.config.match_threshold = self.match_spin.value()
            self.config.min_top2_margin = self.margin_spin.value()
            self.config.detection_threshold = self.det_spin.value()
            self.config.frame_stride = self.stride_spin.value()
            self.config.min_face_size = self.face_size_spin.value()
            self.config.detection_width = self.detection_width_spin.value()
            self.config.detection_height = self.detection_height_spin.value()
            self.config.min_track_hits = self.min_hits_spin.value()
            self.config.min_recognition_quality = self.quality_spin.value()
            self.config.recognition_flip_tta = self.flip_tta_check.isChecked()
            self.config.save()
        except ValueError as exc:
            QMessageBox.warning(self, "参数无效", str(exc))
            return
        self.matcher.threshold = self.config.match_threshold
        self.matcher.min_margin = self.config.min_top2_margin
        self._initialize_engine()

    def closeEvent(self, event) -> None:
        if self.video_worker and self.video_worker.isRunning():
            self.video_worker.stop()
            if not self.video_worker.wait(5000):
                event.ignore()
                QMessageBox.warning(self, "仍在停止", "视频线程仍在结束中，请稍后再关闭。")
                return
        event.accept()
