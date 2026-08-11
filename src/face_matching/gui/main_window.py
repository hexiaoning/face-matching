from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import cv2
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..config import AppConfig
from ..database import FaceDatabase, PersonRecord
from ..engine import FaceEngine, FrameResult
from ..enrollment import EnrollmentService
from ..errors import FaceMatchingError, ModelMissingError
from ..gpu import assert_gpu_available
from ..models import PROFILES, available_profiles, download_profile, profile_spec, required_paths
from ..paths import is_frozen_app
from ..results import VideoMatchList
from ..vision.io import read_image


class EnrollmentDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("录入人员")
        self.resize(560, 430)
        self.name_edit = QLineEdit()
        self.id_edit = QLineEdit()
        form = QFormLayout()
        form.addRow("姓名：", self.name_edit)
        form.addRow("身份证号：", self.id_edit)
        self.photos = QListWidget()
        add_button = QPushButton("选择照片…")
        remove_button = QPushButton("移除选中")
        add_button.clicked.connect(self._add_photos)
        remove_button.clicked.connect(lambda: self.photos.takeItem(self.photos.currentRow()))
        photo_buttons = QHBoxLayout()
        photo_buttons.addWidget(add_button)
        photo_buttons.addWidget(remove_button)
        photo_buttons.addStretch()
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(QLabel("人员照片（可多选，建议包含正面和轻微侧脸）："))
        layout.addWidget(self.photos)
        layout.addLayout(photo_buttons)
        layout.addWidget(buttons)

    def _add_photos(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择人员照片", "", "图片 (*.jpg *.jpeg *.png *.bmp *.webp)"
        )
        existing = {self.photos.item(index).text() for index in range(self.photos.count())}
        for path in paths:
            if path not in existing:
                self.photos.addItem(path)

    def _validate(self) -> None:
        if not self.name_edit.text().strip() or not self.id_edit.text().strip():
            QMessageBox.warning(self, "信息不完整", "姓名和身份证号不能为空。")
            return
        if self.photos.count() == 0:
            QMessageBox.warning(self, "信息不完整", "请至少选择一张照片。")
            return
        self.accept()

    def values(self) -> tuple[str, str, list[str]]:
        return (
            self.name_edit.text().strip(),
            self.id_edit.text().strip(),
            [self.photos.item(index).text() for index in range(self.photos.count())],
        )


class PersonInfoDialog(QDialog):
    def __init__(self, person: PersonRecord, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("编辑人员信息")
        self.name_edit = QLineEdit(person.name)
        self.id_edit = QLineEdit(person.id_card)
        form = QFormLayout(self)
        form.addRow("姓名：", self.name_edit)
        form.addRow("身份证号：", self.id_edit)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)


class VideoWorker(QThread):
    frame_ready = Signal(QImage)
    recognized = Signal(dict)
    stats = Signal(str)
    failed = Signal(str)

    def __init__(self, engine: FaceEngine, source: str, frame_interval: int) -> None:
        super().__init__()
        self.engine = engine
        self.source = source
        self.frame_interval = max(1, int(frame_interval))
        self._running = True

    def stop(self) -> None:
        self._running = False

    @staticmethod
    def _qimage(frame: object, result: FrameResult | None) -> QImage:
        height, width = frame.shape[:2]
        image = QImage(frame.data, width, height, frame.strides[0], QImage.Format_BGR888).copy()
        if result is None:
            return image
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(QFont("Microsoft YaHei UI", max(10, int(height / 55))))
        for track in result.tracks:
            x1, y1, x2, y2 = (int(value) for value in track.bbox)
            known = track.person_id is not None
            color = QColor("#39d98a" if known else "#ffb020")
            painter.setPen(QPen(color, max(2, int(height / 360))))
            painter.drawRect(x1, y1, max(1, x2 - x1), max(1, y2 - y1))
            label = f"#{track.id} {track.name}  {track.score:.2f}  Q:{track.quality:.2f}"
            metrics = painter.fontMetrics()
            label_width = metrics.horizontalAdvance(label) + 12
            label_height = metrics.height() + 6
            top = max(0, y1 - label_height)
            painter.fillRect(x1, top, label_width, label_height, QColor(0, 0, 0, 185))
            painter.setPen(color)
            painter.drawText(x1 + 6, top + metrics.ascent() + 3, label)
        painter.end()
        return image

    def run(self) -> None:
        capture_source: str | int = self.source
        if self.source.isdigit() and not Path(self.source).exists():
            capture_source = int(self.source)
        capture = cv2.VideoCapture()
        # FFmpeg-backed network streams can otherwise block shutdown forever.
        # Unsupported backends simply ignore these properties.
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8_000)
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3_000)
        capture.open(capture_source)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        if not capture.isOpened():
            self.failed.emit(f"无法打开视频源: {self.source}")
            return
        self.engine.reset_video()
        frame_index = 0
        last_result: FrameResult | None = None
        reported_scores: dict[tuple[int, str], float] = {}
        started = time.perf_counter()
        file_source = isinstance(capture_source, str) and not capture_source.lower().startswith(("rtsp://", "rtmp://"))
        source_fps = capture.get(cv2.CAP_PROP_FPS) if file_source else 0.0
        try:
            while self._running:
                loop_started = time.perf_counter()
                ok, frame = capture.read()
                if not ok:
                    break
                frame_index += 1
                if frame_index == 1 or frame_index % self.frame_interval == 0:
                    last_result = self.engine.process_frame(frame, frame_index)
                    for track in last_result.tracks:
                        key = (track.id, track.person_id or "")
                        previous_score = reported_scores.get(key)
                        if (
                            track.person_id
                            and (previous_score is None or track.score > previous_score + 1e-6)
                        ):
                            reported_scores[key] = track.score
                            self.recognized.emit(
                                {
                                    "time": datetime.now().strftime("%H:%M:%S"),
                                    "track_id": track.id,
                                    "person_id": track.person_id,
                                    "name": track.name,
                                    "id_card": track.id_card,
                                    "score": track.score,
                                    "quality": track.quality,
                                    "new_track_match": previous_score is None,
                                }
                            )
                self.frame_ready.emit(self._qimage(frame, last_result))
                if frame_index % 15 == 0:
                    elapsed = max(time.perf_counter() - started, 1e-6)
                    faces = last_result.detected_faces if last_result else 0
                    self.stats.emit(f"GPU 推理中 · {frame_index / elapsed:.1f} FPS · 当前 {faces} 张人脸")
                if file_source and source_fps > 1.0:
                    remaining = 1.0 / source_fps - (time.perf_counter() - loop_started)
                    if remaining > 0:
                        self.msleep(int(remaining * 1000))
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            capture.release()


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self.database = FaceDatabase()
        self.engine: FaceEngine | None = None
        self.enrollment: EnrollmentService | None = None
        self.worker: VideoWorker | None = None
        self.video_matches = VideoMatchList()
        self.setWindowTitle("Face Matching · GPU 视频人脸比对")
        self.resize(1420, 880)
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self._build_video_tab()
        self._build_people_tab()
        self._build_settings_tab()
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("正在检查 GPU…")

    def initialize(self) -> bool:
        try:
            gpu_status = assert_gpu_available(
                self.config.gpu_backend, self.config.gpu_device_id
            )
            try:
                required_paths(self.config.model_profile)
            except ModelMissingError:
                if is_frozen_app():
                    raise ModelMissingError(
                        "离线便携包缺少当前模型。请使用完整的 v2.4 离线包，"
                        "或在联网构建机上重新打包；目标机不会尝试联网下载。"
                    )
                profile = profile_spec(self.config.model_profile)
                answer = QMessageBox.question(
                    self,
                    "需要下载模型",
                    f"尚未安装 {profile.title}（约 {(profile.detector.size + profile.recognizer.size) / 1e6:.0f} MB）。\n\n"
                    f"{profile.note}\n\n现在下载吗？",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if answer != QMessageBox.Yes:
                    return False
                self._download_models()
            self.engine = FaceEngine(self.config, self.database)
            self.enrollment = EnrollmentService(self.database, self.engine)
            self.statusBar().showMessage(f"{gpu_status} · CPU fallback 已禁用")
            self.gpu_label.setText(f"✅ {gpu_status}\nCPU fallback：已禁用")
            self.model_label.setText(
                f"✅ {profile_spec(self.config.model_profile).title}\n"
                f"特征版本：{self.engine.model_id}"
            )
            self._set_video_enabled(True)
            self.refresh_people()
            if self.database.list_stored_samples() and self.engine.matcher.person_count == 0:
                QMessageBox.warning(
                    self,
                    "需要重建特征",
                    "当前特征版本没有可用底库:\n"
                    f"{self.engine.model_id}\n\n"
                    "请在“人员库”点击“重建当前模型特征”。"
                    "在重建前，视频不会产生人员命中。",
                )
            return True
        except (FaceMatchingError, OSError, ValueError) as exc:
            QMessageBox.critical(self, "无法启动 GPU 推理", str(exc))
            self.statusBar().showMessage("GPU 推理不可用")
            return False

    def _download_models(self) -> None:
        progress = QProgressDialog("准备下载…", "取消", 0, 100, self)
        progress.setWindowTitle("下载模型")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        def update(name: str, done: int, total: int) -> None:
            progress.setLabelText(f"正在下载 {name}\n{done / 1e6:.1f} / {total / 1e6:.1f} MB")
            progress.setValue(int(done * 100 / max(total, 1)))
            QApplication.processEvents()

        download_profile(
            self.config.model_profile,
            progress=update,
            cancelled=progress.wasCanceled,
        )
        progress.setValue(100)

    def _build_video_tab(self) -> None:
        tab = QWidget()
        self.tabs.addTab(tab, "视频识别")
        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("视频文件、RTSP/RTMP 地址，或摄像头编号（例如 0）")
        browse = QPushButton("选择视频…")
        camera = QPushButton("摄像头 0")
        self.start_button = QPushButton("开始识别")
        self.stop_button = QPushButton("停止")
        browse.clicked.connect(self._browse_video)
        camera.clicked.connect(lambda: self.source_edit.setText("0"))
        self.start_button.clicked.connect(self.start_video)
        self.stop_button.clicked.connect(self.stop_video)
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("视频源："))
        source_row.addWidget(self.source_edit, 1)
        source_row.addWidget(browse)
        source_row.addWidget(camera)
        source_row.addWidget(self.start_button)
        source_row.addWidget(self.stop_button)
        self.video_label = QLabel("选择视频、输入 RTSP 地址或使用摄像头 0")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(880, 540)
        self.video_label.setStyleSheet("background:#111820;color:#8fa3b8;border-radius:8px;")
        self.events = QTableWidget(0, 6)
        self.events.setHorizontalHeaderLabels(["时间", "轨迹", "姓名", "身份证号", "相似度", "质量"])
        self.events.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.events.setEditTriggers(QTableWidget.NoEditTriggers)
        self.events.setToolTip("每人保留整段视频中的最高分，按相似度从高到低排列")
        body = QHBoxLayout()
        body.addWidget(self.video_label, 3)
        body.addWidget(self.events, 2)
        layout = QVBoxLayout(tab)
        layout.addLayout(source_row)
        layout.addLayout(body, 1)
        self._set_video_enabled(False)

    def _set_video_enabled(self, ready: bool) -> None:
        self.start_button.setEnabled(ready)
        self.stop_button.setEnabled(False)

    def _build_people_tab(self) -> None:
        tab = QWidget()
        self.tabs.addTab(tab, "人员库")
        self.people_table = QTableWidget(0, 5)
        self.people_table.setHorizontalHeaderLabels(["姓名", "身份证号", "照片数", "更新时间", "内部 ID"])
        self.people_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.people_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.people_table.setEditTriggers(QTableWidget.NoEditTriggers)
        add = QPushButton("录入人员…")
        edit = QPushButton("编辑信息…")
        photos = QPushButton("添加照片…")
        rebuild = QPushButton("重建当前模型特征")
        delete = QPushButton("删除人员")
        refresh = QPushButton("刷新")
        add.clicked.connect(self.add_person)
        edit.clicked.connect(self.edit_person)
        photos.clicked.connect(self.add_photos)
        rebuild.clicked.connect(self.rebuild_features)
        delete.clicked.connect(self.delete_person)
        refresh.clicked.connect(self.refresh_people)
        row = QHBoxLayout()
        for button in (add, edit, photos, rebuild, delete, refresh):
            row.addWidget(button)
        row.addStretch()
        layout = QVBoxLayout(tab)
        layout.addLayout(row)
        layout.addWidget(self.people_table)

    def _build_settings_tab(self) -> None:
        tab = QWidget()
        self.tabs.addTab(tab, "设置")
        self.profile_combo = QComboBox()
        selectable_profiles = available_profiles() if is_frozen_app() else list(PROFILES)
        for key in selectable_profiles:
            spec = PROFILES[key]
            self.profile_combo.addItem(spec.title, key)
        self.profile_combo.setCurrentIndex(max(0, self.profile_combo.findData(self.config.model_profile)))
        self.backend_combo = QComboBox()
        self.backend_combo.addItem("自动（优先 NVIDIA CUDA）", "auto")
        self.backend_combo.addItem("NVIDIA CUDA", "cuda")
        self.backend_combo.addItem("Intel OpenVINO GPU", "openvino")
        self.backend_combo.setCurrentIndex(
            max(0, self.backend_combo.findData(self.config.gpu_backend))
        )
        self.gpu_device_id = QSpinBox()
        self.gpu_device_id.setRange(0, 15)
        self.gpu_device_id.setValue(self.config.gpu_device_id)
        self.detector_threshold = QDoubleSpinBox()
        self.detector_threshold.setRange(0.1, 0.95)
        self.detector_threshold.setSingleStep(0.05)
        self.detector_threshold.setValue(self.config.detector_threshold)
        self.match_threshold = QDoubleSpinBox()
        self.match_threshold.setRange(0.1, 0.95)
        self.match_threshold.setSingleStep(0.01)
        self.match_threshold.setValue(self.config.match_threshold)
        self.match_margin = QDoubleSpinBox()
        self.match_margin.setRange(0.0, 0.3)
        self.match_margin.setSingleStep(0.01)
        self.match_margin.setValue(self.config.match_margin)
        self.min_quality = QDoubleSpinBox()
        self.min_quality.setRange(0.0, 0.9)
        self.min_quality.setSingleStep(0.02)
        self.min_quality.setValue(self.config.min_quality)
        self.frame_interval = QSpinBox()
        self.frame_interval.setRange(1, 12)
        self.frame_interval.setValue(self.config.frame_interval)
        self.mirror_augmentation = QCheckBox("启用（提高侧脸稳定性，会增加识别计算量）")
        self.mirror_augmentation.setChecked(self.config.mirror_augmentation)
        self.detector_size = QComboBox()
        for size in (320, 480, 640, 768, 960, 1280):
            self.detector_size.addItem(f"{size} × {size}", size)
        self.detector_size.setCurrentIndex(
            max(0, self.detector_size.findData(self.config.detector_size))
        )
        self.min_face_size = QSpinBox()
        self.min_face_size.setRange(12, 160)
        self.min_face_size.setSuffix(" px")
        self.min_face_size.setValue(self.config.min_face_size)
        self.gpu_label = QLabel("尚未检查")
        self.model_label = QLabel("尚未检查")
        save = QPushButton("保存设置")
        save.clicked.connect(self.save_settings)
        form = QFormLayout()
        form.addRow("识别模型：", self.profile_combo)
        form.addRow("GPU 后端：", self.backend_combo)
        form.addRow("GPU 编号：", self.gpu_device_id)
        form.addRow("检测置信度：", self.detector_threshold)
        form.addRow("身份相似度阈值：", self.match_threshold)
        form.addRow("第一/第二名最小差值：", self.match_margin)
        form.addRow("最低图像质量：", self.min_quality)
        form.addRow("检测输入尺寸：", self.detector_size)
        form.addRow("最小人脸边长：", self.min_face_size)
        form.addRow("镜像测试增强：", self.mirror_augmentation)
        form.addRow("每隔 N 帧推理：", self.frame_interval)
        form.addRow("GPU 状态：", self.gpu_label)
        form.addRow("模型状态：", self.model_label)
        form.addRow("", save)
        note = QLabel(
            "阈值必须用现场摄像头数据校准。高安全场景建议提高相似度阈值，并保留人工复核；"
            "极端侧脸、严重模糊或人脸小于约 28 像素时，系统会宁可不认也不强猜。"
        )
        note.setWordWrap(True)
        layout = QVBoxLayout(tab)
        layout.addLayout(form)
        layout.addWidget(note)
        layout.addStretch()

    def _browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "视频 (*.mp4 *.avi *.mkv *.mov *.m4v);;所有文件 (*)"
        )
        if path:
            self.source_edit.setText(path)

    def start_video(self) -> None:
        if not self.engine:
            return
        source = self.source_edit.text().strip()
        if not source:
            QMessageBox.warning(self, "缺少视频源", "请选择视频、输入流地址或摄像头编号。")
            return
        if not self.stop_video():
            return
        self.events.setRowCount(0)
        self.video_matches.clear()
        self.worker = VideoWorker(self.engine, source, self.config.frame_interval)
        self.worker.frame_ready.connect(self._show_frame)
        self.worker.recognized.connect(self._on_recognized)
        self.worker.stats.connect(self.statusBar().showMessage)
        self.worker.failed.connect(lambda message: QMessageBox.critical(self, "视频处理失败", message))
        self.worker.finished.connect(self._video_finished)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.worker.start()

    def stop_video(self) -> bool:
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            if not self.worker.wait(5_000):
                QMessageBox.warning(
                    self,
                    "正在断开视频流",
                    "视频后端仍在等待网络超时，请稍后再试。",
                )
                return False
        self.worker = None
        if self.engine:
            self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        return True

    def _video_finished(self) -> None:
        self.start_button.setEnabled(bool(self.engine))
        self.stop_button.setEnabled(False)
        self.statusBar().showMessage(
            f"视频处理完成 · 匹配 {len(self.video_matches)} 人 · 已按 score 倒序排列"
        )

    def _show_frame(self, image: QImage) -> None:
        pixmap = QPixmap.fromImage(image)
        self.video_label.setPixmap(
            pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def _on_recognized(self, event: dict) -> None:
        changed = self.video_matches.update(event)
        if changed:
            ranked = self.video_matches.ranked()
            self.events.setRowCount(len(ranked))
            for row, match in enumerate(ranked):
                values = (
                    match.time,
                    f"#{match.track_id}",
                    match.name,
                    match.id_card,
                    f"{match.score:.3f}",
                    f"{match.quality:.3f}",
                )
                for column, value in enumerate(values):
                    self.events.setItem(row, column, QTableWidgetItem(value))
        if self.database and event.get("new_track_match", True):
            self.database.log_event(
                event["person_id"], self.source_edit.text().strip(), event["track_id"],
                event["score"], event["quality"]
            )

    def _selected_person(self) -> PersonRecord | None:
        row = self.people_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "请选择人员", "请先在表格中选择一名人员。")
            return None
        person_id = self.people_table.item(row, 4).data(Qt.UserRole)
        person = self.database.get_person(person_id)
        if not person:
            QMessageBox.warning(self, "人员不存在", "记录已变化，请刷新列表。")
        return person

    def _require_idle(self) -> bool:
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "请先停止视频", "修改人员库前请先停止视频识别。")
            return False
        return self.enrollment is not None

    def add_person(self) -> None:
        if not self._require_idle():
            return
        dialog = EnrollmentDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        name, id_card, photos = dialog.values()
        progress = QProgressDialog("正在使用 GPU 提取人脸特征…", None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        QApplication.processEvents()
        try:
            self.enrollment.add_person(name, id_card, photos)  # type: ignore[union-attr]
            self.refresh_people()
            QMessageBox.information(self, "录入完成", f"已录入 {name}，共 {len(photos)} 张照片。")
        except Exception as exc:
            QMessageBox.critical(self, "录入失败", str(exc))
        finally:
            progress.close()

    def edit_person(self) -> None:
        person = self._selected_person()
        if not person or not self._require_idle():
            return
        dialog = PersonInfoDialog(person, self)
        if dialog.exec() == QDialog.Accepted:
            try:
                self.database.update_person(
                    person.id, dialog.name_edit.text().strip(), dialog.id_edit.text().strip()
                )
                if self.engine:
                    self.engine.refresh_gallery()
                self.refresh_people()
            except Exception as exc:
                QMessageBox.critical(self, "保存失败", str(exc))

    def add_photos(self) -> None:
        person = self._selected_person()
        if not person or not self._require_idle():
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, f"为 {person.name} 添加照片", "", "图片 (*.jpg *.jpeg *.png *.bmp *.webp)"
        )
        if not paths:
            return
        try:
            self.enrollment.add_photos(person.id, paths)  # type: ignore[union-attr]
            self.refresh_people()
            QMessageBox.information(self, "添加完成", f"已添加 {len(paths)} 张照片。")
        except Exception as exc:
            QMessageBox.critical(self, "添加失败", str(exc))

    def delete_person(self) -> None:
        person = self._selected_person()
        if not person or not self._require_idle():
            return
        answer = QMessageBox.question(
            self, "确认删除", f"确定删除 {person.name} 及其 {person.photo_count} 张照片吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        if answer == QMessageBox.Yes:
            try:
                self.enrollment.delete_person(person.id)  # type: ignore[union-attr]
                self.refresh_people()
            except Exception as exc:
                QMessageBox.critical(self, "删除失败", str(exc))

    def rebuild_features(self) -> None:
        if not self.engine or not self._require_idle():
            return
        samples = self.database.list_stored_samples()
        if not samples:
            QMessageBox.information(self, "没有照片", "人员库中没有需要重建的照片。")
            return
        progress = QProgressDialog("准备重建…", "取消", 0, len(samples), self)
        progress.setWindowModality(Qt.WindowModal)
        failures: list[str] = []
        for index, sample in enumerate(samples, 1):
            if progress.wasCanceled():
                break
            progress.setLabelText(f"GPU 重建特征 {index}/{len(samples)}")
            progress.setValue(index - 1)
            QApplication.processEvents()
            image = read_image(sample.image_path)
            if image is None:
                failures.append(Path(sample.image_path).name + "（无法读取）")
                continue
            try:
                feature = self.engine.enrollment_feature(image)
                self.database.update_sample_feature(
                    sample.id, feature.embedding, self.engine.model_id, feature.quality.total
                )
            except Exception as exc:
                failures.append(Path(sample.image_path).name + f"（{exc}）")
        progress.setValue(len(samples))
        self.engine.refresh_gallery()
        if failures:
            QMessageBox.warning(self, "重建完成但有失败", "\n".join(failures[:12]))
        else:
            QMessageBox.information(self, "重建完成", "所有照片特征已更新为当前模型。")

    def refresh_people(self) -> None:
        people = self.database.list_people()
        self.people_table.setRowCount(len(people))
        for row, person in enumerate(people):
            values = (
                person.name,
                person.id_card,
                str(person.photo_count),
                person.updated_at.replace("T", " ")[:19],
                person.id[:10],
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 4:
                    item.setData(Qt.UserRole, person.id)
                self.people_table.setItem(row, column, item)

    def save_settings(self) -> None:
        new_profile = self.profile_combo.currentData()
        new_backend = self.backend_combo.currentData()
        new_gpu_device_id = self.gpu_device_id.value()
        new_mirror_augmentation = self.mirror_augmentation.isChecked()
        changed_model = new_profile != self.config.model_profile
        changed_backend = new_backend != self.config.gpu_backend
        changed_gpu_device = new_gpu_device_id != self.config.gpu_device_id
        changed_tta = new_mirror_augmentation != self.config.mirror_augmentation
        self.config.model_profile = new_profile
        self.config.gpu_backend = new_backend
        self.config.gpu_device_id = new_gpu_device_id
        self.config.detector_threshold = self.detector_threshold.value()
        self.config.match_threshold = self.match_threshold.value()
        self.config.match_margin = self.match_margin.value()
        self.config.min_quality = self.min_quality.value()
        self.config.detector_size = self.detector_size.currentData()
        self.config.min_face_size = self.min_face_size.value()
        self.config.mirror_augmentation = new_mirror_augmentation
        self.config.frame_interval = self.frame_interval.value()
        try:
            self.config.save()
            if self.engine:
                self.engine.detector.threshold = self.config.detector_threshold
                self.engine.matcher.threshold = self.config.match_threshold
                self.engine.matcher.min_margin = self.config.match_margin
            message = "设置已保存。"
            if self.engine and self.config.detector_size != self.engine.detector.input_size[0]:
                message += "\n\n检测输入尺寸将在重启后生效。"
            if changed_model:
                message += "\n\n模型将在重启后生效；首次启动会下载权重。"
            if changed_backend or changed_gpu_device:
                message += "\n\nGPU 后端/编号将在重启后生效。"
            if changed_tta:
                state = "开启" if new_mirror_augmentation else "关闭"
                message += f"\n\n镜像 TTA 将在重启后{state}，特征版本会改变。"
            if changed_model or changed_tta:
                message += "\n重启后必须在人员库点击“重建当前模型特征”。"
            QMessageBox.information(self, "设置", message)
        except Exception as exc:
            QMessageBox.critical(self, "设置无效", str(exc))

    def closeEvent(self, event: object) -> None:
        if self.stop_video():
            event.accept()
        else:
            event.ignore()
