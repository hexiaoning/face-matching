from __future__ import annotations

import time
import uuid
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import cv2
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor, QImage
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
    QMenu,
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

from .. import __version__
from ..annotation_export import ExportReport
from ..config import AppConfig
from ..database import FaceDatabase, PersonRecord
from ..engine import FaceEngine, FrameResult, TrackView
from ..enrollment import EnrollmentService
from ..errors import FaceMatchingError, ModelMissingError
from ..gpu import assert_gpu_available
from ..models import PROFILES, available_profiles, download_profile, profile_spec, required_paths
from ..paths import is_frozen_app
from ..results import VideoMatch, VideoMatchList
from ..vision.io import is_image_source, is_image_data_uri, read_image, store_image_data_uri
from .preview import FacePreview, draw_tracks
from .export_dialog import AnnotationExportDialog, AnnotationExportWorker
from .person_photos import PersonPhotosDialog


def _rate_limit_due(last_emitted_at: float | None, now: float, interval: float) -> bool:
    return last_emitted_at is None or now - last_emitted_at >= interval


def _final_preview_due(reached_end: bool, frame_index: int, last_preview_frame_index: int) -> bool:
    return reached_end and frame_index > 0 and last_preview_frame_index != frame_index


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


class PersonAssociationDialog(QDialog):
    def __init__(self, people: list[PersonRecord], parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("关联已录入人员")
        self.resize(620, 440)
        self.search = QLineEdit()
        self.search.setPlaceholderText("搜索姓名或身份证号")
        self.people = people
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["姓名", "身份证号"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.confirm = buttons.button(QDialogButtonBox.Ok)
        self.confirm.setText("关联")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.search.textChanged.connect(self._filter)
        self.table.itemSelectionChanged.connect(
            lambda: self.confirm.setEnabled(self.table.currentRow() >= 0)
        )
        self.table.cellDoubleClicked.connect(lambda row, column: self.accept())
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("选择此人对应的人员库记录："))
        layout.addWidget(self.search)
        layout.addWidget(self.table)
        layout.addWidget(buttons)
        self._filter("")

    def _filter(self, query: str) -> None:
        query = query.strip().casefold()
        self.filtered = [p for p in self.people
                         if query in p.name.casefold() or query in p.id_card.casefold()]
        self.table.setRowCount(0)
        self.table.setRowCount(len(self.filtered))
        for row, person in enumerate(self.filtered):
            self.table.setItem(row, 0, QTableWidgetItem(person.name))
            self.table.setItem(row, 1, QTableWidgetItem(person.id_card))
        self.confirm.setEnabled(False)

    def selected_person(self) -> PersonRecord | None:
        row = self.table.currentRow()
        return self.filtered[row] if row >= 0 else None


class VideoWorker(QThread):
    PREVIEW_INTERVAL_SECONDS = 1.0 / 15.0
    STATS_INTERVAL_SECONDS = 1.0

    frame_ready = Signal(QImage, object, int)
    recognized = Signal(dict)
    stats = Signal(str)
    failed = Signal(str)

    def __init__(self, engine: FaceEngine, source: str, frame_interval: int) -> None:
        super().__init__()
        self.engine = engine
        self.source = source
        self.frame_interval = max(1, int(frame_interval))
        self._running = True
        self.had_error = False

    def stop(self) -> None:
        self._running = False

    @staticmethod
    def _qimage(frame: object, result: FrameResult | None) -> QImage:
        height, width = frame.shape[:2]
        image = QImage(frame.data, width, height, frame.strides[0], QImage.Format_BGR888).copy()
        return draw_tracks(image, result) if result else image

    def _report_matches(
        self, result: FrameResult, reported_scores: dict[tuple[int, str, bool], float], frame_index: int,
        frame: object,
    ) -> None:
        for track in result.tracks:
            confirmed = track.person_id is not None
            person_id = track.person_id or track.candidate_person_id
            key = (track.id, person_id or "", confirmed)
            previous_score = reported_scores.get(key)
            if person_id and (previous_score is None or track.score > previous_score + 1e-6):
                reported_scores[key] = track.score
                height, width = frame.shape[:2]
                x1, y1, x2, y2 = track.bbox
                crop = frame[max(0, int(y1)):min(height, int(y2)),
                             max(0, int(x1)):min(width, int(x2))]
                face_jpeg = None
                if crop.size:
                    ok, encoded = cv2.imencode(".jpg", crop)
                    if ok:
                        face_jpeg = encoded.tobytes()
                self.recognized.emit({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "track_id": track.id, "person_id": person_id,
                    "name": track.name if confirmed else track.candidate_name,
                    "id_card": track.id_card if confirmed else track.candidate_id_card,
                    "confirmed": confirmed, "status": "自动匹配" if confirmed else "待确认",
                    "reason": track.match_reason,
                    "score": track.score, "quality": track.quality,
                    "frame_index": frame_index, "bbox": track.bbox,
                    "face_jpeg": face_jpeg,
                    "new_track_match": previous_score is None,
                })

    def _fail(self, message: str) -> None:
        self.had_error = True
        self.failed.emit(message)

    def run(self) -> None:
        if is_image_source(self.source):
            try:
                image = read_image(self.source)
                if image is None:
                    label = "图片数据无效或不完整" if is_image_data_uri(self.source) else self.source
                    raise ValueError(f"无法读取图片：{label}")
                if not self._running:
                    return
                result = self.engine.process_image(image)
                self._report_matches(result, {}, 1, image)
                self.frame_ready.emit(self._qimage(image, None), result, 1)
                self.stats.emit(
                    f"图片识别完成 · 检测到 {result.detected_faces} 张人脸 · "
                    f"可用于匹配 {result.usable_faces} 张"
                )
            except Exception as exc:
                self._fail(str(exc))
            return
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
            capture.release()
            self._fail(f"无法打开视频源: {self.source}")
            return
        self.engine.reset_video()
        frame_index = 0
        last_frame: object | None = None
        last_result: FrameResult | None = None
        reported_scores: dict[tuple[int, str, bool], float] = {}
        started = time.perf_counter()
        last_preview_at: float | None = None
        last_preview_frame_index = 0
        last_stats_at = started
        reached_end = False
        try:
            while self._running:
                ok, frame = capture.read()
                if not ok:
                    reached_end = True
                    break
                frame_index += 1
                last_frame = frame
                if frame_index == 1 or frame_index % self.frame_interval == 0:
                    last_result = self.engine.process_frame(frame, frame_index)
                    self._report_matches(last_result, reported_scores, frame_index, frame)
                now = time.perf_counter()
                if _rate_limit_due(last_preview_at, now, self.PREVIEW_INTERVAL_SECONDS):
                    self.frame_ready.emit(self._qimage(frame, None), last_result, frame_index)
                    last_preview_at = now
                    last_preview_frame_index = frame_index
                if _rate_limit_due(last_stats_at, now, self.STATS_INTERVAL_SECONDS):
                    elapsed = max(now - started, 1e-6)
                    faces = last_result.detected_faces if last_result else 0
                    self.stats.emit(f"GPU 推理中 · {frame_index / elapsed:.1f} FPS · 当前 {faces} 张人脸")
                    last_stats_at = now
            if (
                last_frame is not None
                and _final_preview_due(reached_end, frame_index, last_preview_frame_index)
            ):
                self.frame_ready.emit(self._qimage(last_frame, None), last_result, frame_index)
        except Exception as exc:
            self._fail(str(exc))
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
        self.export_worker: AnnotationExportWorker | None = None
        self.export_progress: QProgressDialog | None = None
        self.video_matches = VideoMatchList()
        self.session_id = ""
        self.active_source = ""
        self.preview_image = QImage()
        self.preview_result: FrameResult | None = None
        self.preview_frame_index = 0
        self.match_context: dict[tuple[int, str], dict] = {}
        self.association_context: dict[int, dict] = {}
        self._reviewing = False
        self.setWindowTitle(f"Face Matching v{__version__} · GPU 图片 / 视频人脸检索")
        self.resize(1420, 880)
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self._build_video_tab()
        self._build_people_tab()
        self._build_annotations_tab()
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
                        "离线便携包缺少当前模型。请使用完整的 v2.6 离线包，"
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
            self.statusBar().showMessage(f"{gpu_status} · 神经网络使用 GPU 推理")
            self.gpu_label.setText(f"✅ {gpu_status}\n神经网络：GPU 推理")
            self.model_label.setText(
                f"✅ {profile_spec(self.config.model_profile).title}\n"
                f"特征版本：{self.engine.model_id}"
            )
            self._set_video_enabled(True)
            self.refresh_people()
            self.refresh_annotations()
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
        self.tabs.addTab(tab, "人脸检索")
        self.source_edit = QLineEdit()
        self.source_edit.setMaxLength(16 * 1024 * 1024)
        self.source_edit.setPlaceholderText("图片 / 视频文件、RTSP/RTMP 地址，或摄像头编号（例如 0）")
        self.browse_button = QPushButton("选择图片 / 视频…")
        browse = self.browse_button
        camera = QPushButton("摄像头 0")
        self.start_button = QPushButton("开始识别")
        self.stop_button = QPushButton("停止")
        browse.clicked.connect(self._browse_video)
        camera.clicked.connect(lambda: self.source_edit.setText("0"))
        self.start_button.clicked.connect(self.start_video)
        self.stop_button.clicked.connect(self.stop_video)
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("检索来源："))
        source_row.addWidget(self.source_edit, 1)
        source_row.addWidget(browse)
        source_row.addWidget(camera)
        source_row.addWidget(self.start_button)
        source_row.addWidget(self.stop_button)
        self.video_label = FacePreview()
        self.video_label.track_context_requested.connect(self._track_menu)
        self.events = QTableWidget(0, 8)
        self.events.setHorizontalHeaderLabels([
            "时间", "轨迹", "姓名", "身份证号", "相似度", "质量", "标注", "操作"
        ])
        self.events.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.events.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.events.horizontalHeader().setMinimumSectionSize(58)
        self.events.setSelectionBehavior(QTableWidget.SelectRows)
        self.events.setEditTriggers(QTableWidget.NoEditTriggers)
        self.events.setToolTip("人工关联和自动确认优先；相似候选列为待确认，悬停可查看原因。“不是此人”针对该行轨迹。")
        body = QHBoxLayout()
        body.addWidget(self.video_label, 3)
        body.addWidget(self.events, 2)
        layout = QVBoxLayout(tab)
        layout.addLayout(source_row)
        layout.addLayout(body, 1)
        hint = QHBoxLayout()
        hint.addWidget(QLabel("待确认为相似候选，需人工核实；右键人脸框关联人员，右侧可标注“不是此人”。"))
        hint.addStretch()
        self.export_button = QPushButton("导出标注库…")
        self.export_button.clicked.connect(self.export_annotations)
        hint.addWidget(self.export_button)
        layout.addLayout(hint)
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
        view_photos = QPushButton("查看照片…")
        rebuild = QPushButton("重建当前模型特征")
        delete = QPushButton("删除人员")
        refresh = QPushButton("刷新")
        add.clicked.connect(self.add_person)
        edit.clicked.connect(self.edit_person)
        photos.clicked.connect(self.add_photos)
        view_photos.clicked.connect(self.view_person_photos)
        self.people_table.cellDoubleClicked.connect(lambda row, column: self.view_person_photos())
        rebuild.clicked.connect(self.rebuild_features)
        delete.clicked.connect(self.delete_person)
        refresh.clicked.connect(self.refresh_people)
        row = QHBoxLayout()
        for button in (add, edit, view_photos, photos, rebuild, delete, refresh):
            row.addWidget(button)
        row.addStretch()
        layout = QVBoxLayout(tab)
        layout.addLayout(row)
        layout.addWidget(self.people_table)

    def _build_annotations_tab(self) -> None:
        tab = QWidget()
        self.tabs.addTab(tab, "标注库")
        self.annotation_table = QTableWidget(0, 7)
        self.annotation_table.setHorizontalHeaderLabels([
            "标注时间", "来源", "轨迹", "帧号", "姓名", "身份证号", "操作"
        ])
        self.annotation_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.annotation_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.annotation_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.annotation_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.annotation_count = QLabel("标注会保存在本地，重新启动后仍可查看和导出。")
        refresh = QPushButton("刷新")
        refresh.clicked.connect(self.refresh_annotations)
        self.library_export_button = QPushButton("导出标注库…")
        self.library_export_button.clicked.connect(self.export_annotations)
        row = QHBoxLayout()
        row.addWidget(self.annotation_count, 1)
        row.addWidget(refresh)
        row.addWidget(self.library_export_button)
        layout = QVBoxLayout(tab)
        layout.addLayout(row)
        layout.addWidget(self.annotation_table)
        self.refresh_annotations()

    def refresh_annotations(self) -> None:
        annotations = self.database.list_annotations()
        people = {person.id: person for person in self.database.list_people()}
        self.annotation_table.setRowCount(len(annotations))
        labels = {"reject": "不是此人", "associate": "关联人员",
                  "undo_rejection": "撤销标注", "remove_association": "取消关联"}
        for row, record in enumerate(reversed(annotations)):
            person = people.get(record["person_id"])
            values = (
                datetime.fromisoformat(record["occurred_at"]).astimezone().strftime("%Y-%m-%d %H:%M:%S"),
                record["source"], f"#{record['track_id']}", str(record["frame_index"]),
                person.name if person else "人员已删除", person.id_card if person else "",
                labels[record["action"]],
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                self.annotation_table.setItem(row, column, item)
        self.annotation_count.setText(f"共 {len(annotations)} 条标注及操作历史 · 已保存至本地")

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
            self, "选择图片或视频", "",
            "图片和视频 (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff *.mp4 *.avi *.mkv *.mov *.m4v);;"
            "图片 (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff);;"
            "视频 (*.mp4 *.avi *.mkv *.mov *.m4v);;所有文件 (*)"
        )
        if path:
            self.source_edit.setText(path)

    def start_video(self) -> None:
        if not self.engine:
            return
        source = self.source_edit.text().strip()
        if not source:
            QMessageBox.warning(self, "缺少检索来源", "请选择图片、视频，或输入流地址、摄像头编号。")
            return
        if is_image_data_uri(source):
            try:
                source = str(store_image_data_uri(
                    source, self.database.path.parent / "query_images"
                ).resolve())
                self.source_edit.setText(source)
            except (ValueError, OSError, cv2.error) as exc:
                QMessageBox.warning(self, "图片数据无效", str(exc))
                return
        if Path(source).is_file():
            source = str(Path(source).resolve())
        if not self.stop_video():
            return
        self.events.setRowCount(0)
        self.video_matches.clear()
        self.match_context.clear()
        self.association_context.clear()
        self.session_id = uuid.uuid4().hex
        self.active_source = source
        self.preview_image = QImage()
        self.preview_result = None
        self.preview_frame_index = 0
        self.video_label.set_frame(self.preview_image, None)
        self.worker = VideoWorker(self.engine, source, self.config.frame_interval)
        self.worker.frame_ready.connect(self._show_frame)
        self.worker.recognized.connect(self._on_recognized)
        self.worker.stats.connect(self._worker_stats)
        self.worker.failed.connect(self._worker_failed)
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
        if self.sender() is not None and self.sender() is not self.worker:
            return
        self.start_button.setEnabled(bool(self.engine))
        self.stop_button.setEnabled(False)
        if self.worker and self.worker.had_error:
            return
        kind = "图片" if is_image_source(self.active_source) else "视频"
        faces = f" · 检测到 {self.preview_result.detected_faces} 张人脸" if self.preview_result else ""
        self.statusBar().showMessage(
            f"{kind}处理完成{faces} · 确认匹配 {len(self.video_matches)} 人"
            f" · 待确认 {self.video_matches.pending_count} 人"
        )

    def _worker_stats(self, message: str) -> None:
        if self.sender() is self.worker:
            self.statusBar().showMessage(message)

    def _worker_failed(self, message: str) -> None:
        if self.sender() is self.worker:
            self.statusBar().showMessage("检索处理失败")
            QMessageBox.critical(self, "检索处理失败", message)

    def _show_frame(self, image: QImage, result: FrameResult | None, frame_index: int) -> None:
        if self.sender() is not None and self.sender() is not self.worker:
            return
        if self._reviewing:
            return
        self.preview_image = image
        self.preview_result = result
        self.preview_frame_index = frame_index
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        result = self.preview_result
        if result:
            tracks = []
            for track in result.tracks:
                association = self.video_matches.association(track.id)
                if association:
                    track = replace(track, name=association.name, person_id=association.person_id,
                                    id_card=association.id_card, annotation="人工关联")
                elif (track.person_id or track.candidate_person_id) and self.video_matches.is_rejected(
                    track.id, track.person_id or track.candidate_person_id
                ):
                    track = replace(track, name=f"已排除：{track.candidate_name or track.name}", person_id=None,
                                    id_card="", annotation="不是此人")
                tracks.append(track)
            result = replace(result, tracks=tuple(tracks))
        self.video_label.set_frame(self.preview_image, result)

    def _on_recognized(self, event: dict) -> None:
        if self.sender() is not None and self.sender() is not self.worker:
            return
        changed = self.video_matches.update(event)
        key = (event["track_id"], event["person_id"])
        previous = self.match_context.get(key)
        if not self.video_matches.is_rejected(*key) and (
            previous is None or event["score"] > previous.get("score", -1)
            or (event.get("confirmed", True) and not previous.get("confirmed", True))
        ):
            self.match_context[key] = event.copy()
        if changed:
            self._render_matches()
        if self.database and event.get("confirmed", True) and event.get("new_track_match", True):
            self.database.log_event(
                event["person_id"], self.active_source, event["track_id"],
                event["score"], event["quality"]
            )

    def _render_matches(self) -> None:
        matches = self.video_matches.displayed()
        self.events.setRowCount(len(matches))
        for row, match in enumerate(matches):
            values = (match.time, f"#{match.track_id}", match.name, match.id_card,
                      "—" if match.status == "人工关联" else f"{match.score:.3f}",
                      f"{match.quality:.3f}", match.status)
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                if match.reason:
                    item.setToolTip(f"{value}\n{match.reason}")
                if match.status == "不是此人":
                    item.setForeground(QColor("#c0392b"))
                elif match.status == "待确认":
                    item.setForeground(QColor("#9a6700"))
                self.events.setItem(row, column, item)
            action = "撤销标注" if match.status == "不是此人" else (
                "取消关联" if match.status == "人工关联" else "不是此人"
            )
            button = QPushButton(action)
            button.clicked.connect(lambda checked=False, selected=match: self._review_match(selected))
            self.events.setCellWidget(row, 7, button)

    def _save_annotation(
        self, track_id: int, person_id: str, action: str, context: dict | None = None
    ) -> bool:
        context = context or {}
        try:
            folder = self.database.path.parent / "annotations" / self.session_id
            folder.mkdir(parents=True, exist_ok=True)
            token = uuid.uuid4().hex
            snapshot_path = None
            face_path = None
            if context.get("face_jpeg"):
                face_file = folder / f"{token}-face.jpg"
                face_file.write_bytes(context["face_jpeg"])
                face_path = str(face_file)
            # Preserve the exact visible frame for right-click annotations.
            if context.get("frame_index") == self.preview_frame_index and not self.preview_image.isNull():
                snapshot = folder / f"{token}-frame.jpg"
                if not self.preview_image.save(str(snapshot), "JPG", 95):
                    raise OSError("无法保存标注画面")
                snapshot_path = str(snapshot)
            self.database.log_annotation(
                self.session_id, self.active_source, track_id, action, person_id,
                context.get("frame_index", self.preview_frame_index), context.get("bbox"),
                snapshot_path, face_path,
                score=context.get("score"), quality=context.get("quality"),
                model_id=self.engine.model_id if self.engine else None,
            )
            self.refresh_annotations()
            return True
        except Exception as exc:
            QMessageBox.critical(self, "标注保存失败", str(exc))
            return False

    def _review_match(self, match: VideoMatch) -> None:
        action = {"不是此人": "undo_rejection", "人工关联": "remove_association"}.get(
            match.status, "reject"
        )
        context = self.association_context.get(match.track_id) if match.status == "人工关联" else (
            self.match_context.get((match.track_id, match.person_id))
        )
        if not self._save_annotation(match.track_id, match.person_id, action, context):
            return
        if action == "reject":
            self.video_matches.reject(match.track_id, match.person_id)
        elif action == "undo_rejection":
            self.video_matches.undo_rejection(match.track_id, match.person_id)
        else:
            self.video_matches.remove_association(match.track_id)
            self.association_context.pop(match.track_id, None)
        self._render_matches()
        self._refresh_preview()
        self.statusBar().showMessage(
            f"标注已保存 · 确认匹配 {len(self.video_matches)} 人 · 待确认 {self.video_matches.pending_count} 人"
        )

    def _track_menu(self, track: TrackView, position: object) -> None:
        self._reviewing = True
        try:
            menu = QMenu(self)
            associate = menu.addAction("关联已录入人员…")
            association = self.video_matches.association(track.id)
            remove = menu.addAction("取消关联") if association else None
            chosen = menu.exec(position)
            if remove is not None and chosen == remove:
                self._review_match(association)
                return
            if chosen != associate:
                return
            people = self.database.list_people()
            if not people:
                QMessageBox.information(self, "人员库为空", "请先在人员库录入人员。")
                return
            dialog = PersonAssociationDialog(people, self)
            if dialog.exec() != QDialog.Accepted:
                return
            person = dialog.selected_person()
            if person is None:
                return
            context = {"frame_index": self.preview_frame_index, "bbox": track.bbox,
                       "score": track.score, "quality": track.quality}
            if not self._save_annotation(track.id, person.id, "associate", context):
                return
            self.video_matches.associate(VideoMatch(
                datetime.now().strftime("%H:%M:%S"), track.id, person.id,
                person.name, person.id_card, 0.0, track.quality, "人工关联",
            ))
            self.association_context[track.id] = context
            self._render_matches()
            self._refresh_preview()
            self.statusBar().showMessage(f"轨迹 #{track.id} 已关联 {person.name}")
        finally:
            self._reviewing = False

    def export_annotations(self) -> None:
        if self.export_worker and self.export_worker.isRunning():
            return
        count = len(self.database.list_annotations())
        if not count:
            QMessageBox.information(self, "标注库为空", "请先标注匹配结果或关联人员。")
            return
        dialog = AnnotationExportDialog(count, self)
        if dialog.exec() != QDialog.Accepted:
            return
        options = dialog.options()
        if options.include_media and not options.full_video and not (
            options.seconds_before or options.seconds_after
        ):
            QMessageBox.warning(self, "片段时长无效", "标注前后的时长至少有一项大于 0 秒。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出标注库", f"标注库-{datetime.now():%Y%m%d-%H%M%S}.zip", "ZIP 压缩包 (*.zip)"
        )
        if not path:
            return
        self.export_progress = QProgressDialog("准备导出标注库…", None, 0, 0, self)
        self.export_progress.setWindowTitle("导出标注库")
        self.export_progress.setWindowModality(Qt.WindowModal)
        self.export_progress.show()
        self.export_button.setEnabled(False)
        self.library_export_button.setEnabled(False)
        self.export_worker = AnnotationExportWorker(self.database, Path(path), options)
        self.export_worker.progress.connect(self.export_progress.setLabelText)
        self.export_worker.succeeded.connect(self._export_succeeded)
        self.export_worker.failed.connect(self._export_failed)
        self.export_worker.start()

    def _export_succeeded(self, report: ExportReport) -> None:
        self._close_export_progress()
        message = f"已导出 {report.annotation_count} 条标注。\n{self.export_worker.path}"
        if report.issues:
            QMessageBox.warning(
                self, "导出完成，部分媒体缺失",
                message + "\n\n" + "\n".join(report.issues[:8])
                + "\n完整缺失清单见压缩包内 manifest.json。",
            )
        else:
            QMessageBox.information(self, "导出完成", message)

    def _export_failed(self, message: str) -> None:
        self._close_export_progress()
        QMessageBox.critical(self, "导出失败", message)

    def _close_export_progress(self) -> None:
        if self.export_progress:
            self.export_progress.close()
        self.export_button.setEnabled(True)
        self.library_export_button.setEnabled(True)

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

    def view_person_photos(self) -> None:
        person = self._selected_person()
        if person is None:
            return
        samples = [sample for sample in self.database.list_stored_samples()
                   if sample.person_id == person.id]
        PersonPhotosDialog(person, samples, self).exec()

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
        if self.export_worker and self.export_worker.isRunning():
            QMessageBox.information(self, "正在导出", "请等待标注库导出完成后再关闭。")
            event.ignore()
            return
        if self.stop_video():
            event.accept()
        else:
            event.ignore()
