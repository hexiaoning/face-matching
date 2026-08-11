from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QCloseEvent, QImage
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..database import FaceDatabase
from ..engine import FaceEngine
from ..matcher import GalleryMatcher
from ..pipeline import FrameResult, RecognitionEvent
from ..privacy import redact_source_credentials
from ..video_worker import VideoWorker
from .person_page import PersonPage
from .video_widget import VideoWidget


VIDEO_FILTER = "视频 (*.mp4 *.avi *.mkv *.mov *.wmv *.m4v);;所有文件 (*)"


class MainWindow(QMainWindow):
    def __init__(
        self,
        database: FaceDatabase,
        engine: FaceEngine,
        matcher: GalleryMatcher,
        initial_source: str | int | None = None,
    ) -> None:
        super().__init__()
        self.database = database
        self.engine = engine
        self.matcher = matcher
        self.source: str | int | None = initial_source
        self.worker: VideoWorker | None = None
        self.setWindowTitle("监控视频人脸检索 · CUDA")
        self.resize(1480, 900)

        self.video_widget = VideoWidget()
        self.source_label = QLabel("未选择")
        self.source_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.status_label = QLabel("就绪")
        self.fps_label = QLabel("0.0 FPS")
        self.gallery_label = QLabel()
        self.threshold_label = QLabel()
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(35, 75)
        self.threshold_slider.setValue(round(engine.config.match_threshold * 100))
        self.threshold_slider.valueChanged.connect(self._threshold_changed)
        self.processing_stride = QSpinBox()
        self.processing_stride.setRange(1, 6)
        self.processing_stride.setValue(1)
        self.processing_stride.setToolTip("1 为逐帧检测；数值越大速度越快，但快速经过的人脸可能漏检。")
        self.processing_stride.valueChanged.connect(self._processing_stride_changed)

        self.start_button = QPushButton("开始识别")
        self.pause_button = QPushButton("暂停")
        self.stop_button = QPushButton("停止")
        self.start_button.setProperty("class", "primary")
        self.stop_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_stream)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.stop_button.clicked.connect(self.stop_stream)

        monitor = self._build_monitor_tab()
        self.person_page = PersonPage(database, engine, matcher)
        self.person_page.persons_changed.connect(self._update_gallery_status)
        tabs = QTabWidget()
        tabs.addTab(monitor, "实时识别")
        tabs.addTab(self.person_page, "人员库")
        self.setCentralWidget(tabs)
        self._create_menu()
        self._update_source_label()
        self._update_gallery_status()
        self._threshold_changed(self.threshold_slider.value())
        self.statusBar().showMessage(f"GPU: {engine.provider}  |  模型: {engine.config.model_id}")
        if initial_source is not None:
            self.start_stream()

    def _build_monitor_tab(self) -> QWidget:
        open_button = QPushButton("打开视频…")
        camera_button = QPushButton("摄像头…")
        network_button = QPushButton("RTSP / 网络流…")
        open_button.clicked.connect(self.choose_video)
        camera_button.clicked.connect(self.choose_camera)
        network_button.clicked.connect(self.choose_network)
        source_controls = QHBoxLayout()
        source_controls.addWidget(open_button)
        source_controls.addWidget(camera_button)
        source_controls.addWidget(network_button)
        source_controls.addStretch()

        form = QFormLayout()
        form.addRow("视频源：", self.source_label)
        form.addRow("运行状态：", self.status_label)
        form.addRow("处理速度：", self.fps_label)
        form.addRow("人员库：", self.gallery_label)
        threshold_row = QHBoxLayout()
        threshold_row.addWidget(self.threshold_slider, 1)
        threshold_row.addWidget(self.threshold_label)
        form.addRow("相似度阈值：", threshold_row)
        form.addRow("处理帧步长：", self.processing_stride)
        action_row = QHBoxLayout()
        action_row.addWidget(self.start_button)
        action_row.addWidget(self.pause_button)
        action_row.addWidget(self.stop_button)
        form.addRow(action_row)
        settings = QGroupBox("识别控制")
        settings.setLayout(form)

        self.events_table = QTableWidget(0, 5)
        self.events_table.setHorizontalHeaderLabels(["时间", "姓名", "身份证号", "相似度", "轨迹"])
        self.events_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.events_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.events_table.verticalHeader().setVisible(False)
        self.events_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.events_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.events_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.events_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.events_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        events_group = QGroupBox("识别事件")
        events_layout = QVBoxLayout(events_group)
        events_layout.addWidget(self.events_table)

        side = QWidget()
        side.setMinimumWidth(390)
        side.setMaximumWidth(520)
        side_layout = QVBoxLayout(side)
        side_layout.addLayout(source_controls)
        side_layout.addWidget(settings)
        side_layout.addWidget(events_group, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.video_widget)
        splitter.addWidget(side)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.addWidget(splitter)
        return page

    def _create_menu(self) -> None:
        file_menu = self.menuBar().addMenu("文件")
        open_action = QAction("打开视频…", self)
        exit_action = QAction("退出", self)
        open_action.triggered.connect(self.choose_video)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(open_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)
        help_menu = self.menuBar().addMenu("帮助")
        about_action = QAction("关于", self)
        about_action.triggered.connect(self._about)
        help_menu.addAction(about_action)

    def choose_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", VIDEO_FILTER)
        if path:
            self._set_source(path)

    def choose_camera(self) -> None:
        index, ok = QInputDialog.getInt(self, "选择摄像头", "摄像头编号：", 0, 0, 32)
        if ok:
            self._set_source(index)

    def choose_network(self) -> None:
        value, ok = QInputDialog.getText(
            self, "网络视频流", "RTSP / HTTP 地址：", text="rtsp://"
        )
        if ok and value.strip():
            self._set_source(value.strip())

    def _set_source(self, source: str | int) -> None:
        if self.worker and self.worker.isRunning():
            self.stop_stream()
        self.source = source
        self._update_source_label()

    def _update_source_label(self) -> None:
        self.source_label.setToolTip("")
        if self.source is None:
            text = "未选择"
        elif isinstance(self.source, int):
            text = f"摄像头 {self.source}"
        elif self.source.lower().startswith(("rtsp://", "http://", "https://", "rtmp://")):
            safe_source = redact_source_credentials(self.source)
            text = safe_source[:70] + ("…" if len(safe_source) > 70 else "")
        else:
            text = Path(self.source).name
            self.source_label.setToolTip(self.source)
        self.source_label.setText(text)

    def start_stream(self) -> None:
        if self.worker and self.worker.isRunning():
            return
        if self.source is None:
            QMessageBox.information(self, "请选择视频源", "请先打开视频、摄像头或网络视频流。")
            return
        if self.matcher.sample_count == 0:
            answer = QMessageBox.question(
                self,
                "人员库为空",
                "当前模型下没有已录入的人脸，视频中只会显示“未知”。仍然开始吗？",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.worker = VideoWorker(
            self.engine,
            self.matcher,
            self.source,
            self.threshold_slider.value() / 100.0,
            self.processing_stride.value(),
            self,
        )
        self.worker.frame_ready.connect(self._frame_ready)
        self.worker.recognition.connect(self._add_event)
        self.worker.stream_error.connect(self._stream_error)
        self.worker.stream_ended.connect(self._stream_ended)
        self.worker.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.pause_button.setEnabled(True)
        self.pause_button.setText("暂停")
        self.status_label.setText("GPU 识别中")

    def toggle_pause(self) -> None:
        if not self.worker or not self.worker.isRunning():
            return
        paused = self.pause_button.text() == "暂停"
        self.worker.set_paused(paused)
        self.pause_button.setText("继续" if paused else "暂停")
        self.status_label.setText("已暂停" if paused else "GPU 识别中")

    def stop_stream(self) -> None:
        worker = self.worker
        if worker and worker.isRunning():
            self.status_label.setText("正在停止…")
            worker.stop()
            if not worker.wait(5000):
                self.status_label.setText("视频源仍在断开中")
                QMessageBox.warning(
                    self,
                    "停止超时",
                    "视频源暂未响应，后台仍在等待读取结束。请稍后重试，不会强制终止线程。",
                )
                return
        self._stream_ended()

    def _frame_ready(self, image: QImage, result: FrameResult, fps: float) -> None:
        self.video_widget.set_frame(image, result)
        self.fps_label.setText(f"{fps:.1f} FPS · {len(result.faces)} 张人脸")

    def _add_event(self, event: RecognitionEvent) -> None:
        self.events_table.insertRow(0)
        values = (
            event.time.strftime("%H:%M:%S"),
            event.name,
            event.masked_id_card,
            f"{event.score:.3f}",
            f"#{event.track_id}",
        )
        for column, value in enumerate(values):
            self.events_table.setItem(0, column, QTableWidgetItem(value))
        while self.events_table.rowCount() > 500:
            self.events_table.removeRow(self.events_table.rowCount() - 1)

    def _stream_error(self, message: str) -> None:
        QMessageBox.critical(self, "视频错误", message)

    def _stream_ended(self) -> None:
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.pause_button.setText("暂停")
        self.status_label.setText("已停止")

    def _threshold_changed(self, value: int) -> None:
        threshold = value / 100.0
        self.threshold_label.setText(f"{threshold:.2f}")
        if self.worker:
            self.worker.set_threshold(threshold)

    def _processing_stride_changed(self, value: int) -> None:
        if self.worker:
            self.worker.set_processing_stride(value)

    def _update_gallery_status(self) -> None:
        self.gallery_label.setText(
            f"{self.matcher.person_count} 人 / {self.matcher.sample_count} 张照片"
        )

    def _about(self) -> None:
        QMessageBox.about(
            self,
            "关于",
            "监控视频人脸检索 0.2.2\n\n"
            "SCRFD-10G + LVFace-B + 镜像 TTA + 鲁棒多帧聚合\n"
            "推理强制使用 CUDA，不允许 CPU fallback。\n\n"
            "注意：默认下载的预训练权重仅限非商业研究；生产部署需替换为已获授权的 ONNX 权重。",
        )

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            if not self.worker.wait(5000):
                event.ignore()
                return
        self.database.close()
        event.accept()
