from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import threading

import cv2
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QListWidget, QMessageBox, QPushButton, QSlider,
    QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..models import SEMANTIC_MODEL_ID
from ..person_vehicle import load_person_vehicle_detector
from ..semantic_model import SemanticEncoder
from ..semantic_search import SemanticIndex, parse_query
from ..visual_search import SearchHit, collect_sources, read_hit_frame
from ..vision.io import IMAGE_EXTENSIONS
from .image_tools import ImagePane


def time_label(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remaining = divmod(milliseconds, 3600000)
    minutes, remaining = divmod(remaining, 60000)
    secs, ms = divmod(remaining, 1000)
    return f'{hours:02}:{minutes:02}:{secs:02}.{ms:03}'


class SourceVideoDialog(QDialog):
    """Read the original video at the indexed frame, with local playback."""

    def __init__(self, hit: SearchHit, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f'原视频 · {Path(hit.source).name} · {time_label(hit.seconds or 0)}')
        self.resize(1000, 720)
        self.capture = cv2.VideoCapture(hit.source)
        if not self.capture.isOpened():
            self.capture.release()
            raise ValueError('无法打开原视频，文件可能已被移动。')
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.fps = fps if math.isfinite(fps) and fps > 0 else 25.
        count = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
        self.count = max(1, int(count)) if math.isfinite(count) else 1
        self.pane = ImagePane()
        self.position = QLabel()
        self.play_button = QPushButton('播放')
        self.play_button.clicked.connect(self.toggle)
        self.return_button = QPushButton('返回命中位置')
        self.return_button.clicked.connect(lambda: self.seek(hit.frame_index))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, min(self.count - 1, 2147483647))
        self.slider.sliderPressed.connect(self.pause)
        self.slider.sliderReleased.connect(lambda: self.seek(self.slider.value()))
        self.timer = QTimer(self)
        self.timer.setInterval(max(1, round(1000 / self.fps)))
        self.timer.timeout.connect(self.next_frame)
        layout = QVBoxLayout(self)
        path_label = QLabel(hit.source)
        path_label.setWordWrap(True)
        layout.addWidget(path_label)
        layout.addWidget(self.pane, 1)
        layout.addWidget(self.slider)
        controls = QHBoxLayout()
        for widget in (self.play_button, self.return_button, self.position):
            controls.addWidget(widget)
        layout.addLayout(controls)
        self.seek(hit.frame_index)

    def pause(self) -> None:
        self.timer.stop()
        self.play_button.setText('播放')

    def toggle(self) -> None:
        if self.timer.isActive():
            self.pause()
        else:
            self.timer.start()
            self.play_button.setText('暂停')

    def seek(self, frame_index: int) -> None:
        self.pause()
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        self.next_frame()

    def next_frame(self) -> None:
        ok, frame = self.capture.read()
        if not ok:
            self.pause()
            return
        index = max(0, round(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        milliseconds = self.capture.get(cv2.CAP_PROP_POS_MSEC)
        seconds = milliseconds / 1000 if math.isfinite(milliseconds) and milliseconds > 0 else index / self.fps
        self.pane.set_image(frame)
        if not self.slider.isSliderDown():
            self.slider.setValue(index)
        self.position.setText(f'第{index + 1}帧 · {time_label(seconds)}')

    def done(self, result: int) -> None:
        self.pause()
        self.capture.release()
        super().done(result)


class SemanticSearchWorker(QThread):
    completed = Signal(object, object, object)
    progress = Signal(str)
    failed = Signal(str)

    def __init__(self, query, paths, step, threshold, backend, device_id, parent) -> None:
        super().__init__(parent)
        self.query, self.paths = query, list(paths)
        self.step, self.threshold = step, threshold
        self.backend, self.device_id = backend, device_id
        self.stopped = threading.Event()

    def run(self) -> None:
        index = None
        try:
            sources = collect_sources(self.paths)
            if not sources:
                raise ValueError('所选范围内没有支持的图片或视频。')
            self.progress.emit('正在校验并加载离线中文语义模型…')
            encoder = SemanticEncoder(self.backend, self.device_id)
            # Validate text before loading detection/vision models or writing the index.
            encoder.tokenizer.encode(self.query.text)
            index = SemanticIndex()
            hits, failures, stats = index.search(sources, self.query, encoder,
                lambda: load_person_vehicle_detector(self.backend, self.device_id, threshold=.35),
                step=self.step, threshold=self.threshold, stopped=self.stopped, progress=self.progress.emit)
            stats['providers'] = encoder.providers()
            self.completed.emit(hits, failures, stats)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            if index is not None:
                index.close()


class SemanticSearchWidget(QWidget):
    def __init__(self, config, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.config = config
        self.worker: SemanticSearchWorker | None = None
        self.paths: list[Path] = []
        self.hits: list[SearchHit] = []
        self.failures: list[str] = []
        self.search_info = None
        self.query = QLineEdit()
        self.query.setPlaceholderText('例如：黑色帽子、蓝色上衣、白色裤子的人；或：白色的车')
        self.query.setMaxLength(160)
        self.query.textChanged.connect(self.update_buttons)
        self.query.returnPressed.connect(self.start)
        self.category = QComboBox()
        for title, value in (('自动判断目标', 'auto'), ('人物衣着', 'person'), ('车辆颜色', 'vehicle')):
            self.category.addItem(title, value)
        self.step = QSpinBox()
        self.step.setRange(1, 1000)
        self.step.setValue(25)
        self.step.setSuffix(' 帧')
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0., 1.)
        self.threshold.setSingleStep(.01)
        self.threshold.setValue(.20)
        self.add_files = QPushButton('添加图片 / 视频…')
        self.add_files.clicked.connect(self.choose_files)
        self.add_folder = QPushButton('添加文件夹…')
        self.add_folder.clicked.connect(self.choose_folder)
        self.clear_sources = QPushButton('清空范围')
        self.clear_sources.clicked.connect(lambda: self.set_paths([]))
        self.start_button, self.stop_button = QPushButton('搜索'), QPushButton('停止')
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        self.sources = QListWidget()
        self.sources.setMaximumHeight(75)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['来源', '原视频时间 / 帧号', '相关分值', '目标坐标'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self.show_selected)
        self.table.cellDoubleClicked.connect(lambda *_: self.open_video())
        self.preview = ImagePane()
        self.play_button = QPushButton('从命中位置播放原视频…')
        self.play_button.clicked.connect(self.open_video)
        self.export_button = QPushButton('导出搜索记录 JSON…')
        self.export_button.clicked.connect(self.export)
        self.status = QLabel('添加本地素材，输入中文描述后搜索。首次分析建立索引，后续相同抽帧设置可复用。')
        self.status.setWordWrap(True)
        layout = QVBoxLayout(self)
        note = QLabel('完全离线 · 支持人物衣着、车辆颜色。组合条件在同一目标内匹配，结果为待核实候选。'
                      '夜间黑白、模糊、遮挡会影响颜色判断；不支持行为和身份推断。抽帧可能漏过短暂目标，逐帧分析请选择1帧。')
        note.setWordWrap(True)
        layout.addWidget(note)
        query_row = QHBoxLayout()
        query_row.addWidget(self.query, 1)
        for widget in (self.category, self.start_button, self.stop_button):
            query_row.addWidget(widget)
        layout.addLayout(query_row)
        scope = QHBoxLayout()
        for widget in (self.add_files, self.add_folder, self.clear_sources, QLabel('每隔'), self.step,
                       QLabel('相关度门槛'), self.threshold):
            scope.addWidget(widget)
        scope.addStretch()
        layout.addLayout(scope)
        layout.addWidget(self.sources)
        content = QHBoxLayout()
        content.addWidget(self.table, 3)
        content.addWidget(self.preview, 2)
        layout.addLayout(content, 1)
        footer = QHBoxLayout()
        footer.addWidget(self.play_button)
        footer.addStretch()
        footer.addWidget(self.export_button)
        layout.addLayout(footer)
        layout.addWidget(self.status)
        self.update_buttons()

    def is_busy(self) -> bool:
        return self.worker is not None

    def update_buttons(self) -> None:
        busy = self.is_busy()
        for widget in (self.query, self.category, self.step, self.threshold, self.add_files,
                       self.add_folder, self.clear_sources):
            widget.setEnabled(not busy)
        self.start_button.setEnabled(not busy and bool(self.query.text().strip()) and bool(self.paths))
        self.stop_button.setEnabled(busy)
        row = self.table.currentRow()
        self.play_button.setEnabled(not busy and 0 <= row < len(self.hits)
                                    and self.hits[row].seconds is not None)
        self.export_button.setEnabled(not busy and self.search_info is not None)

    def set_paths(self, paths: list[Path]) -> None:
        if self.is_busy():
            return
        self.paths = list(dict.fromkeys(paths))
        self.sources.clear()
        self.sources.addItems([str(path) for path in self.paths])
        self.update_buttons()

    def choose_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, '添加搜索范围', '',
            '图片和视频 (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff *.mp4 *.avi *.mkv *.mov *.m4v *.wmv *.webm)')
        self.set_paths(self.paths + [Path(path) for path in paths])

    def choose_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, '添加搜索文件夹（包含子文件夹）')
        if path:
            self.set_paths(self.paths + [Path(path)])

    def start(self) -> None:
        if self.is_busy() or not self.paths:
            return
        try:
            query = parse_query(self.query.text(), self.category.currentData())
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self.hits, self.failures = [], []
        self.table.setRowCount(0)
        self.search_info = {'description': query.text, 'category': query.category,
            'conditions': [asdict(clause) for clause in query.clauses], 'model_id': SEMANTIC_MODEL_ID,
            'frame_step': self.step.value(), 'threshold': self.threshold.value(),
            'scope': [str(path) for path in self.paths], 'max_results': 500,
            'started_at': datetime.now(timezone.utc).isoformat()}
        self.worker = SemanticSearchWorker(query, self.paths, self.step.value(), self.threshold.value(),
                                            self.config.gpu_backend, self.config.gpu_device_id, self)
        self.worker.progress.connect(self.status.setText)
        self.worker.failed.connect(self._failed)
        self.worker.completed.connect(self._completed)
        self.worker.finished.connect(self._finished)
        self.worker.start()
        self.update_buttons()

    def stop(self) -> None:
        if self.worker:
            self.worker.stopped.set()
            self.status.setText('正在停止，等待当前模型推理完成…')

    def _completed(self, hits, failures, stats) -> None:
        self.hits, self.failures = hits, failures
        self.search_info['statistics'] = stats
        self.table.setRowCount(len(hits))
        for row, hit in enumerate(hits):
            position = '图片' if hit.seconds is None else f'{time_label(hit.seconds)} / 第{hit.frame_index + 1}帧'
            for col, value in enumerate((Path(hit.source).name, position, f'{hit.score:.3f}', str(hit.bbox))):
                item = QTableWidgetItem(value)
                item.setToolTip(hit.source if col == 0 else value)
                self.table.setItem(row, col, item)
        state = '已停止，保留已完成素材的结果' if self.worker.stopped.is_set() else '搜索完成'
        self.status.setText(f'{state} · 候选{len(hits)}条（最多500条） · 复用索引{stats["reused_sources"]}项'
                            f' · 新建索引{stats["indexed_sources"]}项 · 失败{len(failures)}项'
                            + (f'\n{failures[0]}' if failures else ''))
        if hits:
            self.table.selectRow(0)

    def _failed(self, error: str) -> None:
        self.search_info['error'] = error
        self.status.setText('搜索失败：' + error)

    def _finished(self) -> None:
        worker, self.worker = self.worker, None
        self.search_info['cancelled'] = worker.stopped.is_set()
        self.search_info['finished_at'] = datetime.now(timezone.utc).isoformat()
        worker.deleteLater()
        self.update_buttons()

    def show_selected(self) -> None:
        row = self.table.currentRow()
        if 0 <= row < len(self.hits):
            try:
                hit = self.hits[row]
                frame = read_hit_frame(hit)
                cv2.rectangle(frame, hit.bbox[:2], hit.bbox[2:], (0, 220, 255), 2)
                self.preview.set_image(frame)
            except (ValueError, OSError, cv2.error) as exc:
                self.status.setText(str(exc))
        self.update_buttons()

    def open_video(self) -> None:
        row = self.table.currentRow()
        if self.is_busy() or not 0 <= row < len(self.hits) or self.hits[row].seconds is None:
            return
        try:
            dialog = SourceVideoDialog(self.hits[row], self)
            try:
                dialog.exec()
            finally:
                dialog.capture.release()
                dialog.deleteLater()
        except (ValueError, OSError, cv2.error) as exc:
            self.status.setText(str(exc))

    def export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, '导出语义搜索记录', '语义搜索记录.json', 'JSON (*.json)')
        if not path or self.search_info is None:
            return
        try:
            with Path(path).open('x', encoding='utf-8') as stream:
                json.dump({'search': self.search_info, 'hits': [asdict(hit) for hit in self.hits],
                           'failures': self.failures}, stream, ensure_ascii=False, indent=2)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, '无法导出', str(exc))
