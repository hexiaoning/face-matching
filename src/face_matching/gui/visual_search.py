from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QHeaderView, QLabel,
    QListWidget, QMessageBox, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from ..image_enhancement import encode_png
from ..visual_search import (FaceExampleMatcher, ObjectExampleMatcher, SearchHit,
                             collect_sources, read_hit_frame, search_sources)
from ..vision.io import read_image
from .image_tools import ImagePane


class SearchWorker(QThread):
    completed = Signal(object, object)
    progress = Signal(str)
    failed = Signal(str)

    def __init__(self, query: np.ndarray, paths: list[Path], mode: str, threshold: float,
                 step: int, engine: object, parent: QWidget) -> None:
        super().__init__(parent)
        self.query, self.paths, self.mode = query.copy(), list(paths), mode
        self.threshold, self.step, self.engine = threshold, step, engine
        self.stopped = threading.Event()

    def run(self) -> None:
        try:
            matcher = (FaceExampleMatcher(self.query, self.engine, self.threshold) if self.mode == 'face'
                       else ObjectExampleMatcher(self.query, self.threshold))
            sources = collect_sources(self.paths)
            if not sources:
                raise ValueError("所选范围内没有支持的图片或视频")
            hits, failures = search_sources(sources, matcher, frame_step=self.step,
                                            stopped=self.stopped, on_progress=self.progress.emit)
            self.completed.emit(hits, failures)
        except Exception as exc:
            self.failed.emit(str(exc))


class VisualSearchWidget(QWidget):
    enhance_requested = Signal(object, object)

    def __init__(self, prepare_face_engine, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.prepare_face_engine = prepare_face_engine
        self.original = None
        self.region = None
        self.query_path = ''
        self.paths: list[Path] = []
        self.hits: list[SearchHit] = []
        self.failures: list[str] = []
        self.search_info = None
        self.search_error = None
        self.worker: SearchWorker | None = None
        self.query_view, self.preview = ImagePane(selectable=True), ImagePane()
        self.query_view.region_selected.connect(self.select_region)
        self.mode = QComboBox()
        self.mode.addItem("搜索物体 / 人物外观", 'object')
        self.mode.addItem("搜索同一人脸（GPU）", 'face')
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(.1, .99)
        self.threshold.setSingleStep(.05)
        self.threshold.setValue(.75)
        self.step = QSpinBox()
        self.step.setRange(1, 1000)
        self.step.setValue(1)
        self.step.setSuffix(" 帧")
        self.open_button = QPushButton("打开目标图片…")
        self.open_button.clicked.connect(self.open_query)
        self.clear_region = QPushButton("使用整张图片")
        self.clear_region.clicked.connect(lambda: self.select_region(None))
        self.add_files = QPushButton("添加待搜图片 / 视频…")
        self.add_files.clicked.connect(self.choose_files)
        self.add_folder = QPushButton("添加待搜文件夹…")
        self.add_folder.clicked.connect(self.choose_folder)
        self.clear_sources = QPushButton("清空范围")
        self.clear_sources.clicked.connect(self.clear_paths)
        self.start_button, self.stop_button = QPushButton("开始搜索"), QPushButton("停止")
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        self.export_button = QPushButton("导出搜索记录 JSON…")
        self.export_button.clicked.connect(self.export)
        self.enhance_button = QPushButton("清晰化选中结果")
        self.enhance_button.clicked.connect(self.enhance_selected)
        self.sources = QListWidget()
        self.sources.setMinimumHeight(55)
        self.sources.setMaximumHeight(80)
        self.table = QTableWidget(0, 5)
        self.table.setMinimumHeight(160)
        self.table.setHorizontalHeaderLabels(["来源", "帧 / 时间", "相似分值", "原图坐标", "匹配方式"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self.show_selected)
        self.status = QLabel("打开图片，框选要找的人或物体，再添加本地搜索范围。")
        self.status.setWordWrap(True)
        self.region_label = QLabel("使用整张目标图片")
        top = QHBoxLayout()
        for widget in (self.open_button, self.clear_region, self.mode, QLabel("相似度门槛："),
                       self.threshold, QLabel("视频每隔"), self.step, self.start_button, self.stop_button):
            top.addWidget(widget)
        scope = QHBoxLayout()
        for widget in (self.add_files, self.add_folder, self.clear_sources):
            scope.addWidget(widget)
        scope.addStretch()
        panes = QHBoxLayout()
        for label, pane in (("目标图片 · 拖动框选人脸或物体", self.query_view), ("选中结果 · 框为匹配位置", self.preview)):
            column = QVBoxLayout()
            column.addWidget(QLabel(label))
            column.addWidget(pane)
            panes.addLayout(column, 1)
        footer = QHBoxLayout()
        footer.addWidget(self.region_label)
        footer.addStretch()
        footer.addWidget(self.enhance_button)
        footer.addWidget(self.export_button)
        layout = QVBoxLayout(self)
        note = QLabel("外观搜索查找相同或近似外观，适合纹理清晰、视角接近的目标；不按文字类别搜索。"
                      "同一人员请使用人脸模式。分值仅供排序，结果需人工核实。抽帧间隔大于1时可能漏过短暂出现的目标。")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addLayout(top)
        layout.addLayout(scope)
        layout.addWidget(self.sources)
        layout.addLayout(panes, 2)
        layout.addWidget(self.table, 1)
        layout.addLayout(footer)
        layout.addWidget(self.status)
        self.update_buttons()

    def is_busy(self) -> bool:
        return self.worker is not None

    def update_buttons(self) -> None:
        busy = self.is_busy()
        for widget in (self.open_button, self.clear_region, self.add_files, self.add_folder,
                       self.clear_sources, self.mode, self.threshold, self.step, self.query_view):
            widget.setEnabled(not busy)
        self.start_button.setEnabled(not busy and self.original is not None and bool(self.paths))
        self.stop_button.setEnabled(busy)
        self.export_button.setEnabled(not busy and self.search_info is not None)
        self.enhance_button.setEnabled(not busy and self.table.currentRow() >= 0)

    def _mode_changed(self) -> None:
        self.threshold.setValue(.45 if self.mode.currentData() == 'face' else .75)

    def open_query(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "打开目标图片", '', "图片 (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)")
        if path:
            self.load_query(Path(path))

    def load_query(self, path: Path) -> bool:
        if self.is_busy():
            return False
        try:
            image = read_image(path)
            if image is None or image.shape[0] * image.shape[1] > 40_000_000:
                raise ValueError("图片无法解码或超过4000万像素")
        except (OSError, ValueError, cv2.error) as exc:
            QMessageBox.warning(self, "无法打开目标", str(exc))
            return False
        self.original, self.query_path = image, str(path.resolve())
        self.select_region(None)
        self.query_view.set_image(image)
        self.update_buttons()
        return True

    def select_region(self, region) -> None:
        self.region = region
        self.query_view.set_region(region)
        self.query_view.fit_image()
        self.region_label.setText(f"目标选区：{region}" if region else "使用整张目标图片")

    def choose_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "添加搜索范围", '',
            "图片和视频 (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff *.mp4 *.avi *.mkv *.mov *.m4v *.wmv *.webm)")
        self.set_paths(self.paths + [Path(path) for path in paths])

    def choose_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "添加搜索文件夹（包含子文件夹）")
        if path:
            self.set_paths(self.paths + [Path(path)])

    def set_paths(self, paths: list[Path]) -> None:
        self.paths = list(dict.fromkeys(paths))
        self.sources.clear()
        self.sources.addItems([str(path) for path in self.paths])
        self.update_buttons()

    def clear_paths(self) -> None:
        self.set_paths([])

    def start(self) -> None:
        if self.is_busy() or self.original is None or not self.paths:
            return
        mode = self.mode.currentData()
        engine = self.prepare_face_engine() if mode == 'face' else None
        if mode == 'face' and engine is None:
            return
        query = self.original
        if self.region:
            x1, y1, x2, y2 = self.region
            query = query[y1:y2, x1:x2]
        self.hits, self.failures = [], []
        self.table.setRowCount(0)
        self.search_info = {"query_source": self.query_path, "region_xyxy": self.region,
                            "query_pixels_sha256": hashlib.sha256(encode_png(query)).hexdigest(),
                            "mode": mode, "threshold": self.threshold.value(), "frame_step": self.step.value(),
                            "scope": [str(path) for path in self.paths], "max_results": 500,
                            "started_at": datetime.now(timezone.utc).isoformat()}
        if engine is not None:
            self.search_info['face_model_id'] = engine.model_id
            self.search_info['min_quality'] = engine.config.min_quality
        self.worker = SearchWorker(query, self.paths, mode, self.threshold.value(), self.step.value(), engine, self)
        self.worker.progress.connect(self.status.setText)
        self.search_error = None
        self.worker.failed.connect(self._failed)
        self.worker.completed.connect(self._completed)
        self.worker.finished.connect(self._finished)
        self.worker.start()
        self.update_buttons()

    def stop(self) -> None:
        if self.worker:
            self.worker.stopped.set()
            self.status.setText("正在停止，等待当前帧处理完成…")

    def _completed(self, hits, failures) -> None:
        self.hits, self.failures = hits, failures
        self.table.setRowCount(len(hits))
        for row, hit in enumerate(hits):
            timestamp = ('图片' if hit.seconds is None else f"第 {hit.frame_index + 1} 帧 / {hit.seconds:.2f} 秒")
            for col, value in enumerate((Path(hit.source).name, timestamp, f"{hit.score:.3f}", str(hit.bbox), hit.method)):
                item = QTableWidgetItem(value)
                item.setToolTip(hit.source if col == 0 else value)
                self.table.setItem(row, col, item)
        state = "已停止，保留部分结果" if self.worker.stopped.is_set() else "搜索完成"
        self.status.setText(f"{state} · 候选 {len(hits)} 条（最高分500条） · 失败 {len(failures)} 项"
                            + (f"\n{failures[0]}" if failures else ''))
        self.search_info['cancelled'] = self.worker.stopped.is_set()
        if hits:
            self.table.selectRow(0)

    def _finished(self) -> None:
        worker, self.worker = self.worker, None
        self.search_info['cancelled'] = worker.stopped.is_set()
        if self.search_error:
            self.search_info['error'] = self.search_error
        worker.deleteLater()
        self.update_buttons()

    def _failed(self, error: str) -> None:
        self.search_error = error
        self.status.setText(f"搜索失败：{error}")

    def show_selected(self) -> None:
        row = self.table.currentRow()
        if 0 <= row < len(self.hits):
            try:
                hit = self.hits[row]
                frame = read_hit_frame(hit)
                cv2.rectangle(frame, hit.bbox[:2], hit.bbox[2:], (0, 220, 255), 2)
                self.preview.set_image(frame)
            except (OSError, ValueError, cv2.error) as exc:
                self.status.setText(str(exc))
        self.update_buttons()

    def enhance_selected(self) -> None:
        row = self.table.currentRow()
        if 0 <= row < len(self.hits):
            try:
                hit = self.hits[row]
                self.enhance_requested.emit(read_hit_frame(hit), hit.bbox)
            except (OSError, ValueError, cv2.error) as exc:
                QMessageBox.warning(self, "无法读取结果", str(exc))

    def export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "导出搜索记录", "搜索记录.json", "JSON (*.json)")
        if not path or self.search_info is None:
            return
        try:
            payload = {"search": self.search_info, "hits": [asdict(hit) for hit in self.hits], "failures": self.failures}
            with Path(path).open('x', encoding='utf-8') as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "无法导出", str(exc))
