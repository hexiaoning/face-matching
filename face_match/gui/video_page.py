"""视频分析页：打开视频/流，实时检测识别，结果列表。"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QPushButton, QSlider, QSpinBox, QSplitter,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QAbstractItemView,
)

from .. import config
from ..matcher import MatchResult
from ..pipeline import FrameResult
from .widgets import ModelHub, VideoLabel, draw_annotations


@dataclass
class PersonSighting:
    name: str
    id_number: str
    best_score: float
    first_seen: float
    last_seen: float
    frames: int = 1


class _VideoWorker(QThread):
    """后台读视频 + 跑 pipeline。"""
    frame_ready = Signal(object, object)   # (annotated_bgr, FrameResult)
    finished_all = Signal()
    error = Signal(str)

    def __init__(self, hub: ModelHub, source: str, threshold: float,
                 frame_skip: int, show_quality: bool):
        super().__init__()
        self.hub = hub
        self.source = source
        self.threshold = threshold
        self.frame_skip = frame_skip
        self.show_quality = show_quality
        self._running = True
        self._paused = False
        self.pipeline = hub.new_pipeline(threshold)

    def stop(self) -> None:
        self._running = False

    def set_paused(self, p: bool) -> None:
        self._paused = p

    def set_threshold(self, v: float) -> None:
        self.threshold = v
        self.pipeline.threshold = v

    def run(self) -> None:
        src = self.source
        cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
        if not cap.isOpened():
            self.error.emit(f"无法打开视频源: {self.source}")
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        idx = -1
        last_result: FrameResult | None = None
        try:
            while self._running:
                if self._paused:
                    time.sleep(0.05)
                    continue
                ok, frame = cap.read()
                if not ok:
                    break
                idx += 1
                if idx % (self.frame_skip + 1) != 0 and last_result is not None:
                    continue
                ts = idx / fps
                with self.hub.lock:
                    result = self.pipeline.process_frame(frame, idx, ts)
                last_result = result
                annotated = draw_annotations(frame, result, self.show_quality)
                self.frame_ready.emit(annotated, result)
        except Exception as e:  # noqa: BLE001
            self.error.emit(f"视频处理出错: {e}")
        finally:
            cap.release()
            self.finished_all.emit()


def _fmt_ts(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class VideoPage(QWidget):
    def __init__(self, hub: ModelHub):
        super().__init__()
        self.hub = hub
        self.worker: _VideoWorker | None = None
        self.sightings: dict[str, PersonSighting] = {}

        root = QHBoxLayout(self)
        split = QSplitter()
        root.addWidget(split)

        # 左：视频 + 控制
        left = QWidget()
        ll = QVBoxLayout(left)
        self.video = VideoLabel()
        ll.addWidget(self.video, 1)

        row1 = QHBoxLayout()
        self.btn_open = QPushButton("打开视频文件…")
        self.btn_open.clicked.connect(self._open_file)
        self.edit_url = QLineEdit()
        self.edit_url.setPlaceholderText("或输入视频流地址 (rtsp://… 或摄像头编号 0)")
        self.btn_open_url = QPushButton("打开视频流")
        self.btn_open_url.clicked.connect(self._open_url)
        row1.addWidget(self.btn_open)
        row1.addWidget(self.edit_url, 1)
        row1.addWidget(self.btn_open_url)
        ll.addLayout(row1)

        row2 = QHBoxLayout()
        self.btn_pause = QPushButton("暂停")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self._toggle_pause)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        row2.addWidget(self.btn_pause)
        row2.addWidget(self.btn_stop)

        row2.addWidget(QLabel("匹配阈值"))
        self.spin_th = QDoubleSpinBox()
        self.spin_th.setRange(0.15, 0.80)
        self.spin_th.setSingleStep(0.05)
        self.spin_th.setValue(config.MATCH_THRESHOLD)
        self.spin_th.valueChanged.connect(self._on_threshold)
        row2.addWidget(self.spin_th)

        row2.addWidget(QLabel("跳帧"))
        self.spin_skip = QSpinBox()
        self.spin_skip.setRange(0, 15)
        self.spin_skip.setValue(config.FRAME_SKIP)
        self.spin_skip.valueChanged.connect(self._on_skip)
        row2.addWidget(self.spin_skip)

        self.chk_quality = QCheckBox("显示质量分")
        self.chk_quality.setChecked(True)
        row2.addWidget(self.chk_quality)
        row2.addStretch(1)
        ll.addLayout(row2)

        self.status = QLabel("就绪")
        ll.addWidget(self.status)
        split.addWidget(left)

        # 右：识别结果
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.addWidget(QLabel("识别到的人员"))
        self.results = QTableWidget(0, 5)
        self.results.setHorizontalHeaderLabels(["姓名", "身份证号", "最高分", "首次出现", "最近出现"])
        self.results.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.results.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        rl.addWidget(self.results, 1)
        split.addWidget(right)
        split.setSizes([720, 360])

    # ---- 控制 ----
    def _open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频 (*.mp4 *.avi *.mkv *.mov *.flv *.wmv *.ts);;所有文件 (*)")
        if path:
            self._start(path)

    def _open_url(self) -> None:
        url = self.edit_url.text().strip()
        if url:
            self._start(url)

    def _start(self, source: str) -> None:
        self._stop()
        self.sightings.clear()
        self.results.setRowCount(0)
        self.worker = _VideoWorker(self.hub, source, self.spin_th.value(),
                                   self.spin_skip.value(), self.chk_quality.isChecked())
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.error.connect(lambda m: self.status.setText("错误: " + m))
        self.worker.finished_all.connect(self._on_finished)
        self.worker.start()
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.status.setText(f"处理中: {source}")

    def _toggle_pause(self) -> None:
        if not self.worker:
            return
        paused = self.btn_pause.text() == "暂停"
        self.worker.set_paused(paused)
        self.btn_pause.setText("继续" if paused else "暂停")

    def _stop(self) -> None:
        if self.worker:
            self.worker.stop()
            self.worker.wait(3000)
            self.worker = None
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("暂停")
        self.btn_stop.setEnabled(False)

    def _on_finished(self) -> None:
        self.status.setText(self.status.text().replace("处理中", "已完成"))
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)

    def _on_threshold(self, v: float) -> None:
        if self.worker:
            self.worker.set_threshold(v)

    def _on_skip(self, v: int) -> None:
        if self.worker:
            self.worker.frame_skip = v

    # ---- 结果 ----
    def _on_frame(self, annotated, result: FrameResult) -> None:
        self.video.show_frame(annotated)
        changed = False
        for trk in result.tracks:
            m: MatchResult | None = trk.match
            if m is None:
                continue
            key = m.person.id_number or m.person.name
            s = self.sightings.get(key)
            if s is None:
                self.sightings[key] = PersonSighting(
                    name=m.person.name, id_number=m.person.id_number,
                    best_score=m.score, first_seen=result.timestamp,
                    last_seen=result.timestamp)
                changed = True
            else:
                if m.score > s.best_score or result.timestamp > s.last_seen:
                    s.best_score = max(s.best_score, m.score)
                    s.last_seen = max(s.last_seen, result.timestamp)
                    changed = True
        if changed:
            self._refresh_results()

    def _refresh_results(self) -> None:
        rows = sorted(self.sightings.values(), key=lambda s: s.best_score, reverse=True)
        self.results.setRowCount(len(rows))
        for r, s in enumerate(rows):
            self.results.setItem(r, 0, QTableWidgetItem(s.name))
            self.results.setItem(r, 1, QTableWidgetItem(s.id_number))
            self.results.setItem(r, 2, QTableWidgetItem(f"{s.best_score:.3f}"))
            self.results.setItem(r, 3, QTableWidgetItem(_fmt_ts(s.first_seen)))
            self.results.setItem(r, 4, QTableWidgetItem(_fmt_ts(s.last_seen)))

    def closeEvent(self, e) -> None:  # noqa: N802
        self._stop()
        super().closeEvent(e)
