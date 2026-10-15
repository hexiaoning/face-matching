from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QSpinBox, QWidget,
)

from ..annotation_export import ExportOptions, export_annotation_library
from ..database import FaceDatabase


class AnnotationExportDialog(QDialog):
    def __init__(self, count: int, parent: QWidget, *, selected: bool = False) -> None:
        super().__init__(parent)
        self.setWindowTitle("导出标注库")
        self.resize(480, 330)
        form = QFormLayout(self)
        self.scope_label = QLabel(f"导出{'选中的' if selected else '全部'} {count} 条标注及操作历史（ZIP）")
        form.addRow(self.scope_label)
        if selected:
            note = QLabel("有效状态依据完整历史判断；已撤销或替换的记录仅作历史导出，不计入有效标注。")
            note.setWordWrap(True)
            form.addRow(note)
        self.media = QCheckBox("导出来源图片 / 视频及标注画面")
        self.media.setChecked(True)
        form.addRow(self.media)
        self.video_mode = QComboBox()
        self.video_mode.addItem("标注前后片段（默认，无音轨）", False)
        self.video_mode.addItem("完整视频（保留原文件）", True)
        form.addRow("视频范围：", self.video_mode)
        self.before = QSpinBox()
        self.after = QSpinBox()
        for spin in (self.before, self.after):
            spin.setRange(0, 3600)
            spin.setValue(5)
            spin.setSuffix(" 秒")
        form.addRow("标注前：", self.before)
        form.addRow("标注后：", self.after)
        self.portraits = QCheckBox("人员人像（人员库录入照片）")
        self.info = QCheckBox("人员其它信息（姓名、身份证号等）")
        self.portraits.setChecked(True)
        self.info.setChecked(True)
        form.addRow(self.portraits)
        form.addRow(self.info)
        note = QLabel("仅导出标注涉及的人员；直播流或已移走的文件只导出已保存画面，并列出缺失项。")
        note.setWordWrap(True)
        form.addRow(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Save).setText("导出…")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self.media.toggled.connect(self._update_enabled)
        self.video_mode.currentIndexChanged.connect(self._update_enabled)

    def _update_enabled(self) -> None:
        self.video_mode.setEnabled(self.media.isChecked())
        clips = self.media.isChecked() and not self.video_mode.currentData()
        self.before.setEnabled(clips)
        self.after.setEnabled(clips)

    def options(self) -> ExportOptions:
        return ExportOptions(
            self.media.isChecked(), self.video_mode.currentData(),
            self.before.value(), self.after.value(),
            self.portraits.isChecked(), self.info.isChecked(),
        )


class AnnotationExportWorker(QThread):
    progress = Signal(str)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, database: FaceDatabase, path: Path, options: ExportOptions,
                 annotation_ids: set[int] | None = None) -> None:
        super().__init__()
        self.database, self.path, self.options = database, path, options
        self.annotation_ids = set(annotation_ids) if annotation_ids is not None else None

    def run(self) -> None:
        try:
            self.succeeded.emit(export_annotation_library(
                self.database, self.path, self.options, self.progress.emit,
                annotation_ids=self.annotation_ids,
            ))
        except Exception as exc:
            self.failed.emit(str(exc))
