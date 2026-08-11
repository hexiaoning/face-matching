from __future__ import annotations

import threading
from typing import Any

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QDialog, QLabel, QProgressBar, QPushButton, QVBoxLayout

from face_match.config import AppPaths, AppSettings
from face_match.model_manager import ModelManager
from face_match.services import ApplicationServices, build_services


class _DownloadThread(QThread):
    progress = Signal(str, int, int)
    failed = Signal(str)
    succeeded = Signal()

    def __init__(self, manager: ModelManager, parent: Any | None = None) -> None:
        super().__init__(parent)
        self.manager = manager
        self.cancelled = threading.Event()

    def run(self) -> None:
        try:
            self.manager.ensure_models(
                lambda name, current, total: self.progress.emit(name, current, total),
                self.cancelled.is_set,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.succeeded.emit()


class ModelDownloadDialog(QDialog):
    def __init__(self, manager: ModelManager, parent: Any | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("下载人脸模型")
        self.setModal(True)
        self.setMinimumWidth(520)
        self.label = QLabel("准备下载模型……")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.cancel_button = QPushButton("取消")
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addWidget(self.progress)
        layout.addWidget(self.cancel_button)
        self.thread = _DownloadThread(manager, self)
        self.thread.progress.connect(self._on_progress)
        self.thread.failed.connect(self._on_failed)
        self.thread.succeeded.connect(self.accept)
        self.cancel_button.clicked.connect(self._cancel)
        self.error_message = ""

    def start(self) -> int:
        self.thread.start()
        result = self.exec()
        self.thread.wait(5_000)
        return result

    def _on_progress(self, name: str, current: int, total: int) -> None:
        megabytes = current / (1024 * 1024)
        total_megabytes = total / (1024 * 1024) if total else 0
        self.label.setText(f"正在下载 {name}：{megabytes:.1f} / {total_megabytes:.1f} MiB")
        self.progress.setValue(int(1000 * current / total) if total else 0)

    def _on_failed(self, message: str) -> None:
        self.error_message = message
        QDialog.reject(self)

    def _cancel(self) -> None:
        self.cancel_button.setEnabled(False)
        self.label.setText("正在取消……")
        self.thread.cancelled.set()

    def reject(self) -> None:
        if self.thread.isRunning():
            self.thread.cancelled.set()
            self.cancel_button.setEnabled(False)
            self.label.setText("正在取消……")
            return
        super().reject()


class _ServiceThread(QThread):
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, paths: AppPaths, settings: AppSettings, parent: Any | None = None) -> None:
        super().__init__(parent)
        self.paths = paths
        self.settings = settings

    def run(self) -> None:
        try:
            services = build_services(self.paths, self.settings)
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.succeeded.emit(services)


class ServiceLoadingDialog(QDialog):
    def __init__(self, paths: AppPaths, settings: AppSettings, parent: Any | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("启动 GPU 引擎")
        self.setModal(True)
        self.setMinimumWidth(460)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("正在把检测与识别模型加载到 CUDA GPU，首次启动可能需要几十秒……"))
        progress = QProgressBar()
        progress.setRange(0, 0)
        layout.addWidget(progress)
        self.services: ApplicationServices | None = None
        self.error_message = ""
        self.thread = _ServiceThread(paths, settings, self)
        self.thread.succeeded.connect(self._succeeded)
        self.thread.failed.connect(self._failed)

    def start(self) -> int:
        self.thread.start()
        result = self.exec()
        self.thread.wait(5_000)
        return result

    def _succeeded(self, services: ApplicationServices) -> None:
        self.services = services
        QDialog.accept(self)

    def _failed(self, message: str) -> None:
        self.error_message = message
        QDialog.reject(self)

    def reject(self) -> None:
        if self.thread.isRunning():
            return
        super().reject()
