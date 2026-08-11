from __future__ import annotations

import os
import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from face_match.config import AppPaths, AppSettings
from face_match.gpu import prepare_cuda_runtime
from face_match.model_manager import MODEL_LICENSE_NOTICE, ModelManager
from face_match.ui.main_window import MainWindow
from face_match.ui.startup import ModelDownloadDialog, ServiceLoadingDialog
from face_match.ui.styles import APP_STYLE


def _critical(title: str, message: str) -> None:
    box = QMessageBox(QMessageBox.Icon.Critical, title, message)
    box.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    box.exec()


def main() -> int:
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
    application = QApplication(sys.argv)
    application.setApplicationName("监控视频人脸比对")
    application.setOrganizationName("FaceMatch")
    application.setStyle("Fusion")
    application.setStyleSheet(APP_STYLE)

    paths = AppPaths.create()
    settings = AppSettings.load(paths.settings)
    try:
        prepare_cuda_runtime()
    except Exception as exc:
        _critical("GPU 不可用，程序无法启动", str(exc))
        return 2

    models = ModelManager(paths.models)
    if models.missing_models() and not settings.model_license_accepted:
        answer = QMessageBox.question(
            None,
            "模型许可确认",
            MODEL_LICENSE_NOTICE + "\n\n首次运行约需下载 710 MiB。是否同意上述用途限制并继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return 1
        settings.model_license_accepted = True
        settings.save(paths.settings)

    if models.missing_models():
        download = ModelDownloadDialog(models)
        if download.start() != QDialog.DialogCode.Accepted:
            if download.error_message:
                _critical("模型下载失败", download.error_message)
            return 3

    loading = ServiceLoadingDialog(paths, settings)
    if loading.start() != QDialog.DialogCode.Accepted or loading.services is None:
        _critical("GPU 引擎启动失败", loading.error_message or "未知错误")
        return 4

    window = MainWindow(loading.services)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
