from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import AppPaths, RecognitionSettings
from .gpu import GPUUnavailableError, require_cuda_runtime


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU-only surveillance face matching")
    parser.add_argument("--data-dir", type=Path, help="数据库、照片与模型存储目录")
    parser.add_argument(
        "--check-gpu", action="store_true", help="只检查 NVIDIA/CUDA 推理环境后退出"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    if args.check_gpu:
        try:
            ort, info = require_cuda_runtime()
        except GPUUnavailableError as exc:
            print(f"GPU 检查失败：{exc}", file=sys.stderr)
            return 3
        print(
            f"GPU 可用：{info.name}，驱动 {info.driver_version}，"
            f"ONNX Runtime {ort.__version__}，CUDAExecutionProvider"
        )
        return 0

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QMessageBox, QProgressDialog

    from .database import Database
    from .face_engine import FaceEngine
    from .model_manager import DownloadCancelled, ModelDownloadError, ModelManager
    from .ui.main_window import MainWindow
    from .ui.theme import APP_STYLE

    app = QApplication(sys.argv[:1])
    app.setApplicationName("监控视频人脸比对")
    app.setOrganizationName("FaceMatching")
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLE)
    try:
        ort, gpu_info = require_cuda_runtime()
    except GPUUnavailableError as exc:
        QMessageBox.critical(None, "GPU 不可用", str(exc))
        return 3

    paths = AppPaths.create(args.data_dir)
    paths.ensure()
    manager = ModelManager(paths.models)
    model_paths = manager.locate()
    if model_paths is None:
        answer = QMessageBox.question(
            None,
            "下载识别模型",
            "首次运行需要从 InsightFace GitHub 下载 AntelopeV2 模型（约 344 MiB）。\n\n"
            "重要：该预训练权重仅限非商业研究；商业使用需另行取得 InsightFace 授权。\n"
            "是否继续下载？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return 2
        progress = QProgressDialog("正在下载并校验模型…", "取消", 0, 100)
        progress.setWindowTitle("首次运行")
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.show()

        def update_progress(downloaded: int, total: int) -> None:
            if progress.wasCanceled():
                raise DownloadCancelled("模型下载已取消")
            if total > 0:
                progress.setMaximum(100)
                progress.setValue(min(99, int(downloaded * 100 / total)))
                progress.setLabelText(
                    f"正在下载模型… {downloaded / 1024 / 1024:.1f} / {total / 1024 / 1024:.1f} MiB"
                )
            else:
                progress.setMaximum(0)
            QApplication.processEvents()

        try:
            model_paths = manager.ensure_models(update_progress)
            progress.setMaximum(100)
            progress.setValue(100)
        except DownloadCancelled:
            progress.close()
            return 2
        except ModelDownloadError as exc:
            progress.close()
            QMessageBox.critical(None, "模型下载失败", str(exc))
            return 2
        finally:
            progress.close()

    settings = RecognitionSettings.load(paths.settings)
    try:
        engine = FaceEngine(ort, model_paths, detector_size=settings.detector_size)
    except GPUUnavailableError as exc:
        QMessageBox.critical(None, "CUDA 模型加载失败", str(exc))
        return 3
    except Exception as exc:
        QMessageBox.critical(None, "模型加载失败", f"无法初始化人脸模型：{exc}")
        return 2
    try:
        database = Database(paths.database)
    except Exception as exc:
        QMessageBox.critical(None, "数据库错误", f"无法打开人员库：{exc}")
        return 2
    window = MainWindow(engine, database, paths, gpu_info, settings)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

