from __future__ import annotations

import argparse
import sys

from .config import AppConfig
from .gpu import assert_cuda_available
from .models import PROFILES, download_profile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GPU-only 视频人脸比对")
    parser.add_argument("--check-gpu", action="store_true", help="检查 CUDA 并退出")
    parser.add_argument("--download-models", action="store_true", help="下载模型并退出")
    parser.add_argument("--profile", choices=sorted(PROFILES), help="覆盖配置中的模型档")
    args = parser.parse_args(argv)
    config = AppConfig.load()
    if args.profile:
        config.model_profile = args.profile
    if args.download_models:
        download_profile(config.model_profile)
        print(f"模型 {config.model_profile} 已就绪")
        return 0
    if args.check_gpu:
        try:
            print(assert_cuda_available())
            return 0
        except Exception as exc:
            print(f"GPU 不可用: {exc}", file=sys.stderr)
            return 2

    from PySide6.QtWidgets import QApplication
    from .gui.main_window import MainWindow

    application = QApplication(sys.argv[:1])
    application.setApplicationName("Face Matching")
    application.setStyle("Fusion")
    window = MainWindow(config)
    if not window.initialize():
        return 2
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
