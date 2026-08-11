"""程序入口：GPU 检查 -> 加载模型 -> 启动图形界面。"""
from __future__ import annotations

import argparse
import sys

from PyQt6.QtWidgets import QApplication, QMessageBox


def main() -> int:
    parser = argparse.ArgumentParser(description="视频人脸比对系统（GPU 版）")
    parser.add_argument(
        "--backend",
        default=None,
        choices=["auto", "cuda", "directml"],
        help="GPU 后端，默认 auto（优先 CUDA，备选 DirectML；无 GPU 直接报错）",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="insightface 模型包名，默认 antelopev2（可选 buffalo_l 等）",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)

    # GPU 强制检查 + 模型加载，失败则弹窗报错退出，绝不回退 CPU
    from . import config
    from .engine import FaceEngine

    backend = args.backend or config.GPU_BACKEND
    try:
        engine = FaceEngine(backend=backend, model_name=args.model)
    except Exception as e:  # noqa: BLE001
        QMessageBox.critical(
            None,
            "GPU 初始化失败",
            f"{e}\n\n本项目必须使用 GPU 运行，程序将退出。",
        )
        return 2

    from .db import Database
    from .gallery import Gallery
    from .gui.main_window import MainWindow

    db = Database()
    gallery = Gallery(db)
    window = MainWindow(engine, db, gallery)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
