from __future__ import annotations

import argparse
import sys

from PySide6.QtWidgets import QApplication, QMessageBox

from .config import EngineConfig
from .database import FaceDatabase
from .engine import FaceEngine
from .errors import FaceMatchingError
from .matcher import GalleryMatcher
from .theme import STYLESHEET


def _source(value: str | None) -> str | int | None:
    if value is None:
        return None
    if value.isdigit() and len(value) <= 2:
        return int(value)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CUDA surveillance face matching desktop app")
    parser.add_argument("--source", help="video path, camera index, or RTSP URL")
    args = parser.parse_args(argv)
    app = QApplication(sys.argv[:1])
    app.setApplicationName("监控视频人脸检索")
    app.setOrganizationName("FaceMatching")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    try:
        config = EngineConfig()
        engine = FaceEngine(config)
        database = FaceDatabase()
        matcher = GalleryMatcher(database, config.model_id)
    except FaceMatchingError as exc:
        QMessageBox.critical(None, "无法启动", str(exc))
        return 2
    except Exception as exc:
        QMessageBox.critical(None, "初始化失败", f"程序初始化失败：\n{exc}")
        return 3
    from .gui.main_window import MainWindow

    window = MainWindow(database, engine, matcher, _source(args.source))
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
