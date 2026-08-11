from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
    parser.add_argument(
        "--diagnose-file",
        type=Path,
        help="run the bundled GPU/model self-test, write JSON, and exit",
    )
    args = parser.parse_args(argv)
    if args.diagnose_file:
        from .diagnostics import collect, exit_code

        try:
            result = collect()
        except Exception as exc:
            result = {"gpu_ready": False, "inference_ready": False, "fatal_error": str(exc)}
            code = 4
        else:
            code = exit_code(result)
        args.diagnose_file.parent.mkdir(parents=True, exist_ok=True)
        args.diagnose_file.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return code
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
