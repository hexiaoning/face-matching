from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QTabWidget

from face_matching.config import EngineConfig
from face_matching.database import FaceDatabase
from face_matching.gui.main_window import MainWindow
from face_matching.matcher import GalleryMatcher


class FakeEngine:
    def __init__(self) -> None:
        self.config = EngineConfig(
            detector_model=Path("detector.onnx"),
            recognizer_model=Path("recognizer.onnx"),
        )
        self.provider = "CUDAExecutionProvider"
        self.detector = SimpleNamespace(input_size=(960, 960))

    def detect(self, frame: np.ndarray) -> list:
        return []

    def embed(self, faces: list[np.ndarray]) -> np.ndarray:
        return np.empty((0, 512), dtype=np.float32)


def test_main_window_can_be_constructed_with_mouse_controls(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    database = FaceDatabase(tmp_path / "faces.sqlite3", tmp_path / "photos")
    engine = FakeEngine()
    matcher = GalleryMatcher(database, engine.config.model_id)

    window = MainWindow(database, engine, matcher)
    assert isinstance(window.centralWidget(), QTabWidget)
    assert window.centralWidget().count() == 2
    assert window.start_button.text() == "开始识别"
    assert window.person_page.table.columnCount() == 4

    window.close()
    app.processEvents()
