import os
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from face_matching.gui.video_detection import VideoDetectionDialog, VideoDetectionWidget
from face_matching.person_vehicle import ObjectDetection


_application = None


@pytest.fixture(scope="module")
def app():
    global _application
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    _application = QApplication.instance() or QApplication([])
    return _application


@pytest.fixture(autouse=True)
def cleanup_widgets(app):
    existing = list(app.topLevelWidgets())
    yield
    # These tests drive processEvents instead of app.exec; explicitly flush
    # DeferredDelete before Python tears down the offscreen Windows Qt runtime.
    for widget in app.topLevelWidgets():
        if widget not in existing:
            widget.close()
            widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 12, (96, 64))
    assert writer.isOpened()
    for _ in range(6):
        writer.write(np.full((64, 96, 3), 60, np.uint8))
    writer.release()
    return path


def wait_for_worker(app, tools):
    deadline = time.monotonic() + 5
    while tools.worker is not None and time.monotonic() < deadline:
        app.processEvents()
        QTest.qWait(5)
    assert tools.worker is None


def test_background_export_updates_frame_counts_scores_and_restores_controls(app, video, tmp_path, monkeypatch):
    detector = SimpleNamespace(
        session=SimpleNamespace(get_providers=lambda: ["test-GPU"]),
        detect=lambda frame: [ObjectDetection(0, 0.91, (10, 20, 30, 50)),
                              ObjectDetection(2, 0.82, (40, 20, 80, 50))],
    )
    monkeypatch.setattr("face_matching.gui.video_detection.load_person_vehicle_detector", lambda *args: detector)
    tools = VideoDetectionWidget()
    assert tools.load_video(video)
    tools.start(tmp_path / "output.mp4")
    assert not tools.open_button.isEnabled()
    wait_for_worker(app, tools)
    assert tools.open_button.isEnabled()
    assert tools.export_button.isEnabled()
    assert tools.table.item(0, 1).text() == "0.91"
    assert tools.table.item(1, 1).text() == "0.82"
    assert "人 1 · 车 1" in tools.counts.text()
    assert "第 6 帧" in tools.counts.text()
    assert "已导出完整" in tools.status.text()
    tools.close()


def test_model_failure_restores_controls_without_starting_download(app, video, tmp_path, monkeypatch):
    def fail(*args):
        raise ValueError("检测模型缺失")

    monkeypatch.setattr("face_matching.gui.video_detection.load_person_vehicle_detector", fail)
    tools = VideoDetectionWidget()
    tools.load_video(video)
    tools.start(tmp_path / "failed.mp4")
    wait_for_worker(app, tools)
    assert tools.start_button.isEnabled()
    assert "检测模型缺失" in tools.status.text()
    assert not (tmp_path / "failed.mp4").exists()
    tools.close()


def test_escape_requests_stop_and_cannot_destroy_active_worker(app, monkeypatch):
    dialog = VideoDetectionDialog()
    dialog.show()
    stopped = []
    monkeypatch.setattr(dialog.tools, "is_busy", lambda: True)
    monkeypatch.setattr(dialog.tools, "stop", lambda: stopped.append(True))
    QTest.keyClick(dialog, Qt.Key_Escape)
    assert dialog.isVisible() and stopped
    assert not dialog.close()
    monkeypatch.setattr(dialog.tools, "is_busy", lambda: False)
    assert dialog.close()


def test_standalone_entry_does_not_initialize_face_models_or_identity_engine(app, monkeypatch):
    from face_matching import __main__

    def unexpected(*args, **kwargs):
        raise AssertionError("不应初始化人脸推理")

    monkeypatch.setattr(__main__.AppConfig, "load", lambda: SimpleNamespace(gpu_backend="auto", gpu_device_id=0))
    monkeypatch.setattr(__main__, "required_paths", unexpected)
    monkeypatch.setattr(__main__, "assert_gpu_available", unexpected)
    monkeypatch.setattr(QApplication, "exec", lambda self: 0)
    assert __main__.main(["--video-detection"]) == 0
