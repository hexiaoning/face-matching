import os
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from face_matching.config import AppConfig
from face_matching.database import FaceSampleInput
from face_matching.engine import FrameResult, TrackView
from face_matching.gui.main_window import MainWindow
from face_matching.vision.io import read_image, write_jpeg


@pytest.fixture
def batch(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    monkeypatch.setenv("FACE_MATCHING_HOME", str(tmp_path))
    window = MainWindow(AppConfig())
    person = window.database.add_person("Alice", "ID-a", [
        FaceSampleInput("photo.jpg", np.array([1, 0]), "test", 0.9)
    ])
    result = FrameResult((TrackView(1, (20, 20, 80, 80), "Alice", person, "ID-a", 0.8, 0.9, 1),), 1, 1)
    window.engine = SimpleNamespace(process_image=lambda frame: result,
        process_frame=lambda frame, index: result, reset_video=lambda: None, model_id="test")
    paths = []
    for index, value in enumerate((40, 180)):
        path = tmp_path / f"图片 {index}.jpg"
        write_jpeg(path, np.full((100, 100, 3), value, np.uint8))
        paths.append(str(path.resolve()))
    yield app, window, paths, result
    window.close()
    app.processEvents()


def wait_until(app, predicate, seconds=5):
    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        QTest.qWait(5)
    assert predicate()


def test_multi_select_mixed_media_starts_and_accumulates_correct_source_annotations(batch, tmp_path, monkeypatch):
    app, window, paths, _ = batch
    video = tmp_path / "来源视频.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (100, 100))
    for _ in range(4):
        writer.write(np.full((100, 100, 3), 100, np.uint8))
    writer.release()
    sources = [paths[0], str(video.resolve()), paths[1]]
    monkeypatch.setattr("face_matching.gui.main_window.QFileDialog.getOpenFileNames",
                        lambda *args: (sources, ""))
    window._browse_video()
    assert window.worker is not None
    wait_until(app, lambda: window.batch_completed == 3 and window.worker is None)
    assert window.batch_total == 3
    assert window.events.rowCount() == 3
    assert len(window.video_matches) == 1
    contexts = window.match_context.values()
    assert {item["track_id"] for item in contexts} == {1, 2, 3}
    assert {item["source"] for item in contexts} == set(sources)
    first = next(match for match in window.video_matches.displayed() if match.source == paths[0])
    window._review_match(first)
    annotation = window.database.list_annotations()[0]
    assert annotation["source"] == paths[0]
    assert annotation["track_id"] == 1
    assert annotation["snapshot_path"] is None
    assert abs(int(read_image(annotation["face_path"])[10, 10, 0]) - 40) < 3


@pytest.mark.parametrize("trigger", ["timer", "editingFinished"])
def test_entering_source_starts_automatically_once(batch, trigger):
    app, window, paths, _ = batch
    window.source_edit.setText(paths[0])
    if trigger == "timer":
        window.source_timer.setInterval(10)
        window.source_edit.textEdited.emit(paths[0])
    else:
        window.source_edit.editingFinished.emit()
    wait_until(app, lambda: window.batch_completed == 1 and window.worker is None)
    session = window.session_id
    window.source_edit.editingFinished.emit()
    app.processEvents()
    assert window.worker is None
    assert window.session_id == session


def test_adding_sources_during_processing_preserves_batch_and_deduplicates_queue(batch):
    app, window, paths, result = batch
    entered, release = threading.Event(), threading.Event()

    def process(frame):
        entered.set()
        assert release.wait(3)
        return result

    window.engine.process_image = process
    window._submit_sources([paths[0]])
    try:
        wait_until(app, entered.is_set)
        session = window.session_id
        window._submit_sources([paths[0], paths[1], paths[1]])
        assert window.pending_sources == [paths[1]]
        assert window.session_id == session
        assert window.batch_total == 2
    finally:
        release.set()
    wait_until(app, lambda: window.batch_completed == 2 and window.worker is None)
    assert window.events.rowCount() == 2


def test_stop_cancels_entire_pending_batch(batch):
    app, window, paths, result = batch
    entered, release = threading.Event(), threading.Event()

    def process(frame):
        entered.set()
        assert release.wait(3)
        return result

    window.engine.process_image = process
    window._submit_sources(paths)
    try:
        wait_until(app, entered.is_set)
        threading.Timer(0.05, release.set).start()
        assert window.stop_video()
    finally:
        release.set()
    app.processEvents()
    assert window.worker is None
    assert not window.pending_sources
    assert window.batch_completed == 0
    assert window.events.rowCount() == 0


def test_batch_continues_after_bad_source_and_reports_failures(batch, tmp_path, monkeypatch):
    app, window, paths, _ = batch
    warnings = []
    monkeypatch.setattr("face_matching.gui.main_window.QMessageBox.warning",
                        lambda *args: warnings.append(args))
    window._submit_sources([str(tmp_path / "不存在.jpg"), paths[0]])
    wait_until(app, lambda: window.batch_completed == 2 and window.worker is None)
    assert len(window.batch_errors) == 1
    assert len(warnings) == 1
    assert window.events.rowCount() == 1


def test_queue_waits_for_open_annotation_menu_before_switching_source(batch):
    app, window, paths, _ = batch
    window._reviewing = True
    window._submit_sources(paths)
    wait_until(app, lambda: window.worker is not None and not window.worker.isRunning())
    app.processEvents()
    assert window.active_source == paths[0]
    assert window.pending_sources == [paths[1]]
    window._reviewing = False
    wait_until(app, lambda: window.batch_completed == 2 and window.worker is None)


def test_multiple_typed_paths_are_parsed_but_inline_images_are_not_split(batch):
    _, window, paths, _ = batch
    assert window._parse_sources('; '.join(paths)) == paths
    assert window._parse_sources('data:image/png;base64,payload') == ['data:image/png;base64,payload']
    window.source_edit.setText('C:/still typing')
    window._auto_start_source()
    assert window.worker is None
