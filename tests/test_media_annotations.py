import os
import base64
from types import SimpleNamespace

import numpy as np
import pytest
import cv2
from PySide6.QtCore import QPoint, QPointF, QItemSelectionModel
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QDialog, QPushButton

from face_matching.config import AppConfig
from face_matching.database import FaceDatabase, FaceSampleInput
from face_matching.engine import FaceEngine, FrameResult, TrackView
from face_matching.gui.export_dialog import AnnotationExportDialog
from face_matching.gui.main_window import MainWindow, PersonAssociationDialog, VideoWorker
from face_matching.gui.preview import FacePreview
from face_matching.gui.person_photos import PersonPhotosDialog
from face_matching.gui.annotation_compare import AnnotationCompareDialog, source_frame
from face_matching.matching import MatchResult
from face_matching.vision.detector import Detection
from face_matching.vision.io import write_jpeg
from face_matching.vision.tracker import FaceTracker
from face_matching.annotations import annotation_states


@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


def track(identifier=1, person_id="a", name="Alice", bbox=(20, 20, 80, 80)):
    return TrackView(identifier, bbox, name, person_id, "ID-a", 0.8, 0.9, 1)


def test_image_matches_multiple_faces_once_without_video_confirmation(monkeypatch):
    engine = FaceEngine.__new__(FaceEngine)
    engine.config = AppConfig()
    engine.mirror_augmentation = True
    engine.tracker = FaceTracker()
    detections = [Detection(np.array(box), 0.9, np.zeros((5, 2)))
                  for box in ((0, 0, 80, 80), (100, 0, 180, 80))]
    engine.detector = SimpleNamespace(detect=lambda image: detections)
    monkeypatch.setattr("face_matching.engine.align_face", lambda *args: np.zeros((112, 112, 3)))
    monkeypatch.setattr("face_matching.engine.assess_quality", lambda *args: SimpleNamespace(
        total=0.9, blur=0.9, pose=0.9,
    ))
    engine.embedder = SimpleNamespace(embed_many=lambda faces, **kwargs: [
        np.array([1, 0], dtype=np.float32), np.array([0, 1], dtype=np.float32)
    ])
    engine.matcher = SimpleNamespace(match_track=lambda embedding, observations: MatchResult(
        True, "a" if embedding[0] else "b", "Person", "ID", 0.9, 0.1, 0.8
    ))
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    video = engine.process_frame(frame, 1)
    assert all(view.person_id is None for view in video.tracks)
    image = engine.process_image(frame)
    assert {view.person_id for view in image.tracks} == {"a", "b"}
    assert all(view.observations == 1 for view in image.tracks)


def test_image_worker_reads_unicode_path_and_reports_real_matches(tmp_path, monkeypatch, app):
    path = tmp_path / "多人合照.JPG"
    write_jpeg(path, np.zeros((100, 200, 3), np.uint8))
    result = FrameResult((track(), track(2, None, "陌生人", (100, 20, 160, 80))), 2, 2)
    calls = []
    engine = SimpleNamespace(process_image=lambda image: calls.append(image.shape) or result)
    worker = VideoWorker(engine, str(path), 10)
    frames, recognized, errors = [], [], []
    worker.frame_ready.connect(lambda image, view, index: frames.append((image, view, index)))
    worker.recognized.connect(recognized.append)
    worker.failed.connect(errors.append)
    monkeypatch.setattr("face_matching.gui.main_window.cv2.VideoCapture", lambda *args: pytest.fail(
        "An image must not open VideoCapture"
    ))
    worker.run()

    assert calls == [(100, 200, 3)]
    assert len(recognized) == 1
    assert recognized[0]["person_id"] == "a"
    assert recognized[0]["face_jpeg"]
    assert frames[0][1] == result
    assert frames[0][2] == 1
    assert not errors


def test_invalid_image_fails_without_a_successful_preview(tmp_path, app):
    worker = VideoWorker(SimpleNamespace(), str(tmp_path / "missing.png"), 1)
    errors = []
    worker.failed.connect(errors.append)
    worker.run()
    assert errors and worker.had_error


def test_inline_image_worker_reports_pending_candidate_without_video_capture(monkeypatch, app):
    _, payload = cv2.imencode(".webp", np.zeros((100, 100, 3), np.uint8))
    source = "data:image/webp;base64," + base64.b64encode(payload).decode()
    view = TrackView(1, (20, 20, 80, 80), "待确认：Candidate", None, "", 0.413, 0.816, 1,
                     candidate_person_id="a", candidate_name="Candidate", candidate_id_card="ID-a",
                     match_reason="相似度低于确认阈值")
    engine = SimpleNamespace(process_image=lambda image: FrameResult((view,), 1, 1))
    worker = VideoWorker(engine, source, 1)
    events, errors = [], []
    worker.recognized.connect(events.append)
    worker.failed.connect(errors.append)
    monkeypatch.setattr("face_matching.gui.main_window.cv2.VideoCapture", lambda *args: pytest.fail(
        "Inline images must use process_image"
    ))
    worker.run()
    assert not errors
    assert len(events) == 1
    assert events[0]["status"] == "待确认" and not events[0]["confirmed"]
    assert events[0]["name"] == "Candidate"


def test_gui_persists_inline_source_before_starting_the_worker(tmp_path, monkeypatch, app):
    monkeypatch.setenv("FACE_MATCHING_HOME", str(tmp_path))
    window = MainWindow(AppConfig())
    window.engine = SimpleNamespace()
    _, payload = cv2.imencode(".png", np.zeros((100, 100, 3), np.uint8))
    window.source_edit.setText("data:image/png;base64," + base64.b64encode(payload).decode())

    class PreparedWorker(VideoWorker):
        def start(self):
            self.was_started = True

    monkeypatch.setattr("face_matching.gui.main_window.VideoWorker", PreparedWorker)
    window.start_video()
    assert window.worker.was_started
    assert window.active_source.endswith(".png")
    assert window.source_edit.text() == window.active_source
    assert not window.active_source.startswith("data:")
    window.close()


def test_pending_candidate_is_visible_and_rejectable_without_a_confirmed_event(tmp_path, monkeypatch, app):
    monkeypatch.setenv("FACE_MATCHING_HOME", str(tmp_path))
    window = MainWindow(AppConfig())
    person = window.database.add_person("Candidate", "ID", [
        FaceSampleInput("photo.jpg", np.array([1, 0]), "model", 0.9)
    ])
    window.session_id, window.active_source = "test", "photo.webp"
    view = TrackView(1, (20, 20, 80, 80), "待确认：Candidate", None, "", 0.413, 0.816, 1,
                     candidate_person_id=person, candidate_name="Candidate", candidate_id_card="ID",
                     match_reason="相似度 0.413 低于确认阈值 0.450")
    image = QImage(100, 100, QImage.Format_RGB32)
    image.fill(0)
    window._show_frame(image, FrameResult((view,), 1, 1), 1)
    window._on_recognized({"time": "12:00:00", "track_id": 1, "person_id": person,
                           "name": "Candidate", "id_card": "ID", "score": .413, "quality": .816,
                           "status": "待确认", "confirmed": False, "reason": view.match_reason,
                           "frame_index": 1, "bbox": view.bbox})
    assert window.events.item(0, 2).text() == "Candidate"
    assert "0.450" in window.events.item(0, 6).toolTip()
    assert len(window.video_matches) == 0
    with window.database._connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM recognition_events").fetchone()[0] == 0
    window.events.cellWidget(0, 7).findChild(QPushButton, "reject_person").click()
    assert window.video_label.result.tracks[0].annotation == "不是此人"
    assert window.database.list_annotations()[0]["action"] == "reject"
    window.close()


def test_two_review_buttons_confirm_reject_switch_and_undo_with_consistent_history(tmp_path, monkeypatch, app):
    monkeypatch.setenv("FACE_MATCHING_HOME", str(tmp_path))
    window = MainWindow(AppConfig())
    person = window.database.add_person("Candidate", "ID", [
        FaceSampleInput("photo.jpg", np.array([1, 0]), "model", 0.9)
    ])
    window.session_id, window.active_source = "test", "original.webp"
    window._on_recognized({"time": "12:00:00", "track_id": 1, "person_id": person,
        "name": "Candidate", "id_card": "ID", "score": .413, "quality": .816,
        "status": "待确认", "confirmed": False, "frame_index": 1, "bbox": (0, 0, 50, 50)})

    def buttons():
        return {button.text(): button for button in window.events.cellWidget(0, 7).findChildren(QPushButton)}

    assert set(buttons()) == {"是他", "不是他"}
    window.show()
    app.processEvents()
    assert window.events.horizontalHeader().visualIndex(2) == 0
    assert window.events.horizontalHeader().visualIndex(7) == 1
    assert window.events.columnViewportPosition(7) + window.events.columnWidth(7) <= window.events.viewport().width()
    assert not any(button.isChecked() for button in buttons().values())
    window.active_source = "another-source.mp4"
    buttons()["是他"].click()
    assert window.events.item(0, 6).text() == "人工确认"
    assert window.events.item(0, 4).text() == "0.413"
    assert buttons()["是他"].isChecked() and not buttons()["不是他"].isChecked()
    assert len(window.video_matches) == 1
    buttons()["不是他"].click()
    assert window.video_matches.association(1) is None
    assert buttons()["不是他"].isChecked() and not buttons()["是他"].isChecked()
    assert len(window.video_matches) == 0
    buttons()["是他"].click()
    assert window.video_matches.association(1).person_id == person
    assert not window.video_matches.is_rejected(1, person)
    records = window.database.list_annotations()
    states = annotation_states(records)
    assert [record["action"] for record in records if states[record["id"]]["is_current"]] == ["associate"]
    buttons()["是他"].click()
    assert window.events.item(0, 6).text() == "待确认"
    assert not any(button.isChecked() for button in buttons().values())
    buttons()["不是他"].click()
    buttons()["不是他"].click()
    records = window.database.list_annotations()
    assert [record["action"] for record in records] == [
        "associate", "remove_association", "reject", "undo_rejection", "associate",
        "remove_association", "reject", "undo_rejection",
    ]
    assert all(record["source"] == "original.webp" and record["frame_index"] == 1 for record in records)
    assert not any(state["is_current"] for state in annotation_states(records).values())
    with window.database._connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM recognition_events").fetchone()[0] == 0
    window.close()


@pytest.mark.parametrize("choice", ["confirm_person", "reject_person"])
def test_failed_review_restores_both_button_states(tmp_path, monkeypatch, app, choice):
    monkeypatch.setenv("FACE_MATCHING_HOME", str(tmp_path))
    window = MainWindow(AppConfig())
    person = window.database.add_person("Alice", "ID", [
        FaceSampleInput("photo.jpg", np.array([1, 0]), "model", 0.9)
    ])
    window._on_recognized({"time": "12:00:00", "track_id": 1, "person_id": person,
        "name": "Alice", "id_card": "ID", "score": .8, "quality": .9})
    errors = []
    monkeypatch.setattr("face_matching.gui.main_window.QMessageBox.critical", lambda *args: errors.append(args))
    monkeypatch.setattr(window.database, "log_annotation", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")))
    window.events.cellWidget(0, 7).findChild(QPushButton, choice).click()
    assert errors
    assert not any(button.isChecked() for button in window.events.cellWidget(0, 7).findChildren(QPushButton))
    assert window.events.item(0, 6).text() == "自动匹配"
    assert window.video_matches.association(1) is None
    assert not window.video_matches.is_rejected(1, person)
    window.close()


def test_person_photo_viewer_shows_all_photos_and_handles_missing_files(tmp_path, monkeypatch, app):
    monkeypatch.setenv("FACE_MATCHING_HOME", str(tmp_path))
    database = FaceDatabase(tmp_path / "photos.db")
    photo = tmp_path / "人员照片.jpg"
    write_jpeg(photo, np.zeros((80, 120, 3), np.uint8))
    person_id = database.add_person("Person", "ID", [
        FaceSampleInput(str(tmp_path / "missing.jpg"), np.array([1, 0]), "model", 0.9),
        FaceSampleInput(str(photo), np.array([1, 0]), "model", 0.9),
    ])
    parent = FacePreview()
    dialog = PersonPhotosDialog(database.get_person(person_id), database.list_stored_samples(), parent)
    assert dialog.photos.count() == 2
    assert "无法读取" in dialog.preview.text()
    assert not dialog.previous.isEnabled() and dialog.next.isEnabled()
    dialog.next.click()
    assert not dialog.image.isNull() and dialog.image.size().width() == 120
    assert "2 / 2" in dialog.caption.text()
    assert dialog.previous.isEnabled() and not dialog.next.isEnabled()
    dialog.close()


def test_video_worker_keeps_frame_sampling_and_final_preview(tmp_path, app):
    path = tmp_path / "video.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (100, 100))
    assert writer.isOpened()
    for _ in range(9):
        writer.write(np.zeros((100, 100, 3), np.uint8))
    writer.release()
    result = FrameResult((track(),), 1, 1)
    calls, resets = [], []
    engine = SimpleNamespace(
        reset_video=lambda: resets.append(True),
        process_frame=lambda frame, index: calls.append(index) or result,
    )
    worker = VideoWorker(engine, str(path), 4)
    frames, events = [], []
    worker.frame_ready.connect(lambda image, result, index: frames.append(index))
    worker.recognized.connect(events.append)
    worker.run()
    assert resets == [True]
    assert calls == [1, 4, 8]
    assert frames[-1] == 9
    assert len(events) == 1
    assert events[0]["frame_index"] == 1


def test_preview_hit_testing_handles_letterboxing_resize_and_empty_space(app):
    preview = FacePreview()
    preview.resize(800, 600)
    image = QImage(400, 200, QImage.Format_RGB32)
    image.fill(0)
    preview.set_frame(image, FrameResult((track(bbox=(100, 50, 200, 150)),), 1, 1))
    assert preview.track_at(QPointF(300, 300)).id == 1
    assert preview.track_at(QPointF(300, 50)) is None
    assert preview.track_at(QPointF(700, 300)) is None
    preview.resize(1000, 1000)
    rect = preview.image_rect()
    point = QPointF(rect.x() + 150 / 400 * rect.width(), rect.y() + 100 / 200 * rect.height())
    assert preview.track_at(point).id == 1
    preview.close()


def test_gui_review_and_right_click_association_persist_and_refresh(tmp_path, monkeypatch, app):
    monkeypatch.setenv("FACE_MATCHING_HOME", str(tmp_path))
    window = MainWindow(AppConfig())
    sample = FaceSampleInput("photo.jpg", np.array([1, 0]), "test", 0.9)
    alice = window.database.add_person("Alice", "ID-a", [sample])
    bob = window.database.add_person("Bob", "ID-b", [sample])
    window.session_id = "test-session"
    window.active_source = "original-video.mp4"
    window.source_edit.setText("edited-source.mp4")
    current = track(person_id=alice)
    image = QImage(200, 100, QImage.Format_RGB32)
    image.fill(0)
    window._show_frame(image, FrameResult((current,), 1, 1), 5)
    window._on_recognized({
        "time": "12:00:00", "track_id": 1, "person_id": alice,
        "name": "Alice", "id_card": "ID-a", "score": 0.8, "quality": 0.9,
        "frame_index": 5, "bbox": current.bbox,
    })
    window.events.cellWidget(0, 7).findChild(QPushButton, "reject_person").click()
    assert window.events.item(0, 6).text() == "不是此人"
    assert window.video_label.result.tracks[0].person_id is None
    assert len(window.video_matches) == 0
    window.events.cellWidget(0, 7).findChild(QPushButton, "reject_person").click()
    assert window.video_label.result.tracks[0].person_id == alice

    class ChosenMenu:
        def __init__(self, parent):
            self.action = object()

        def addAction(self, label):
            return self.action

        def exec(self, position):
            return self.action

    monkeypatch.setattr("face_matching.gui.main_window.QMenu", ChosenMenu)

    def select_bob(dialog):
        dialog.search.setText("ID-b")
        dialog.table.selectRow(0)
        return QDialog.Accepted

    monkeypatch.setattr(PersonAssociationDialog, "exec", select_bob)
    window._track_menu(current, QPoint(0, 0))
    assert window.events.item(0, 2).text() == "Bob"
    assert window.events.item(0, 4).text() == "—"
    assert window.events.item(0, 6).text() == "人工关联"
    assert window.video_label.result.tracks[0].person_id == bob
    window.events.cellWidget(0, 7).findChild(QPushButton, "confirm_person").click()
    assert window.video_label.result.tracks[0].person_id == alice
    annotations = FaceDatabase(window.database.path).list_annotations()
    assert [row["action"] for row in annotations] == [
        "reject", "undo_rejection", "associate", "remove_association"
    ]
    assert all(row["source"] == "original-video.mp4" for row in annotations)
    assert all(row["frame_index"] == 5 for row in annotations)
    assert all(row["snapshot_path"] for row in annotations)
    window.close()


def test_export_dialog_defaults_to_clips_with_independent_people_choices(app):
    parent = FacePreview()
    dialog = AnnotationExportDialog(3, parent)
    assert not dialog.options().full_video
    assert dialog.options().seconds_before == dialog.options().seconds_after == 5
    dialog.portraits.setChecked(False)
    assert not dialog.options().include_portraits
    assert dialog.options().include_person_info
    dialog.video_mode.setCurrentIndex(1)
    assert not dialog.before.isEnabled()
    dialog.close()


def test_annotation_selection_survives_refresh_by_id_and_opens_comparison(tmp_path, monkeypatch, app):
    monkeypatch.setenv("FACE_MATCHING_HOME", str(tmp_path))
    window = MainWindow(AppConfig())
    person = window.database.add_person("Alice", "ID-a", [
        FaceSampleInput("photo.jpg", np.array([1, 0]), "test", 0.9)
    ])
    window.database.log_annotation("s", "missing.jpg", 1, "reject", person, 1)
    window.database.log_annotation("s", "missing.jpg", 1, "undo_rejection", person, 1)
    window.refresh_annotations()
    window.annotation_table.selectRow(1)
    assert window._selected_annotation_ids() == {1}
    assert window.library_export_button.text() == "导出选中 1 条…"
    assert window.annotation_table.item(1, 7).text() == "历史记录"
    seen = []
    monkeypatch.setattr(AnnotationCompareDialog, "exec", lambda dialog: seen.append(dialog))
    window.compare_annotation()
    assert len(seen) == 1
    assert seen[0].sample_preview.image.isNull()
    window.database.log_annotation("s", "missing.jpg", 1, "reject", person, 1)
    window.refresh_annotations()
    assert window._selected_annotation_ids() == {1}
    window.compare_annotation()
    assert seen[-1].windowTitle() == "标注样本对比 · #1"
    window.annotation_table.selectionModel().select(window.annotation_table.model().index(0, 0),
        QItemSelectionModel.Select | QItemSelectionModel.Rows)
    assert window._selected_annotation_ids() == {1, 3}
    window.annotation_table.clearSelection()
    assert not window._selected_annotation_ids()
    assert window.library_export_button.text() == "导出全部标注…"
    window.close()


def test_annotation_comparison_saved_frame_crop_and_multiple_portraits(tmp_path, monkeypatch, app):
    monkeypatch.setenv("FACE_MATCHING_HOME", str(tmp_path))
    database = FaceDatabase(tmp_path / "faces.db")
    photos = []
    for index in range(2):
        path = tmp_path / f"人像{index}.jpg"
        write_jpeg(path, np.full((100, 80, 3), 40 + index * 100, np.uint8))
        photos.append(FaceSampleInput(str(path), np.array([1, 0]), "test", 0.9))
    person = database.add_person("Alice", "ID-a", photos)
    database.log_annotation("s", "missing-video.mp4", 1, "reject", person, 5,
        (10, 10, 50, 70), snapshot_path=photos[0].image_path, face_path=photos[1].image_path)
    parent = FacePreview()
    dialog = AnnotationCompareDialog(database.list_annotations()[0], database.get_person(person),
                                      database.list_stored_samples(), parent)
    dialog.show()
    app.processEvents()
    assert dialog.sample_choices.count() == 2
    assert dialog.sample_preview.image.pixelColor(10, 10).red() > 200
    dialog.sample_choices.setCurrentIndex(1)
    assert dialog.sample_preview.image.pixelColor(30, 30).red() > 100
    dialog.portrait_choices.setCurrentIndex(1)
    assert dialog.portrait_preview.image.pixelColor(30, 30).red() > 100
    dialog.resize(1200, 750)
    app.processEvents()
    assert not dialog.portrait_preview.pixmap().isNull()
    dialog.close()
    parent.close()


def test_comparison_falls_back_to_original_image_or_video_frame(tmp_path, app):
    image_path = tmp_path / "来源图片.webp"
    write_jpeg(image_path, np.full((48, 64, 3), 70, np.uint8))
    record = {"source": str(image_path), "frame_index": 1}
    assert source_frame(record).pixelColor(20, 20).red() == 70
    video = tmp_path / "来源视频.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for value in (30, 80, 150):
        writer.write(np.full((48, 64, 3), value, np.uint8))
    writer.release()
    record.update(source=str(video), frame_index=3)
    assert abs(source_frame(record).pixelColor(20, 20).red() - 150) < 5
    record["frame_index"] = 999
    assert source_frame(record).isNull()
    record["source"] = "rtsp://camera/live"
    assert source_frame(record).isNull()


def test_export_selection_is_passed_to_worker(tmp_path, monkeypatch, app):
    monkeypatch.setenv("FACE_MATCHING_HOME", str(tmp_path))
    window = MainWindow(AppConfig())
    person = window.database.add_person("Alice", "ID-a", [
        FaceSampleInput("photo.jpg", np.array([1, 0]), "test", 0.9)
    ])
    for track_id in (1, 2):
        window.database.log_annotation("s", "missing.jpg", track_id, "reject", person, 1)
    window.refresh_annotations()
    window.annotation_table.selectRow(1)
    monkeypatch.setattr(AnnotationExportDialog, "exec", lambda dialog: QDialog.Accepted)
    monkeypatch.setattr("face_matching.gui.main_window.QFileDialog.getSaveFileName",
                        lambda *args: (str(tmp_path / "selected.zip"), "ZIP"))
    monkeypatch.setattr("face_matching.gui.main_window.AnnotationExportWorker.start", lambda worker: None)
    window.export_annotations()
    assert window.export_worker.annotation_ids == {1}
    window._close_export_progress()
    window.close()
