import json
import sqlite3
import zipfile

import cv2
import numpy as np
import pytest

from face_matching.annotation_export import ExportOptions, export_annotation_library, write_video_clip
from face_matching.database import FaceDatabase, FaceSampleInput
from face_matching.vision.io import write_jpeg
from face_matching.annotations import annotation_states


@pytest.fixture
def library(tmp_path):
    database = FaceDatabase(tmp_path / "faces.db")
    portrait = tmp_path / "人员照.jpg"
    write_jpeg(portrait, np.zeros((48, 64, 3), np.uint8))
    person = database.add_person("Alice", "ID-001", [
        FaceSampleInput(str(portrait), np.array([1, 0]), "model", 0.9)
    ])
    video = tmp_path / "来源视频.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    assert writer.isOpened()
    for index in range(30):
        writer.write(np.full((48, 64, 3), index * 5, np.uint8))
    writer.release()
    database.log_annotation("s1", str(video), 1, "associate", person, 15, (0, 0, 32, 32))
    return database, person, portrait, video


def read_export(path):
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def test_export_defaults_to_clips_and_includes_linked_people_and_annotations(library, tmp_path):
    database, person, _, _ = library
    target = tmp_path / "标注库.zip"
    report = export_annotation_library(database, target, ExportOptions(seconds_before=1, seconds_after=1))
    files = read_export(target)
    annotations = json.loads(files["annotations.json"])
    people = json.loads(files["people.json"])
    media = annotations[0]["media"]
    assert not report.issues
    assert report.annotation_count == 1
    assert annotations[0]["bbox"] == [0, 0, 32, 32]
    assert media["mode"] == "clip" and media["audio_included"] is False
    assert media["start_frame"] == 5
    assert media["frame_count"] == 21
    assert media["annotation_frame"] == 11
    assert people[0]["person_id"] == person
    assert people[0]["name"] == "Alice"
    assert people[0]["portraits"][0] in files
    clip = tmp_path / "check.mp4"
    clip.write_bytes(files[media["path"]])
    capture = cv2.VideoCapture(str(clip))
    assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 21
    capture.release()


@pytest.mark.parametrize("portraits,info", [(False, False), (True, False), (False, True)])
def test_export_person_options_are_independent(library, tmp_path, portraits, info):
    database, _, _, _ = library
    target = tmp_path / "options.zip"
    export_annotation_library(database, target, ExportOptions(
        include_media=False, include_portraits=portraits, include_person_info=info
    ))
    files = read_export(target)
    assert not any(name.startswith("media/") for name in files)
    if not portraits and not info:
        assert "people.json" not in files
    else:
        person = json.loads(files["people.json"])[0]
        assert ("name" in person) == info
        assert ("id_card" in person) == info
        assert ("portraits" in person) == portraits
    assert any(name.startswith("people/") for name in files) == portraits


def test_full_video_is_copied_verbatim_and_deduplicated(library, tmp_path):
    database, person, _, video = library
    database.log_annotation("s1", str(video), 1, "remove_association", person, 15)
    target = tmp_path / "full.zip"
    export_annotation_library(database, target, ExportOptions(full_video=True))
    files = read_export(target)
    annotations = json.loads(files["annotations.json"])
    assert annotations[0]["media"]["path"] == annotations[1]["media"]["path"]
    assert files[annotations[0]["media"]["path"]] == video.read_bytes()


@pytest.mark.parametrize("frame,expected_anchor", [(1, 1), (30, 11)])
def test_clips_keep_annotation_at_video_boundaries(library, tmp_path, frame, expected_anchor):
    *_, video = library
    metadata = write_video_clip(video, tmp_path / "edge.mp4", frame, 1, 1)
    assert metadata["annotation_frame"] == expected_anchor
    assert metadata["frame_count"] == 11


def test_unavailable_source_keeps_annotations_and_saved_snapshot(library, tmp_path):
    database, person, portrait, _ = library
    database.log_annotation("live", "rtsp://camera/live", 2, "reject", person, 7,
                            snapshot_path=str(portrait))
    target = tmp_path / "missing.zip"
    report = export_annotation_library(database, target, ExportOptions(full_video=True))
    files = read_export(target)
    annotations = json.loads(files["annotations.json"])
    assert report.issues
    assert annotations[-1]["media"] is None
    assert annotations[-1]["snapshot"] in files
    assert json.loads(files["manifest.json"])["issues"]


def test_source_image_exports_without_video_decoding(library, tmp_path):
    database, person, portrait, _ = library
    database.log_annotation("image", str(portrait), 1, "reject", person, 1)
    target = tmp_path / "image.zip"
    export_annotation_library(database, target, ExportOptions(full_video=True))
    files = read_export(target)
    last = json.loads(files["annotations.json"])[-1]
    assert last["media"]["mode"] == "image"
    assert files[last["media"]["path"]] == portrait.read_bytes()


def test_failed_export_does_not_replace_existing_file(library, tmp_path, monkeypatch):
    database, *_ = library
    target = tmp_path / "existing.zip"
    target.write_bytes(b"existing")
    monkeypatch.setattr(database, "list_stored_samples", lambda: (_ for _ in ()).throw(
        OSError("read failed")
    ))
    with pytest.raises(OSError):
        export_annotation_library(database, target, ExportOptions(include_media=False))
    assert target.read_bytes() == b"existing"


def test_annotations_survive_person_deletion_and_invalid_action_is_rejected(library):
    database, person, *_ = library
    with pytest.raises(ValueError, match="未知"):
        database.log_annotation("s", "input", 1, "invalid", person)
    database.delete_person(person)
    assert database.list_annotations()[0]["person_id"] is None


def test_effective_annotations_replay_undo_and_separate_sessions(library, tmp_path):
    database, person, _, video = library
    database.log_annotation("s1", str(video), 1, "remove_association", person, 15)
    database.log_annotation("s1", str(video), 1, "reject", person, 15)
    database.log_annotation("s1", str(video), 1, "undo_rejection", person, 15)
    database.log_annotation("s2", str(video), 1, "associate", person, 15)
    target = tmp_path / "current.zip"
    export_annotation_library(database, target, ExportOptions(include_media=False))
    files = read_export(target)
    assert len(json.loads(files["annotations.json"])) == 5
    current = json.loads(files["current_annotations.json"])
    assert len(current) == 1
    assert current[0]["session_id"] == "s2"
    assert current[0]["action"] == "associate"


@pytest.mark.parametrize("selected,active", [({1}, []), ({2}, []), ({3}, [3]),
                                                  ({1, 3}, [3]), ({4}, []), ({5}, [5])])
def test_selected_export_resolves_against_unselected_history(library, tmp_path, selected, active):
    database, person, portrait, video = library
    other = database.add_person("Bob", "ID-002", [
        FaceSampleInput(str(portrait), np.array([0, 1]), "model", 0.9)
    ])
    database.log_annotation("s1", str(video), 1, "remove_association", person, 15)
    database.log_annotation("s1", str(video), 1, "associate", other, 15)
    database.log_annotation("s1", str(video), 1, "reject", person, 15)
    database.log_annotation("s2", str(video), 1, "reject", person, 15)
    database.log_annotation("s1", str(video), 1, "undo_rejection", person, 15)
    target = tmp_path / "selected.zip"
    report = export_annotation_library(database, target, ExportOptions(include_media=False),
                                       annotation_ids=selected)
    files = read_export(target)
    records = json.loads(files["annotations.json"])
    assert {record["id"] for record in records} == selected
    assert [record["id"] for record in json.loads(files["current_annotations.json"])] == active
    assert report.annotation_count == len(selected)
    people = json.loads(files["people.json"])
    assert {entry["person_id"] for entry in people} == {record["person_id"] for record in records}
    manifest = json.loads(files["manifest.json"])
    assert manifest["scope"] == "selected"
    assert manifest["selected_annotation_ids"] == sorted(selected)
    assert manifest["active_annotation_count"] == len(active)
    if 1 in selected:
        assert records[0]["is_current"] is False
        assert records[0]["superseded_by"] == 2


@pytest.mark.parametrize("selected", [set(), {999}])
def test_invalid_or_empty_selection_does_not_export_everything(library, tmp_path, selected):
    database, *_ = library
    target = tmp_path / "keep.zip"
    target.write_bytes(b"keep")
    with pytest.raises(ValueError):
        export_annotation_library(database, target, ExportOptions(), annotation_ids=selected)
    assert target.read_bytes() == b"keep"


def test_selected_export_copies_only_selected_media(library, tmp_path):
    database, person, portrait, _ = library
    database.log_annotation("image", str(portrait), 1, "reject", person, 1,
                            face_path=str(portrait))
    target = tmp_path / "image-only.zip"
    export_annotation_library(database, target, ExportOptions(), annotation_ids={2})
    files = read_export(target)
    assert not any(name.endswith(".mp4") for name in files)
    assert "annotations/2-face.jpg" in files
    assert json.loads(files["annotations.json"])[0]["media"]["mode"] == "image"


def test_selection_does_not_revive_replaced_association_or_old_rejection(library, tmp_path):
    database, person, portrait, video = library
    other = database.add_person("Bob", "ID-002", [
        FaceSampleInput(str(portrait), np.array([0, 1]), "model", 0.9)
    ])
    database.log_annotation("s1", str(video), 1, "associate", other, 15)
    database.log_annotation("s1", str(video), 1, "reject", person, 15)
    database.log_annotation("s1", str(video), 1, "undo_rejection", person, 15)
    database.log_annotation("s1", str(video), 1, "reject", person, 15)
    target = tmp_path / "replacement.zip"
    export_annotation_library(database, target, ExportOptions(include_media=False),
                               annotation_ids={1, 2, 3, 5})
    files = read_export(target)
    assert [record["id"] for record in json.loads(files["current_annotations.json"])] == [2, 5]
    records = {record["id"]: record for record in json.loads(files["annotations.json"])}
    assert records[1]["superseded_by"] == 2
    assert records[3]["superseded_by"] == 4


@pytest.mark.parametrize("initial,next_action", [("associate", "reject"), ("reject", "associate")])
def test_review_transition_is_atomic_when_new_choice_cannot_be_saved(library, initial, next_action):
    database, person, _, video = library
    database.log_annotation("transition", str(video), 1, initial, person)
    before = database.list_annotations()
    with database._connect() as connection:
        connection.executescript(f"""CREATE TRIGGER fail_choice BEFORE INSERT ON recognition_annotations
            WHEN NEW.action='{next_action}' BEGIN SELECT RAISE(ABORT, 'cannot save'); END;""")
    with pytest.raises(sqlite3.IntegrityError, match="cannot save"):
        database.log_annotation("transition", str(video), 1, next_action, person, resolve_conflicts=True)
    assert database.list_annotations() == before
    assert annotation_states(before)[before[-1]["id"]]["is_current"]


def test_selected_old_positive_label_is_inactive_after_negative_review(library, tmp_path):
    database, person, _, video = library
    database.log_annotation("s1", str(video), 1, "reject", person, 15, resolve_conflicts=True)
    target = tmp_path / "old-positive.zip"
    export_annotation_library(database, target, ExportOptions(include_media=False), annotation_ids={1})
    files = read_export(target)
    assert json.loads(files["annotations.json"])[0]["is_current"] is False
    assert json.loads(files["current_annotations.json"]) == []
