import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from face_matching.detection_video import VideoControl, analyze_video
from face_matching.person_vehicle import ObjectDetection


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 12, (96, 64))
    assert writer.isOpened()
    for index in range(12):
        writer.write(np.full((64, 96, 3), 20 + index * 10, np.uint8))
    writer.release()
    return path


class CountingDetector:
    def __init__(self):
        self.frames = []
        self.session = SimpleNamespace(get_providers=lambda: ["test-GPU"])

    def detect(self, frame):
        self.frames.append(float(frame.mean()))
        return [ObjectDetection(0, 0.9, (10, 30, 30, 50))]


def test_export_processes_all_frames_at_original_rate_even_when_previews_are_throttled(video, tmp_path, monkeypatch):
    monkeypatch.setattr("face_matching.detection_video.time.monotonic", lambda: 1.0)
    detector = CountingDetector()
    events = []
    output = tmp_path / "annotated.mp4"
    report = analyze_video(video, detector, output, lambda image, detections, index, total, fps: events.append(index))
    assert not report.cancelled and report.frames == 12
    assert len(detector.frames) == 12
    assert detector.frames == sorted(detector.frames)
    assert events == [1, 12]
    capture = cv2.VideoCapture(str(output))
    assert capture.isOpened()
    assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(12)
    assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 12
    capture.release()
    assert not list(tmp_path.glob(".detection-*"))


def test_cancel_removes_partial_video_and_emits_no_success_path(video, tmp_path):
    output = tmp_path / "cancelled.mp4"
    control = VideoControl()
    report = analyze_video(video, CountingDetector(), output,
                           lambda *args: control.stopped.set(), control)
    assert report.cancelled and report.output is None and report.frames == 1
    assert not output.exists()
    assert not list(tmp_path.glob(".detection-*"))


def test_failure_removes_partial_video(video, tmp_path):
    detector = CountingDetector()
    original_detect = detector.detect

    def fail(frame):
        if len(detector.frames) == 3:
            raise RuntimeError("test inference failure")
        return original_detect(frame)

    detector.detect = fail
    output = tmp_path / "failed.mp4"
    with pytest.raises(RuntimeError, match="test inference failure"):
        analyze_video(video, detector, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".detection-*"))


def test_existing_source_and_racing_destination_are_not_overwritten(video, tmp_path):
    payload = video.read_bytes()
    with pytest.raises(FileExistsError):
        analyze_video(video, CountingDetector(), video)
    assert video.read_bytes() == payload
    output = tmp_path / "created-during-analysis.mp4"

    def race(*args):
        if not output.exists():
            output.write_bytes(b"keep")

    with pytest.raises(FileExistsError):
        analyze_video(video, CountingDetector(), output, race)
    assert output.read_bytes() == b"keep"
    assert not list(tmp_path.glob(".detection-*"))


def test_stopping_wakes_a_paused_worker():
    control = VideoControl()
    control.paused.set()
    waiting = threading.Event()
    results = []

    def wait():
        waiting.set()
        results.append(control.wait_for_resume())

    thread = threading.Thread(target=wait)
    thread.start()
    assert waiting.wait(1)
    assert thread.is_alive()
    control.stopped.set()
    thread.join(timeout=1)
    assert not thread.is_alive() and results == [False]
