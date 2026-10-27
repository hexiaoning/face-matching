from pathlib import Path
import threading
from types import SimpleNamespace
import time

import cv2
import numpy as np
import pytest
from PySide6.QtTest import QTest

from face_matching.image_enhancement import encode_png
from face_matching.visual_search import (FaceExampleMatcher, ObjectExampleMatcher, collect_sources,
                                         read_hit_frame, search_sources)
from face_matching.gui.visual_search import VisualSearchWidget


def target():
    rng = np.random.default_rng(6)
    return rng.integers(0, 255, (100, 120, 3), np.uint8)


def scene():
    query = target()
    frame = np.full((260, 400, 3), 30, np.uint8)
    frame[80:180, 150:270] = query
    return query, frame


def test_object_search_locates_selected_instance_and_rejects_absent_target():
    query, frame = scene()
    matcher = ObjectExampleMatcher(query, .75)
    hits = matcher.find(frame)
    assert hits
    assert hits[0][0] > .95
    assert hits[0][1] == pytest.approx((150, 80, 270, 180), abs=2)
    assert not matcher.find(np.full_like(frame, 50))
    with pytest.raises(ValueError, match='纹理'):
        ObjectExampleMatcher(np.zeros((50, 50, 3), np.uint8))


def test_object_search_recovers_scaled_instance():
    query = target()
    frame = np.full((260, 400, 3), 30, np.uint8)
    frame[70:210, 120:288] = cv2.resize(query, (168, 140))
    found = ObjectExampleMatcher(query, .75).find(frame)
    assert found and found[0][1] == pytest.approx((120, 70, 288, 210), abs=3)


def test_recursive_search_video_sampling_sources_failure_and_partial_cancel(tmp_path):
    query, frame = scene()
    folder = tmp_path / 'nested'
    folder.mkdir()
    image = folder / '图像.png'
    image.write_bytes(encode_png(frame))
    broken = tmp_path / '坏图.png'
    broken.write_bytes(b'bad')
    video = tmp_path / 'test.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (400, 260))
    assert writer.isOpened()
    for _ in range(4):
        writer.write(frame)
    writer.release()
    sources = collect_sources([tmp_path, folder, image])
    assert len(sources) == 3
    hits, failures = search_sources(sources, ObjectExampleMatcher(query), frame_step=2)
    assert len(failures) == 1 and str(broken) in failures[0]
    video_hits = [hit for hit in hits if hit.source == str(video)]
    assert {hit.frame_index for hit in video_hits} == {0, 2}
    assert {hit.seconds for hit in video_hits} == {0, .2}
    assert read_hit_frame(video_hits[0]).shape == frame.shape
    image_hit = next(hit for hit in hits if hit.source == str(image))
    np.testing.assert_array_equal(read_hit_frame(image_hit), frame)
    stopped = threading.Event()
    def find(frame):
        stopped.set()
        return [(1., (0, 0, 10, 10), 'test')]
    partial, failures = search_sources([image, video], SimpleNamespace(find=find), stopped=stopped)
    assert len(partial) == 1 and not failures


def test_face_search_uses_reference_embedding_without_mutating_identity_gallery(monkeypatch):
    query, frame = scene()
    detection = SimpleNamespace(width=50, height=50, bbox=(5, 10, 55, 60), landmarks=np.zeros((5, 2)))
    engine = SimpleNamespace(enrollment_feature=lambda image: SimpleNamespace(embedding=np.array([1., 0.])),
        detector=SimpleNamespace(detect=lambda image: [detection]),
        embedder=SimpleNamespace(embed_many=lambda *args, **kwargs: [np.array([.9, .1])]),
        config=SimpleNamespace(min_face_size=20, min_quality=.3), mirror_augmentation=True)
    monkeypatch.setattr('face_matching.vision.alignment.align_face', lambda *args: query)
    monkeypatch.setattr('face_matching.vision.quality.assess_quality', lambda *args: SimpleNamespace(total=.8))
    hits = FaceExampleMatcher(query, engine, .45).find(frame)
    assert hits[0][0] == pytest.approx(.9)
    assert hits[0][1] == (5, 10, 55, 60)


def test_search_widget_cropped_query_and_sends_unannotated_frame_to_enhancement(qt_runtime, tmp_path):
    query, frame = scene()
    source = tmp_path / '场景.png'
    source.write_bytes(encode_png(frame))
    tool = VisualSearchWidget(lambda: None)
    try:
        assert tool.load_query(source)
        tool.select_region((150, 80, 270, 180))
        tool.set_paths([source])
        tool.start()
        deadline = time.monotonic() + 10
        while tool.is_busy() and time.monotonic() < deadline:
            qt_runtime.processEvents()
            QTest.qWait(5)
        assert not tool.is_busy()
        assert len(tool.hits) == 1
        sent = []
        tool.enhance_requested.connect(lambda pixels, bbox: sent.append((pixels, bbox)))
        tool.enhance_selected()
        np.testing.assert_array_equal(sent[0][0], frame)
        assert sent[0][1] == (150, 80, 270, 180)
        assert tool.search_info['region_xyxy'] == (150, 80, 270, 180)
    finally:
        tool.stop()
        if tool.worker:
            tool.worker.wait(10000)
        tool.close()
        tool.deleteLater()
