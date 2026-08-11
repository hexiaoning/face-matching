from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from face_matching.detector import SCRFDDetector, distance_to_bbox, distance_to_landmarks, nms


def test_distance_decoders() -> None:
    points = np.array([[10.0, 20.0]], dtype=np.float32)
    boxes = distance_to_bbox(points, np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32))
    landmarks = distance_to_landmarks(
        points,
        np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]], dtype=np.float32),
    )

    np.testing.assert_allclose(boxes, [[9.0, 18.0, 13.0, 24.0]])
    np.testing.assert_allclose(
        landmarks,
        [[11.0, 22.0, 13.0, 24.0, 15.0, 26.0, 17.0, 28.0, 19.0, 30.0]],
    )


def test_nms_keeps_best_overlapping_box_and_separate_box() -> None:
    boxes = np.array(
        [
            [0.0, 0.0, 20.0, 20.0, 0.95],
            [1.0, 1.0, 21.0, 21.0, 0.80],
            [100.0, 100.0, 120.0, 120.0, 0.70],
        ],
        dtype=np.float32,
    )

    assert nms(boxes, 0.4) == [0, 2]
    assert nms(np.empty((0, 5), dtype=np.float32), 0.4) == []


def test_detector_accepts_fixed_input_and_two_class_scrfd_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSession:
        def get_inputs(self):
            return [SimpleNamespace(name="input", shape=[1, 3, 32, 32])]

        def get_outputs(self):
            return [SimpleNamespace(name=f"output-{index}") for index in range(9)]

        def run(self, names, inputs):
            rows = [32, 8, 2]  # two anchors at strides 8/16/32
            scores = [np.zeros((count, 2), dtype=np.float32) for count in rows]
            scores[0][0, 1] = 0.9
            boxes = [np.ones((count, 4), dtype=np.float32) for count in rows]
            landmarks = [np.zeros((count, 10), dtype=np.float32) for count in rows]
            return scores + boxes + landmarks

    monkeypatch.setattr(
        "face_matching.detector.create_gpu_session", lambda *args, **kwargs: FakeSession()
    )
    detector = SCRFDDetector("detector.onnx", input_size=(64, 64), threshold=0.4)

    detections = detector.detect(np.zeros((32, 32, 3), dtype=np.uint8))

    assert detector.input_size == (32, 32)
    assert len(detections) == 1
    assert detections[0].score == pytest.approx(0.9)
