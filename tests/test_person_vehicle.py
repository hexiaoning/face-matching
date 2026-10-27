from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from face_matching import models
from face_matching.errors import ModelMissingError
from face_matching.person_vehicle import (
    ObjectDetection, PERSON_COLOR, VEHICLE_COLOR, PersonVehicleDetector, annotate_objects,
)


class FakeSession:
    def __init__(self):
        self.logits = np.full((1, 300, 80), -20, np.float32)
        self.boxes = np.zeros((1, 300, 4), np.float32)
        self.tensor = None

    def get_inputs(self):
        return [SimpleNamespace(name="pixel_values")]

    def run(self, names, inputs):
        assert names == ["logits", "pred_boxes"]
        self.tensor = inputs["pixel_values"]
        return [self.logits, self.boxes]


def prediction(detector, index, class_id, box, probability=0.9, width=640, height=640):
    x1, y1, x2, y2 = box
    detector.session.boxes[0, index] = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                                       (x2 - x1) / width, (y2 - y1) / height)
    detector.session.logits[0, index, class_id] = np.log(probability / (1 - probability))


def test_rgb_rescale_and_sigmoid_scores_do_not_mutate_source():
    session = FakeSession()
    detector = PersonVehicleDetector(session)
    prediction(detector, 0, 0, (50, 100, 150, 300), height=320)
    image = np.full((320, 640, 3), (10, 20, 30), np.uint8)
    before = image.copy()
    detections = detector.detect(image)
    assert session.tensor.shape == (1, 3, 640, 640)
    np.testing.assert_allclose(session.tensor[0, :, 0, 0], np.array((30, 20, 10)) / 255)
    np.testing.assert_array_equal(session.tensor[0, :, 639, 0], session.tensor[0, :, 0, 0])
    np.testing.assert_array_equal(image, before)
    assert len(detections) == 1
    assert detections[0].score == pytest.approx(0.9)
    np.testing.assert_allclose(detections[0].bbox, (50, 100, 150, 300), atol=0.001)


def test_class_filters_preserve_overlapping_distinct_queries_without_nms():
    session = FakeSession()
    detector = PersonVehicleDetector(session)
    for index, class_id in enumerate((0, 2, 16)):
        prediction(detector, index, class_id, (50, 100, 150, 300), probability=0.95 - index * 0.01)
    detections = detector.detect(np.zeros((640, 640, 3), np.uint8))
    assert {item.class_id for item in detections} == {0, 2}
    vehicles_only = PersonVehicleDetector(session, people=False)
    assert [item.class_id for item in vehicles_only.detect(np.zeros((640, 640, 3), np.uint8))] == [2]


def test_normalized_coordinates_map_back_to_source_and_reject_nonfinite_or_empty_boxes():
    session = FakeSession()
    detector = PersonVehicleDetector(session)
    prediction(detector, 0, 2, (100, 100, 300, 300), width=1280)
    prediction(detector, 1, 0, (-100, 20, -10, 60))  # Entirely outside the image.
    prediction(detector, 2, 0, (20, 20, 80, 80))
    session.boxes[0, 2, 0] = np.nan
    detections = detector.detect(np.zeros((640, 1280, 3), np.uint8))
    assert len(detections) == 1
    np.testing.assert_allclose(detections[0].bbox, (100, 100, 300, 300), atol=0.001)


def test_unsupported_highest_class_is_not_relabelled_as_person():
    session = FakeSession()
    detector = PersonVehicleDetector(session)
    prediction(detector, 0, 0, (30, 30, 80, 80))
    session.logits[0, 0, 16] = 8
    assert not detector.detect(np.zeros((640, 640, 3), np.uint8))


def test_small_targets_are_retained_without_size_filtering():
    session = FakeSession()
    detector = PersonVehicleDetector(session)
    prediction(detector, 0, 2, (40, 40, 64, 64))
    image = np.zeros((640, 640, 3), np.uint8)
    assert len(detector.detect(image)) == 1


def test_red_people_yellow_vehicles_and_top_left_scores(monkeypatch):
    texts = []
    put_text = cv2.putText

    def record(image, text, position, *args):
        texts.append((text, position))
        return put_text(image, text, position, *args)

    monkeypatch.setattr(cv2, "putText", record)
    frame = np.zeros((300, 400, 3), np.uint8)
    annotated = annotate_objects(frame, [
        ObjectDetection(0, 0.91, (40, 100, 120, 200)),
        ObjectDetection(2, 0.82, (200, 100, 350, 220)),
    ])
    np.testing.assert_array_equal(annotated[150, 40], PERSON_COLOR)
    np.testing.assert_array_equal(annotated[160, 200], VEHICLE_COLOR)
    assert texts[0][0] == "person 0.91" and texts[0][1][1] < 100
    assert texts[1][0] == "car 0.82" and texts[1][1][1] < 100
    assert not frame.any()


@pytest.mark.parametrize("kwargs", [{"threshold": float("nan")}, {"threshold": 0},
                                     {"people": False, "vehicles": False}])
def test_invalid_detector_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        PersonVehicleDetector(FakeSession(), **kwargs)


def test_person_vehicle_model_uses_hash_validation_and_bundled_fallback(tmp_path, monkeypatch):
    local, bundled = tmp_path / "local", tmp_path / "bundled"
    local.mkdir()
    bundled.mkdir()
    import hashlib
    spec = replace(models.PERSON_VEHICLE_MODEL, size=4, sha256=hashlib.sha256(b"good").hexdigest())
    monkeypatch.setattr(models, "PERSON_VEHICLE_MODEL", spec)
    monkeypatch.setattr(models, "models_dir", lambda: local)
    monkeypatch.setattr(models, "packaged_models_dir", lambda: bundled)
    (local / spec.name).write_bytes(b"bad!")
    (bundled / spec.name).write_bytes(b"good")
    assert models.required_person_vehicle_model() == bundled / spec.name
    (bundled / spec.name).unlink()
    with pytest.raises(ModelMissingError):
        models.required_person_vehicle_model()


def test_detector_requests_static_gpu_input_shape(monkeypatch, tmp_path):
    from face_matching.person_vehicle import load_person_vehicle_detector
    from face_matching import gpu

    captured = []
    path = tmp_path / "model.onnx"
    monkeypatch.setattr(models, "required_person_vehicle_model", lambda: path)
    monkeypatch.setattr(gpu, "create_gpu_session", lambda *args, **kwargs: (
        captured.append((args, kwargs)) or FakeSession()
    ))
    load_person_vehicle_detector("openvino", device_id=2)
    assert captured == [((path,), {"device_id": 2, "backend": "openvino", "input_shape": (1, 3, 640, 640)})]
