"""RT-DETRv2 category detection, independent of identity and gallery code.

The pinned model consumes RGB 0..1 resized to 640x640 and returns class logits
and normalized center/width/height boxes. No minimum target size is imposed.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


MODEL_TITLE = "RT-DETRv2-R50"
CLASS_NAMES = {
    0: ("person", "人"), 1: ("bicycle", "自行车"), 2: ("car", "汽车"),
    3: ("motorcycle", "摩托车"), 5: ("bus", "公交车"),
    6: ("train", "列车"), 7: ("truck", "货车"),
}
PERSON_COLOR = (0, 0, 255)  # BGR, deliberately red.
VEHICLE_COLOR = (0, 255, 255)  # BGR, deliberately yellow.


@dataclass(frozen=True)
class ObjectDetection:
    class_id: int
    score: float
    bbox: tuple[float, float, float, float]

    @property
    def is_person(self) -> bool:
        return self.class_id == 0

    @property
    def label(self) -> str:
        return CLASS_NAMES[self.class_id][1]

    @property
    def color(self) -> tuple[int, int, int]:
        return PERSON_COLOR if self.is_person else VEHICLE_COLOR


class PersonVehicleDetector:
    def __init__(self, session: object, threshold: float = 0.5,
                 people: bool = True, vehicles: bool = True) -> None:
        if not np.isfinite(threshold) or not 0.05 <= threshold <= 0.99:
            raise ValueError("置信度门槛必须在 0.05～0.99 之间")
        if not people and not vehicles:
            raise ValueError("请至少选择人或车辆")
        self.session = session
        self.threshold = threshold
        self.allowed = {key for key in CLASS_NAMES if (people if key == 0 else vehicles)}
        self.input_name = session.get_inputs()[0].name
        self.output_names = ["logits", "pred_boxes"]

    def detect(self, image: np.ndarray) -> list[ObjectDetection]:
        if image is None or not image.size:
            return []
        height, width = image.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(image, (640, 640)), cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        if (len(outputs) != 2 or np.asarray(outputs[0]).shape != (1, 300, 80)
                or np.asarray(outputs[1]).shape != (1, 300, 4)):
            raise ValueError("人车检测模型输出格式不正确，需要 RT-DETRv2 logits 与 pred_boxes")
        return self.decode(np.asarray(outputs[0])[0], np.asarray(outputs[1])[0], width, height)

    def decode(self, logits: np.ndarray, predictions: np.ndarray,
               width: int, height: int) -> list[ObjectDetection]:
        # Choose the global best class first; never relabel a dog's second-best
        # person class as a person just because the other classes are hidden.
        classes = logits.argmax(axis=1)
        top_logits = logits[np.arange(len(logits)), classes]
        scores = 1 / (1 + np.exp(-np.clip(top_logits, -80, 80)))
        keep = (np.isfinite(logits).all(axis=1) & np.isfinite(predictions).all(axis=1)
                & (scores >= self.threshold) & np.isin(classes, list(self.allowed)))
        rows, classes, scores = predictions[keep], classes[keep], scores[keep]
        if not len(rows):
            return []
        centers, sizes = rows[:, :2], rows[:, 2:]
        boxes = np.column_stack((centers - sizes / 2, centers + sizes / 2))
        boxes *= np.array((width, height, width, height), np.float32)
        boxes[:, (0, 2)] = np.clip(boxes[:, (0, 2)], 0, width - 1)
        boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)], 0, height - 1)
        valid = ((boxes[:, 2] - boxes[:, 0]) > 0) & ((boxes[:, 3] - boxes[:, 1]) > 0)
        boxes, classes, scores = boxes[valid], classes[valid], scores[valid]
        # This end-to-end detector needs no NMS or minimum-size suppression.
        return [ObjectDetection(int(classes[index]), float(scores[index]), tuple(float(v) for v in boxes[index]))
                for index in np.argsort(-scores)]


def load_person_vehicle_detector(backend: str = "auto", device_id: int = 0,
                                 threshold: float = 0.5, people: bool = True,
                                 vehicles: bool = True) -> PersonVehicleDetector:
    from .gpu import create_gpu_session
    from .models import required_person_vehicle_model

    return PersonVehicleDetector(
        create_gpu_session(required_person_vehicle_model(), device_id=device_id,
                           backend=backend, input_shape=(1, 3, 640, 640)),
        threshold, people, vehicles,
    )


def annotate_objects(frame: np.ndarray, detections: list[ObjectDetection]) -> np.ndarray:
    result = frame.copy()
    height, width = result.shape[:2]
    thickness = max(2, round(min(width, height) / 360))
    font_scale = max(0.45, min(1.2, min(width, height) / 1000))
    for detection in detections:
        x1, y1, x2, y2 = (int(round(value)) for value in detection.bbox)
        x1, x2 = np.clip((x1, x2), 0, width - 1).tolist()
        y1, y2 = np.clip((y1, y2), 0, height - 1).tolist()
        if x2 <= x1 or y2 <= y1:
            continue
        color = detection.color
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
        # Hershey fonts are ASCII-only. The UI provides the Chinese class names.
        text = f"{CLASS_NAMES[detection.class_id][0]} {detection.score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        left = max(0, min(x1, width - text_width - 6))
        top = y1 - text_height - baseline - 6
        if top < 0:
            top = y1
        bottom = min(height - 1, top + text_height + baseline + 6)
        cv2.rectangle(result, (left, top), (min(width - 1, left + text_width + 6), bottom), color, -1)
        cv2.putText(result, text, (left + 3, min(height - baseline - 1, top + text_height + 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
    return result
