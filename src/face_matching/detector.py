from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .gpu import create_gpu_session


@dataclass(slots=True)
class FaceDetection:
    bbox: np.ndarray
    landmarks: np.ndarray
    score: float
    aligned: np.ndarray | None = None
    quality: object | None = None


def distance_to_bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance_to_landmarks(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    landmarks = []
    for index in range(0, distance.shape[1], 2):
        landmarks.append(points[:, 0] + distance[:, index])
        landmarks.append(points[:, 1] + distance[:, index + 1])
    return np.stack(landmarks, axis=-1)


def nms(boxes: np.ndarray, threshold: float) -> list[int]:
    if boxes.size == 0:
        return []
    x1, y1, x2, y2, scores = boxes.T
    areas = np.maximum(0.0, x2 - x1 + 1.0) * np.maximum(0.0, y2 - y1 + 1.0)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        index = int(order[0])
        keep.append(index)
        xx1 = np.maximum(x1[index], x1[order[1:]])
        yy1 = np.maximum(y1[index], y1[order[1:]])
        xx2 = np.minimum(x2[index], x2[order[1:]])
        yy2 = np.minimum(y2[index], y2[order[1:]])
        width = np.maximum(0.0, xx2 - xx1 + 1.0)
        height = np.maximum(0.0, yy2 - yy1 + 1.0)
        intersection = width * height
        overlap = intersection / np.maximum(areas[index] + areas[order[1:]] - intersection, 1e-8)
        order = order[np.where(overlap <= threshold)[0] + 1]
    return keep


class SCRFDDetector:
    """Minimal ONNX SCRFD inference with mandatory GPU execution."""

    def __init__(
        self,
        model_path: str,
        input_size: tuple[int, int] = (640, 640),
        threshold: float = 0.45,
        nms_threshold: float = 0.40,
        prefer_tensorrt: bool = False,
        backend: str = "auto",
    ) -> None:
        self.session = create_gpu_session(model_path, prefer_tensorrt, backend)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.input_size = input_size
        self.threshold = threshold
        self.nms_threshold = nms_threshold
        output_count = len(self.output_names)
        if output_count in (6, 9):
            self.feature_maps = 3
            self.strides = (8, 16, 32)
            self.anchors = 2
        elif output_count in (10, 15):
            self.feature_maps = 5
            self.strides = (8, 16, 32, 64, 128)
            self.anchors = 1
        else:
            raise ValueError(f"Unsupported SCRFD output count: {output_count}")
        self.has_landmarks = output_count in (9, 15)
        if not self.has_landmarks:
            raise ValueError("SCRFD model must include five-point landmarks")
        self._centers: dict[tuple[int, int, int], np.ndarray] = {}

    def _anchor_centers(self, height: int, width: int, stride: int) -> np.ndarray:
        key = height, width, stride
        cached = self._centers.get(key)
        if cached is not None:
            return cached
        grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
        centers = np.stack([grid_x, grid_y], axis=-1).astype(np.float32) * stride
        centers = centers.reshape(-1, 2)
        if self.anchors > 1:
            centers = np.repeat(centers, self.anchors, axis=0)
        if len(self._centers) < 32:
            self._centers[key] = centers
        return centers

    def detect(self, frame_bgr: np.ndarray) -> list[FaceDetection]:
        image_height, image_width = frame_bgr.shape[:2]
        target_width, target_height = self.input_size
        scale = min(target_width / image_width, target_height / image_height)
        resized_width = max(1, int(round(image_width * scale)))
        resized_height = max(1, int(round(image_height * scale)))
        resized = cv2.resize(frame_bgr, (resized_width, resized_height))
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        canvas[:resized_height, :resized_width] = resized
        blob = cv2.dnn.blobFromImage(
            canvas,
            scalefactor=1.0 / 128.0,
            size=(target_width, target_height),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        )
        outputs = self.session.run(self.output_names, {self.input_name: blob})

        all_scores: list[np.ndarray] = []
        all_boxes: list[np.ndarray] = []
        all_landmarks: list[np.ndarray] = []
        for index, stride in enumerate(self.strides):
            scores = np.asarray(outputs[index]).reshape(-1)
            box_distances = np.asarray(outputs[index + self.feature_maps]).reshape(-1, 4) * stride
            landmark_distances = (
                np.asarray(outputs[index + self.feature_maps * 2]).reshape(-1, 10) * stride
            )
            feature_height = target_height // stride
            feature_width = target_width // stride
            centers = self._anchor_centers(feature_height, feature_width, stride)
            count = min(len(scores), len(box_distances), len(landmark_distances), len(centers))
            scores = scores[:count]
            positive = np.where(scores >= self.threshold)[0]
            if not len(positive):
                continue
            centers = centers[:count]
            boxes = distance_to_bbox(centers, box_distances[:count])[positive]
            landmarks = distance_to_landmarks(centers, landmark_distances[:count])[positive]
            all_scores.append(scores[positive])
            all_boxes.append(boxes)
            all_landmarks.append(landmarks)

        if not all_scores:
            return []
        scores = np.concatenate(all_scores).astype(np.float32)
        boxes = np.concatenate(all_boxes).astype(np.float32) / scale
        landmarks = np.concatenate(all_landmarks).astype(np.float32).reshape(-1, 5, 2) / scale
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, image_width - 1)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, image_height - 1)
        landmarks[:, :, 0] = np.clip(landmarks[:, :, 0], 0, image_width - 1)
        landmarks[:, :, 1] = np.clip(landmarks[:, :, 1], 0, image_height - 1)
        candidates = np.column_stack([boxes, scores])
        keep = nms(candidates, self.nms_threshold)
        return [
            FaceDetection(boxes[index], landmarks[index], float(scores[index]))
            for index in keep
        ]
