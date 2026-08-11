from __future__ import annotations

from pathlib import Path

import numpy as np

from face_match.domain import FaceDetection
from face_match.errors import FaceMatchError
from face_match.gpu import create_cuda_session


def _distance_to_boxes(points: np.ndarray, distances: np.ndarray) -> np.ndarray:
    boxes = np.empty_like(distances, dtype=np.float32)
    boxes[:, 0] = points[:, 0] - distances[:, 0]
    boxes[:, 1] = points[:, 1] - distances[:, 1]
    boxes[:, 2] = points[:, 0] + distances[:, 2]
    boxes[:, 3] = points[:, 1] + distances[:, 3]
    return boxes


def _distance_to_landmarks(points: np.ndarray, distances: np.ndarray) -> np.ndarray:
    landmarks = np.empty((distances.shape[0], 5, 2), dtype=np.float32)
    reshaped = distances.reshape(-1, 5, 2)
    landmarks[:, :, 0] = points[:, None, 0] + reshaped[:, :, 0]
    landmarks[:, :, 1] = points[:, None, 1] + reshaped[:, :, 1]
    return landmarks


def non_maximum_suppression(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
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
        overlap = width * height
        union = areas[index] + areas[order[1:]] - overlap
        iou = overlap / np.maximum(union, 1e-8)
        order = order[np.where(iou <= threshold)[0] + 1]
    return keep


class ScrfdDetector:
    """Direct ONNX implementation of the 3-level InsightFace detector."""

    def __init__(self, model_path: Path, input_size: int = 640) -> None:
        self.session = create_cuda_session(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [item.name for item in self.session.get_outputs()]
        if len(self.output_names) != 9:
            raise FaceMatchError(
                f"检测模型输出数量应为 9，实际为 {len(self.output_names)}；请重新下载模型。"
            )
        self.input_size = int(input_size)
        self.strides = (8, 16, 32)
        self._anchor_cache: dict[tuple[int, int, int, int], np.ndarray] = {}
        # A real run catches missing driver/DLL errors that provider enumeration alone cannot catch.
        self.detect(np.zeros((64, 64, 3), dtype=np.uint8), score_threshold=0.99)

    def detect(
        self, frame_bgr: np.ndarray, score_threshold: float = 0.55, nms_threshold: float = 0.4
    ) -> list[FaceDetection]:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("检测输入必须是 BGR 三通道图像")
        import cv2

        frame_height, frame_width = frame_bgr.shape[:2]
        scale = min(self.input_size / frame_width, self.input_size / frame_height)
        resized_width = max(1, round(frame_width * scale))
        resized_height = max(1, round(frame_height * scale))
        resized = cv2.resize(
            frame_bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
        )
        canvas = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        canvas[:resized_height, :resized_width] = resized
        rgb = canvas[:, :, ::-1].astype(np.float32)
        tensor = ((rgb - 127.5) / 128.0).transpose(2, 0, 1)[None, ...]
        outputs = self.session.run(self.output_names, {self.input_name: tensor})

        all_scores: list[np.ndarray] = []
        all_boxes: list[np.ndarray] = []
        all_landmarks: list[np.ndarray] = []
        for level, stride in enumerate(self.strides):
            scores = np.asarray(outputs[level]).reshape(-1)
            box_distances = np.asarray(outputs[level + 3]).reshape(-1, 4) * float(stride)
            landmark_distances = np.asarray(outputs[level + 6]).reshape(-1, 10) * float(stride)
            feature_height = self.input_size // stride
            feature_width = self.input_size // stride
            locations = feature_height * feature_width
            if locations <= 0 or scores.size % locations:
                raise FaceMatchError("检测模型输出形状不兼容")
            anchors = scores.size // locations
            centers = self._anchor_centers(feature_height, feature_width, stride, anchors)
            positive = np.flatnonzero(scores >= score_threshold)
            if positive.size == 0:
                continue
            all_scores.append(scores[positive].astype(np.float32))
            all_boxes.append(_distance_to_boxes(centers, box_distances)[positive])
            all_landmarks.append(_distance_to_landmarks(centers, landmark_distances)[positive])

        if not all_scores:
            return []
        scores = np.concatenate(all_scores)
        boxes = np.concatenate(all_boxes) / scale
        landmarks = np.concatenate(all_landmarks) / scale
        boxes[:, (0, 2)] = np.clip(boxes[:, (0, 2)], 0, frame_width - 1)
        boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)], 0, frame_height - 1)
        landmarks[:, :, 0] = np.clip(landmarks[:, :, 0], 0, frame_width - 1)
        landmarks[:, :, 1] = np.clip(landmarks[:, :, 1], 0, frame_height - 1)
        keep = non_maximum_suppression(boxes, scores, nms_threshold)
        return [
            FaceDetection(boxes[index].copy(), float(scores[index]), landmarks[index].copy())
            for index in keep[:100]
            if boxes[index, 2] > boxes[index, 0] and boxes[index, 3] > boxes[index, 1]
        ]

    def _anchor_centers(self, height: int, width: int, stride: int, anchors: int) -> np.ndarray:
        key = (height, width, stride, anchors)
        cached = self._anchor_cache.get(key)
        if cached is not None:
            return cached
        grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
        centers = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)
        centers = (centers * float(stride)).reshape(-1, 2)
        if anchors > 1:
            centers = np.repeat(centers, anchors, axis=0)
        self._anchor_cache[key] = centers
        return centers
