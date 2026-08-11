from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class Detection:
    bbox: np.ndarray
    score: float
    landmarks: np.ndarray

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])


def distance_to_bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack((x1, y1, x2, y2), axis=-1)


def distance_to_landmarks(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    values: list[np.ndarray] = []
    for index in range(0, distance.shape[1], 2):
        values.extend((points[:, 0] + distance[:, index], points[:, 1] + distance[:, index + 1]))
    return np.stack(values, axis=-1)


def nms(boxes: np.ndarray, threshold: float) -> list[int]:
    if boxes.size == 0:
        return []
    x1, y1, x2, y2, scores = (boxes[:, index] for index in range(5))
    areas = np.maximum(0.0, x2 - x1 + 1.0) * np.maximum(0.0, y2 - y1 + 1.0)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        xx1 = np.maximum(x1[current], x1[remaining])
        yy1 = np.maximum(y1[current], y1[remaining])
        xx2 = np.minimum(x2[current], x2[remaining])
        yy2 = np.minimum(y2[current], y2[remaining])
        width = np.maximum(0.0, xx2 - xx1 + 1.0)
        height = np.maximum(0.0, yy2 - yy1 + 1.0)
        intersection = width * height
        overlap = intersection / np.maximum(areas[current] + areas[remaining] - intersection, 1e-9)
        order = remaining[np.where(overlap <= threshold)[0]]
    return keep


class SCRFDDetector:
    """Minimal SCRFD ONNX decoder, intentionally independent of InsightFace runtime."""

    def __init__(
        self,
        session: object,
        input_size: int = 640,
        threshold: float = 0.5,
        nms_threshold: float = 0.4,
    ) -> None:
        self.session = session
        self.input_size = (int(input_size), int(input_size))
        self.threshold = float(threshold)
        self.nms_threshold = float(nms_threshold)
        self.input_name = session.get_inputs()[0].name
        self.output_names = [item.name for item in session.get_outputs()]
        count = len(self.output_names)
        if count == 9:
            self.feature_map_count = 3
            self.strides = (8, 16, 32)
            self.num_anchors = 2
        elif count == 15:
            self.feature_map_count = 5
            self.strides = (8, 16, 32, 64, 128)
            self.num_anchors = 1
        else:
            raise ValueError(f"不支持的 SCRFD 输出数量: {count}（需要 9 或 15）")
        self._centers: dict[tuple[int, int, int], np.ndarray] = {}

    def _anchor_centers(self, height: int, width: int, stride: int) -> np.ndarray:
        key = (height, width, stride)
        cached = self._centers.get(key)
        if cached is not None:
            return cached
        centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
        centers = (centers * stride).reshape(-1, 2)
        if self.num_anchors > 1:
            centers = np.stack([centers] * self.num_anchors, axis=1).reshape(-1, 2)
        self._centers[key] = centers
        return centers

    @staticmethod
    def _remove_batch(array: np.ndarray) -> np.ndarray:
        value = np.asarray(array)
        if value.ndim == 3 and value.shape[0] == 1:
            value = value[0]
        return value

    def detect(
        self,
        image: np.ndarray,
        max_faces: int = 100,
        threshold: float | None = None,
    ) -> list[Detection]:
        if image is None or image.size == 0:
            return []
        input_width, input_height = self.input_size
        image_height, image_width = image.shape[:2]
        scale = min(input_width / image_width, input_height / image_height)
        new_width = max(1, int(round(image_width * scale)))
        new_height = max(1, int(round(image_height * scale)))
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((input_height, input_width, 3), dtype=np.uint8)
        canvas[:new_height, :new_width] = resized
        blob = cv2.dnn.blobFromImage(
            canvas, 1.0 / 128.0, self.input_size, (127.5, 127.5, 127.5), swapRB=True
        )
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        score_chunks: list[np.ndarray] = []
        bbox_chunks: list[np.ndarray] = []
        landmark_chunks: list[np.ndarray] = []
        fmc = self.feature_map_count
        score_threshold = self.threshold if threshold is None else float(threshold)
        for index, stride in enumerate(self.strides):
            scores = self._remove_batch(outputs[index]).reshape(-1)
            bbox_predictions = self._remove_batch(outputs[index + fmc]).reshape(-1, 4) * stride
            landmark_predictions = self._remove_batch(outputs[index + fmc * 2]).reshape(-1, 10) * stride
            height, width = input_height // stride, input_width // stride
            centers = self._anchor_centers(height, width, stride)
            usable = min(len(scores), len(bbox_predictions), len(landmark_predictions), len(centers))
            positive = np.where(scores[:usable] >= score_threshold)[0]
            if positive.size == 0:
                continue
            score_chunks.append(scores[positive])
            bbox_chunks.append(distance_to_bbox(centers[:usable], bbox_predictions[:usable])[positive])
            landmarks = distance_to_landmarks(centers[:usable], landmark_predictions[:usable])
            landmark_chunks.append(landmarks[positive].reshape(-1, 5, 2))
        if not score_chunks:
            return []
        scores = np.concatenate(score_chunks).astype(np.float32)
        boxes = np.vstack(bbox_chunks).astype(np.float32) / scale
        landmarks = np.vstack(landmark_chunks).astype(np.float32) / scale
        boxes[:, (0, 2)] = np.clip(boxes[:, (0, 2)], 0, image_width - 1)
        boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)], 0, image_height - 1)
        candidates = np.column_stack((boxes, scores))
        order = scores.argsort()[::-1]
        candidates, landmarks = candidates[order], landmarks[order]
        keep = nms(candidates, self.nms_threshold)[:max_faces]
        return [
            Detection(candidates[index, :4].copy(), float(candidates[index, 4]), landmarks[index].copy())
            for index in keep
        ]

    def detect_reference(self, image: np.ndarray, max_faces: int = 100) -> list[Detection]:
        """Detect a face in a still portrait even when it is cropped very tightly.

        Adding neutral canvas around a portrait restores the face-to-image scale
        expected by SCRFD. No facial pixels are generated, and all geometry is
        mapped back to the original photo before it is returned.
        """
        if image is None or image.size == 0:
            return []
        image_height, image_width = image.shape[:2]
        reference_threshold = min(self.threshold, 0.25)
        candidates = list(
            self.detect(image, max_faces=max_faces, threshold=reference_threshold)
        )
        longest_side = max(image_height, image_width)
        for ratio in (0.25, 0.50):
            padding = max(1, int(round(longest_side * ratio)))
            padded = cv2.copyMakeBorder(
                image,
                padding,
                padding,
                padding,
                padding,
                cv2.BORDER_CONSTANT,
                value=(127, 127, 127),
            )
            box_offset = np.asarray(
                (padding, padding, padding, padding), dtype=np.float32
            )
            point_offset = np.asarray((padding, padding), dtype=np.float32)
            for detection in self.detect(
                padded, max_faces=max_faces, threshold=reference_threshold
            ):
                box = np.asarray(detection.bbox, dtype=np.float32) - box_offset
                center_x = float((box[0] + box[2]) * 0.5)
                center_y = float((box[1] + box[3]) * 0.5)
                if not (0.0 <= center_x < image_width and 0.0 <= center_y < image_height):
                    continue
                original_area = max(
                    float(box[2] - box[0]) * float(box[3] - box[1]), 1.0
                )
                clipped = box.copy()
                clipped[[0, 2]] = np.clip(clipped[[0, 2]], 0, image_width - 1)
                clipped[[1, 3]] = np.clip(clipped[[1, 3]], 0, image_height - 1)
                visible_area = max(
                    float(clipped[2] - clipped[0])
                    * float(clipped[3] - clipped[1]),
                    0.0,
                )
                if visible_area / original_area < 0.55:
                    continue
                candidates.append(
                    Detection(
                        clipped,
                        detection.score,
                        np.asarray(detection.landmarks, dtype=np.float32) - point_offset,
                    )
                )

        # The context scales normally rediscover the same face. Keep the most
        # confident mapped detection from each overlap cluster.
        kept: list[Detection] = []
        for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
            box = candidate.bbox
            duplicate = False
            for existing in kept:
                other = existing.bbox
                x1 = max(float(box[0]), float(other[0]))
                y1 = max(float(box[1]), float(other[1]))
                x2 = min(float(box[2]), float(other[2]))
                y2 = min(float(box[3]), float(other[3]))
                intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                box_area = max(0.0, float(box[2] - box[0])) * max(
                    0.0, float(box[3] - box[1])
                )
                other_area = max(0.0, float(other[2] - other[0])) * max(
                    0.0, float(other[3] - other[1])
                )
                overlap = intersection / max(
                    box_area + other_area - intersection, 1e-9
                )
                if overlap >= self.nms_threshold:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)
                if len(kept) >= max_faces:
                    break
        return kept
