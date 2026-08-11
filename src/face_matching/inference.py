from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from face_matching.errors import GpuUnavailableError, ModelError
from face_matching.gpu import create_cuda_session
from face_matching.model_manager import locate_model_files

ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True, slots=True)
class DetectedFace:
    bbox: np.ndarray
    landmarks: np.ndarray
    detection_score: float


@dataclass(frozen=True, slots=True)
class EmbeddedFace:
    bbox: np.ndarray
    landmarks: np.ndarray
    detection_score: float
    embedding: np.ndarray
    quality: float


def _distance_to_bbox(points: np.ndarray, distances: np.ndarray) -> np.ndarray:
    x1 = points[:, 0] - distances[:, 0]
    y1 = points[:, 1] - distances[:, 1]
    x2 = points[:, 0] + distances[:, 2]
    y2 = points[:, 1] + distances[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance_to_landmarks(points: np.ndarray, distances: np.ndarray) -> np.ndarray:
    result = np.empty_like(distances, dtype=np.float32)
    for index in range(0, distances.shape[1], 2):
        result[:, index] = points[:, 0] + distances[:, index]
        result[:, index + 1] = points[:, 1] + distances[:, index + 1]
    return result


def _nms(boxes: np.ndarray, threshold: float) -> list[int]:
    if boxes.size == 0:
        return []
    x1, y1, x2, y2, scores = boxes.T
    areas = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        index = int(order[0])
        keep.append(index)
        xx1 = np.maximum(x1[index], x1[order[1:]])
        yy1 = np.maximum(y1[index], y1[order[1:]])
        xx2 = np.minimum(x2[index], x2[order[1:]])
        yy2 = np.minimum(y2[index], y2[order[1:]])
        width = np.maximum(0, xx2 - xx1 + 1)
        height = np.maximum(0, yy2 - yy1 + 1)
        overlap = (
            width * height / np.maximum(areas[index] + areas[order[1:]] - width * height, 1e-8)
        )
        order = order[np.where(overlap <= threshold)[0] + 1]
    return keep


class ScrfdDetector:
    def __init__(
        self,
        model_path: Path,
        device_id: int,
        input_size: tuple[int, int],
        threshold: float,
    ):
        self.session = create_cuda_session(str(model_path), device_id)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size
        self.threshold = threshold
        output_count = len(self.session.get_outputs())
        if output_count not in {6, 9, 10, 15}:
            raise ModelError(f"不支持的 SCRFD 输出数量：{output_count}")
        self.feature_map_count = 3 if output_count in {6, 9} else 5
        self.has_landmarks = output_count in {9, 15}
        if not self.has_landmarks:
            raise ModelError("检测模型没有 5 点关键点输出，无法进行可靠人脸对齐。")
        self.strides = (8, 16, 32) if self.feature_map_count == 3 else (8, 16, 32, 64, 128)
        self.anchor_cache: dict[tuple[int, int, int, int], np.ndarray] = {}

    def _anchor_centers(self, height: int, width: int, stride: int, count: int) -> np.ndarray:
        key = (height, width, stride, count)
        if key not in self.anchor_cache:
            grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
            centers = np.stack([grid_x, grid_y], axis=-1).astype(np.float32) * stride
            centers = centers.reshape(-1, 2)
            if count > 1:
                centers = np.repeat(centers, count, axis=0)
            self.anchor_cache[key] = centers
        return self.anchor_cache[key]

    def detect(self, frame: np.ndarray) -> list[DetectedFace]:
        image_height, image_width = frame.shape[:2]
        target_width, target_height = self.input_size
        ratio = min(target_width / image_width, target_height / image_height)
        resized_width = max(1, int(round(image_width * ratio)))
        resized_height = max(1, int(round(image_height * ratio)))
        resized = cv2.resize(frame, (resized_width, resized_height))
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        canvas[:resized_height, :resized_width] = resized
        blob = cv2.dnn.blobFromImage(
            canvas, 1.0 / 128.0, (target_width, target_height), (127.5, 127.5, 127.5), True
        )
        outputs = self.session.run(None, {self.input_name: blob})
        score_parts: list[np.ndarray] = []
        box_parts: list[np.ndarray] = []
        landmark_parts: list[np.ndarray] = []
        for level, stride in enumerate(self.strides):
            scores = np.asarray(outputs[level]).reshape(-1)
            distances = np.asarray(outputs[level + self.feature_map_count]).reshape(-1, 4) * stride
            locations = max(1, (target_height // stride) * (target_width // stride))
            anchors_per_location = max(1, scores.size // locations)
            centers = self._anchor_centers(
                target_height // stride, target_width // stride, stride, anchors_per_location
            )
            count = min(scores.size, distances.shape[0], centers.shape[0])
            selected = np.where(scores[:count] >= self.threshold)[0]
            if selected.size == 0:
                continue
            score_parts.append(scores[selected])
            box_parts.append(_distance_to_bbox(centers[selected], distances[selected]))
            if self.has_landmarks:
                landmark_distances = (
                    np.asarray(outputs[level + self.feature_map_count * 2]).reshape(-1, 10) * stride
                )
                landmark_parts.append(
                    _distance_to_landmarks(centers[selected], landmark_distances[selected])
                )
        if not score_parts:
            return []
        scores = np.concatenate(score_parts)
        boxes = np.concatenate(box_parts) / ratio
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, image_width - 1)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, image_height - 1)
        landmarks = np.concatenate(landmark_parts).reshape(-1, 5, 2) / ratio
        candidates = np.column_stack([boxes, scores])
        return [
            DetectedFace(boxes[index], landmarks[index], float(scores[index]))
            for index in _nms(candidates, 0.4)
        ]


class ArcFaceRecognizer:
    def __init__(
        self,
        model_path: Path,
        device_id: int,
        *,
        color_order: str = "rgb",
        input_mean: float = 127.5,
        input_std: float = 127.5,
    ):
        self.model_path = model_path
        if color_order not in {"rgb", "bgr"}:
            raise ModelError("recognizer_color_order 只能是 rgb 或 bgr")
        if input_std == 0:
            raise ModelError("recognizer_std 不得为 0")
        self.color_order = color_order
        self.input_mean = float(input_mean)
        self.input_std = float(input_std)
        self.session = create_cuda_session(str(model_path), device_id)
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        if len(shape) != 4 or tuple(shape[-2:]) != (112, 112):
            raise ModelError(f"识别模型输入应为 N×3×112×112，实际为 {shape}")
        self.static_batch_size = int(shape[0]) if isinstance(shape[0], int) else None
        if self.static_batch_size is not None and self.static_batch_size < 1:
            raise ModelError(f"识别模型批量维度无效：{shape[0]}")
        digest = hashlib.sha256()
        with model_path.open("rb") as model_file:
            for block in iter(lambda: model_file.read(1024 * 1024), b""):
                digest.update(block)
        preprocessing = f"{color_order}:{self.input_mean}:{self.input_std}"
        preprocessing_hash = hashlib.sha256(preprocessing.encode()).hexdigest()[:8]
        self.model_id = f"onnx-face-112:{digest.hexdigest()[:20]}:{preprocessing_hash}"

    @staticmethod
    def align(frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        transform, _ = cv2.estimateAffinePartial2D(
            np.asarray(landmarks, dtype=np.float32), ARCFACE_TEMPLATE, method=cv2.LMEDS
        )
        if transform is None:
            raise ModelError("无法根据人脸关键点完成对齐。")
        return cv2.warpAffine(frame, transform, (112, 112), borderValue=0)

    def _infer(self, faces: list[np.ndarray]) -> np.ndarray:
        if not faces:
            return np.empty((0, 512), dtype=np.float32)
        prepared = [face[:, :, ::-1] if self.color_order == "rgb" else face for face in faces]
        batch = np.ascontiguousarray(
            np.stack([np.transpose(face.astype(np.float32), (2, 0, 1)) for face in prepared])
        )
        batch = (batch - self.input_mean) / self.input_std
        if self.static_batch_size is None:
            output = np.asarray(
                self.session.run(None, {self.input_name: batch})[0], dtype=np.float32
            ).reshape(len(faces), -1)
        else:
            outputs: list[np.ndarray] = []
            size = self.static_batch_size
            for start in range(0, len(batch), size):
                chunk = batch[start : start + size]
                valid = len(chunk)
                if valid < size:
                    chunk = np.concatenate([chunk, np.repeat(chunk[-1:], size - valid, axis=0)])
                result = np.asarray(
                    self.session.run(None, {self.input_name: chunk})[0], dtype=np.float32
                ).reshape(size, -1)
                outputs.append(result[:valid])
            output = np.concatenate(outputs, axis=0)
        return output

    def embed_aligned(self, faces: list[np.ndarray], *, flip_tta: bool) -> np.ndarray:
        if not faces:
            return np.empty((0, 512), dtype=np.float32)
        if flip_tta:
            flipped = [np.ascontiguousarray(face[:, ::-1]) for face in faces]
            raw = self._infer([*faces, *flipped])
            output = raw[: len(faces)] + raw[len(faces) :]
        else:
            output = self._infer(faces)
        norms = np.linalg.norm(output, axis=1, keepdims=True)
        if np.any(norms <= 1e-8):
            raise ModelError("识别模型产生了无效的零向量。")
        return output / norms


def face_quality(frame: np.ndarray, face: DetectedFace) -> float:
    x1, y1, x2, y2 = face.bbox.astype(int)
    crop = frame[max(0, y1) : max(y1 + 1, y2), max(0, x1) : max(x1 + 1, x2)]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_score = 1.0 - math.exp(-blur_variance / 80.0)
    size_score = min(1.0, min(x2 - x1, y2 - y1) / 120.0)
    left_eye, right_eye, nose = face.landmarks[:3]
    eye_distance = max(float(np.linalg.norm(right_eye - left_eye)), 1.0)
    eye_middle = (left_eye + right_eye) * 0.5
    yaw_proxy = abs(float(nose[0] - eye_middle[0])) / (eye_distance * 0.5)
    pose_score = max(0.15, 1.0 - min(1.0, yaw_proxy))
    quality = (
        0.35 * face.detection_score + 0.30 * blur_score + 0.25 * size_score + 0.10 * pose_score
    )
    return float(np.clip(quality, 0.0, 1.0))


class FaceEngine:
    def __init__(
        self,
        model_dir: Path,
        device_id: int,
        detection_size: tuple[int, int],
        detection_threshold: float,
        min_face_size: int,
        flip_tta: bool,
    ):
        detector_path, recognizer_path = locate_model_files(model_dir)
        metadata_path = model_dir / "model.json"
        metadata: dict[str, Any] = {}
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ModelError(f"无法读取模型配置 model.json：{exc}") from exc
        self.detector = ScrfdDetector(detector_path, device_id, detection_size, detection_threshold)
        self.recognizer = ArcFaceRecognizer(
            recognizer_path,
            device_id,
            color_order=str(metadata.get("recognizer_color_order", "rgb")).lower(),
            input_mean=float(metadata.get("recognizer_mean", 127.5)),
            input_std=float(metadata.get("recognizer_std", 127.5)),
        )
        self.min_face_size = min_face_size
        self.flip_tta = flip_tta
        self._model_id = f"{self.recognizer.model_id}:flip-tta-{int(flip_tta)}"
        self._warm_up()

    @property
    def model_id(self) -> str:
        return self._model_id

    def _warm_up(self) -> None:
        detector_width, detector_height = self.detector.input_size
        detector_blank = np.zeros((detector_height, detector_width, 3), dtype=np.uint8)
        recognizer_blank = np.full((112, 112, 3), 127, dtype=np.uint8)
        try:
            self.detector.detect(detector_blank)
            self.recognizer.embed_aligned([recognizer_blank], flip_tta=self.flip_tta)
        except ModelError:
            raise
        except Exception as exc:
            raise GpuUnavailableError(
                f"检测或识别模型的 CUDA 预热失败；CPU 回退已禁用。详情：{exc}"
            ) from exc

    def analyze(self, frame: np.ndarray) -> list[EmbeddedFace]:
        detections = [
            face
            for face in self.detector.detect(frame)
            if min(face.bbox[2] - face.bbox[0], face.bbox[3] - face.bbox[1]) >= self.min_face_size
        ]
        if not detections:
            return []
        aligned = [self.recognizer.align(frame, face.landmarks) for face in detections]
        embeddings = self.recognizer.embed_aligned(aligned, flip_tta=self.flip_tta)
        return [
            EmbeddedFace(
                bbox=face.bbox,
                landmarks=face.landmarks,
                detection_score=face.detection_score,
                embedding=embedding,
                quality=face_quality(frame, face),
            )
            for face, embedding in zip(detections, embeddings, strict=True)
        ]

    def enroll_image(self, frame: np.ndarray) -> EmbeddedFace:
        faces = self.analyze(frame)
        if not faces:
            raise ModelError("照片中未检测到达到最小尺寸要求的人脸。")
        if len(faces) > 1:
            raise ModelError("照片中检测到多张人脸；注册照必须只包含当前人员的一张人脸。")
        return faces[0]
