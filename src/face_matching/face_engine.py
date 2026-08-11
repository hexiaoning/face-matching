from __future__ import annotations

import math
import threading
from collections.abc import Sequence

import cv2
import numpy as np

from .domain import FaceObservation, ModelPaths, normalize_embedding
from .gpu import create_cuda_session


class FaceEngineError(RuntimeError):
    pass


def _distance_to_bbox(points: np.ndarray, distances: np.ndarray) -> np.ndarray:
    x1 = points[:, 0] - distances[:, 0]
    y1 = points[:, 1] - distances[:, 1]
    x2 = points[:, 0] + distances[:, 2]
    y2 = points[:, 1] + distances[:, 3]
    return np.stack((x1, y1, x2, y2), axis=-1)


def _distance_to_landmarks(points: np.ndarray, distances: np.ndarray) -> np.ndarray:
    landmarks = np.empty_like(distances, dtype=np.float32)
    for index in range(0, distances.shape[1], 2):
        landmarks[:, index] = points[:, 0] + distances[:, index]
        landmarks[:, index + 1] = points[:, 1] + distances[:, index + 1]
    return landmarks


def _nms(boxes: np.ndarray, threshold: float) -> list[int]:
    if boxes.size == 0:
        return []
    x1, y1, x2, y2, scores = (boxes[:, index] for index in range(5))
    areas = np.maximum(0.0, x2 - x1 + 1) * np.maximum(0.0, y2 - y1 + 1)
    order = scores.argsort()[::-1]
    kept: list[int] = []
    while order.size:
        current = int(order[0])
        kept.append(current)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[current], x1[order[1:]])
        yy1 = np.maximum(y1[current], y1[order[1:]])
        xx2 = np.minimum(x2[current], x2[order[1:]])
        yy2 = np.minimum(y2[current], y2[order[1:]])
        width = np.maximum(0.0, xx2 - xx1 + 1)
        height = np.maximum(0.0, yy2 - yy1 + 1)
        overlap = width * height
        union = areas[current] + areas[order[1:]] - overlap
        iou = np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)
        order = order[np.where(iou <= threshold)[0] + 1]
    return kept


class SCRFDDetector:
    def __init__(self, session, input_size: int = 960) -> None:
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.output_names = [output.name for output in session.get_outputs()]
        output_count = len(self.output_names)
        if output_count in (6, 9):
            self.strides = (8, 16, 32)
            self.use_landmarks = output_count == 9
        elif output_count in (10, 15):
            self.strides = (8, 16, 32, 64, 128)
            self.use_landmarks = output_count == 15
        else:
            raise FaceEngineError(f"不支持的 SCRFD 输出数量：{output_count}")
        if not self.use_landmarks:
            raise FaceEngineError("检测模型不输出 5 点关键点，无法进行可靠的人脸对齐。")
        self.feature_levels = len(self.strides)
        input_shape = session.get_inputs()[0].shape
        fixed_height = input_shape[2] if len(input_shape) == 4 else None
        fixed_width = input_shape[3] if len(input_shape) == 4 else None
        if isinstance(fixed_height, int) and isinstance(fixed_width, int):
            self.input_height, self.input_width = fixed_height, fixed_width
        else:
            self.input_height = self.input_width = int(input_size)
        self._anchor_cache: dict[tuple[int, int, int, int], np.ndarray] = {}

    def detect(
        self, image: np.ndarray, score_threshold: float = 0.5, nms_threshold: float = 0.4
    ) -> list[FaceObservation]:
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("输入必须是 BGR 三通道图像")
        image_height, image_width = image.shape[:2]
        scale = min(self.input_width / image_width, self.input_height / image_height)
        resized_width = max(1, int(round(image_width * scale)))
        resized_height = max(1, int(round(image_height * scale)))
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8)
        canvas[:resized_height, :resized_width] = resized
        blob = cv2.dnn.blobFromImage(
            canvas,
            scalefactor=1.0 / 128.0,
            size=(self.input_width, self.input_height),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        )
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        score_groups: list[np.ndarray] = []
        bbox_groups: list[np.ndarray] = []
        landmark_groups: list[np.ndarray] = []

        for level, stride in enumerate(self.strides):
            scores = np.asarray(outputs[level]).reshape(-1)
            bbox_distances = np.asarray(outputs[level + self.feature_levels]).reshape(-1, 4)
            landmark_distances = np.asarray(
                outputs[level + self.feature_levels * 2]
            ).reshape(-1, 10)
            feature_height = self.input_height // stride
            feature_width = self.input_width // stride
            locations = feature_height * feature_width
            if locations <= 0 or bbox_distances.shape[0] % locations:
                raise FaceEngineError("SCRFD 输出形状与检测输入不匹配")
            anchor_count = bbox_distances.shape[0] // locations
            if scores.size != bbox_distances.shape[0]:
                # A few exported detectors retain two class scores; the final channel is face.
                if scores.size == bbox_distances.shape[0] * 2:
                    scores = scores.reshape(-1, 2)[:, -1]
                else:
                    raise FaceEngineError("SCRFD 分类与边框输出数量不一致")

            cache_key = (feature_height, feature_width, stride, anchor_count)
            centers = self._anchor_cache.get(cache_key)
            if centers is None:
                grid_x, grid_y = np.meshgrid(
                    np.arange(feature_width, dtype=np.float32),
                    np.arange(feature_height, dtype=np.float32),
                )
                centers = np.stack((grid_x, grid_y), axis=-1).reshape(-1, 2) * stride
                if anchor_count > 1:
                    centers = np.repeat(centers, anchor_count, axis=0)
                self._anchor_cache[cache_key] = centers

            positive = np.where(scores >= score_threshold)[0]
            if positive.size == 0:
                continue
            predicted_boxes = _distance_to_bbox(
                centers, bbox_distances.astype(np.float32, copy=False) * stride
            )
            predicted_landmarks = _distance_to_landmarks(
                centers, landmark_distances.astype(np.float32, copy=False) * stride
            ).reshape(-1, 5, 2)
            score_groups.append(scores[positive].astype(np.float32, copy=False))
            bbox_groups.append(predicted_boxes[positive])
            landmark_groups.append(predicted_landmarks[positive])

        if not score_groups:
            return []
        scores = np.concatenate(score_groups)
        boxes = np.concatenate(bbox_groups)
        landmarks = np.concatenate(landmark_groups)
        scale_x = resized_width / image_width
        scale_y = resized_height / image_height
        boxes[:, (0, 2)] /= scale_x
        boxes[:, (1, 3)] /= scale_y
        landmarks[:, :, 0] /= scale_x
        landmarks[:, :, 1] /= scale_y
        boxes[:, (0, 2)] = np.clip(boxes[:, (0, 2)], 0, image_width - 1)
        boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)], 0, image_height - 1)
        order = scores.argsort()[::-1]
        candidates = np.hstack((boxes[order], scores[order, None]))
        kept = _nms(candidates, nms_threshold)
        observations: list[FaceObservation] = []
        for ordered_index in kept:
            original_index = int(order[ordered_index])
            box = boxes[original_index].astype(np.float32)
            if box[2] - box[0] < 2 or box[3] - box[1] < 2:
                continue
            observations.append(
                FaceObservation(
                    bbox=box,
                    landmarks=landmarks[original_index].astype(np.float32),
                    detection_score=float(scores[original_index]),
                )
            )
        return observations


_ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def similarity_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the least-squares 2D similarity transform from source to target."""

    source = np.asarray(source, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 2)
    if source.shape != target.shape or source.shape[0] < 2:
        raise ValueError("关键点形状无效")
    matrix = np.zeros((source.shape[0] * 2, 4), dtype=np.float64)
    values = target.reshape(-1)
    for index, (x_coord, y_coord) in enumerate(source):
        matrix[index * 2] = (x_coord, -y_coord, 1.0, 0.0)
        matrix[index * 2 + 1] = (y_coord, x_coord, 0.0, 1.0)
    coefficients, _, _, _ = np.linalg.lstsq(matrix, values, rcond=None)
    scale_cos, scale_sin, translate_x, translate_y = coefficients
    return np.array(
        [
            [scale_cos, -scale_sin, translate_x],
            [scale_sin, scale_cos, translate_y],
        ],
        dtype=np.float32,
    )


def align_face(image: np.ndarray, landmarks: np.ndarray, output_size: int = 112) -> np.ndarray:
    template = _ARCFACE_TEMPLATE.copy()
    if output_size != 112:
        template *= output_size / 112.0
    transform = similarity_transform(landmarks, template)
    return cv2.warpAffine(
        image,
        transform,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


class ArcFaceRecognizer:
    def __init__(self, session) -> None:
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name
        shape = session.get_inputs()[0].shape
        self.fixed_batch_size = int(shape[0]) if isinstance(shape[0], int) else None
        self.input_height = int(shape[2]) if isinstance(shape[2], int) else 112
        self.input_width = int(shape[3]) if isinstance(shape[3], int) else 112

    def embed_aligned(self, faces: Sequence[np.ndarray]) -> list[np.ndarray]:
        if not faces:
            return []
        batches: list[np.ndarray] = []
        chunk_size = self.fixed_batch_size or len(faces)
        for offset in range(0, len(faces), chunk_size):
            chunk = list(faces[offset : offset + chunk_size])
            actual_size = len(chunk)
            if self.fixed_batch_size and actual_size < self.fixed_batch_size:
                chunk.extend([chunk[-1]] * (self.fixed_batch_size - actual_size))
            blob = cv2.dnn.blobFromImages(
                chunk,
                scalefactor=1.0 / 127.5,
                size=(self.input_width, self.input_height),
                mean=(127.5, 127.5, 127.5),
                swapRB=True,
            )
            output = np.asarray(
                self.session.run([self.output_name], {self.input_name: blob})[0],
                dtype=np.float32,
            ).reshape(len(chunk), -1)
            batches.append(output[:actual_size])
        features = np.concatenate(batches, axis=0)
        return [normalize_embedding(vector) for vector in features]


def face_quality(image: np.ndarray, observation: FaceObservation) -> float:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = observation.bbox
    ix1, iy1 = max(0, int(x1)), max(0, int(y1))
    ix2, iy2 = min(width, int(math.ceil(x2))), min(height, int(math.ceil(y2)))
    if ix2 - ix1 < 4 or iy2 - iy1 < 4:
        return 0.0
    crop = image[iy1:iy2, ix1:ix2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    variance = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    sharpness = float(np.clip(math.log1p(variance) / math.log1p(900.0), 0.0, 1.0))
    face_size = min(ix2 - ix1, iy2 - iy1)
    size_score = float(np.clip((face_size - 20.0) / 100.0, 0.0, 1.0))
    brightness = float(gray.mean())
    exposure = float(np.clip(1.0 - abs(brightness - 127.5) / 127.5, 0.0, 1.0))

    landmarks = observation.landmarks
    left_eye, right_eye, nose, left_mouth, right_mouth = landmarks
    eye_distance = max(float(np.linalg.norm(right_eye - left_eye)), 1e-6)
    left_span = float(np.linalg.norm(nose - left_eye))
    right_span = float(np.linalg.norm(nose - right_eye))
    yaw_balance = min(left_span, right_span) / max(left_span, right_span, 1e-6)
    eye_midpoint = (left_eye + right_eye) * 0.5
    mouth_midpoint = (left_mouth + right_mouth) * 0.5
    vertical_ratio = float(np.linalg.norm(nose - eye_midpoint) / eye_distance)
    lower_ratio = float(np.linalg.norm(mouth_midpoint - nose) / eye_distance)
    pitch_score = float(
        np.clip(1.0 - abs(vertical_ratio - 0.45) - 0.5 * abs(lower_ratio - 0.35), 0.0, 1.0)
    )
    pose_score = float(np.clip(0.65 * yaw_balance + 0.35 * pitch_score, 0.0, 1.0))
    detection = float(np.clip(observation.detection_score, 0.0, 1.0))
    quality = (
        0.32 * sharpness
        + 0.24 * size_score
        + 0.22 * pose_score
        + 0.12 * exposure
        + 0.10 * detection
    )
    return float(np.clip(quality, 0.0, 1.0))


class FaceEngine:
    """GPU-only detector and recognizer. ONNX Runtime sessions never receive a CPU provider."""

    def __init__(self, ort, models: ModelPaths, detector_size: int = 960) -> None:
        self._lock = threading.RLock()
        detector_session = create_cuda_session(ort, str(models.detector))
        recognizer_session = create_cuda_session(ort, str(models.recognizer))
        self.detector = SCRFDDetector(detector_session, detector_size)
        self.recognizer = ArcFaceRecognizer(recognizer_session)
        self.model_name = models.model_name

    def detect(self, image: np.ndarray, threshold: float = 0.5) -> list[FaceObservation]:
        with self._lock:
            observations = self.detector.detect(image, score_threshold=threshold)
        for observation in observations:
            observation.quality = face_quality(image, observation)
        return observations

    def add_embeddings(
        self, image: np.ndarray, observations: Sequence[FaceObservation]
    ) -> list[FaceObservation]:
        if not observations:
            return list(observations)
        aligned = [
            align_face(
                image,
                observation.landmarks,
                output_size=self.recognizer.input_width,
            )
            for observation in observations
        ]
        with self._lock:
            embeddings = self.recognizer.embed_aligned(aligned)
        for observation, embedding in zip(observations, embeddings, strict=True):
            observation.embedding = embedding
        return list(observations)

    def analyze(self, image: np.ndarray, threshold: float = 0.5) -> list[FaceObservation]:
        observations = self.detect(image, threshold)
        return self.add_embeddings(image, observations)

    def set_detector_size(self, size: int) -> None:
        if size not in {640, 960, 1280}:
            raise ValueError("检测尺寸只能是 640、960 或 1280")
        # AntelopeV2 SCRFD has dynamic spatial inputs. Fixed exports stay fixed.
        input_shape = self.detector.session.get_inputs()[0].shape
        fixed_height = input_shape[2] if len(input_shape) == 4 else None
        fixed_width = input_shape[3] if len(input_shape) == 4 else None
        if isinstance(fixed_height, int) and isinstance(fixed_width, int):
            if size != fixed_height or size != fixed_width:
                raise ValueError(f"当前检测模型固定为 {fixed_width} × {fixed_height}")
            return
        self.detector.input_height = size
        self.detector.input_width = size
        self.detector._anchor_cache.clear()
