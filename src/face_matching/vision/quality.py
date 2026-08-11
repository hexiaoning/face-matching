from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .detector import Detection


@dataclass(frozen=True, slots=True)
class Quality:
    total: float
    blur: float
    pose: float
    size: float
    illumination: float
    detection: float


def _clamp(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def assess_quality(aligned_face: np.ndarray, detection: Detection) -> Quality:
    gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    blur = _clamp(laplacian_variance / (laplacian_variance + 100.0))

    points = np.asarray(detection.landmarks, dtype=np.float32)
    left_eye, right_eye, nose, left_mouth, right_mouth = points
    eye_distance = max(float(np.linalg.norm(right_eye - left_eye)), 1e-6)
    nose_position = float((nose[0] - left_eye[0]) / max(right_eye[0] - left_eye[0], 1e-6))
    yaw_penalty = min(abs(nose_position - 0.5) / 0.55, 1.0)
    eye_tilt = min(abs(float(right_eye[1] - left_eye[1])) / (eye_distance * 0.35), 1.0)
    mouth_center = (left_mouth + right_mouth) * 0.5
    vertical_ok = _clamp(float((mouth_center[1] - nose[1]) / eye_distance) / 0.75)
    pose = _clamp((1.0 - yaw_penalty) * 0.65 + (1.0 - eye_tilt) * 0.2 + vertical_ok * 0.15)

    face_size = min(detection.width, detection.height)
    size_score = _clamp((face_size - 20.0) / 80.0)
    mean = float(gray.mean())
    illumination = _clamp(1.0 - abs(mean - 127.5) / 127.5)
    detection_score = _clamp((detection.score - 0.45) / 0.55)
    total = _clamp(
        0.28 * blur
        + 0.27 * pose
        + 0.20 * size_score
        + 0.15 * detection_score
        + 0.10 * illumination
    )
    return Quality(total, blur, pose, size_score, illumination, detection_score)
