from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class FaceQuality:
    overall: float
    detector: float
    resolution: float
    sharpness: float
    frontal: float


def _clip(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def assess_quality(
    aligned_bgr: np.ndarray,
    bbox: np.ndarray,
    landmarks: np.ndarray,
    detector_score: float,
) -> FaceQuality:
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32)
    face_size = max(0.0, min(float(x2 - x1), float(y2 - y1)))
    resolution = _clip((face_size - 24.0) / 88.0)

    gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    low, high = math.log1p(12.0), math.log1p(260.0)
    sharpness = _clip((math.log1p(laplacian_variance) - low) / (high - low))

    points = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
    left_eye, right_eye, nose, left_mouth, right_mouth = points
    eye_distance = max(float(np.linalg.norm(right_eye - left_eye)), 1e-6)
    eye_mid = (left_eye + right_eye) * 0.5
    mouth_mid = (left_mouth + right_mouth) * 0.5
    center_line = mouth_mid - eye_mid
    center_norm = max(float(np.linalg.norm(center_line)), 1e-6)
    nose_offset = nose - eye_mid
    cross_product = center_line[0] * nose_offset[1] - center_line[1] * nose_offset[0]
    lateral = abs(float(cross_product)) / (center_norm * eye_distance)
    eye_ratio = float(np.linalg.norm(nose - left_eye)) / max(float(np.linalg.norm(nose - right_eye)), 1e-6)
    symmetry_penalty = abs(math.log(max(eye_ratio, 1e-6)))
    frontal = _clip(math.exp(-1.8 * lateral - 1.15 * symmetry_penalty))

    detector = _clip((float(detector_score) - 0.35) / 0.60)
    # Geometric mean prevents one bad dimension from being hidden by the rest,
    # while the floor still permits difficult side-view frames into aggregation.
    dimensions = np.array(
        [max(detector, 0.05), max(resolution, 0.05), max(sharpness, 0.05), max(frontal, 0.05)]
    )
    overall = float(np.prod(dimensions ** np.array([0.25, 0.30, 0.25, 0.20])))
    return FaceQuality(overall, detector, resolution, sharpness, frontal)
