from __future__ import annotations

import math

import numpy as np

from face_match.domain import FaceDetection, FaceQuality
from face_match.vision.alignment import alignment_error


def assess_face_quality(
    aligned_bgr: np.ndarray, detection: FaceDetection, frame_shape: tuple[int, ...]
) -> FaceQuality:
    import cv2

    gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness = float(np.clip(1.0 - math.exp(-laplacian_variance / 90.0), 0.0, 1.0))

    try:
        residual = alignment_error(detection.landmarks)
        pose = float(np.clip(math.exp(-32.0 * residual), 0.0, 1.0))
    except (ValueError, np.linalg.LinAlgError):
        pose = 0.0

    width = max(0.0, float(detection.bbox[2] - detection.bbox[0]))
    height = max(0.0, float(detection.bbox[3] - detection.bbox[1]))
    short_side = min(width, height)
    frame_short_side = float(min(frame_shape[0], frame_shape[1]))
    relative_size = short_side / max(frame_short_side, 1.0)
    absolute_resolution = float(np.clip((short_side - 32.0) / 96.0, 0.0, 1.0))
    relative_resolution = float(np.clip((relative_size - 0.035) / 0.18, 0.0, 1.0))
    resolution = max(absolute_resolution, relative_resolution)

    mean = float(gray.mean())
    contrast = float(gray.std())
    exposure = math.exp(-(((mean - 128.0) / 100.0) ** 2))
    contrast_term = float(np.clip(contrast / 48.0, 0.0, 1.0))
    illumination = float(np.clip(0.65 * exposure + 0.35 * contrast_term, 0.0, 1.0))

    detector_quality = float(np.clip((detection.score - 0.45) / 0.5, 0.0, 1.0))
    overall = (
        0.28 * sharpness
        + 0.25 * pose
        + 0.25 * resolution
        + 0.12 * illumination
        + 0.10 * detector_quality
    )
    return FaceQuality(float(np.clip(overall, 0.0, 1.0)), sharpness, pose, resolution, illumination)
