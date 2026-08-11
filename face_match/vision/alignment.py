from __future__ import annotations

import numpy as np

ARCFACE_REFERENCE = np.array(
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
    """Least-squares 2-D similarity transform returned as a 2x3 matrix."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("关键点必须是相同形状的 Nx2 数组")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = (target_centered.T @ source_centered) / source.shape[0]
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[-1, -1] = -1
    rotation = u @ correction @ vt
    variance = float(np.sum(source_centered**2) / source.shape[0])
    if variance <= 1e-12:
        raise ValueError("关键点退化，无法对齐")
    scale = float(np.sum(singular * np.diag(correction)) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    matrix = np.concatenate([scale * rotation, translation[:, None]], axis=1)
    return matrix.astype(np.float32)


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
    return homogeneous @ np.asarray(matrix, dtype=np.float32).T


def alignment_error(landmarks: np.ndarray) -> float:
    matrix = similarity_transform(landmarks, ARCFACE_REFERENCE)
    aligned = transform_points(landmarks, matrix)
    return float(np.mean(np.linalg.norm(aligned - ARCFACE_REFERENCE, axis=1)) / 112.0)


def align_face(image: np.ndarray, landmarks: np.ndarray, size: int = 112) -> np.ndarray:
    import cv2

    reference = ARCFACE_REFERENCE * (float(size) / 112.0)
    matrix = similarity_transform(landmarks, reference)
    return cv2.warpAffine(
        image,
        matrix,
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
