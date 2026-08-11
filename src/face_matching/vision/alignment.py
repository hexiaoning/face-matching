from __future__ import annotations

import cv2
import numpy as np


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


def estimate_similarity_transform(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float32).reshape(-1, 2)
    destination = np.asarray(destination, dtype=np.float32).reshape(-1, 2)
    if source.shape != destination.shape or source.shape[0] < 2:
        raise ValueError("源点与目标点必须是相同形状的二维坐标")
    matrix, _ = cv2.estimateAffinePartial2D(source, destination, method=cv2.LMEDS)
    if matrix is None or not np.isfinite(matrix).all():
        raise ValueError("无法根据人脸关键点计算对齐变换")
    return matrix.astype(np.float32)


def align_face(image: np.ndarray, landmarks: np.ndarray, size: int = 112) -> np.ndarray:
    if size != 112:
        destination = ARCFACE_TEMPLATE * (float(size) / 112.0)
    else:
        destination = ARCFACE_TEMPLATE
    matrix = estimate_similarity_transform(landmarks, destination)
    return cv2.warpAffine(
        image,
        matrix,
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def align_face_variants(
    image: np.ndarray,
    landmarks: np.ndarray,
    horizontal_offsets: tuple[float, ...] = (-4.0, 0.0, 4.0),
) -> list[np.ndarray]:
    """Build nearby alignments that tolerate noisy surveillance landmarks."""
    variants: list[np.ndarray] = []
    for offset in horizontal_offsets:
        destination = ARCFACE_TEMPLATE.copy()
        destination[:, 0] += float(offset)
        matrix = estimate_similarity_transform(landmarks, destination)
        variants.append(
            cv2.warpAffine(
                image,
                matrix,
                (112, 112),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        )
    return variants
