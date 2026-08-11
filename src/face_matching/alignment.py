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


def align_face(frame_bgr: np.ndarray, landmarks: np.ndarray, size: int = 112) -> np.ndarray:
    landmarks = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
    template = ARCFACE_TEMPLATE * (size / 112.0)
    transform, _ = cv2.estimateAffinePartial2D(landmarks, template, method=cv2.LMEDS)
    if transform is None:
        raise ValueError("could not estimate five-point face alignment")
    return cv2.warpAffine(
        frame_bgr,
        transform,
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
