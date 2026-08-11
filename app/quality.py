"""人脸质量评估：清晰度（拉普拉斯方差）+ 姿态（关键点估计偏航角）+ 尺寸。

监控视频中的模糊帧和大侧脸帧会显著拉低识别精度，
这里给每张检测到的人脸打质量分，用于：
1. 门控：质量过差不参与比对；
2. 加权：多帧特征融合时按质量加权。
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from . import config


def _yaw_degrees(kps: np.ndarray) -> float:
    """由 5 点关键点粗略估计左右偏航角（度）。

    利用鼻尖到左右眼角距离的不对称性，正面时接近 0。
    """
    left_eye, right_eye, nose = kps[0], kps[1], kps[2]
    eye_dist = np.linalg.norm(left_eye - right_eye) + 1e-6
    ratio = (np.linalg.norm(nose - left_eye) - np.linalg.norm(nose - right_eye)) / eye_dist
    return math.degrees(math.asin(max(-1.0, min(1.0, ratio))))


def face_quality(frame_bgr: np.ndarray, face) -> dict:
    """返回 {quality, size, blur, yaw}。quality ∈ [0,1]，越大越好。"""
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    size = float(min(x2 - x1, y2 - y1))

    blur = 0.0
    if x2 > x1 and y2 > y1:
        crop = frame_bgr[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    yaw = _yaw_degrees(face.kps) if getattr(face, "kps", None) is not None else 90.0

    size_score = min(1.0, max(0.0, (size - config.MIN_FACE_SIZE) / 120.0))
    blur_score = min(1.0, max(0.0, (blur - config.MIN_BLUR_VAR) / 160.0))
    pose_score = min(1.0, max(0.0, 1.0 - abs(yaw) / 60.0))
    quality = float(face.det_score) * (0.4 * size_score + 0.3 * blur_score + 0.3 * pose_score)

    return {"quality": quality, "size": size, "blur": blur, "yaw": yaw}


def passes_gate(q: dict) -> bool:
    """是否达到参与比对的质量门槛。"""
    return (
        q["size"] >= config.MIN_FACE_SIZE
        and q["blur"] >= config.MIN_BLUR_VAR
        and abs(q["yaw"]) <= config.MAX_YAW_DEG
    )
