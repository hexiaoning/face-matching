"""人脸质量评估：清晰度（模糊）、姿态（正脸程度）、尺寸 → 综合质量分。

监控视频中的模糊帧和侧脸帧会严重拉低识别准确率，因此每一帧先算质量分：
- 低于阈值的帧不参与识别；
- track 内多帧融合时按质量加权。

分数均在 0~1，越大越好。
"""
from __future__ import annotations

import cv2
import numpy as np

from . import config
from .detector import FaceDet


def blur_score(aligned_face: np.ndarray) -> float:
    """Laplacian 方差归一化到 0~1。aligned_face 为对齐后的人脸图。"""
    gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
    var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return min(1.0, var / config.BLUR_REF)


def pose_angles(kps: np.ndarray) -> tuple[float, float]:
    """由 5 点关键点粗估 yaw / pitch（度）。

    yaw: 鼻尖到左右眼距离的不对称程度；pitch: 鼻尖与眼/嘴连线的纵向比例。
    """
    le, re, nose, lm, rm = kps[0], kps[1], kps[2], kps[3], kps[4]
    eye_span = float(np.linalg.norm(le - re)) + 1e-6
    d_l = float(np.linalg.norm(nose - le))
    d_r = float(np.linalg.norm(nose - re))
    yaw_ratio = abs(d_l - d_r) / eye_span          # 0=正脸
    yaw = min(90.0, yaw_ratio * 90.0)

    eye_mid = (le + re) / 2.0
    mouth_mid = (lm + rm) / 2.0
    full = float(np.linalg.norm(eye_mid - mouth_mid)) + 1e-6
    upper = float(np.linalg.norm(nose - eye_mid))
    pitch_ratio = upper / full                      # 正脸约 0.55~0.7
    pitch = min(60.0, abs(pitch_ratio - 0.6) * 120.0)
    return yaw, pitch


def pose_score(kps: np.ndarray) -> float:
    yaw, pitch = pose_angles(kps)
    yaw_s = max(0.0, 1.0 - yaw / config.MAX_YAW_DEG)
    pitch_s = max(0.0, 1.0 - pitch / 45.0)
    return min(yaw_s, pitch_s)


def size_score(det: FaceDet) -> float:
    """人脸越大越好，128px 以上视为满分。"""
    return min(1.0, det.size / 128.0)


def quality_score(det: FaceDet, aligned_face: np.ndarray | None = None) -> float:
    """综合质量分 0~1。"""
    s_det = min(1.0, det.score / 0.8)
    s_size = size_score(det)
    s_pose = pose_score(det.kps) if det.kps is not None and det.kps.any() else 0.5
    s_blur = blur_score(aligned_face) if aligned_face is not None else 0.5
    # 模糊和姿态是监控场景的主要杀手，权重给高
    q = 0.15 * s_det + 0.15 * s_size + 0.35 * s_pose + 0.35 * s_blur
    return float(max(0.0, min(1.0, q)))


def usable(det: FaceDet, min_quality: float = 0.2,
           aligned_face: np.ndarray | None = None) -> bool:
    """该人脸是否值得参与识别（硬过滤）。"""
    if det.size < config.MIN_FACE_SIZE:
        return False
    yaw, _ = pose_angles(det.kps) if det.kps is not None and det.kps.any() else (0.0, 0.0)
    if yaw > config.MAX_YAW_DEG:
        return False
    if aligned_face is not None and quality_score(det, aligned_face) < min_quality:
        return False
    return True
