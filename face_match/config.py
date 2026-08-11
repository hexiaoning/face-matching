"""全局配置：路径与可调参数的默认值。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "FaceMatch"


def app_dir() -> Path:
    """程序所在目录（兼容 PyInstaller 打包）。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    """数据目录：模型、数据库、照片。可用 FACEMATCH_DATA_DIR 覆盖。"""
    env = os.environ.get("FACEMATCH_DATA_DIR")
    base = Path(env) if env else app_dir() / "data"
    base.mkdir(parents=True, exist_ok=True)
    return base


def models_dir() -> Path:
    d = data_dir() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def photos_dir() -> Path:
    d = data_dir() / "photos"
    d.mkdir(parents=True, exist_ok=True)
    return d


def db_path() -> Path:
    return data_dir() / "face_match.db"


# ---- 模型 ----
BUFFALO_L_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
DETECTOR_MODEL = "det_10g.onnx"       # SCRFD 10G
RECOGNIZER_MODEL = "w600k_r50.onnx"   # ArcFace ResNet50 @ WebFace600K
DET_INPUT_SIZE = 640

# ---- 质量评估默认值 ----
MIN_FACE_SIZE = 32        # 人脸框短边低于该像素直接忽略
MAX_YAW_DEG = 55.0        # 姿态 yaw 超过该角度的帧质量分打折扣
BLUR_REF = 80.0           # Laplacian 方差达到该值视为清晰

# ---- 视频处理默认值 ----
FRAME_SKIP = 3            # 每处理 1 帧跳过几帧（GUI 可调）
TRACK_TOP_K = 8           # 每个 track 融合质量最高的 K 帧
TRACK_MIN_SAMPLES = 2     # track 至少积累几帧才输出识别结果
TRACK_MAX_AGE = 15        # 丢失多少帧后删除 track

# ---- 匹配 ----
MATCH_THRESHOLD = 0.35    # 余弦相似度阈值（GUI 可调，典型范围 0.25~0.5）

# ---- 性能 ----
DET_INTERVAL = 1          # 每 N 个采样帧做一次检测（其余帧复用上次结果）
