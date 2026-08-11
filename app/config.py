"""全局配置。"""
import os
from pathlib import Path

# ---- 路径 ----
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "faces.db"
PHOTO_DIR = DATA_DIR / "photos"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
MODEL_ROOT = DATA_DIR / "models"  # insightface 模型缓存目录

# 让 insightface 把模型下载到项目内（一键部署、可离线拷贝）
os.environ.setdefault("INSIGHTFACE_HOME", str(MODEL_ROOT))

# ---- 模型 ----
# antelopev2: SCRFD 检测 + GlintR100 识别，监控/非约束场景开源最强
# 备选 "buffalo_l"（更小更快，精度略低）
MODEL_NAME = "antelopev2"
DET_SIZE = (640, 640)

# GPU 后端: "cuda" | "directml" | "auto"
# auto: 优先 CUDA，不可用则 DirectML，再不行报错退出（绝不用 CPU）
GPU_BACKEND = os.environ.get("FACE_GPU_BACKEND", "auto")

# ---- 质量门控（监控画面模糊/侧脸过滤）----
MIN_FACE_SIZE = 48        # 人脸最小边（像素），小于此不参与比对
MIN_BLUR_VAR = 60.0       # 拉普拉斯方差，低于此视为模糊
MAX_YAW_DEG = 45.0        # 允许的最大侧脸角度
MIN_DET_SCORE = 0.5       # 检测置信度

# ---- 跟踪与多帧融合 ----
DET_STRIDE = 2            # 每隔几帧检测一次（中间的帧沿用跟踪框）
IOU_THRESHOLD = 0.3       # 跟踪匹配 IoU
MAX_MISSED = 12           # 跟踪目标最多消失多少帧后删除
TOP_K_SAMPLES = 8         # 每个跟踪目标保留质量最高的 K 个特征做融合
MIN_SAMPLES_TO_MATCH = 3  # 至少积累多少个有效样本才开始比对
REMATCH_INTERVAL = 30     # 已命中目标每隔多少帧重新融合比对一次（结果可随样本增多而更新）

# ---- 比对 ----
DEFAULT_THRESHOLD = 0.45  # 余弦相似度阈值（antelopev2 建议 0.4~0.5，可在界面调）
