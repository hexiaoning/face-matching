"""人脸引擎：SCRFD 检测 + GlintR100 识别（antelopev2），强制 GPU 推理。"""
from __future__ import annotations

import threading

import numpy as np

from . import config
from .gpu import assert_on_gpu, resolve_providers


class FaceEngine:
    """封装 insightface FaceAnalysis。

    - 所有模型会话必须运行在 GPU provider 上，否则直接抛错。
    - detect() / embed_photo() 线程安全（内部加锁），GUI 线程与视频线程可共用。
    """

    def __init__(self, backend: str = "auto", model_name: str | None = None):
        self.providers, self.backend = resolve_providers(backend)
        self.model_name = model_name or config.MODEL_NAME

        from insightface.app import FaceAnalysis

        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._app = FaceAnalysis(
            name=self.model_name,
            root=str(config.MODEL_ROOT),
            providers=self.providers,
            allowed_modules=["detection", "recognition"],
        )
        self._app.prepare(ctx_id=0, det_size=config.DET_SIZE)

        # 校验每个模型确实在 GPU 上
        for model in self._app.models.values():
            sess = getattr(model, "session", None)
            if sess is not None:
                assert_on_gpu(sess.get_providers(), self.backend)

        self._lock = threading.Lock()

    def detect(self, frame_bgr: np.ndarray) -> list:
        """检测一帧画面中的所有人脸，返回 insightface Face 列表。"""
        with self._lock:
            return self._app.get(frame_bgr)

    def embed_photo(self, image_bgr: np.ndarray) -> tuple[np.ndarray, object]:
        """从一张登记照中提取特征。

        取画面最大的一张人脸。返回 (512 维归一化特征, Face)。
        未检测到人脸时抛出 ValueError。
        """
        faces = self.detect(image_bgr)
        if not faces:
            raise ValueError("照片中未检测到人脸")
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return face.normed_embedding.astype(np.float32), face
