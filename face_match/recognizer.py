"""人脸识别器：5 点对齐到 112x112 + ArcFace/AdaFace 风格 ONNX 提 512 维特征。"""
from __future__ import annotations

import cv2
import numpy as np

# ArcFace 标准 5 点参考位置（112x112）
ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def align_face(img: np.ndarray, kps: np.ndarray, out_size: int = 112) -> np.ndarray:
    """用相似变换把人脸对齐到标准位置。"""
    if out_size != 112:
        dst = ARCFACE_DST * (out_size / 112.0)
    else:
        dst = ARCFACE_DST
    M, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), dst, method=cv2.LMEDS)
    if M is None:  # 关键点退化时退化为按框缩放，调用方应避免
        M = np.array([[1.0, 0, 0], [0, 1.0, 0]], dtype=np.float32)
    return cv2.warpAffine(img, M, (out_size, out_size), borderValue=0.0)


class FaceRecognizer:
    """支持 InsightFace 格式的识别 ONNX（输入 112x112，输出 512 维）。

    AdaFace 等模型可导出为同样格式后直接替换模型文件使用。
    """

    def __init__(self, model_path: str, providers: list[str] | None = None):
        import onnxruntime as ort

        providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_size = int(inp.shape[-1]) if isinstance(inp.shape[-1], int) else 112
        self.output_name = self.session.get_outputs()[0].name

    def _preprocess(self, aligned: np.ndarray) -> np.ndarray:
        blob = ((aligned.astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
        return blob

    def embed_aligned(self, aligned: np.ndarray) -> np.ndarray:
        """对已对齐的 112x112 人脸提特征，返回 L2 归一化的 embedding。"""
        out = self.session.run([self.output_name], {self.input_name: self._preprocess(aligned)})[0]
        emb = out[0].astype(np.float32)
        n = np.linalg.norm(emb)
        return emb / n if n > 0 else emb

    def embed(self, img: np.ndarray, kps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """原图 + 5 点关键点 → (embedding, 对齐后的人脸图)。"""
        aligned = align_face(img, kps, self.input_size)
        return self.embed_aligned(aligned), aligned
