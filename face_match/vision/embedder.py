from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from face_match.errors import FaceMatchError
from face_match.gpu import create_cuda_session


class LvFaceEmbedder:
    def __init__(self, model_path: Path) -> None:
        self.session = create_cuda_session(model_path)
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.output_name = self.session.get_outputs()[0].name
        self._dynamic_batch = not isinstance(model_input.shape[0], int) or model_input.shape[0] != 1
        probe = self.embed([np.zeros((112, 112, 3), dtype=np.uint8)])
        if probe.shape[0] != 1 or probe.shape[1] < 128:
            raise FaceMatchError(f"识别模型输出形状异常：{probe.shape}")
        self.embedding_dimension = int(probe.shape[1])

    @staticmethod
    def _preprocess(faces_bgr: Sequence[np.ndarray]) -> np.ndarray:
        import cv2

        tensors: list[np.ndarray] = []
        for face in faces_bgr:
            if face.shape[:2] != (112, 112):
                face = cv2.resize(face, (112, 112), interpolation=cv2.INTER_LINEAR)
            rgb = face[:, :, ::-1].astype(np.float32)
            tensors.append(((rgb / 127.5) - 1.0).transpose(2, 0, 1))
        return np.ascontiguousarray(np.stack(tensors), dtype=np.float32)

    def embed(self, faces_bgr: Sequence[np.ndarray]) -> np.ndarray:
        if not faces_bgr:
            dimension = getattr(self, "embedding_dimension", 0)
            return np.empty((0, dimension), dtype=np.float32)
        tensor = self._preprocess(faces_bgr)
        if self._dynamic_batch or len(faces_bgr) == 1:
            output = self.session.run([self.output_name], {self.input_name: tensor})[0]
            vectors = np.asarray(output, dtype=np.float32).reshape(len(faces_bgr), -1)
        else:
            vectors = np.concatenate(
                [
                    np.asarray(
                        self.session.run([self.output_name], {self.input_name: item[None, ...]})[0],
                        dtype=np.float32,
                    ).reshape(1, -1)
                    for item in tensor
                ],
                axis=0,
            )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms <= 1e-8) or not np.isfinite(vectors).all():
            raise FaceMatchError("识别模型产生了无效特征")
        return vectors / norms
