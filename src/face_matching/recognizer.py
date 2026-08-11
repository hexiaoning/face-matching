from __future__ import annotations

import cv2
import numpy as np

from .gpu import create_gpu_session


def l2_normalize(values: np.ndarray, axis: int = -1) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=axis, keepdims=True)
    return values / np.maximum(norms, 1e-12)


class LVFaceRecognizer:
    def __init__(self, model_path: str, prefer_tensorrt: bool = False) -> None:
        self.session = create_gpu_session(model_path, prefer_tensorrt)
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        self.output_name = self.session.get_outputs()[0].name
        shape = input_meta.shape
        self.dynamic_batch = not isinstance(shape[0], int) or shape[0] != 1
        self.embedding_size = _output_dimension(self.session.get_outputs()[0].shape)

    @staticmethod
    def preprocess(aligned_faces_bgr: list[np.ndarray]) -> np.ndarray:
        tensors = []
        for face in aligned_faces_bgr:
            if face.shape[:2] != (112, 112):
                face = cv2.resize(face, (112, 112), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            tensor = np.transpose(rgb, (2, 0, 1)).astype(np.float32)
            tensors.append((tensor - 127.5) / 127.5)
        return np.stack(tensors, axis=0)

    def embed_batch(self, aligned_faces_bgr: list[np.ndarray]) -> np.ndarray:
        if not aligned_faces_bgr:
            width = self.embedding_size or 0
            return np.empty((0, width), dtype=np.float32)
        if self.dynamic_batch:
            output = self.session.run(
                [self.output_name], {self.input_name: self.preprocess(aligned_faces_bgr)}
            )[0]
            return l2_normalize(np.asarray(output).reshape(len(aligned_faces_bgr), -1))
        embeddings = []
        for face in aligned_faces_bgr:
            output = self.session.run(
                [self.output_name], {self.input_name: self.preprocess([face])}
            )[0]
            embeddings.append(np.asarray(output).reshape(-1))
        return l2_normalize(np.stack(embeddings, axis=0))


def _output_dimension(shape: list[object]) -> int | None:
    for dimension in reversed(shape):
        if isinstance(dimension, int) and dimension > 1:
            return dimension
    return None
