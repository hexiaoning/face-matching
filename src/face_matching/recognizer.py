from __future__ import annotations

import cv2
import numpy as np

from .gpu import create_gpu_session


def l2_normalize(values: np.ndarray, axis: int = -1) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=axis, keepdims=True)
    if not np.isfinite(values).all() or np.any(norms <= 1e-12):
        raise ValueError("识别模型返回了无效特征")
    return values / np.maximum(norms, 1e-12)


class LVFaceRecognizer:
    def __init__(
        self,
        model_path: str,
        prefer_tensorrt: bool = False,
        device_id: int = 0,
        gpu_backend: str = "auto",
    ) -> None:
        self.session = create_gpu_session(
            model_path, prefer_tensorrt, device_id, gpu_backend
        )
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        self.output_name = self.session.get_outputs()[0].name
        shape = input_meta.shape
        self.fixed_batch = int(shape[0]) if isinstance(shape[0], int) else None
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

    def _infer(self, tensor: np.ndarray) -> np.ndarray:
        if self.fixed_batch is None:
            rows: list[np.ndarray] = []
            for offset in range(0, len(tensor), 32):
                chunk = tensor[offset : offset + 32]
                output = self.session.run([self.output_name], {self.input_name: chunk})[0]
                rows.append(np.asarray(output, dtype=np.float32).reshape(len(chunk), -1))
            return np.concatenate(rows, axis=0)
        if len(tensor) == self.fixed_batch:
            output = self.session.run([self.output_name], {self.input_name: tensor})[0]
            return np.asarray(output, dtype=np.float32).reshape(len(tensor), -1)
        if self.fixed_batch < 1:
            raise ValueError("识别模型批大小无效")
        rows = []
        for offset in range(0, len(tensor), self.fixed_batch):
            chunk = tensor[offset : offset + self.fixed_batch]
            usable = len(chunk)
            if usable < self.fixed_batch:
                padding = np.repeat(chunk[-1:], self.fixed_batch - usable, axis=0)
                chunk = np.concatenate((chunk, padding), axis=0)
            output = self.session.run([self.output_name], {self.input_name: chunk})[0]
            rows.append(
                np.asarray(output, dtype=np.float32).reshape(self.fixed_batch, -1)[:usable]
            )
        return np.concatenate(rows, axis=0)

    def embed_batch(
        self,
        aligned_faces_bgr: list[np.ndarray],
        mirror_augmentation: bool = True,
    ) -> np.ndarray:
        """Return normalized embeddings with optional horizontal-flip TTA.

        Enrollment and video probes use the same augmentation. The additional
        recognizer work is batched on dynamic graphs and improves robustness to
        pose/asymmetry without inventing identity details.
        """
        if not aligned_faces_bgr:
            width = self.embedding_size or 0
            return np.empty((0, width), dtype=np.float32)
        inference_faces: list[np.ndarray] = []
        for face in aligned_faces_bgr:
            inference_faces.append(face)
            if mirror_augmentation:
                inference_faces.append(np.ascontiguousarray(face[:, ::-1]))
        output = self._infer(self.preprocess(inference_faces))
        if not mirror_augmentation:
            return l2_normalize(output)
        fused = output[0::2] + output[1::2]
        return l2_normalize(fused)


def _output_dimension(shape: list[object]) -> int | None:
    for dimension in reversed(shape):
        if isinstance(dimension, int) and dimension > 1:
            return dimension
    return None
