from __future__ import annotations

import cv2
import numpy as np


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("识别模型返回了无效的零向量")
    return value / norm


class FaceEmbedder:
    def __init__(self, session: object) -> None:
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name
        shape = session.get_inputs()[0].shape
        self.size = int(shape[-1]) if isinstance(shape[-1], int) else 112
        self.fixed_batch = int(shape[0]) if isinstance(shape[0], int) else None

    def _tensor(self, aligned_bgr: np.ndarray) -> np.ndarray:
        if aligned_bgr.shape[:2] != (self.size, self.size):
            aligned_bgr = cv2.resize(aligned_bgr, (self.size, self.size))
        rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        tensor = np.transpose(rgb, (2, 0, 1)).astype(np.float32)
        return (tensor - 127.5) / 127.5

    def _infer(self, tensors: np.ndarray) -> np.ndarray:
        if self.fixed_batch is None or len(tensors) == self.fixed_batch:
            output = self.session.run([self.output_name], {self.input_name: tensors})[0]
            return np.asarray(output, dtype=np.float32).reshape(len(tensors), -1)
        batch_size = self.fixed_batch
        if batch_size is None or batch_size < 1:
            raise ValueError("识别模型的批大小无效")
        rows: list[np.ndarray] = []
        for start in range(0, len(tensors), batch_size):
            chunk = tensors[start : start + batch_size]
            usable = len(chunk)
            if usable < batch_size:
                padding = np.repeat(chunk[-1:], batch_size - usable, axis=0)
                chunk = np.concatenate((chunk, padding), axis=0)
            output = self.session.run([self.output_name], {self.input_name: chunk})[0]
            rows.extend(np.asarray(output, dtype=np.float32).reshape(batch_size, -1)[:usable])
        return np.vstack(rows)

    def embed_many(
        self,
        aligned_faces: list[np.ndarray],
        mirror_augmentation: bool = True,
    ) -> list[np.ndarray]:
        """Extract normalized embeddings, batching when the ONNX graph permits it.

        Horizontal-flip test-time augmentation is deliberately applied to both
        enrollment and probes. It costs one extra recognizer pass but is useful
        for the pose and image-quality variation found in surveillance video.
        """
        if not aligned_faces:
            return []
        tensors: list[np.ndarray] = []
        for face in aligned_faces:
            tensors.append(self._tensor(face))
            if mirror_augmentation:
                tensors.append(self._tensor(np.ascontiguousarray(face[:, ::-1])))
        output = self._infer(np.stack(tensors).astype(np.float32, copy=False))
        step = 2 if mirror_augmentation else 1
        result: list[np.ndarray] = []
        for index in range(0, len(output), step):
            if mirror_augmentation:
                result.append(l2_normalize(output[index] + output[index + 1]))
            else:
                result.append(l2_normalize(output[index]))
        return result

    def embed(self, aligned_bgr: np.ndarray, mirror_augmentation: bool = True) -> np.ndarray:
        return self.embed_many([aligned_bgr], mirror_augmentation=mirror_augmentation)[0]
