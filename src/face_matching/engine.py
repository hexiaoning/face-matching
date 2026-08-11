from __future__ import annotations

import threading

import numpy as np

from .alignment import align_face
from .config import EngineConfig
from .detector import FaceDetection, SCRFDDetector
from .errors import GPUUnavailableError
from .gpu import resolve_gpu_backend
from .model_manager import assert_models_present
from .quality import assess_quality
from .recognizer import LVFaceRecognizer


class FaceEngine:
    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        assert_models_present(self.config)
        self.backend = resolve_gpu_backend(self.config.gpu_backend)
        self.detector = SCRFDDetector(
            str(self.config.detector_model),
            input_size=self.config.detector_size,
            threshold=self.config.detector_threshold,
            nms_threshold=self.config.nms_threshold,
            prefer_tensorrt=self.config.prefer_tensorrt,
            backend=self.backend,
        )
        self.recognizer = LVFaceRecognizer(
            str(self.config.recognizer_model), self.config.prefer_tensorrt, self.backend
        )
        self._inference_lock = threading.RLock()
        self._warm_up_gpu()

    @property
    def provider(self) -> str:
        return self.recognizer.session.get_providers()[0]

    def detect(self, frame_bgr: np.ndarray) -> list[FaceDetection]:
        with self._inference_lock:
            detections = self.detector.detect(frame_bgr)
        accepted: list[FaceDetection] = []
        for detection in detections:
            try:
                aligned = align_face(frame_bgr, detection.landmarks)
            except (ValueError, cv2_error()):
                continue
            detection.aligned = aligned
            detection.quality = assess_quality(
                aligned, detection.bbox, detection.landmarks, detection.score
            )
            accepted.append(detection)
        return accepted

    def embed(
        self,
        aligned_faces: list[np.ndarray],
        mirror_augmentation: bool | None = None,
    ) -> np.ndarray:
        use_mirror = (
            self.config.mirror_augmentation
            if mirror_augmentation is None
            else mirror_augmentation
        )
        with self._inference_lock:
            return self.recognizer.embed_batch(aligned_faces, use_mirror)

    def _warm_up_gpu(self) -> None:
        """Run both networks once so GPU failures happen before the GUI opens."""
        width, height = self.config.detector_size
        try:
            with self._inference_lock:
                self.detector.detect(np.zeros((height, width, 3), dtype=np.uint8))
                self.recognizer.embed_batch(
                    [np.zeros((112, 112, 3), dtype=np.uint8)],
                    mirror_augmentation=False,
                )
        except Exception as exc:
            raise GPUUnavailableError(
                "GPU 推理自检失败，程序拒绝切换到 CPU。\n"
                "请检查 NVIDIA/Intel 显卡驱动、GPU 运行库和 ONNX 模型。\n"
                f"后端: {self.backend}\n原因: {exc}"
            ) from exc


def cv2_error() -> type[Exception]:
    # Kept behind a helper to avoid making OpenCV part of type-checking imports.
    import cv2

    return cv2.error
