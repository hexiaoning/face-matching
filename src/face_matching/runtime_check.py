"""Exercise the complete offline payload without requiring a GPU by default."""
from __future__ import annotations

import importlib.metadata
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

from . import __version__
from .config import AppConfig
from .gui.main_window import MainWindow
from .image_enhancement import PRESETS, encode_png, enhance_image
from .models import available_profiles, file_sha256, required_paths, required_person_vehicle_model, required_semantic_models


def check_runtime(config: AppConfig, *, diagnose: bool = False) -> dict:
    application = QApplication.instance() or QApplication(sys.argv[:1])
    window = MainWindow(config)
    try:
        profiles = available_profiles()
        if not profiles:
            required_paths(config.model_profile, verify_hash=True)
            profiles = [config.model_profile]
        models = {required_person_vehicle_model()}
        models.update(required_semantic_models())
        for profile in profiles:
            models.update(required_paths(profile, verify_hash=True))
        pixels = np.full((32, 48, 3), 70, np.uint8)
        result = enhance_image(pixels, PRESETS["屏幕翻拍"])
        if cv2.imdecode(np.frombuffer(encode_png(result), np.uint8), 1).shape != pixels.shape:
            raise RuntimeError("PNG 编解码自检失败")
        with tempfile.TemporaryDirectory(prefix="face-matching-check-") as temporary:
            video_path = Path(temporary) / "sample.mp4"
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (96, 64))
            try:
                if not writer.isOpened():
                    raise RuntimeError("MP4 编码器无法打开")
                writer.write(cv2.resize(result, (96, 64)))
            finally:
                writer.release()
            reader = cv2.VideoCapture(str(video_path))
            try:
                ok, decoded = reader.read()
                if not ok or decoded.shape[:2] != (64, 96):
                    raise RuntimeError("MP4 解码自检失败")
            finally:
                reader.release()
        for index in range(window.tabs.count()):
            window.tabs.setCurrentIndex(index)
            application.processEvents()
            if window.grab().isNull():
                raise RuntimeError("GUI 渲染检查失败")
        report = {
            "ok": True, "application": "FaceMatching", "version": __version__,
            "tabs": [window.tabs.tabText(index) for index in range(window.tabs.count())],
            "profiles": profiles, "video_codec": "mp4v",
            "models": [{"name": model.name, "size": model.stat().st_size, "sha256": file_sha256(model)}
                       for model in sorted(models)],
            "runtime": {name: importlib.metadata.version(name) for name in (
                "numpy", "opencv-python-headless", "PySide6", "onnxruntime-gpu", "openvino",
            )},
        }
        if diagnose:
            from .diagnostics import run_diagnostics
            from .person_vehicle import load_person_vehicle_detector

            report["face_gpu"] = {profile: run_diagnostics(
                profile, backend=config.gpu_backend, device_id=config.gpu_device_id,
                mirror_augmentation=config.mirror_augmentation,
            ) for profile in profiles}
            detector = load_person_vehicle_detector(config.gpu_backend, config.gpu_device_id)
            raw = detector.session.run(None, {detector.input_name: np.zeros((1, 3, 640, 640), np.float32)})
            if len(raw) != 2 or any(not np.isfinite(output).all() for output in raw):
                raise RuntimeError("人车 GPU 推理输出无效")
            report["person_vehicle_gpu"] = {"providers": detector.session.get_providers(),
                                            "sample_detections": len(detector.detect(pixels))}
            from .semantic_model import SemanticEncoder

            encoder = SemanticEncoder(config.gpu_backend, config.gpu_device_id)
            text_feature = encoder.text('白色的车')
            image_feature = encoder.image(pixels)
            report['semantic_gpu'] = {'providers': encoder.providers(),
                                      'embedding_dim': int(text_feature.size),
                                      'sample_similarity': float(text_feature @ image_feature)}
        return report
    finally:
        window.close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        application.processEvents()
