"""Standalone offline application for category detection and image adjustments."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path

from . import __version__


def write_report(path: Path | None, report: dict) -> None:
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    if sys.stdout is not None:
        print(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="离线人车视频检测与图像增强")
    parser.add_argument("--version", action="store_true")
    parser.add_argument("--image-tools", action="store_true", help="直接打开图像增强")
    parser.add_argument("--video-detection", action="store_true", help="直接打开人车检测")
    parser.add_argument("--backend", choices=("auto", "cuda", "openvino"), default="auto")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--check-runtime", action="store_true", help="离线校验 GUI、运行库和模型")
    parser.add_argument("--diagnose", action="store_true", help="额外执行真实 GPU 推理")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.version:
        write_report(args.report, {"application": "VideoAnalysis", "version": __version__})
        return 0
    application, window = None, None
    owns_application = False
    try:
        import cv2
        import numpy as np
        from PySide6.QtCore import QCoreApplication, QEvent
        from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget
        from .gui.image_tools import ImageToolsWidget
        from .gui.video_detection import VideoDetectionWidget
        from .image_enhancement import PRESETS, encode_png, enhance_image

        class AnalysisWindow(QMainWindow):
            def __init__(self) -> None:
                super().__init__()
                self.setWindowTitle(f"离线图像与人车分析 v{__version__}")
                self.resize(1400, 880)
                self.tabs = QTabWidget()
                self.video = VideoDetectionWidget(args.backend, args.device_id, self)
                self.images = ImageToolsWidget(self)
                self.tabs.addTab(self.video, "人车检测")
                self.tabs.addTab(self.images, "图像增强")
                self.tabs.setCurrentIndex(1 if args.image_tools else 0)
                self.setCentralWidget(self.tabs)

            def closeEvent(self, event: object) -> None:
                if self.video.is_busy():
                    self.video.stop()
                    event.ignore()
                elif self.images.is_busy():
                    self.images.status.setText("请等待处理或导出完成后再关闭。")
                    event.ignore()
                else:
                    event.accept()

        owns_application = QApplication.instance() is None
        application = QApplication.instance() or QApplication(sys.argv[:1])
        application.setApplicationName("VideoAnalysis")
        application.setApplicationVersion(__version__)
        application.setStyle("Fusion")
        window = AnalysisWindow()
        if args.check_runtime or args.diagnose:
            from .models import PERSON_VEHICLE_MODEL, file_sha256, required_person_vehicle_model

            model = required_person_vehicle_model()
            pixels = np.full((32, 48, 3), 70, np.uint8)
            result = enhance_image(pixels, PRESETS["屏幕翻拍"])
            assert cv2.imdecode(np.frombuffer(encode_png(result), np.uint8), 1).shape == pixels.shape
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temporary:
                video_path = Path(temporary.name)
            writer, reader = None, None
            try:
                writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (96, 64))
                if not writer.isOpened():
                    raise RuntimeError("MP4 编码器无法打开")
                writer.write(cv2.resize(result, (96, 64)))
                writer.release()
                reader = cv2.VideoCapture(str(video_path))
                ok, decoded = reader.read()
                if not ok or decoded.shape[:2] != (64, 96):
                    raise RuntimeError("MP4 解码自检失败")
            finally:
                if writer is not None:
                    writer.release()
                if reader is not None:
                    reader.release()
                video_path.unlink(missing_ok=True)
            report = {
                "ok": True, "application": "VideoAnalysis", "version": __version__,
                "tabs": [window.tabs.tabText(index) for index in range(window.tabs.count())],
                "video_codec": "mp4v",
                "model": {"name": model.name, "size": model.stat().st_size,
                          "sha256": file_sha256(model), "expected_sha256": PERSON_VEHICLE_MODEL.sha256},
                "runtime": {name: importlib.metadata.version(name) for name in (
                    "numpy", "opencv-python-headless", "PySide6", "onnxruntime-gpu", "openvino",
                )},
                "loaded_identity_modules": [name for name in sys.modules if name in {
                    "face_matching.engine", "face_matching.database", "face_matching.enrollment",
                    "face_matching.matching", "face_matching.vision.recognizer", "face_matching.gui.main_window",
                }],
            }
            for index in range(window.tabs.count()):
                window.tabs.setCurrentIndex(index)
                application.processEvents()
                if window.grab().isNull():
                    raise RuntimeError("GUI 渲染检查失败")
            if args.diagnose:
                from .person_vehicle import load_person_vehicle_detector

                started = time.perf_counter()
                detector = load_person_vehicle_detector(args.backend, args.device_id)
                raw = detector.session.run(None, {detector.input_name: np.zeros((1, 3, 640, 640), np.float32)})
                if len(raw) != 2 or any(not np.isfinite(output).all() for output in raw):
                    raise RuntimeError("GPU 推理输出无效")
                detections = detector.detect(pixels)
                report["gpu"] = {"providers": detector.session.get_providers(),
                                 "load_and_inference_seconds": round(time.perf_counter() - started, 3),
                                 "sample_detections": len(detections)}
            write_report(args.report, report)
            return 0
        window.show()
        return application.exec()
    except Exception as exc:
        write_report(args.report, {"ok": False, "application": "VideoAnalysis", "version": __version__, "error": str(exc)})
        if not (args.check_runtime or args.diagnose or args.report):
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.critical(None, "无法启动离线分析", str(exc))
        return 2
    finally:
        if application is not None and window is not None:
            window.close()
            window.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            application.processEvents()
        if owns_application and application is not None:
            application.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
