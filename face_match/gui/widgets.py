"""共享模型容器与通用控件。"""
from __future__ import annotations

import threading

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy

from ..detector import SCRFDDetector
from ..recognizer import FaceRecognizer
from ..matcher import GalleryIndex
from ..pipeline import FacePipeline
from .. import config, models as models_mod


class ModelHub:
    """全局唯一的检测/识别模型 + 人员库索引，线程安全。"""

    def __init__(self, providers: list[str]):
        mdir = config.models_dir()
        self.detector = SCRFDDetector(str(models_mod.model_path(config.DETECTOR_MODEL, mdir)),
                                      providers=providers)
        self.recognizer = FaceRecognizer(str(models_mod.model_path(config.RECOGNIZER_MODEL, mdir)),
                                         providers=providers)
        self.gallery = GalleryIndex()
        self.lock = threading.Lock()  # onnxruntime session 不做并发推理

    def new_pipeline(self, threshold: float) -> FacePipeline:
        return FacePipeline(self.detector, self.recognizer, self.gallery, threshold)

    def embed_photo(self, img: np.ndarray):
        from ..pipeline import embed_gallery_photo
        with self.lock:
            return embed_gallery_photo(self.detector, self.recognizer, img)


def bgr_to_qpixmap(frame: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img.copy())


class VideoLabel(QLabel):
    """按比例缩放显示视频帧的控件。"""

    def __init__(self):
        super().__init__("打开视频文件或输入视频流地址")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background:#202020; color:#888; font-size:16px;")
        self._pixmap: QPixmap | None = None

    def show_frame(self, frame: np.ndarray) -> None:
        self._pixmap = bgr_to_qpixmap(frame)
        self.update()

    def resizeEvent(self, e) -> None:  # noqa: N802
        super().resizeEvent(e)
        self.update()

    def paintEvent(self, e) -> None:  # noqa: N802
        if self._pixmap is None:
            super().paintEvent(e)
            return
        from PySide6.QtGui import QPainter
        p = QPainter(self)
        scaled = self._pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation)
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        p.drawPixmap(x, y, scaled)
        p.end()


def draw_annotations(frame: np.ndarray, result, show_quality: bool = True) -> np.ndarray:
    """在帧上画检测框和识别结果。已识别=绿色，未识别=黄色，样本不足=灰色。"""
    out = frame.copy()
    for trk in result.tracks:
        x1, y1, x2, y2 = [int(v) for v in trk.box]
        if trk.match is not None:
            color = (60, 200, 60)
            label = f"{trk.match.person.name} {trk.match.score:.2f}"
        elif trk.fused_ready:
            color = (0, 200, 220)
            label = f"未知 #{trk.tid}"
        else:
            color = (150, 150, 150)
            label = f"采样中 #{trk.tid}"
        if show_quality:
            label += f" q={trk.quality:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = max(y1 - 6, th + 4)
        cv2.rectangle(out, (x1, ty - th - 6), (x1 + tw + 4, ty + 2), color, -1)
        cv2.putText(out, label, (x1 + 2, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 2, cv2.LINE_AA)
    return out
