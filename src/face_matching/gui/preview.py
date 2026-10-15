from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QLabel

from ..engine import FrameResult, TrackView


def draw_tracks(image: QImage, result: FrameResult | None) -> QImage:
    image = image.copy()
    if result is None:
        return image
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setFont(QFont("Microsoft YaHei UI", max(10, int(image.height() / 55))))
    for track in result.tracks:
        x1, y1, x2, y2 = (int(value) for value in track.bbox)
        color = QColor(
            "#58a6ff" if track.annotation == "人工关联" else
            "#ff6b6b" if track.annotation == "不是此人" else
            "#39d98a" if track.person_id else "#ffb020"
        )
        painter.setPen(QPen(color, max(2, int(image.height() / 360))))
        painter.drawRect(x1, y1, max(1, x2 - x1), max(1, y2 - y1))
        detail = track.annotation or f"{track.score:.2f}  Q:{track.quality:.2f}"
        label = f"#{track.id} {track.name}  {detail}"
        metrics = painter.fontMetrics()
        label_width = metrics.horizontalAdvance(label) + 12
        label_height = metrics.height() + 6
        top = max(0, y1 - label_height)
        painter.fillRect(x1, top, label_width, label_height, QColor(0, 0, 0, 185))
        painter.setPen(color)
        painter.drawText(x1 + 6, top + metrics.ascent() + 3, label)
    painter.end()
    return image


class FacePreview(QLabel):
    track_context_requested = Signal(object, object)

    def __init__(self) -> None:
        super().__init__("选择图片、视频，或输入流地址、摄像头编号")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 420)
        self.setStyleSheet("background:#111820;color:#8fa3b8;border-radius:8px;")
        self.setToolTip("右键点击人脸框，可关联已录入人员")
        self.image = QImage()
        self.result: FrameResult | None = None
        self._rendered = QImage()

    def set_frame(self, image: QImage, result: FrameResult | None) -> None:
        self.image = image
        self.result = result
        self._rendered = draw_tracks(image, result)
        self.update()

    def image_rect(self) -> QRectF:
        if self.image.isNull():
            return QRectF()
        bounds = self.contentsRect()
        scale = min(bounds.width() / self.image.width(), bounds.height() / self.image.height())
        width, height = self.image.width() * scale, self.image.height() * scale
        return QRectF(
            bounds.x() + (bounds.width() - width) / 2,
            bounds.y() + (bounds.height() - height) / 2, width, height,
        )

    def track_at(self, position: QPointF) -> TrackView | None:
        rect = self.image_rect()
        if not self.result or rect.isEmpty() or not rect.contains(position):
            return None
        x = (position.x() - rect.x()) * self.image.width() / rect.width()
        y = (position.y() - rect.y()) * self.image.height() / rect.height()
        hits = [track for track in self.result.tracks
                if track.bbox[0] <= x <= track.bbox[2] and track.bbox[1] <= y <= track.bbox[3]]
        return min(hits, key=lambda track: (
            (track.bbox[2] - track.bbox[0]) * (track.bbox[3] - track.bbox[1])
        ), default=None)

    def paintEvent(self, event: object) -> None:
        super().paintEvent(event)
        if self._rendered.isNull():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.drawImage(self.image_rect(), self._rendered)
        painter.end()

    def contextMenuEvent(self, event: object) -> None:
        track = self.track_at(QPointF(event.pos()))
        if track is not None:
            self.track_context_requested.emit(track, event.globalPos())
            event.accept()
        else:
            event.ignore()
