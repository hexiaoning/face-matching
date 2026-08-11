from __future__ import annotations

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..pipeline import FrameResult


class VideoWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._image = QImage()
        self._result = FrameResult([], [])
        self.setMinimumSize(720, 480)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def sizeHint(self) -> QSize:
        return QSize(960, 640)

    def clear(self) -> None:
        self._image = QImage()
        self._result = FrameResult([], [])
        self.update()

    def set_frame(self, image: QImage, result: FrameResult) -> None:
        self._image = image
        self._result = result
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0b1020"))
        if self._image.isNull():
            painter.setPen(QColor("#8e9bb8"))
            painter.setFont(QFont("Microsoft YaHei UI", 16))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "请选择视频、摄像头或 RTSP 视频源")
            return
        target = self._target_rect()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(target, self._image)
        scale_x = target.width() / self._image.width()
        scale_y = target.height() / self._image.height()
        painter.setFont(QFont("Microsoft YaHei UI", 10, QFont.Weight.DemiBold))
        for face in self._result.faces:
            x1, y1, x2, y2 = face.bbox
            rect = QRectF(
                target.left() + x1 * scale_x,
                target.top() + y1 * scale_y,
                max(1.0, (x2 - x1) * scale_x),
                max(1.0, (y2 - y1) * scale_y),
            )
            color = QColor("#39d98a") if face.accepted else (
                QColor("#ffbf69") if face.state in {"采集中", "待确认"} else QColor("#ff6b6b")
            )
            painter.setPen(QPen(color, 2.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(rect, 4, 4)
            if face.accepted:
                label = f"{face.name}  {face.score:.3f}  {face.id_card}"
            else:
                label = f"{face.state}  Q:{face.quality:.2f}  #{face.track_id}"
            metrics = painter.fontMetrics()
            label_width = metrics.horizontalAdvance(label) + 14
            label_height = metrics.height() + 8
            label_y = max(target.top(), rect.top() - label_height)
            label_rect = QRectF(rect.left(), label_y, label_width, label_height)
            painter.setPen(Qt.PenStyle.NoPen)
            background = QColor(color)
            background.setAlpha(220)
            painter.setBrush(background)
            painter.drawRoundedRect(label_rect, 4, 4)
            painter.setPen(QColor("#08111e"))
            painter.drawText(label_rect.adjusted(7, 3, -7, -3), Qt.AlignmentFlag.AlignVCenter, label)

    def _target_rect(self) -> QRectF:
        source_ratio = self._image.width() / self._image.height()
        widget_ratio = self.width() / max(self.height(), 1)
        if widget_ratio > source_ratio:
            height = float(self.height())
            width = height * source_ratio
            return QRectF((self.width() - width) / 2, 0, width, height)
        width = float(self.width())
        height = width / source_ratio
        return QRectF(0, (self.height() - height) / 2, width, height)
