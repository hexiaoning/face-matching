from __future__ import annotations

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..domain import TrackView, mask_id_number


class VideoWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._image = QImage()
        self._tracks: list[TrackView] = []
        self.setMinimumSize(640, 420)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_frame(self, image: QImage, tracks: list[TrackView]) -> None:
        self._image = image
        self._tracks = tracks
        self.update()

    def clear(self) -> None:
        self._image = QImage()
        self._tracks = []
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#050910"))
        if self._image.isNull():
            painter.setPen(QColor("#6f829a"))
            font = painter.font()
            font.setPointSize(16)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "请选择视频、摄像头或 RTSP 视频流")
            return
        scaled = self._image.size().scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio
        )
        target = QRect(
            (self.width() - scaled.width()) // 2,
            (self.height() - scaled.height()) // 2,
            scaled.width(),
            scaled.height(),
        )
        painter.drawImage(target, self._image)
        scale_x = target.width() / self._image.width()
        scale_y = target.height() / self._image.height()
        font = QFont("Microsoft YaHei UI", 10)
        font.setBold(True)
        painter.setFont(font)
        for track in self._tracks:
            x1, y1, x2, y2 = track.bbox
            box = QRect(
                int(target.left() + x1 * scale_x),
                int(target.top() + y1 * scale_y),
                max(1, int((x2 - x1) * scale_x)),
                max(1, int((y2 - y1) * scale_y)),
            )
            color = QColor("#38d996") if track.confirmed else QColor("#ffbf4b")
            if track.label == "质量不足":
                color = QColor("#ef6f78")
            painter.setPen(QPen(color, 3))
            painter.drawRect(box)
            details = track.label
            if track.confirmed:
                details += f"  {track.score:.3f}  {mask_id_number(track.id_number)}"
            else:
                details += f"  Q:{track.quality:.2f}"
            metrics = painter.fontMetrics()
            text_width = min(max(metrics.horizontalAdvance(details) + 16, 100), target.width())
            text_height = metrics.height() + 10
            text_y = box.top() - text_height
            if text_y < target.top():
                text_y = box.top()
            label_rect = QRect(box.left(), text_y, text_width, text_height)
            painter.fillRect(label_rect, QColor(color.red(), color.green(), color.blue(), 215))
            painter.setPen(QColor("#071018"))
            painter.drawText(
                label_rect.adjusted(8, 0, -4, 0),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                details,
            )

