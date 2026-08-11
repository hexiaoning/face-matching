"""主窗口：人员库 / 视频分析两个页签。"""
from __future__ import annotations

from PySide6.QtWidgets import QLabel, QMainWindow, QTabWidget

from ..database import PersonDB
from ..matcher import GalleryIndex
from .persons_page import PersonsPage
from .video_page import VideoPage
from .widgets import ModelHub


class MainWindow(QMainWindow):
    def __init__(self, db: PersonDB, hub: ModelHub, gpu_info: dict):
        super().__init__()
        self.db = db
        self.hub = hub
        self.setWindowTitle("FaceMatch 监控视频人脸比对")
        self.resize(1200, 800)

        self.tabs = QTabWidget()
        self.persons = PersonsPage(db, hub, self._reload_gallery)
        self.video = VideoPage(hub)
        self.tabs.addTab(self.video, "视频分析")
        self.tabs.addTab(self.persons, "人员库管理")
        self.setCentralWidget(self.tabs)

        device = gpu_info.get("device", "GPU")
        self.status_label = QLabel(f"GPU 加速已启用 ({device})")
        self.statusBar().addPermanentWidget(self.status_label)
        self._reload_gallery()
        self._update_person_count()

    def _reload_gallery(self) -> None:
        self.hub.gallery.rebuild(self.db.gallery())
        self._update_person_count()

    def _update_person_count(self) -> None:
        n_persons = len(self.db.list_persons())
        n_embs = len(self.hub.gallery)
        self.statusBar().showMessage(f"人员库: {n_persons} 人 / {n_embs} 张照片特征")

    def closeEvent(self, e) -> None:  # noqa: N802
        self.video._stop()
        self.db.close()
        super().closeEvent(e)
