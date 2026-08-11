from __future__ import annotations

from typing import Any

from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QMainWindow, QTabWidget

from face_match.services import ApplicationServices
from face_match.ui.live_tab import LiveTab
from face_match.ui.person_tab import PersonTab
from face_match.ui.system_tab import SystemTab


class MainWindow(QMainWindow):
    def __init__(self, services: ApplicationServices, parent: Any | None = None) -> None:
        super().__init__(parent)
        self.services = services
        self.setWindowTitle("监控视频人脸比对")
        self.resize(1440, 860)
        tabs = QTabWidget()
        self.live_tab = LiveTab(services)
        self.person_tab = PersonTab(services)
        self.system_tab = SystemTab(services)
        tabs.addTab(self.live_tab, "实时识别")
        tabs.addTab(self.person_tab, "人员库")
        tabs.addTab(self.system_tab, "系统状态")
        self.setCentralWidget(tabs)
        self.live_tab.running_changed.connect(self.person_tab.set_video_running)
        self.person_tab.busy_changed.connect(self.live_tab.set_enrollment_busy)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.live_tab.shutdown()
        try:
            self.services.settings.save(self.services.paths.settings)
        except OSError:
            pass
        event.accept()
