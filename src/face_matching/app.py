from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication, QMessageBox

from face_matching.config import AppConfig
from face_matching.ui import MainWindow

STYLESHEET = """
QWidget { font-family: "Microsoft YaHei UI", "Segoe UI"; font-size: 10pt; }
QMainWindow { background: #f4f7f9; }
QGroupBox { font-weight: 600; border: 1px solid #ccd6dd; border-radius: 6px;
            margin-top: 12px; padding-top: 10px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
QPushButton { padding: 7px 13px; border: 1px solid #aab7c0; border-radius: 5px;
              background: #ffffff; }
QPushButton:hover { background: #edf5ff; border-color: #3b82f6; }
QPushButton:disabled { color: #9aa5ad; background: #eef1f3; }
QLineEdit, QSpinBox, QDoubleSpinBox { padding: 6px; border: 1px solid #b8c4cc;
                                     border-radius: 4px; background: white; }
QHeaderView::section { background: #e8eef2; padding: 7px; border: 0;
                       border-right: 1px solid #d0d9df; font-weight: 600; }
QTableWidget { background: white; border: 1px solid #d0d9df; gridline-color: #e5eaee; }
QTabWidget::pane { border: 0; }
QTabBar::tab { padding: 10px 22px; }
QTabBar::tab:selected { color: #1565c0; border-bottom: 2px solid #1565c0; }
"""


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    application = QApplication(sys.argv)
    application.setApplicationName("FaceMatching")
    application.setOrganizationName("FaceMatching")
    application.setStyleSheet(STYLESHEET)
    try:
        config = AppConfig.load()
        window = MainWindow(config)
    except Exception as exc:
        QMessageBox.critical(None, "启动失败", str(exc))
        return 1
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
