STYLESHEET = """
QWidget {
    font-family: "Microsoft YaHei UI", "Segoe UI";
    font-size: 13px;
    color: #dce6f7;
    background: #111827;
}
QMainWindow, QTabWidget::pane { background: #0f172a; }
QMenuBar, QMenu { background: #162033; }
QMenuBar::item:selected, QMenu::item:selected { background: #26344f; }
QTabBar::tab {
    background: #162033; color: #aebbd1; padding: 10px 24px;
    border-top-left-radius: 6px; border-top-right-radius: 6px;
}
QTabBar::tab:selected { background: #23304a; color: white; }
QGroupBox {
    border: 1px solid #2c3b58; border-radius: 8px; margin-top: 12px;
    padding: 12px 8px 8px 8px; font-weight: 600;
}
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; }
QPushButton {
    background: #26344f; border: 1px solid #354766; border-radius: 6px;
    padding: 8px 14px; min-height: 20px;
}
QPushButton:hover { background: #30415f; }
QPushButton:pressed { background: #1e2a40; }
QPushButton:disabled { color: #6f7c92; background: #1c273b; }
QPushButton[class="primary"] { background: #2f7df6; border-color: #4b91ff; color: white; }
QPushButton[class="primary"]:hover { background: #428cff; }
QLineEdit, QListWidget, QTableWidget {
    background: #0d1525; border: 1px solid #2c3b58; border-radius: 5px;
    selection-background-color: #2f7df6; padding: 5px;
}
QHeaderView::section {
    background: #1d2a40; border: none; border-right: 1px solid #31405b;
    padding: 8px; font-weight: 600;
}
QSlider::groove:horizontal { height: 5px; background: #35445f; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #2f7df6; border-radius: 2px; }
QSlider::handle:horizontal { width: 16px; margin: -6px 0; border-radius: 8px; background: #69a2ff; }
QLabel[class="hint"] { color: #91a0b8; }
QStatusBar { background: #101a2b; color: #9baccc; }
"""
