APP_STYLE = """
QMainWindow, QWidget {
    background: #f4f6f9;
    color: #1f2937;
    font-size: 14px;
}
QPushButton {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 7px 14px;
}
QPushButton:hover { background: #eef4ff; border-color: #6b93d6; }
QPushButton:pressed { background: #dfeaff; }
QPushButton:disabled { color: #94a3b8; background: #eef1f5; }
QLineEdit, QDoubleSpinBox, QListWidget, QTableWidget {
    background: #ffffff;
    border: 1px solid #d5dbe5;
    border-radius: 5px;
    padding: 4px;
}
QHeaderView::section {
    background: #e9eef6;
    border: none;
    border-right: 1px solid #d5dbe5;
    padding: 7px;
    font-weight: 600;
}
QTabBar::tab { padding: 9px 20px; }
QTabBar::tab:selected { color: #245fb5; border-bottom: 2px solid #3979ce; }
QGroupBox {
    background: #ffffff;
    border: 1px solid #d5dbe5;
    border-radius: 7px;
    margin-top: 12px;
    padding: 12px;
    font-weight: 600;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
#videoSurface { background: #101318; color: #aeb8c6; border-radius: 8px; }
#statusLabel { color: #53657d; padding: 4px; }
#hintLabel { color: #53657d; }
#successLabel { color: #18864b; font-weight: 600; }
#warningLabel {
    color: #8a4b08;
    background: #fff6df;
    border: 1px solid #efd18c;
    border-radius: 6px;
    padding: 10px;
}
"""
