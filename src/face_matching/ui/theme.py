APP_STYLE = """
QWidget {
    font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
    font-size: 14px;
    color: #dce5f2;
}
QMainWindow, QDialog, QWidget#root {
    background: #0d1420;
}
QTabWidget::pane {
    border: 1px solid #243247;
    background: #111b2a;
    border-radius: 8px;
}
QTabBar::tab {
    background: #111b2a;
    color: #8fa2ba;
    padding: 12px 24px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    color: #ffffff;
    background: #1a2a40;
    border-bottom: 3px solid #39a0ff;
}
QPushButton {
    background: #1d3048;
    border: 1px solid #2d4868;
    border-radius: 6px;
    padding: 8px 15px;
    min-height: 20px;
}
QPushButton:hover { background: #27405e; border-color: #4a78a8; }
QPushButton:pressed { background: #15263a; }
QPushButton:disabled { color: #607086; background: #152031; border-color: #243247; }
QPushButton[primary="true"] { background: #1677c8; border-color: #2999f0; color: white; }
QPushButton[primary="true"]:hover { background: #2389da; }
QPushButton[danger="true"] { background: #4b2530; border-color: #78404d; }
QPushButton[danger="true"]:hover { background: #63303d; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: #0c1522;
    border: 1px solid #2b3c52;
    border-radius: 5px;
    padding: 7px;
    selection-background-color: #277ec0;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: #409fe8;
}
QTableWidget, QListWidget {
    background: #0c1522;
    alternate-background-color: #101d2d;
    border: 1px solid #243247;
    border-radius: 6px;
    gridline-color: #1f2d40;
}
QHeaderView::section {
    background: #172539;
    color: #aebed1;
    border: none;
    border-right: 1px solid #26374e;
    padding: 8px;
}
QTableWidget::item { padding: 6px; }
QTableWidget::item:selected, QListWidget::item:selected { background: #214f78; }
QGroupBox {
    border: 1px solid #27384d;
    border-radius: 7px;
    margin-top: 12px;
    padding-top: 12px;
    font-weight: bold;
}
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; }
QLabel[muted="true"] { color: #8699b1; }
QStatusBar { background: #0a111b; color: #91a3b8; }
QSplitter::handle { background: #243247; width: 2px; }
QScrollBar:vertical { background: #111b2a; width: 11px; }
QScrollBar::handle:vertical { background: #344960; border-radius: 5px; min-height: 30px; }
QToolTip { color: white; background: #16263a; border: 1px solid #3a5878; }
"""

