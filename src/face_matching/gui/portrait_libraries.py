from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QInputDialog,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
    QSpinBox, QVBoxLayout, QWidget,
)

from ..database import FaceDatabase
from ..identity_filters import PersonFilter


class LibrarySelection(QListWidget):
    def __init__(self, database: FaceDatabase, selected: tuple[str, ...] = ('default',)) -> None:
        super().__init__()
        self.setMaximumHeight(100)
        for identifier, name in database.list_libraries():
            item = QListWidgetItem(name, self)
            item.setData(Qt.UserRole, identifier)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if identifier in selected else Qt.Unchecked)

    def selected_ids(self) -> list[str]:
        return [self.item(i).data(Qt.UserRole) for i in range(self.count())
                if self.item(i).checkState() == Qt.Checked] or ['default']


class LibrariesDialog(QDialog):
    def __init__(self, database: FaceDatabase, parent: QWidget) -> None:
        super().__init__(parent)
        self.database = database
        self.setWindowTitle("管理人像库")
        self.resize(420, 360)
        self.libraries = QListWidget()
        row = QHBoxLayout()
        for label, callback in (("新建库", self.create), ("重命名", self.rename), ("删除库", self.remove)):
            button = QPushButton(label)
            button.clicked.connect(callback)
            row.addWidget(button)
        close = QDialogButtonBox(QDialogButtonBox.Close)
        close.rejected.connect(self.accept)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("同一人员可归入多个库；删除库会保留人员及照片。"))
        layout.addWidget(self.libraries)
        layout.addLayout(row)
        layout.addWidget(close)
        self.refresh()

    def refresh(self) -> None:
        self.libraries.clear()
        for identifier, name in self.database.list_libraries():
            item = QListWidgetItem(name, self.libraries)
            item.setData(Qt.UserRole, identifier)

    def create(self) -> None:
        name, ok = QInputDialog.getText(self, "新建人像库", "库名称（例如 xx库、yy库）：")
        if ok:
            try:
                self.database.create_library(name)
                self.refresh()
            except ValueError as exc:
                QMessageBox.warning(self, "无法新建", str(exc))

    def rename(self) -> None:
        item = self.libraries.currentItem()
        if item is None:
            return
        name, ok = QInputDialog.getText(self, "重命名人像库", "库名称：", text=item.text())
        if ok:
            try:
                self.database.rename_library(item.data(Qt.UserRole), name)
                self.refresh()
            except (ValueError, KeyError) as exc:
                QMessageBox.warning(self, "无法重命名", str(exc))

    def remove(self) -> None:
        item = self.libraries.currentItem()
        if item is None:
            return
        if QMessageBox.question(self, "删除人像库", f"删除“{item.text()}”？人员及照片会保留；无其它归属的人员移入默认库。") != QMessageBox.Yes:
            return
        try:
            self.database.delete_library(item.data(Qt.UserRole))
            self.refresh()
        except ValueError as exc:
            QMessageBox.warning(self, "无法删除", str(exc))


class PersonFilterWidget(QWidget):
    def __init__(self, database: FaceDatabase) -> None:
        super().__init__()
        self.database = database
        self.library = QComboBox()
        self.refresh_libraries()
        self.sex = QComboBox()
        for title, value in (("性别不限", None), ("男", "男"), ("女", "女")):
            self.sex.addItem(title, value)
        self.regions = json.loads((Path(__file__).resolve().parents[1] / 'resources/regions-2023.json').read_text(encoding='utf-8'))
        self.province, self.city, self.county = QComboBox(), QComboBox(), QComboBox()
        self.province.addItem("省份不限", '')
        for province in self.regions:
            self.province.addItem(province['name'], province['code'])
        self.province.currentIndexChanged.connect(self._province_changed)
        self.city.currentIndexChanged.connect(self._city_changed)
        self._province_changed()
        self.code = QLineEdit()
        self.code.setPlaceholderText("或输入历史地区码")
        self.code.setMaximumWidth(145)
        self.code.setMaxLength(6)
        self.age_enabled = QCheckBox("年龄")
        self.min_age, self.max_age = QSpinBox(), QSpinBox()
        for spin in (self.min_age, self.max_age):
            spin.setRange(0, 150)
            spin.setSuffix(" 岁")
            spin.setEnabled(False)
            self.age_enabled.toggled.connect(spin.setEnabled)
        self.max_age.setValue(150)
        self.apply = QPushButton("应用筛选")
        row = QHBoxLayout()
        for widget in (QLabel("人像库："), self.library, self.sex, self.province, self.city, self.county,
                       self.code, self.age_enabled, self.min_age, QLabel("至"), self.max_age, self.apply):
            row.addWidget(widget)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(row)
        note = QLabel("性别、周岁和地区按有效18位身份证筛选；地址码不等于当前户籍。地区名称参考2023年数据，历史码可手动输入。")
        note.setWordWrap(True)
        layout.addWidget(note)

    def refresh_libraries(self) -> None:
        current = self.library.currentData()
        self.library.clear()
        self.library.addItem("全部库", None)
        for identifier, name in self.database.list_libraries():
            self.library.addItem(name, identifier)
        self.library.setCurrentIndex(max(0, self.library.findData(current)))

    def _province_changed(self) -> None:
        self.city.blockSignals(True)
        self.city.clear()
        self.city.addItem("城市不限", '')
        province = next((p for p in self.regions if p['code'] == self.province.currentData()), {})
        for city in province.get('children', []):
            self.city.addItem(city['name'], city['code'])
        self.city.blockSignals(False)
        self._city_changed()

    def _city_changed(self) -> None:
        self.county.clear()
        self.county.addItem("区县不限", '')
        province = next((p for p in self.regions if p['code'] == self.province.currentData()), {})
        city = next((c for c in province.get('children', []) if c['code'] == self.city.currentData()), {})
        for county in city.get('children', []):
            self.county.addItem(county['name'], county['code'])

    def value(self) -> PersonFilter:
        return PersonFilter(
            library_id=self.library.currentData(), sex=self.sex.currentData(),
            region_prefix=self.code.text().strip() or self.county.currentData() or self.city.currentData() or self.province.currentData() or '',
            min_age=self.min_age.value() if self.age_enabled.isChecked() else None,
            max_age=self.max_age.value() if self.age_enabled.isChecked() else None,
        )
