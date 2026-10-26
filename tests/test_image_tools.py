import os
import time
from zipfile import ZipFile

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from face_matching.gui.image_tools import ImageToolsDialog, ImageToolsWidget
from face_matching.image_enhancement import encode_png


@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


def wait_for_worker(app, tools):
    deadline = time.monotonic() + 5
    while tools.worker is not None and time.monotonic() < deadline:
        app.processEvents()
        QTest.qWait(5)
    assert tools.worker is None


def test_apply_export_state_and_region_reset_when_opening_another_image(app, tmp_path):
    source = tmp_path / "样本.png"
    pixels = np.full((120, 160, 3), 60, np.uint8)
    source.write_bytes(encode_png(pixels))
    tool = ImageToolsWidget()
    assert not tool.apply_button.isEnabled()
    assert tool.load_image(source)
    assert tool.zip_button.isEnabled()
    tool.preset.setCurrentText("暗部提亮")
    assert not tool.zip_button.isEnabled()
    tool._select_region((10, 10, 80, 90))
    tool._apply()
    assert not tool.open_button.isEnabled()
    wait_for_worker(app, tool)
    assert tool.zip_button.isEnabled()
    assert tool.processed[0, 0, 0] > pixels[0, 0, 0]
    np.testing.assert_array_equal(tool.original, pixels)
    source.unlink()
    assert tool.source_bytes == encode_png(pixels)
    source.write_bytes(encode_png(pixels[:30, :40]))
    assert tool.load_image(source)
    assert tool.region is None
    assert tool.original_view.region is None
    assert tool.processed_view.region is None
    assert tool.processed.shape == (30, 40, 3)
    assert not tool.zip_button.isEnabled()  # New image still needs selected parameters applied.
    tool.close()


def test_worker_failures_restore_controls_and_cannot_export_stale_parameters(app, tmp_path, monkeypatch):
    source = tmp_path / "a.png"
    source.write_bytes(encode_png(np.full((40, 40, 3), 60, np.uint8)))
    tool = ImageToolsWidget()
    tool.load_image(source)
    tool.controls["gamma"].setValue(1.5)

    def fail(*args):
        raise ValueError("test failure")

    monkeypatch.setattr("face_matching.gui.image_tools.enhance_image", fail)
    tool._apply()
    wait_for_worker(app, tool)
    assert tool.open_button.isEnabled()
    assert not tool.zip_button.isEnabled()
    assert "test failure" in tool.status.text()
    tool.close()


def test_escape_cannot_destroy_dialog_while_processing(app, monkeypatch):
    dialog = ImageToolsDialog()
    dialog.show()
    monkeypatch.setattr(dialog.tools, "is_busy", lambda: True)
    QTest.keyClick(dialog, Qt.Key_Escape)
    assert dialog.isVisible()
    assert not dialog.close()
    monkeypatch.setattr(dialog.tools, "is_busy", lambda: False)
    assert dialog.close()


def test_standalone_entry_does_not_load_config_or_check_gpu(app, monkeypatch):
    from face_matching import __main__

    def unexpected(*args, **kwargs):
        raise AssertionError("独立图像工具不应初始化配置或 GPU")

    monkeypatch.setattr(__main__.AppConfig, "load", unexpected)
    monkeypatch.setattr(__main__, "assert_gpu_available", unexpected)
    monkeypatch.setattr(QApplication, "exec", lambda self: 0)
    assert __main__.main(["--image-tools"]) == 0


def test_export_button_uses_background_worker_and_keeps_original_bytes(app, tmp_path, monkeypatch):
    source = tmp_path / "原图.png"
    payload = encode_png(np.full((40, 40, 3), 60, np.uint8))
    source.write_bytes(payload)
    destination = tmp_path / "导出.zip"
    monkeypatch.setattr("face_matching.gui.image_tools.QFileDialog.getSaveFileName",
                        lambda *args: (str(destination), "ZIP (*.zip)"))
    tool = ImageToolsWidget()
    tool.load_image(source)
    tool._export(False)
    deadline = time.monotonic() + 5
    while tool.export_worker is not None and time.monotonic() < deadline:
        app.processEvents()
        QTest.qWait(5)
    assert tool.export_worker is None
    assert not tool.export_failed
    assert tool.zip_button.isEnabled()
    with ZipFile(destination) as archive:
        assert archive.read("original/source.png") == payload
    tool.close()
