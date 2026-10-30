import json
from pathlib import Path
import re

import pytest
from PySide6.QtWidgets import QApplication

from face_matching import __version__, __main__
from face_matching.video_analysis import main

ROOT = Path(__file__).resolve().parents[1]
TABS = ["人脸检索", "人车检测", "图像增强", "以图搜索", "语义搜索", "人员库", "标注库", "设置"]


def test_release_version_matches_project_and_portable_instructions(tmp_path):
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'^version = "([^"]+)"', project, re.MULTILINE).group(1) == __version__
    assert f"v{__version__}" in (ROOT / "packaging/FaceMatching-README.txt").read_text(encoding="utf-8")
    report = tmp_path / "version.json"
    assert main(["--version", "--report", str(report)]) == 0
    assert json.loads(report.read_text(encoding="utf-8")) == {"application": "FaceMatching", "version": __version__}


@pytest.mark.parametrize("entry, args, selected", [
    (__main__.main, [], "人脸检索"),
    (__main__.main, ["--image-tools"], "图像增强"),
    (__main__.main, ["--video-detection"], "人车检测"),
    (__main__.main, ["--semantic-search"], "语义搜索"),
    (main, [], "人脸检索"),
    (main, ["--image-tools"], "图像增强"),
])
def test_every_entry_opens_complete_app_without_initializing_gpu(qt_runtime, tmp_path, monkeypatch, entry, args, selected):
    monkeypatch.setenv("FACE_MATCHING_HOME", str(tmp_path))
    def unexpected(*args, **kwargs):
        raise AssertionError("Opening the unified app must not require GPU or face models")
    monkeypatch.setattr(__main__, "assert_gpu_available", unexpected)
    monkeypatch.setattr("face_matching.gui.main_window.assert_gpu_available", unexpected)
    def inspect_window(application):
        windows = [w for w in application.topLevelWidgets() if w.__class__.__name__ == "MainWindow" and w.isVisible()]
        assert len(windows) == 1
        window = windows[0]
        assert f"v{__version__}" in window.windowTitle()
        assert [window.tabs.tabText(i) for i in range(window.tabs.count())] == TABS
        assert window.tabs.tabText(window.tabs.currentIndex()) == selected
        assert window.engine is None
        return 0
    monkeypatch.setattr(QApplication, "exec", inspect_window)
    assert entry(args) == 0
