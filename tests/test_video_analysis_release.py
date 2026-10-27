import json
from pathlib import Path
import re

from PySide6.QtWidgets import QApplication

from face_matching import __version__
from face_matching.video_analysis import main


ROOT = Path(__file__).resolve().parents[1]


def test_release_version_matches_project_and_portable_instructions(tmp_path):
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'^version = "([^"]+)"', project, re.MULTILINE).group(1) == __version__
    assert f"v{__version__}" in (ROOT / "packaging/VideoAnalysis-README.txt").read_text(encoding="utf-8")
    report = tmp_path / "version.json"
    assert main(["--version", "--report", str(report)]) == 0
    assert json.loads(report.read_text(encoding="utf-8")) == {"application": "VideoAnalysis", "version": __version__}


def test_standalone_release_has_only_analysis_tabs(qt_runtime, monkeypatch):
    def inspect_window(application):
        windows = [window for window in application.topLevelWidgets() if window.__class__.__name__ == "AnalysisWindow"]
        assert len(windows) == 1
        window = windows[0]
        assert window.windowTitle().endswith(f"v{__version__}")
        assert [window.tabs.tabText(index) for index in range(window.tabs.count())] == ["人车检测", "图像增强"]
        assert window.tabs.currentIndex() == 1
        return 0

    monkeypatch.setattr(QApplication, "exec", inspect_window)
    assert main(["--image-tools"]) == 0
