from pathlib import Path

from face_matching import paths


def test_bundled_model_is_preferred(monkeypatch, tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    detector = models / "scrfd_10g_bnkps.onnx"
    detector.write_bytes(b"bundled")
    monkeypatch.setattr(paths, "resource_root", lambda: tmp_path)
    monkeypatch.setattr(paths, "is_frozen", lambda: True)

    assert paths.default_model_path(detector.name) == detector
