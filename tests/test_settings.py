from __future__ import annotations

from pathlib import Path

from face_match.config import AppSettings


def test_settings_round_trip_and_corrupt_fallback(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    settings = AppSettings(similarity_threshold=0.61, model_license_accepted=True)
    settings.save(path)
    loaded = AppSettings.load(path)
    assert loaded.similarity_threshold == 0.61
    assert loaded.model_license_accepted

    path.write_text("not json", encoding="utf-8")
    assert AppSettings.load(path).similarity_threshold == AppSettings().similarity_threshold
