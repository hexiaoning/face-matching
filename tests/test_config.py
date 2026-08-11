from pathlib import Path

from face_matching.config import RecognitionSettings


def test_settings_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    expected = RecognitionSettings(similarity_threshold=0.61, detector_size=1280)
    expected.save(path)
    actual = RecognitionSettings.load(path)
    assert actual.similarity_threshold == 0.61
    assert actual.detector_size == 1280


def test_invalid_settings_fall_back_to_defaults(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text('{"detector_size": 123}', encoding="utf-8")
    assert RecognitionSettings.load(path).detector_size == 960

