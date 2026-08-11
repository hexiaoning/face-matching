import json

import pytest

from face_matching.config import AppConfig


def test_config_round_trip(tmp_path):
    config = AppConfig(data_dir=str(tmp_path), match_threshold=0.51)
    config.save()
    loaded = AppConfig.load(tmp_path)
    assert loaded.match_threshold == 0.51
    assert loaded.root == tmp_path.resolve()
    assert json.loads(config.config_path.read_text(encoding="utf-8"))["frame_stride"] == 2
    assert loaded.detection_width == 960
    assert loaded.detection_height == 544
    assert loaded.recognition_flip_tta is True


def test_invalid_detection_size_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        AppConfig(data_dir=str(tmp_path), detection_width=630)


def test_invalid_temporal_settings_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        AppConfig(data_dir=str(tmp_path), min_track_hits=0)
    with pytest.raises(ValueError):
        AppConfig(data_dir=str(tmp_path), min_recognition_quality=1.1)
