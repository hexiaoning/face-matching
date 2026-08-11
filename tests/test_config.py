import json

import pytest

from face_matching.config import AppConfig


def test_config_round_trip_and_ignores_future_keys(tmp_path):
    target = tmp_path / "config.json"
    config = AppConfig(detector_size=960, mirror_augmentation=False, match_threshold=0.61)
    config.save(target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["future_option"] = "ignored"
    target.write_text(json.dumps(payload), encoding="utf-8")

    loaded = AppConfig.load(target)
    assert loaded.detector_size == 960
    assert loaded.mirror_augmentation is False
    assert loaded.match_threshold == pytest.approx(0.61)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gpu_device_id", -1),
        ("detector_size", 100),
        ("match_threshold", 1.1),
        ("match_margin", 0.9),
        ("min_face_size", 8),
        ("frame_interval", 0),
        ("mirror_augmentation", 1),
    ],
)
def test_config_rejects_invalid_values(field, value):
    config = AppConfig()
    setattr(config, field, value)
    with pytest.raises(ValueError):
        config.validate()
