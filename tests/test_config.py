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
    assert loaded.feature_model_id.endswith("-single")


def test_tta_state_is_part_of_feature_model_id():
    enabled = AppConfig(model_profile="lvface-b", mirror_augmentation=True)
    disabled = AppConfig(model_profile="lvface-b", mirror_augmentation=False)

    assert enabled.feature_model_id.endswith("-tta")
    assert disabled.feature_model_id.endswith("-single")
    assert enabled.feature_model_id != disabled.feature_model_id


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gpu_device_id", -1),
        ("gpu_backend", "directml"),
        ("detector_size", 100),
        ("match_threshold", 1.1),
        ("match_margin", 0.9),
        ("target_match_threshold", 1.1),
        ("target_review_threshold", -0.1),
        ("target_min_support", 1),
        ("min_face_size", 8),
        ("frame_interval", 0),
        ("mirror_augmentation", 1),
        ("enrollment_min_quality", 1.1),
        ("track_consistency_threshold", 1.1),
        ("confirmation_matches", 0),
    ],
)
def test_config_rejects_invalid_values(field, value):
    config = AppConfig()
    setattr(config, field, value)
    with pytest.raises(ValueError):
        config.validate()


def test_config_rejects_target_review_threshold_above_confirmation():
    with pytest.raises(ValueError, match="疑似目标阈值"):
        AppConfig(target_match_threshold=0.18, target_review_threshold=0.19).validate()
