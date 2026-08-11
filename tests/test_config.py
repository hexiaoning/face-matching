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


def test_target_search_defaults_require_multi_frame_evidence():
    config = AppConfig()

    assert config.target_review_threshold < config.target_match_threshold
    assert config.target_match_threshold == pytest.approx(0.19)
    assert config.target_min_support == 3
    assert config.target_min_evidence_gap == pytest.approx(0.75)
    assert config.target_top_k == 20
    assert config.target_auto_confirm is False
    assert config.fast_file_scan is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gpu_device_id", -1),
        ("gpu_backend", "directml"),
        ("detector_size", 100),
        ("match_threshold", 1.1),
        ("target_match_threshold", 1.1),
        ("target_review_threshold", -0.1),
        ("match_margin", 0.9),
        ("min_face_size", 8),
        ("frame_interval", 0),
        ("mirror_augmentation", 1),
        ("enrollment_min_quality", 1.1),
        ("track_consistency_threshold", 1.1),
        ("confirmation_matches", 0),
        ("target_min_support", 0),
        ("target_min_evidence_gap", -0.1),
        ("target_top_k", 0),
        ("target_auto_confirm", 1),
        ("fast_file_scan", 1),
    ],
)
def test_config_rejects_invalid_values(field, value):
    config = AppConfig()
    setattr(config, field, value)
    with pytest.raises(ValueError):
        config.validate()


def test_config_requires_review_threshold_below_confirmation_threshold():
    with pytest.raises(ValueError, match="复核阈值"):
        AppConfig(target_match_threshold=0.20, target_review_threshold=0.20).validate()
