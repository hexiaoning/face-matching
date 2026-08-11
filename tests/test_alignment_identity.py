from __future__ import annotations

import numpy as np
import pytest

from face_match.identity import is_valid_chinese_id, mask_id_number, validate_identity
from face_match.vision.alignment import (
    ARCFACE_REFERENCE,
    similarity_transform,
    transform_points,
)


def test_similarity_transform_recovers_known_mapping() -> None:
    angle = np.deg2rad(18)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32
    )
    source = (ARCFACE_REFERENCE @ rotation.T) * 1.3 + np.array([20, -7])
    matrix = similarity_transform(source, ARCFACE_REFERENCE)
    recovered = transform_points(source, matrix)
    assert np.max(np.abs(recovered - ARCFACE_REFERENCE)) < 1e-4


def test_identity_validation_and_masking() -> None:
    assert is_valid_chinese_id("11010519491231002X")
    with pytest.raises(ValueError, match="校验位"):
        validate_identity("某人", "110105194912310021")
    assert validate_identity(" Alice  Smith ", " pass 123 ") == ("Alice Smith", "PASS123")
    assert mask_id_number("11010519491231002X") == "110105********002X"
