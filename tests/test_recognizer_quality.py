from __future__ import annotations

import numpy as np

from face_matching.quality import assess_quality
from face_matching.recognizer import LVFaceRecognizer, l2_normalize


def test_lvface_preprocessing_matches_official_rgb_normalization() -> None:
    image = np.zeros((112, 112, 3), dtype=np.uint8)
    image[:, :] = [0, 127, 255]  # OpenCV BGR

    tensor = LVFaceRecognizer.preprocess([image])

    assert tensor.shape == (1, 3, 112, 112)
    np.testing.assert_allclose(tensor[0, :, 0, 0], [1.0, -1 / 255, -1.0], atol=1e-6)


def test_normalization_and_quality_are_bounded() -> None:
    normalized = l2_normalize(np.array([[3.0, 4.0]], dtype=np.float32))
    np.testing.assert_allclose(normalized, [[0.6, 0.8]], atol=1e-6)

    image = np.indices((112, 112)).sum(axis=0).astype(np.uint8)
    image = np.repeat(image[:, :, None], 3, axis=2)
    landmarks = np.array(
        [[38.0, 52.0], [74.0, 52.0], [56.0, 72.0], [42.0, 92.0], [71.0, 92.0]],
        dtype=np.float32,
    )
    quality = assess_quality(image, np.array([0, 0, 112, 112]), landmarks, 0.95)
    assert 0.0 <= quality.overall <= 1.0
    assert quality.frontal > 0.8
