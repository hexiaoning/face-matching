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


class _FixedBatchSession:
    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size
        self.calls: list[tuple[int, ...]] = []

    def run(self, _outputs, inputs):
        tensors = next(iter(inputs.values()))
        self.calls.append(tensors.shape)
        values = np.stack(
            [tensors[:, 0].mean(axis=(1, 2)) + 2.0, tensors[:, 1].mean(axis=(1, 2)) + 2.0],
            axis=1,
        )
        return [values.astype(np.float32)]


def test_recognizer_honors_fixed_batch_and_mirror_tta() -> None:
    recognizer = object.__new__(LVFaceRecognizer)
    recognizer.session = _FixedBatchSession(4)
    recognizer.input_name = "input"
    recognizer.output_name = "output"
    recognizer.fixed_batch = 4
    recognizer.embedding_size = 2
    faces = [np.full((112, 112, 3), value, dtype=np.uint8) for value in (20, 80, 140)]

    embeddings = recognizer.embed_batch(faces, mirror_augmentation=True)

    assert embeddings.shape == (3, 2)
    assert recognizer.session.calls == [(4, 3, 112, 112), (4, 3, 112, 112)]
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-6)
