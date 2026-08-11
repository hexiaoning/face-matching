import numpy as np
import pytest

from face_matching.errors import ModelError
from face_matching.inference import ArcFaceRecognizer, FaceEngine


class _StaticBatchSession:
    def __init__(self):
        self.batch_sizes = []

    def run(self, _outputs, inputs):
        batch = next(iter(inputs.values()))
        self.batch_sizes.append(len(batch))
        values = batch.reshape(len(batch), -1)
        return [np.column_stack([values[:, 0] + 2.0, values[:, -1] + 1.0])]


def test_flip_tta_supports_static_batch_one_models():
    recognizer = ArcFaceRecognizer.__new__(ArcFaceRecognizer)
    recognizer.session = _StaticBatchSession()
    recognizer.input_name = "input"
    recognizer.color_order = "bgr"
    recognizer.input_mean = 0.0
    recognizer.input_std = 1.0
    recognizer.static_batch_size = 1
    faces = [np.zeros((112, 112, 3), dtype=np.uint8) for _ in range(2)]

    embeddings = recognizer.embed_aligned(faces, flip_tta=True)

    assert embeddings.shape == (2, 2)
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), [1, 1])
    assert recognizer.session.batch_sizes == [1, 1, 1, 1]


def test_enrollment_rejects_images_with_multiple_faces():
    engine = FaceEngine.__new__(FaceEngine)
    engine.analyze = lambda _frame: [object(), object()]

    with pytest.raises(ModelError, match="多张人脸"):
        engine.enroll_image(np.zeros((10, 10, 3), dtype=np.uint8))
