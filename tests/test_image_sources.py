import base64
from urllib.parse import quote_from_bytes

import cv2
import numpy as np
import pytest

from face_matching.vision.io import (
    image_data_payload, is_image_source, read_image, store_image_data_uri,
)


def data_uri(extension=".webp"):
    pixels = np.full((48, 64, 3), (20, 60, 120), dtype=np.uint8)
    ok, encoded = cv2.imencode(extension, pixels)
    assert ok
    payload = encoded.tobytes()
    return f"data:image/{extension[1:]};base64," + base64.b64encode(payload).decode(), payload


@pytest.mark.parametrize("extension", [".webp", ".png", ".jpg"])
def test_inline_images_are_decoded_as_images(extension):
    source, payload = data_uri(extension)
    assert is_image_source(source)
    assert image_data_payload(source) == payload
    np.testing.assert_array_equal(read_image(source), cv2.imdecode(np.frombuffer(payload, np.uint8), 1))


def test_percent_encoded_binary_image_data_is_supported():
    _, payload = data_uri(".png")
    source = "data:image/png," + quote_from_bytes(payload)
    assert read_image(source).shape == (48, 64, 3)


def test_materialized_image_preserves_pixels_and_can_be_read_after_restart(tmp_path):
    source, _ = data_uri()
    path = store_image_data_uri(source, tmp_path / "query_images")
    assert path.suffix == ".png" and path.is_file()
    np.testing.assert_array_equal(read_image(path), read_image(source))
    assert store_image_data_uri(source, path.parent) == path


@pytest.mark.parametrize("source", ["data:image/webp;base64,***", "data:image/png;base64,", "data:image/png"])
def test_invalid_inline_image_fails_cleanly(source, tmp_path):
    assert read_image(source) is None
    with pytest.raises((ValueError, cv2.error)):
        store_image_data_uri(source, tmp_path)
