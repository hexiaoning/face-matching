import hashlib
import json
from zipfile import ZipFile

import cv2
import numpy as np
import pytest

from face_matching.image_enhancement import (
    PRESETS, EnhancementSettings, encode_png, enhance_image, enhance_region, export_enhancement, save_new_png,
)


def test_local_face_plate_enhancement_preserves_outside_region_pixel_for_pixel(tmp_path):
    original = sample_image()
    region = (20, 10, 80, 60)
    result = enhance_region(original, PRESETS['车牌清晰化（保守）'], region)
    mask = np.ones(original.shape[:2], bool)
    mask[10:60, 20:80] = False
    np.testing.assert_array_equal(result[mask], original[mask])
    assert np.any(result[~mask] != original[~mask])
    output = tmp_path / '记录.zip'
    export_enhancement(output, '原图.png', encode_png(original), original, result,
                       PRESETS['车牌清晰化（保守）'], region, region)
    with ZipFile(output) as archive:
        assert json.loads(archive.read('manifest.json'))['processing_region_xyxy'] == list(region)


def sample_image():
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(80, 120, 3), dtype=np.uint8)


def test_zero_settings_preserve_every_pixel_and_return_independent_buffer():
    original = sample_image()
    result = enhance_image(original, EnhancementSettings())
    np.testing.assert_array_equal(result, original)
    result[:] = 0
    assert original.any()


@pytest.mark.parametrize("settings", PRESETS.values())
def test_presets_keep_source_pixels_and_geometry(settings):
    original = sample_image()
    before = original.copy()
    result = enhance_image(original, settings)
    assert result.shape == original.shape
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(original, before)


def test_gamma_lifts_shadows_and_does_not_invent_values_in_clipped_white():
    image = np.full((32, 32, 3), 30, np.uint8)
    image[:, 16:] = 255
    result = enhance_image(image, EnhancementSettings(gamma=1.5))
    assert result[0, 0, 0] > 30
    np.testing.assert_array_equal(result[:, 16:], image[:, 16:])


def test_screen_suppression_reduces_synthetic_pixel_grid():
    grid = np.indices((100, 100)).sum(axis=0) % 2
    image = np.repeat((grid * 160 + 40).astype(np.uint8)[:, :, None], 3, axis=2)
    result = enhance_image(image, EnhancementSettings(texture=0.9))
    assert result[5:-5, 5:-5].std() < image.std() * 0.2


@pytest.mark.parametrize("settings", [
    EnhancementSettings(gamma=float("nan")), EnhancementSettings(texture=-1),
    EnhancementSettings(sharpen=2), EnhancementSettings(denoise=float("inf")),
])
def test_invalid_settings_fail_before_processing(settings):
    with pytest.raises(ValueError):
        enhance_image(sample_image(), settings)


def test_zip_keeps_exact_original_bytes_params_crops_and_hashes(tmp_path):
    original = sample_image()
    source_bytes = encode_png(original)
    settings = PRESETS["屏幕翻拍"]
    result = enhance_image(original, settings)
    destination = tmp_path / "处理记录.zip"
    export_enhancement(destination, "测试原图.png", source_bytes, original, result, settings, (10, 20, 70, 60))
    with ZipFile(destination) as archive:
        assert archive.read("original/source.png") == source_bytes
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["settings"]["texture"] == settings.texture
        assert manifest["region_xyxy"] == [10, 20, 70, 60]
        for name, expected_hash in manifest["sha256"].items():
            payload = archive.read(name)
            assert hashlib.sha256(payload).hexdigest() == expected_hash
        pixels = cv2.imdecode(np.frombuffer(archive.read("processed.png"), np.uint8), 1)
        np.testing.assert_array_equal(pixels, result)
        crop = cv2.imdecode(np.frombuffer(archive.read("region/original.png"), np.uint8), 1)
        np.testing.assert_array_equal(crop, original[20:60, 10:70])


def test_export_never_overwrites_source_or_existing_results(tmp_path):
    image = sample_image()
    payload = encode_png(image)
    destination = tmp_path / "原图.png"
    destination.write_bytes(payload)
    with pytest.raises(FileExistsError):
        export_enhancement(destination, destination.name, payload, image, image, EnhancementSettings())
    assert destination.read_bytes() == payload
    with pytest.raises(FileExistsError):
        save_new_png(destination, image)
    assert destination.read_bytes() == payload


def test_invalid_crop_does_not_leave_export_file(tmp_path):
    image = sample_image()
    destination = tmp_path / "invalid.zip"
    with pytest.raises(ValueError):
        export_enhancement(destination, "a.png", b"a", image, image, EnhancementSettings(), (1, 1, 200, 300))
    assert not destination.exists()


def test_failed_zip_write_removes_partial_export(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(ZipFile, "writestr", fail)
    image = sample_image()
    destination = tmp_path / "failed.zip"
    with pytest.raises(OSError, match="disk full"):
        export_enhancement(destination, "a.png", b"a", image, image, EnhancementSettings())
    assert not destination.exists()
