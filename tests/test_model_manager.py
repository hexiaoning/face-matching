from __future__ import annotations

import hashlib
import zipfile

import pytest

from face_matching.model_manager import DownloadSpec, _existing_is_valid, _materialize


def test_materialize_verifies_hash_and_extracts_selected_model(tmp_path) -> None:
    archive_path = tmp_path / "models.zip"
    model_bytes = b"onnx-model-placeholder"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("pack/det_10g.onnx", model_bytes)
        archive.writestr("pack/other.onnx", b"other")
    spec = DownloadSpec(
        name="detector",
        url="https://example.invalid/model.zip",
        destination="detector.onnx",
        size=archive_path.stat().st_size,
        sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        zip_member_suffix="det_10g.onnx",
    )
    destination = tmp_path / "output" / spec.destination

    _materialize(spec, archive_path, destination)

    assert destination.read_bytes() == model_bytes
    assert not destination.with_suffix(".onnx.partial").exists()


def test_materialize_rejects_bad_checksum(tmp_path) -> None:
    download = tmp_path / "model.download"
    download.write_bytes(b"bad")
    spec = DownloadSpec(
        name="recognizer",
        url="https://example.invalid/model.onnx",
        destination="recognizer.onnx",
        size=3,
        sha256="0" * 64,
    )

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        _materialize(spec, download, tmp_path / spec.destination)


def test_existing_direct_model_is_rehashed(tmp_path) -> None:
    destination = tmp_path / "recognizer.onnx"
    destination.write_bytes(b"valid-model")
    spec = DownloadSpec(
        name="recognizer",
        url="https://example.invalid/model.onnx",
        destination=destination.name,
        size=destination.stat().st_size,
        sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        destination_min_size=5,
    )

    assert _existing_is_valid(spec, destination)
    destination.write_bytes(b"corrupt-model")
    assert not _existing_is_valid(spec, destination)
