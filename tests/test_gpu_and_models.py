from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from face_match.errors import GpuUnavailableError
from face_match.gpu import prepare_cuda_runtime
from face_match.model_manager import ModelAsset, ModelManager


class FakeOrt:
    __version__ = "test"

    def __init__(self, providers: list[str]) -> None:
        self.providers = providers
        self.preloaded = False

    def preload_dlls(self, directory: str = "") -> None:
        assert directory == ""
        self.preloaded = True

    def get_available_providers(self) -> list[str]:
        return self.providers


def test_gpu_guard_accepts_cuda_and_rejects_cpu_only() -> None:
    gpu = FakeOrt(["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert prepare_cuda_runtime(gpu) is gpu
    assert gpu.preloaded
    with pytest.raises(GpuUnavailableError, match="禁止使用 CPU"):
        prepare_cuda_runtime(FakeOrt(["CPUExecutionProvider"]))


def test_model_download_verifies_hash(tmp_path: Path) -> None:
    payload = b"small deterministic model payload"
    source = tmp_path / "source.onnx"
    source.write_bytes(payload)
    asset = ModelAsset(
        name="test",
        filename="installed.onnx",
        url=source.as_uri(),
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
    )
    manager = ModelManager(tmp_path / "models")
    manager._install_asset(asset, lambda *_: None, lambda: False)
    assert (tmp_path / "models" / "installed.onnx").read_bytes() == payload
    assert (tmp_path / "models" / "installed.onnx.verified").exists()


def test_archive_model_checks_extracted_file(tmp_path: Path) -> None:
    model = b"detector bytes"
    archive_path = tmp_path / "models.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("nested/detector.onnx", model)
    archive_bytes = archive_path.read_bytes()
    asset = ModelAsset(
        name="detector",
        filename="detector.onnx",
        url=archive_path.as_uri(),
        sha256=hashlib.sha256(archive_bytes).hexdigest(),
        size=len(archive_bytes),
        archive_member="detector.onnx",
        installed_size=len(model),
        installed_sha256=hashlib.sha256(model).hexdigest(),
    )
    manager = ModelManager(tmp_path / "installed")
    manager._install_asset(asset, lambda *_: None, lambda: False)
    assert (tmp_path / "installed" / "detector.onnx").read_bytes() == model
