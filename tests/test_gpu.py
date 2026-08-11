from __future__ import annotations

from pathlib import Path

import pytest

from face_matching import gpu
from face_matching.errors import GPUUnavailableError


def test_gpu_is_mandatory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu, "preload_cuda_runtime", lambda: None)
    monkeypatch.setattr(gpu.ort, "get_available_providers", lambda: ["CPUExecutionProvider"])

    with pytest.raises(GPUUnavailableError, match="拒绝使用 CPU"):
        gpu.assert_gpu_available()


def test_requested_backend_never_crosses_to_another_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gpu, "available_gpu_providers", lambda: ["DmlExecutionProvider"]
    )
    assert gpu.assert_gpu_available("directml") == "DmlExecutionProvider"
    with pytest.raises(GPUUnavailableError, match="拒绝使用 CPU"):
        gpu.assert_gpu_available("cuda")


def test_session_rejects_provider_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    class CpuOnlySession:
        def get_providers(self) -> list[str]:
            return ["CPUExecutionProvider"]

    monkeypatch.setattr(
        gpu, "assert_gpu_available", lambda *args, **kwargs: "CUDAExecutionProvider"
    )
    monkeypatch.setattr(gpu.ort, "InferenceSession", lambda *args, **kwargs: CpuOnlySession())

    with pytest.raises(GPUUnavailableError, match="未在 GPU provider"):
        gpu.create_gpu_session(Path("model.onnx"))


def test_directml_uses_required_sequential_session_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class DmlSession:
        def get_providers(self) -> list[str]:
            return ["DmlExecutionProvider", "CPUExecutionProvider"]

    def create_session(*args, **kwargs):
        captured.update(kwargs)
        return DmlSession()

    monkeypatch.setattr(
        gpu, "assert_gpu_available", lambda *args, **kwargs: "DmlExecutionProvider"
    )
    monkeypatch.setattr(gpu.ort, "InferenceSession", create_session)

    gpu.create_gpu_session(Path("model.onnx"), backend="directml")

    options = captured["sess_options"]
    assert options.enable_mem_pattern is False
    assert options.execution_mode == gpu.ort.ExecutionMode.ORT_SEQUENTIAL
    assert captured["providers"] == [("DmlExecutionProvider", {"device_id": 0})]
