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


def test_session_rejects_provider_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    class CpuOnlySession:
        def get_providers(self) -> list[str]:
            return ["CPUExecutionProvider"]

    monkeypatch.setattr(gpu, "assert_gpu_available", lambda: ["CUDAExecutionProvider"])
    monkeypatch.setattr(gpu.ort, "InferenceSession", lambda *args, **kwargs: CpuOnlySession())

    with pytest.raises(GPUUnavailableError, match="未在 GPU provider"):
        gpu.create_gpu_session(Path("model.onnx"))
