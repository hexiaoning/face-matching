import sys
from types import SimpleNamespace

import pytest

from face_matching.errors import GpuUnavailableError
from face_matching.gpu import load_onnxruntime_gpu, verify_cuda_device


def test_missing_cuda_driver_is_rejected(monkeypatch):
    def missing_driver(_name):
        raise OSError("driver missing")

    monkeypatch.setattr("face_matching.gpu.sys.platform", "win32")
    monkeypatch.setattr("face_matching.gpu.ctypes.WinDLL", missing_driver)

    with pytest.raises(GpuUnavailableError, match="NVIDIA CUDA 驱动"):
        verify_cuda_device()


def test_cpu_only_onnxruntime_is_rejected(monkeypatch):
    monkeypatch.setattr("face_matching.gpu.verify_cuda_device", lambda: 1)
    fake_ort = SimpleNamespace(
        preload_dlls=lambda directory: None,
        get_available_providers=lambda: ["CPUExecutionProvider"],
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    with pytest.raises(GpuUnavailableError, match="CUDAExecutionProvider"):
        load_onnxruntime_gpu()
