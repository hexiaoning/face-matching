from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from face_matching import gpu
from face_matching.errors import GPUUnavailableError


def test_gpu_is_mandatory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu, "preload_cuda_runtime", lambda: None)
    monkeypatch.setattr(gpu.ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(gpu, "available_openvino_devices", lambda: ["CPU"])

    with pytest.raises(GPUUnavailableError, match="拒绝使用 CPU"):
        gpu.assert_gpu_available()


def test_auto_backend_prefers_cuda_then_openvino(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpu, "available_cuda_providers", lambda: ["CUDAExecutionProvider"])
    monkeypatch.setattr(gpu, "available_openvino_devices", lambda: ["GPU.0"])
    assert gpu.resolve_gpu_backend("auto") == "cuda"

    monkeypatch.setattr(gpu, "available_cuda_providers", lambda: [])
    assert gpu.resolve_gpu_backend("auto") == "openvino"
    assert gpu.resolve_gpu_backend("openvino") == "openvino"


def test_session_rejects_provider_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    class CpuOnlySession:
        def get_providers(self) -> list[str]:
            return ["CPUExecutionProvider"]

    monkeypatch.setattr(gpu, "available_cuda_providers", lambda: ["CUDAExecutionProvider"])
    monkeypatch.setattr(gpu.ort, "InferenceSession", lambda *args, **kwargs: CpuOnlySession())

    with pytest.raises(GPUUnavailableError, match="未在 CUDA/TensorRT GPU provider"):
        gpu._create_cuda_session(Path("model.onnx"))


def test_openvino_session_runs_on_gpu_without_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    class Port:
        def __init__(self, name: str, shape: list[int]) -> None:
            self.name = name
            self.partial_shape = shape

        def get_any_name(self) -> str:
            return self.name

    class Tensor:
        def __init__(self, data) -> None:
            self.data = np.asarray(data)

    class Request:
        def set_input_tensor(self, index: int, tensor: Tensor) -> None:
            assert index == 0
            self.input = tensor.data

        def infer(self) -> None:
            self.output = np.asarray([[self.input.mean(), self.input.max()]], dtype=np.float32)

        def get_output_tensor(self, index: int) -> Tensor:
            assert index == 0
            return Tensor(self.output)

    class Compiled:
        inputs = [Port("input", [1, 3, 2, 2])]
        outputs = [Port("embedding", [1, 2])]

        def create_infer_request(self) -> Request:
            return Request()

    class Core:
        available_devices = ["CPU", "GPU.0"]

        def read_model(self, path: str):
            return path

        def compile_model(self, model, device: str, properties: dict):
            assert device == "GPU"
            assert properties["PERFORMANCE_HINT"] == "LATENCY"
            assert properties["INFERENCE_PRECISION_HINT"] == "f32"
            return Compiled()

    fake_openvino = SimpleNamespace(Core=Core, Tensor=Tensor)
    monkeypatch.setattr(gpu, "_load_openvino", lambda: fake_openvino)
    session = gpu.OpenVINOInferenceSession(Path("model.onnx"))
    output = session.run(
        ["embedding"],
        {"input": np.ones((1, 3, 2, 2), dtype=np.float32)},
    )

    assert session.get_providers() == [gpu.OPENVINO_PROVIDER]
    assert session.get_inputs()[0].shape == [1, 3, 2, 2]
    np.testing.assert_allclose(output[0], [[1.0, 1.0]])
