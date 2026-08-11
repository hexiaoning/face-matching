from __future__ import annotations

import importlib.metadata
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .errors import GPUUnavailableError


GPUBackend = Literal["cuda", "openvino"]
OPENVINO_PROVIDER = "OpenVINO-GPU"


@dataclass(frozen=True, slots=True)
class TensorMetadata:
    name: str
    shape: list[object]


def _load_ort() -> Any:
    try:
        import onnxruntime as ort
    except Exception as exc:
        raise GPUUnavailableError(
            "未安装 ONNX Runtime GPU。请双击 install.cmd 完成安装。"
        ) from exc
    try:
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls(directory="")
    except Exception as exc:
        raise GPUUnavailableError(
            "CUDA/cuDNN 运行库加载失败。请重新运行 install.cmd，并确认 NVIDIA 驱动已更新。"
        ) from exc
    return ort


def _load_openvino() -> Any:
    try:
        import openvino as ov
    except Exception as exc:
        raise GPUUnavailableError(
            "未安装 OpenVINO，无法使用 Intel GPU。请重新运行 install.cmd。"
        ) from exc
    return ov


def assert_cuda_available() -> str:
    try:
        package_version = importlib.metadata.version("onnxruntime-gpu")
    except importlib.metadata.PackageNotFoundError as exc:
        raise GPUUnavailableError(
            "未安装 onnxruntime-gpu；CUDA 后端不允许 CPU 回退。"
        ) from exc
    ort = _load_ort()
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise GPUUnavailableError(
            "未发现 CUDAExecutionProvider；请确认 NVIDIA 驱动和 GPU 运行环境。"
        )
    if os.name == "nt":
        try:
            import ctypes

            ctypes.WinDLL("nvcuda.dll")
        except OSError as exc:
            raise GPUUnavailableError(
                "无法加载 NVIDIA 驱动 nvcuda.dll。请安装/更新 NVIDIA 显卡驱动。"
            ) from exc
    return (
        f"ONNX Runtime GPU {package_version} "
        f"(runtime {ort.__version__}) / CUDAExecutionProvider"
    )


def available_openvino_devices() -> list[str]:
    try:
        return [str(device) for device in _load_openvino().Core().available_devices]
    except GPUUnavailableError:
        return []
    except Exception:
        return []


def _select_openvino_device(device_id: int = 0) -> str:
    devices = available_openvino_devices()
    numbered = f"GPU.{device_id}"
    if numbered in devices:
        return numbered
    if device_id == 0 and "GPU" in devices:
        return "GPU"
    gpu_devices = [item for item in devices if item.startswith("GPU.")]
    if device_id == 0 and gpu_devices:
        return gpu_devices[0]
    raise GPUUnavailableError(
        f"OpenVINO 未发现 Intel GPU {device_id}；可用设备: {devices or ['none']}"
    )


def assert_openvino_available(device_id: int = 0) -> str:
    ov = _load_openvino()
    device = _select_openvino_device(device_id)
    try:
        version = importlib.metadata.version("openvino")
    except importlib.metadata.PackageNotFoundError:
        version = str(getattr(ov, "__version__", "unknown"))
    return f"OpenVINO {version} / {device} (FP32)"


def available_gpu_backends(device_id: int = 0) -> list[str]:
    backends: list[str] = []
    try:
        assert_cuda_available()
        backends.append("cuda")
    except GPUUnavailableError:
        pass
    try:
        assert_openvino_available(device_id)
        backends.append("openvino")
    except GPUUnavailableError:
        pass
    return backends


def resolve_gpu_backend(requested: str = "auto", device_id: int = 0) -> GPUBackend:
    value = requested.strip().lower()
    if value not in {"auto", "cuda", "openvino"}:
        raise GPUUnavailableError(
            f"未知 GPU 后端: {requested}。可选值为 auto、cuda、openvino。"
        )
    cuda_error: GPUUnavailableError | None = None
    if value in {"auto", "cuda"}:
        try:
            assert_cuda_available()
            return "cuda"
        except GPUUnavailableError as exc:
            cuda_error = exc
            if value == "cuda":
                raise
    try:
        assert_openvino_available(device_id)
        return "openvino"
    except GPUUnavailableError as openvino_error:
        if value == "openvino":
            raise
        raise GPUUnavailableError(
            "未检测到可用的 NVIDIA CUDA 或 Intel OpenVINO GPU，"
            "程序拒绝使用 CPU 推理。\n"
            f"CUDA: {cuda_error}\n"
            f"OpenVINO: {openvino_error}"
        ) from openvino_error


def assert_gpu_available(backend: str = "auto", device_id: int = 0) -> str:
    selected = resolve_gpu_backend(backend, device_id)
    if selected == "cuda":
        return assert_cuda_available()
    return assert_openvino_available(device_id)


def _port_name(port: Any, index: int, prefix: str) -> str:
    try:
        name = str(port.get_any_name())
        if name:
            return name
    except Exception:
        pass
    return f"{prefix}_{index}"


def _port_shape(port: Any) -> list[object]:
    result: list[object] = []
    for dimension in port.partial_shape:
        try:
            result.append(int(dimension.get_length()) if dimension.is_static else None)
        except Exception:
            try:
                result.append(int(dimension))
            except Exception:
                result.append(None)
    return result


class OpenVINOInferenceSession:
    """Small ORT-compatible adapter that compiles an ONNX model on Intel GPU."""

    def __init__(self, model_path: str | Path, device_id: int = 0) -> None:
        ov = _load_openvino()
        self._ov = ov
        self._core = ov.Core()
        self.device = _select_openvino_device(device_id)
        path = Path(model_path).resolve()
        try:
            model = self._core.read_model(str(path))
            self._compiled = self._core.compile_model(
                model,
                self.device,
                {
                    "PERFORMANCE_HINT": "LATENCY",
                    "INFERENCE_PRECISION_HINT": "f32",
                },
            )
            self._request = self._compiled.create_infer_request()
        except Exception as exc:
            raise GPUUnavailableError(
                f"OpenVINO 无法在 {self.device} 上完整加载模型（不会回退 CPU）:\n"
                f"模型: {path}\n原因: {exc}"
            ) from exc
        self._input_ports = list(self._compiled.inputs)
        self._output_ports = list(self._compiled.outputs)
        self._inputs = [
            TensorMetadata(_port_name(port, index, "input"), _port_shape(port))
            for index, port in enumerate(self._input_ports)
        ]
        self._outputs = [
            TensorMetadata(_port_name(port, index, "output"), _port_shape(port))
            for index, port in enumerate(self._output_ports)
        ]
        self._output_indices = {item.name: index for index, item in enumerate(self._outputs)}

    def get_inputs(self) -> list[TensorMetadata]:
        return self._inputs

    def get_outputs(self) -> list[TensorMetadata]:
        return self._outputs

    def get_providers(self) -> list[str]:
        return [f"{OPENVINO_PROVIDER}:{self.device}"]

    def run(
        self, output_names: list[str] | None, inputs: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        if len(self._inputs) != 1:
            raise RuntimeError("人脸模型必须只有一个输入")
        array = np.ascontiguousarray(inputs[self._inputs[0].name])
        self._request.set_input_tensor(0, self._ov.Tensor(array))
        self._request.infer()
        names = output_names or [item.name for item in self._outputs]
        return [
            np.array(self._request.get_output_tensor(self._output_indices[name]).data, copy=True)
            for name in names
        ]


def _create_cuda_session(model_path: Path, device_id: int = 0) -> Any:
    assert_cuda_available()
    ort = _load_ort()
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    provider_options = {
        "device_id": str(device_id),
        "arena_extend_strategy": "kNextPowerOfTwo",
        "cudnn_conv_algo_search": "HEURISTIC",
        "do_copy_in_default_stream": "1",
        "use_tf32": "1",
    }
    try:
        session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=[("CUDAExecutionProvider", provider_options)],
        )
    except Exception as exc:
        raise GPUUnavailableError(
            f"模型无法在 CUDA GPU {device_id} 上完整加载（不会回退 CPU）: "
            f"{model_path.name}\n{exc}"
        ) from exc
    if not session.get_providers() or session.get_providers()[0] != "CUDAExecutionProvider":
        raise GPUUnavailableError(f"{model_path.name} 未使用 CUDAExecutionProvider")
    return session


def create_gpu_session(
    model_path: Path, device_id: int = 0, backend: str = "auto"
) -> Any:
    selected = resolve_gpu_backend(backend, device_id)
    if selected == "cuda":
        return _create_cuda_session(model_path, device_id)
    return OpenVINOInferenceSession(model_path, device_id)
