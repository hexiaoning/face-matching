from __future__ import annotations

import os
import site
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import onnxruntime as ort

from .errors import GPUUnavailableError


GPUBackend = Literal["auto", "cuda", "openvino"]
CUDA_PROVIDERS = ("TensorrtExecutionProvider", "CUDAExecutionProvider")
OPENVINO_PROVIDER = "OpenVINO-GPU"
_DLL_DIRECTORY_HANDLES: list[Any] = []


@dataclass(frozen=True, slots=True)
class TensorMetadata:
    name: str
    shape: list[object]


def _runtime_library_dirs() -> list[Path]:
    roots = [Path(sys.prefix) / "Lib" / "site-packages"]
    roots.extend(Path(value) for value in site.getsitepackages())
    usersite = site.getusersitepackages()
    if usersite:
        roots.append(Path(usersite))
    bundled = getattr(sys, "_MEIPASS", None)
    if bundled:
        roots.insert(0, Path(bundled))
    roots.insert(0, Path(sys.executable).resolve().parent)

    discovered: list[Path] = []
    for root in roots:
        nvidia = root / "nvidia"
        if not nvidia.is_dir():
            continue
        for package in nvidia.iterdir():
            for child in (package / "bin", package / "lib"):
                if child.is_dir() and child not in discovered:
                    discovered.append(child)
    return discovered


def preload_cuda_runtime() -> None:
    """Load packaged CUDA/cuDNN DLLs before CUDA provider discovery."""
    for directory in _runtime_library_dirs():
        value = str(directory)
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            try:
                _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(value))
            except OSError:
                pass
        if value not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = value + os.pathsep + os.environ.get("PATH", "")
    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        return
    try:
        preload(directory="")
    except Exception:
        # Provider/session validation below produces the actionable error.
        pass


def available_cuda_providers() -> list[str]:
    preload_cuda_runtime()
    available = ort.get_available_providers()
    return [provider for provider in CUDA_PROVIDERS if provider in available]


def _load_openvino() -> Any:
    try:
        import openvino as ov
    except Exception as exc:
        raise GPUUnavailableError(
            "未安装 OpenVINO，无法使用 Intel GPU。请重新运行 install.bat。"
        ) from exc
    return ov


def available_openvino_devices() -> list[str]:
    try:
        core = _load_openvino().Core()
        return [str(device) for device in core.available_devices]
    except Exception:
        return []


def available_gpu_backends() -> list[str]:
    backends: list[str] = []
    if "CUDAExecutionProvider" in available_cuda_providers():
        backends.append("cuda")
    if any(device == "GPU" or device.startswith("GPU.") for device in available_openvino_devices()):
        backends.append("openvino")
    return backends


def resolve_gpu_backend(requested: str = "auto") -> Literal["cuda", "openvino"]:
    value = requested.strip().lower()
    if value not in {"auto", "cuda", "openvino"}:
        raise GPUUnavailableError(
            f"未知 GPU 后端：{requested}。可选值为 auto、cuda、openvino。"
        )
    cuda_ready = "CUDAExecutionProvider" in available_cuda_providers()
    if value == "cuda" and cuda_ready:
        return "cuda"
    if value == "auto" and cuda_ready:
        return "cuda"

    openvino_devices = available_openvino_devices()
    openvino_ready = any(
        device == "GPU" or device.startswith("GPU.") for device in openvino_devices
    )
    if value == "auto":
        if openvino_ready:
            return "openvino"
    elif value == "openvino" and openvino_ready:
        return "openvino"

    ort_providers = ", ".join(ort.get_available_providers()) or "none"
    ov_devices = ", ".join(openvino_devices) or "none"
    raise GPUUnavailableError(
        "未检测到可用的 NVIDIA CUDA 或 Intel OpenVINO GPU，程序拒绝使用 CPU 推理。\n\n"
        f"请求后端: {value}\nONNX Runtime provider: {ort_providers}\n"
        f"OpenVINO device: {ov_devices}\n"
        "RTX 5070 请安装 NVIDIA 驱动；Intel UHD 请安装/更新 Intel 显卡驱动，"
        "然后重新运行 install.bat。"
    )


def assert_gpu_available(backend: str = "auto") -> str:
    return resolve_gpu_backend(backend)


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
        if isinstance(dimension, int):
            result.append(dimension)
            continue
        try:
            result.append(int(dimension.get_length()) if dimension.is_static else None)
        except Exception:
            result.append(None)
    return result


class OpenVINOInferenceSession:
    """Small ORT-compatible adapter that compiles an ONNX model on Intel GPU only."""

    def __init__(self, model_path: str | Path) -> None:
        ov = _load_openvino()
        self._ov = ov
        self._core = ov.Core()
        devices = [str(device) for device in self._core.available_devices]
        if not any(device == "GPU" or device.startswith("GPU.") for device in devices):
            raise GPUUnavailableError(f"OpenVINO 未发现 Intel GPU：{devices or ['none']}")
        try:
            model = self._core.read_model(str(Path(model_path).resolve()))
            self._compiled = self._core.compile_model(
                model,
                "GPU",
                {
                    "PERFORMANCE_HINT": "LATENCY",
                    # Keep gallery embeddings stable across CUDA and OpenVINO.
                    # Accuracy takes priority over implicit FP16 conversion.
                    "INFERENCE_PRECISION_HINT": "f32",
                },
            )
            self._request = self._compiled.create_infer_request()
        except Exception as exc:
            raise GPUUnavailableError(
                f"OpenVINO 无法在 Intel GPU 上完整加载模型（不会回退 CPU）：\n"
                f"模型: {Path(model_path).resolve()}\n原因: {exc}"
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
        return [OPENVINO_PROVIDER]

    def run(self, output_names: list[str], inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        if len(self._inputs) != 1:
            raise RuntimeError("face models must have exactly one input")
        array = np.ascontiguousarray(inputs[self._inputs[0].name])
        self._request.set_input_tensor(0, self._ov.Tensor(array))
        self._request.infer()
        result: list[np.ndarray] = []
        for name in output_names:
            index = self._output_indices[name]
            result.append(np.array(self._request.get_output_tensor(index).data, copy=True))
        return result


def _create_cuda_session(
    model_path: str | Path,
    prefer_tensorrt: bool = False,
) -> ort.InferenceSession:
    available = available_cuda_providers()
    if "CUDAExecutionProvider" not in available:
        raise GPUUnavailableError("CUDAExecutionProvider 不可用，程序拒绝使用 CPU 推理。")
    path = str(Path(model_path).resolve())
    provider_config: list[Any] = []
    if prefer_tensorrt and "TensorrtExecutionProvider" in available:
        cache_dir = str(Path(path).parent / "tensorrt_cache")
        os.makedirs(cache_dir, exist_ok=True)
        provider_config.append(("TensorrtExecutionProvider", {
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": cache_dir,
            "trt_fp16_enable": True,
        }))
    provider_config.append(("CUDAExecutionProvider", {
        "device_id": 0,
        "cudnn_conv_algo_search": "EXHAUSTIVE",
        "arena_extend_strategy": "kNextPowerOfTwo",
        "do_copy_in_default_stream": True,
    }))

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    try:
        session = ort.InferenceSession(path, sess_options=options, providers=provider_config)
    except Exception as exc:
        raise GPUUnavailableError(
            f"CUDA 模型加载失败，CPU fallback 已禁用。\n模型: {path}\n原因: {exc}"
        ) from exc
    active = session.get_providers()
    if not active or active[0] not in CUDA_PROVIDERS:
        raise GPUUnavailableError(f"模型未在 CUDA/TensorRT GPU provider 上运行：{active}")
    return session


def create_gpu_session(
    model_path: str | Path,
    prefer_tensorrt: bool = False,
    backend: str = "auto",
) -> Any:
    selected = resolve_gpu_backend(backend)
    if selected == "cuda":
        return _create_cuda_session(model_path, prefer_tensorrt)
    return OpenVINOInferenceSession(model_path)
