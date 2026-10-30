from __future__ import annotations

import importlib.metadata
import json
import os
import tempfile
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

    def __init__(self, model_path: str | Path, device_id: int = 0,
                 input_shape: tuple[int, int, int, int] | None = None,
                 input_shapes: dict[str, tuple[int, ...]] | None = None) -> None:
        ov = _load_openvino()
        self._ov = ov
        self._core = ov.Core()
        self.device = _select_openvino_device(device_id)
        path = Path(model_path).resolve()
        try:
            model = self._core.read_model(str(path))
            if input_shape is not None:
                model.reshape({0: list(input_shape)})
            if input_shapes is not None:
                model.reshape({name: list(shape) for name, shape in input_shapes.items()})
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
        for index, metadata in enumerate(self._inputs):
            array = np.ascontiguousarray(inputs[metadata.name])
            self._request.set_input_tensor(index, self._ov.Tensor(array))
        self._request.infer()
        names = output_names or [item.name for item in self._outputs]
        return [
            np.array(self._request.get_output_tensor(self._output_indices[name]).data, copy=True)
            for name in names
        ]


_CPU_SHAPE_OPS = frozenset({
    "Shape", "Size", "Gather", "GatherElements", "Slice", "Concat",
    "Unsqueeze", "Squeeze", "Reshape", "Cast", "Identity",
    "Add", "Sub", "Mul", "Div", "Mod", "Min", "Max", "Equal", "Where",
})


def _small_integer_tensors(metadata: object) -> bool:
    """Shape helpers may manipulate small integer/bool tensors, never features."""
    if not isinstance(metadata, list) or not metadata:
        return False
    for tensor in metadata:
        if not isinstance(tensor, dict) or len(tensor) != 1:
            return False
        dtype, shape = next(iter(tensor.items()))
        if dtype not in {"int32", "int64", "bool"} or not isinstance(shape, list):
            return False
        size = 1
        for dimension in shape:
            if not isinstance(dimension, int) or dimension < 0:
                return False
            size *= dimension
        if size > 64:
            return False
    return True


def _check_cuda_profile(events: list[dict[str, Any]]) -> None:
    """Require real CUDA kernels and permit only CPU shape bookkeeping."""
    cuda_nodes = 0
    for event in events:
        if event.get("cat") != "Node":
            continue
        args = event.get("args", {})
        provider = args.get("provider")
        # ORT also records scheduling/fence events without a kernel provider.
        if not provider:
            if args.get("op_name") or event.get("name", "").endswith("_kernel_time"):
                raise GPUUnavailableError("模型计算节点缺少执行后端记录，无法确认 GPU 推理。")
            continue
        op = args.get("op_name", "")
        if provider == "CUDAExecutionProvider":
            if op and op not in {"MemcpyFromHost", "MemcpyToHost"}:
                cuda_nodes += 1
            continue
        if (
            provider == "CPUExecutionProvider"
            and op in _CPU_SHAPE_OPS
            and _small_integer_tensors(args.get("output_type_shape"))
            and (op in {"Shape", "Size"} or _small_integer_tensors(args.get("input_type_shape")))
        ):
            continue
        raise GPUUnavailableError(
            f"模型计算节点未在 CUDA GPU 上执行: {event.get('name', op)} "
            f"({op}, {provider})；仅允许 CPU 处理张量形状。"
        )
    if not cuda_nodes:
        raise GPUUnavailableError("模型自检未发现真实 CUDA 计算节点，拒绝使用 CPU 推理。")


def _verify_cuda_execution(session: Any, sample_inputs: dict[str, np.ndarray] | None = None) -> None:
    inputs = session.get_inputs()
    if sample_inputs is None:
        if len(inputs) != 1 or len(inputs[0].shape) != 4 or inputs[0].type != "tensor(float)":
            raise GPUUnavailableError("人脸模型必须只有一个 NCHW float32 输入")
        shape = [
            value if isinstance(value, int) and value > 0 else (1, 3, 640, 640)[index]
            for index, value in enumerate(inputs[0].shape)
        ]
        sample_inputs = {inputs[0].name: np.zeros(shape, dtype=np.float32)}
    outputs = session.run(None, sample_inputs)
    if not outputs or any(not np.isfinite(output).all() for output in outputs):
        raise GPUUnavailableError("CUDA 模型自检没有返回有效输出")
    profile = Path(session.end_profiling())
    _check_cuda_profile(json.loads(profile.read_text(encoding="utf-8")))


def _create_cuda_session(model_path: Path, device_id: int = 0,
                         sample_inputs: dict[str, np.ndarray] | None = None) -> Any:
    assert_cuda_available()
    ort = _load_ort()
    options = ort.SessionOptions()
    options.log_severity_level = 3
    # ORT deliberately places shape subgraphs on CPU. Rejecting every CPU node
    # prevents SCRFD from loading even when its neural network uses CUDA.
    # Audit an actual inference below instead of accepting CPU model execution.
    options.enable_profiling = True
    provider_options = {
        "device_id": str(device_id),
        "arena_extend_strategy": "kNextPowerOfTwo",
        "cudnn_conv_algo_search": "HEURISTIC",
        "do_copy_in_default_stream": "1",
        "use_tf32": "1",
    }
    try:
        with tempfile.TemporaryDirectory(prefix="face-matching-cuda-") as directory:
            options.profile_file_prefix = str(Path(directory) / "execution")
            session = None
            try:
                session = ort.InferenceSession(
                    str(model_path),
                    sess_options=options,
                    providers=[("CUDAExecutionProvider", provider_options)],
                    enable_fallback=False,
                )
                session.disable_fallback()
                providers = session.get_providers()
                if not providers or providers[0] != "CUDAExecutionProvider":
                    raise GPUUnavailableError(f"{model_path.name} 未使用 CUDAExecutionProvider")
                _verify_cuda_execution(session, sample_inputs)
            finally:
                if session is not None:
                    # Also flush on failures so Windows can remove the temporary
                    # profile. Profiling stays disabled during normal inference.
                    session.end_profiling()
    except Exception as exc:
        raise GPUUnavailableError(
            f"模型无法在 CUDA GPU {device_id} 上加载并通过推理校验: "
            f"{model_path.name}\n{exc}"
        ) from exc
    return session


def create_gpu_session(
    model_path: Path, device_id: int = 0, backend: str = "auto",
    input_shape: tuple[int, int, int, int] | None = None,
    sample_inputs: dict[str, np.ndarray] | None = None,
) -> Any:
    selected = resolve_gpu_backend(backend, device_id)
    if selected == "cuda":
        return _create_cuda_session(model_path, device_id, sample_inputs)
    if sample_inputs is not None:
        return OpenVINOInferenceSession(model_path, device_id,
                                       input_shapes={name: array.shape for name, array in sample_inputs.items()})
    if input_shape is not None:
        return OpenVINOInferenceSession(model_path, device_id, input_shape=input_shape)
    return OpenVINOInferenceSession(model_path, device_id)
