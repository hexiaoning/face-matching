from __future__ import annotations

import subprocess
from typing import Any

from .domain import GPUInfo


class GPUUnavailableError(RuntimeError):
    """Raised when the mandatory CUDA execution path cannot be used."""


def _query_nvidia_smi() -> GPUInfo:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        raise GPUUnavailableError(
            "未检测到可用的 NVIDIA 驱动（nvidia-smi 不可用）。\n"
            "本项目禁止 CPU 回退，请先安装或更新 NVIDIA 显卡驱动；不需要单独安装 CUDA Toolkit。"
        ) from exc

    first_line = next((line for line in completed.stdout.splitlines() if line.strip()), "")
    parts = [part.strip() for part in first_line.split(",")]
    if len(parts) < 2:
        raise GPUUnavailableError("nvidia-smi 未返回有效的 GPU 信息。")
    try:
        memory = int(float(parts[2])) if len(parts) >= 3 else None
    except ValueError:
        memory = None
    return GPUInfo(name=parts[0], driver_version=parts[1], memory_mb=memory)


def require_cuda_runtime() -> tuple[Any, GPUInfo]:
    """Load ONNX Runtime and reject machines that cannot expose its CUDA EP."""

    gpu_info = _query_nvidia_smi()
    try:
        import onnxruntime as ort
    except (ImportError, OSError) as exc:
        raise GPUUnavailableError(
            "未安装 ONNX Runtime GPU 运行时。请重新运行 install.bat。"
        ) from exc

    try:
        preload = getattr(ort, "preload_dlls", None)
        if preload is not None:
            preload()
    except Exception as exc:
        raise GPUUnavailableError(
            "CUDA/cuDNN 运行库加载失败。请重新运行 install.bat，并确认 NVIDIA 驱动为较新版本。\n"
            f"底层错误：{exc}"
        ) from exc

    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        raise GPUUnavailableError(
            "当前 Python 环境没有 CUDAExecutionProvider。可能误装了 CPU 版 onnxruntime；"
            "请重新运行 install.bat。"
        )
    return ort, gpu_info


def create_cuda_session(ort: Any, model_path: str) -> Any:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    # If CUDA cannot execute a node, fail instead of silently making inference slow on CPU.
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    provider_options = {
        "device_id": "0",
        "arena_extend_strategy": "kSameAsRequested",
        "cudnn_conv_algo_search": "HEURISTIC",
        "do_copy_in_default_stream": "1",
        "use_tf32": "1",
    }
    try:
        session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=[("CUDAExecutionProvider", provider_options)],
        )
    except Exception as exc:
        raise GPUUnavailableError(
            "模型无法在 CUDA 上加载，程序不会回退到 CPU。请检查 NVIDIA 驱动及 CUDA/cuDNN pip 依赖。\n"
            f"模型：{model_path}\n底层错误：{exc}"
        ) from exc
    if "CUDAExecutionProvider" not in session.get_providers():
        raise GPUUnavailableError("模型会话未启用 CUDAExecutionProvider，已拒绝启动。")
    return session

