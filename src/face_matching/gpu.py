from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
from typing import Any

from .errors import GPUUnavailableError


def _load_ort() -> Any:
    try:
        import onnxruntime as ort
    except Exception as exc:
        raise GPUUnavailableError(
            "未安装 ONNX Runtime GPU。请双击 install.ps1 完成安装。"
        ) from exc
    try:
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls(directory="")
    except Exception as exc:
        raise GPUUnavailableError(
            "CUDA/cuDNN 运行库加载失败。请重新运行 install.ps1，并确认 NVIDIA 驱动已更新。"
        ) from exc
    return ort


def assert_cuda_available() -> str:
    try:
        gpu_package_version = importlib.metadata.version("onnxruntime-gpu")
    except importlib.metadata.PackageNotFoundError as exc:
        cpu_hint = ""
        try:
            cpu_hint = f" 检测到 CPU 包 onnxruntime {importlib.metadata.version('onnxruntime')}。"
        except importlib.metadata.PackageNotFoundError:
            pass
        raise GPUUnavailableError(
            "未安装 onnxruntime-gpu；本程序不允许 CPU 推理。"
            "请运行 install.cmd 安装隔离的 GPU 运行环境。" + cpu_hint
        ) from exc
    ort = _load_ort()
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        raise GPUUnavailableError(
            "未发现 CUDAExecutionProvider；本程序禁止 CPU 推理。"
            "请确认使用 NVIDIA 显卡、安装最新版驱动并运行 install.cmd。"
        )
    # The Windows NVIDIA display driver exposes nvcuda.dll even when the full
    # CUDA toolkit is intentionally not installed. The pip extras provide the
    # remaining CUDA/cuDNN runtime DLLs used by ONNX Runtime.
    if os.name == "nt":
        try:
            import ctypes

            ctypes.WinDLL("nvcuda.dll")
        except OSError as exc:
            raise GPUUnavailableError(
                "无法加载 NVIDIA 驱动 nvcuda.dll。请安装/更新 NVIDIA 显卡驱动。"
            ) from exc
    return (
        f"ONNX Runtime GPU {gpu_package_version} "
        f"(runtime {ort.__version__}) / CUDAExecutionProvider"
    )


def create_gpu_session(model_path: Path, device_id: int = 0) -> Any:
    ort = _load_ort()
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        assert_cuda_available()
    options = ort.SessionOptions()
    options.log_severity_level = 3
    # CPU EP is normally appended by ORT. This makes any unsupported CUDA node a hard error.
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
            f"模型无法在 CUDA GPU 上完整加载（不会回退 CPU）: {model_path.name}\n{exc}"
        ) from exc
    if not session.get_providers() or session.get_providers()[0] != "CUDAExecutionProvider":
        raise GPUUnavailableError(f"{model_path.name} 未使用 CUDAExecutionProvider")
    return session
