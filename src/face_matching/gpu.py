from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import onnxruntime as ort

from .errors import GPUUnavailableError


GPU_PROVIDERS = ("TensorrtExecutionProvider", "CUDAExecutionProvider")


def preload_cuda_runtime() -> None:
    """Load pip-installed CUDA/cuDNN DLLs before provider discovery on Windows."""
    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        return
    try:
        preload(directory="")
    except Exception:
        # Provider validation below produces a concise, actionable error.
        pass


def available_gpu_providers() -> list[str]:
    preload_cuda_runtime()
    available = ort.get_available_providers()
    return [provider for provider in GPU_PROVIDERS if provider in available]


def assert_gpu_available() -> list[str]:
    providers = available_gpu_providers()
    if "CUDAExecutionProvider" not in providers:
        found = ", ".join(ort.get_available_providers()) or "none"
        raise GPUUnavailableError(
            "未检测到 ONNX Runtime CUDAExecutionProvider，程序拒绝使用 CPU 推理。\n\n"
            f"当前 provider: {found}\n"
            "请确认 NVIDIA 驱动正常，并重新运行 install.bat 安装 onnxruntime-gpu[cuda,cudnn]。"
        )
    return providers


def create_gpu_session(model_path: str | Path, prefer_tensorrt: bool = False) -> ort.InferenceSession:
    available = assert_gpu_available()
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
    if not any(provider in active for provider in GPU_PROVIDERS):
        raise GPUUnavailableError(f"模型未在 GPU provider 上运行：{active}")
    return session
