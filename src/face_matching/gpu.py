from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import onnxruntime as ort

from .errors import GPUUnavailableError


NVIDIA_PROVIDERS = ("TensorrtExecutionProvider", "CUDAExecutionProvider")
GPU_PROVIDERS = NVIDIA_PROVIDERS + ("DmlExecutionProvider",)
_DLL_DIRECTORY_HANDLES: list[Any] = []


def preload_cuda_runtime() -> None:
    """Load pip-installed CUDA/cuDNN DLLs before provider discovery on Windows."""
    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        return
    directory = ""
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        bundled = Path(frozen_root) / "cuda_dlls"
        if bundled.is_dir():
            directory = str(bundled)
            if os.name == "nt" and hasattr(os, "add_dll_directory"):
                handle = os.add_dll_directory(directory)
                _DLL_DIRECTORY_HANDLES.append(handle)
    try:
        preload(directory=directory)
    except Exception:
        # Provider validation below produces a concise, actionable error.
        pass


def available_gpu_providers() -> list[str]:
    preload_cuda_runtime()
    available = ort.get_available_providers()
    return [provider for provider in GPU_PROVIDERS if provider in available]


def assert_gpu_available(
    backend: str = "auto",
    prefer_tensorrt: bool = False,
) -> str:
    providers = available_gpu_providers()
    backend = backend.strip().lower()
    if backend not in {"auto", "cuda", "directml"}:
        raise ValueError(f"不支持的 GPU 后端：{backend}")
    selected: str | None = None
    if backend in {"auto", "cuda"} and "CUDAExecutionProvider" in providers:
        selected = (
            "TensorrtExecutionProvider"
            if prefer_tensorrt and "TensorrtExecutionProvider" in providers
            else "CUDAExecutionProvider"
        )
    if selected is None and backend in {"auto", "directml"} and "DmlExecutionProvider" in providers:
        selected = "DmlExecutionProvider"
    if selected is None:
        found = ", ".join(ort.get_available_providers()) or "none"
        if getattr(sys, "frozen", False):
            action = (
                "请确认 NVIDIA 驱动正常，并重新复制完整离线包后运行 GPU诊断.bat。"
                if backend == "cuda"
                else "请确认 Intel 核显驱动和 DirectX 12 正常，并使用 DirectML 离线包。"
            )
        else:
            action = (
                "请确认 NVIDIA 驱动正常，并重新运行 install.bat 安装 "
                "onnxruntime-gpu[cuda,cudnn]。"
                if backend == "cuda"
                else "请运行 install_intel.bat 安装独立的 DirectML 环境。"
            )
        raise GPUUnavailableError(
            f"未检测到请求的 GPU 后端（{backend}），程序拒绝使用 CPU 推理。\n\n"
            f"当前 provider: {found}\n"
            f"{action}"
        )
    return selected


def create_gpu_session(
    model_path: str | Path,
    prefer_tensorrt: bool = False,
    device_id: int = 0,
    backend: str = "auto",
) -> ort.InferenceSession:
    selected = assert_gpu_available(backend, prefer_tensorrt)
    path = str(Path(model_path).resolve())
    provider_config: list[Any] = []
    if selected == "TensorrtExecutionProvider":
        cache_dir = str(Path(path).parent / "tensorrt_cache")
        os.makedirs(cache_dir, exist_ok=True)
        provider_config.append(("TensorrtExecutionProvider", {
            "device_id": device_id,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": cache_dir,
            "trt_fp16_enable": True,
        }))
        provider_config.append(("CUDAExecutionProvider", {
            "device_id": device_id,
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "arena_extend_strategy": "kNextPowerOfTwo",
            "do_copy_in_default_stream": True,
        }))
    elif selected == "CUDAExecutionProvider":
        provider_config.append(("CUDAExecutionProvider", {
            "device_id": device_id,
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "arena_extend_strategy": "kNextPowerOfTwo",
            "do_copy_in_default_stream": True,
        }))
    else:
        provider_config.append(("DmlExecutionProvider", {"device_id": device_id}))

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if selected == "DmlExecutionProvider":
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    try:
        session = ort.InferenceSession(path, sess_options=options, providers=provider_config)
    except Exception as exc:
        raise GPUUnavailableError(
            f"GPU 模型加载失败，CPU fallback 已禁用。\n模型: {path}\n原因: {exc}"
        ) from exc
    active = session.get_providers()
    if selected not in active:
        raise GPUUnavailableError(f"模型未在 GPU provider 上运行：{active}")
    return session
