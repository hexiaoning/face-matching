from __future__ import annotations

import ctypes
import sys
from typing import Any

from face_matching.errors import GpuUnavailableError


def verify_cuda_device() -> int:
    """Initialize the NVIDIA CUDA driver and return the number of visible devices."""
    library_name = "nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1"
    loader = ctypes.WinDLL if sys.platform == "win32" else ctypes.CDLL
    try:
        driver = loader(library_name)
    except OSError as exc:
        raise GpuUnavailableError(
            "无法加载 NVIDIA CUDA 驱动。请安装或更新 NVIDIA 官方显卡驱动；本程序禁止 CPU 推理。"
        ) from exc
    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuInit.restype = ctypes.c_int
    driver.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    driver.cuDeviceGetCount.restype = ctypes.c_int
    init_result = int(driver.cuInit(0))
    if init_result != 0:
        raise GpuUnavailableError(
            f"NVIDIA CUDA 驱动初始化失败（错误码 {init_result}）；本程序禁止 CPU 推理。"
        )
    count = ctypes.c_int()
    count_result = int(driver.cuDeviceGetCount(ctypes.byref(count)))
    if count_result != 0 or count.value < 1:
        raise GpuUnavailableError(
            f"未发现可用 NVIDIA CUDA GPU（错误码 {count_result}）；本程序禁止 CPU 推理。"
        )
    return count.value


def load_onnxruntime_gpu() -> Any:
    verify_cuda_device()
    try:
        import onnxruntime as ort
    except (ImportError, OSError) as exc:
        raise GpuUnavailableError(
            "无法加载 ONNX Runtime GPU。请运行 install.ps1 安装 CUDA/cuDNN 运行库。"
        ) from exc

    try:
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls(directory="")
    except Exception as exc:
        raise GpuUnavailableError(
            "CUDA/cuDNN DLL 加载失败。请更新 NVIDIA 驱动后重新运行 install.ps1。"
        ) from exc

    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        raise GpuUnavailableError(
            "未发现 CUDAExecutionProvider；本程序禁止 CPU 推理。"
            "请确认 NVIDIA 驱动与 GPU 版依赖已安装。"
        )
    return ort


def create_cuda_session(model_path: str, device_id: int = 0) -> Any:
    ort = load_onnxruntime_gpu()
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    provider_options = {
        "device_id": str(device_id),
        "arena_extend_strategy": "kNextPowerOfTwo",
        "cudnn_conv_algo_search": "DEFAULT",
        "do_copy_in_default_stream": "1",
    }
    try:
        session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=[("CUDAExecutionProvider", provider_options)],
        )
    except Exception as exc:
        raise GpuUnavailableError(
            f"模型无法在 CUDA GPU {device_id} 上创建会话；CPU 回退已禁用。详情：{exc}"
        ) from exc
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise GpuUnavailableError("CUDA 会话未成为首选执行器；已拒绝继续运行。")
    return session
