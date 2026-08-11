from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from face_match.errors import GpuUnavailableError

CUDA_PROVIDER = "CUDAExecutionProvider"


@dataclass(frozen=True)
class GpuInfo:
    provider: str
    device_name: str
    driver_version: str
    memory_total: str
    runtime_version: str


def _import_onnxruntime() -> ModuleType:
    try:
        import onnxruntime as ort
    except Exception as exc:  # DLL load errors must be presented as GPU setup failures.
        raise GpuUnavailableError(
            f"无法加载 ONNX Runtime GPU。请运行 install_and_run.bat 重新安装依赖。\n底层错误：{exc}"
        ) from exc
    return ort


def prepare_cuda_runtime(ort: ModuleType | Any | None = None) -> ModuleType | Any:
    """Load bundled CUDA libraries and fail instead of allowing CPU inference."""
    ort = ort or _import_onnxruntime()
    preload = getattr(ort, "preload_dlls", None)
    if preload is not None:
        try:
            # Empty directory tells ORT to load NVIDIA wheels from site-packages.
            preload(directory="")
        except TypeError:
            preload()
        except Exception as exc:
            raise GpuUnavailableError(
                "CUDA/cuDNN 运行库加载失败。无需安装完整 CUDA Toolkit；请重新运行一键安装脚本，"
                "并确认 NVIDIA 驱动为最新版本。"
                f"\n底层错误：{exc}"
            ) from exc
    providers = list(ort.get_available_providers())
    if CUDA_PROVIDER not in providers:
        raise GpuUnavailableError(
            "未检测到 CUDAExecutionProvider。程序按要求禁止使用 CPU 推理。\n"
            f"当前可用执行器：{', '.join(providers) or '无'}\n"
            "请确认使用 NVIDIA 显卡驱动，并安装 onnxruntime-gpu（不能安装 onnxruntime CPU 包）。"
        )
    return ort


def create_cuda_session(model_path: Path, ort: ModuleType | Any | None = None) -> Any:
    ort = prepare_cuda_runtime(ort)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # This makes session construction fail when any model node would need CPU inference.
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    provider_options = {
        "device_id": "0",
        "arena_extend_strategy": "kNextPowerOfTwo",
        "cudnn_conv_algo_search": "HEURISTIC",
        "do_copy_in_default_stream": "1",
        "use_tf32": "1",
    }
    try:
        session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=[(CUDA_PROVIDER, provider_options)],
        )
    except Exception as exc:
        raise GpuUnavailableError(
            f"模型无法在 CUDA GPU 上完整加载：{model_path.name}。CPU 回退已禁用。\n"
            "请更新 NVIDIA 驱动或重新安装 GPU 依赖。"
            f"\n底层错误：{exc}"
        ) from exc
    active = list(session.get_providers())
    if not active or active[0] != CUDA_PROVIDER:
        raise GpuUnavailableError(
            f"模型 {model_path.name} 未使用 CUDA（实际执行器：{active}），已拒绝启动。"
        )
    disable_fallback = getattr(session, "disable_fallback", None)
    if disable_fallback is not None:
        disable_fallback()
    return session


def _nvidia_smi_path() -> str | None:
    found = shutil.which("nvidia-smi")
    if found:
        return found
    if os.name == "nt":
        candidates = [
            Path(os.environ.get("ProgramFiles", "C:/Program Files"))
            / "NVIDIA Corporation"
            / "NVSMI"
            / "nvidia-smi.exe",
            Path("C:/Windows/System32/nvidia-smi.exe"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
    return None


def gpu_info(ort: ModuleType | Any | None = None) -> GpuInfo:
    ort = prepare_cuda_runtime(ort)
    name, memory, driver = "NVIDIA GPU", "未知", "未知"
    executable = _nvidia_smi_path()
    if executable:
        try:
            completed = subprocess.run(
                [
                    executable,
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            first_line = completed.stdout.strip().splitlines()[0]
            fields = [part.strip() for part in first_line.split(",")]
            if len(fields) >= 3:
                name, memory, driver = fields[0], f"{fields[1]} MiB", fields[2]
        except (OSError, subprocess.SubprocessError, IndexError):
            pass
    return GpuInfo(CUDA_PROVIDER, name, driver, memory, str(ort.__version__))
