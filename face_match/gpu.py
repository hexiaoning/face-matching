"""GPU 强制检查与 CUDA 运行时集成。

目标机器（Windows）不要求预装 CUDA Toolkit：CUDA 运行时库通过 pip 的
nvidia-cuda-runtime-cu12 / nvidia-cudnn-cu12 / nvidia-cublas-cu12 等包提供，
启动时把它们的 bin 目录加入 DLL 搜索路径。
"""
from __future__ import annotations

import os
import site
import sys
from pathlib import Path


class GPUUnavailableError(RuntimeError):
    """没有可用 GPU / CUDA 环境时抛出，程序应报错退出。"""


def _nvidia_lib_dirs() -> list[Path]:
    """找到 pip 安装的 nvidia-* 包中的动态库目录。"""
    dirs: list[Path] = []
    roots: list[Path] = []
    for p in site.getsitepackages() + [site.getusersitepackages()]:
        roots.append(Path(p))
    # venv / conda 场景
    roots.append(Path(sys.prefix) / "Lib" / "site-packages")
    roots.append(Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages")
    for root in roots:
        nv = root / "nvidia"
        if not nv.is_dir():
            continue
        for sub in ("bin", "lib"):
            for pkg_dir in nv.iterdir():
                cand = pkg_dir / sub
                if cand.is_dir() and cand not in dirs:
                    dirs.append(cand)
    return dirs


def integrate_cuda_libs() -> list[str]:
    """把 nvidia pip 包自带的 CUDA DLL/so 目录注册进动态库搜索路径。

    返回已注册的目录列表（便于日志/诊断）。Linux 下同步追加到 LD_LIBRARY_PATH
    对本进程无效，因此仅做 PATH / add_dll_directory 处理；Linux 部署建议
    通过安装脚本写环境变量。
    """
    added: list[str] = []
    for d in _nvidia_lib_dirs():
        s = str(d)
        if os.name == "nt":
            try:
                os.add_dll_directory(s)
            except (OSError, AttributeError):
                pass
        os.environ["PATH"] = s + os.pathsep + os.environ.get("PATH", "")
        added.append(s)
    return added


def cuda_providers() -> list[str]:
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def ensure_gpu(det_model_path: str | None = None) -> dict:
    """强制检查 GPU 可用性。

    1. 集成 pip CUDA 运行时库；
    2. CUDAExecutionProvider 必须在可用 provider 列表中；
    3. 若给出检测模型路径，则实际创建一个 CUDA session 做最终验证。

    失败一律抛出 GPUUnavailableError —— 不允许回退 CPU。
    """
    added = integrate_cuda_libs()

    try:
        import onnxruntime as ort
    except ImportError as e:
        raise GPUUnavailableError(
            "未安装 onnxruntime-gpu。请运行 install.bat / pip install -r requirements.txt"
        ) from e

    avail = ort.get_available_providers()
    if "CUDAExecutionProvider" not in avail:
        raise GPUUnavailableError(
            "未检测到可用的 CUDA GPU 环境。\n"
            f"onnxruntime 可用 provider: {avail}\n"
            "请确认：1) 机器有 NVIDIA 显卡且驱动正常(nvidia-smi 可用)；"
            "2) 已安装 onnxruntime-gpu 而非 onnxruntime；"
            "3) CUDA 运行时库已随依赖安装。"
        )

    info = {"providers": avail, "cuda_dll_dirs": added, "device": "unknown"}
    try:
        info["device"] = ort.get_device()
    except Exception:
        pass

    if det_model_path:
        try:
            sess = ort.InferenceSession(det_model_path, providers=["CUDAExecutionProvider"])
            used = sess.get_providers()
            if "CUDAExecutionProvider" not in used:
                raise GPUUnavailableError(f"模型会话未使用 GPU: {used}")
            info["session_providers"] = used
        except GPUUnavailableError:
            raise
        except Exception as e:
            raise GPUUnavailableError(f"GPU 推理会话创建失败: {e}") from e

    return info
