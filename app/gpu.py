"""GPU 环境初始化与强制检查。

项目必须使用 GPU 推理。检测不到 GPU（CUDA 或 DirectML 均不可用）时
抛出 RuntimeError，绝不允许静默回退 CPU。
"""
from __future__ import annotations

import glob
import os
import sys


def setup_cuda_dlls() -> None:
    """Windows 下把 pip 安装的 nvidia-*-cu12 运行库加入 DLL 搜索路径。

    目标机不安装 CUDA 工具包，CUDA/cuDNN 以 pip 包形式随 requirements 安装，
    位于 site-packages/nvidia/*/bin。
    """
    if sys.platform != "win32":
        return
    import site

    search_dirs = set()
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        for d in glob.glob(os.path.join(sp, "nvidia", "*", "bin")):
            search_dirs.add(d)
    for d in sorted(search_dirs):
        try:
            os.add_dll_directory(d)
        except (OSError, AttributeError):
            pass
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


def _cuda_providers() -> list[str]:
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def _dml_providers() -> list[str]:
    return ["DmlExecutionProvider", "CPUExecutionProvider"]


def resolve_providers(backend: str = "auto") -> tuple[list[str], str]:
    """返回 (onnxruntime providers, 实际后端名)。

    找不到可用 GPU 后端时抛出 RuntimeError。
    """
    setup_cuda_dlls()
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    backend = (backend or "auto").lower()

    if backend in ("auto", "cuda"):
        if "CUDAExecutionProvider" in available and ort.get_device() == "GPU":
            return _cuda_providers(), "CUDA"
        if backend == "cuda":
            raise RuntimeError(
                "未检测到可用的 CUDA GPU。\n"
                "请确认：1) 已安装 NVIDIA 显卡驱动；2) 显卡未被其他程序独占；\n"
                "3) 已运行 run.bat 安装 nvidia-cudnn-cu12 等运行库。"
            )
    if backend in ("auto", "directml"):
        if "DmlExecutionProvider" in available:
            return _dml_providers(), "DirectML"
        if backend == "directml":
            raise RuntimeError("未检测到可用的 DirectML GPU。")

    raise RuntimeError(
        "未检测到任何可用的 GPU 推理后端（CUDA / DirectML）。\n"
        "本项目必须使用 GPU 运行，请安装 NVIDIA 驱动后重试。"
    )


def assert_on_gpu(session_providers: list[str], backend: str) -> None:
    """模型会话创建后校验其确实落在 GPU provider 上。"""
    expected = {"CUDA": "CUDAExecutionProvider", "DirectML": "DmlExecutionProvider"}[backend]
    if not session_providers or session_providers[0] != expected:
        raise RuntimeError(
            f"模型未能加载到 {backend} GPU 上（实际 providers: {session_providers}）。\n"
            "本项目禁止使用 CPU 推理，请检查显卡驱动与 CUDA 运行库。"
        )


def gpu_summary() -> str:
    """返回 GPU 描述信息，用于界面状态栏。"""
    try:
        setup_cuda_dlls()
        import onnxruntime as ort

        dev = ort.get_device()
        providers = ",".join(ort.get_available_providers())
        return f"onnxruntime device={dev}, providers=[{providers}]"
    except Exception as e:  # noqa: BLE001
        return f"onnxruntime 不可用: {e}"
