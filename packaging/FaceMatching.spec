from pathlib import Path
from importlib.util import find_spec
import os

from PyInstaller.utils.hooks import collect_all, copy_metadata


project_root = Path(os.environ.get("FACE_MATCHING_PROJECT_ROOT", Path.cwd())).resolve()
model_root_value = os.environ.get("FACE_MATCHING_BUNDLE_MODEL_DIR")
if not model_root_value:
    raise SystemExit("FACE_MATCHING_BUNDLE_MODEL_DIR must point to the verified offline models")
model_root = Path(model_root_value).resolve()
if not model_root.is_dir():
    raise SystemExit(f"Offline model directory does not exist: {model_root}")

datas = [
    (str(model_root), "models"),
    (str(project_root / "README.md"), "."),
    (str(project_root / "THIRD_PARTY_NOTICES.md"), "."),
    (str(project_root / "docs" / "offline-deployment.md"), "docs"),
    (str(project_root / "packaging" / "verify_offline.ps1"), "."),
]
binaries = []
hiddenimports = []


def module_available(name):
    try:
        return find_spec(name) is not None
    except (ImportError, ModuleNotFoundError):
        return False

# PyInstaller's standard hooks cover Qt/OpenCV. ONNX Runtime, OpenVINO, and the
# CUDA wheel namespace need explicit collection so no target-machine package
# installation or CUDA/OpenVINO Toolkit is required.
for module_name in (
    "onnxruntime",
    "openvino",
    "nvidia.cuda_runtime",
    "nvidia.cublas",
    "nvidia.cudnn",
    "nvidia.cufft",
    "nvidia.curand",
    "nvidia.nvjitlink",
):
    if module_available(module_name):
        module_datas, module_binaries, module_hidden = collect_all(module_name)
        datas += module_datas
        binaries += module_binaries
        hiddenimports += module_hidden

for distribution in (
    "face-matching",
    "numpy",
    "opencv-python-headless",
    "PySide6",
    "shiboken6",
    "onnxruntime-gpu",
    "openvino",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cublas-cu12",
    "nvidia-cudnn-cu12",
    "nvidia-cufft-cu12",
    "nvidia-curand-cu12",
    "nvidia-nvjitlink-cu12",
):
    try:
        datas += copy_metadata(distribution, recursive=True)
    except Exception:
        pass

a = Analysis(
    [str(project_root / "packaging" / "launcher.py")],
    pathex=[str(project_root / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(project_root / "packaging" / "pyi_runtime_hook.py")],
    excludes=["IPython", "matplotlib", "pytest", "tkinter"],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="FaceMatching",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch="x86_64",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="FaceMatching-v2.4.0-windows-x64",
)
