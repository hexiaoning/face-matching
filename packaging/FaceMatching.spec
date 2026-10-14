from pathlib import Path
from importlib.util import find_spec
import os
import sys

from PyInstaller.utils.hooks import collect_all, copy_metadata
from PyInstaller.building.datastruct import TOC


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

# PySide6's Qt6Core uses the unversioned Windows ICU compatibility shim.
# Never let an unrelated icuuc.dll discovered on the build machine's PATH
# shadow the system shim. This previously collected Codex/Poppler's ICU,
# whose version-suffixed exports made QtWidgets fail with ERROR_PROC_NOT_FOUND.
runtime_root = Path(sys.prefix).resolve()


def is_external_icu(entry):
    destination = Path(entry[0]).name.lower()
    is_icu = (
        destination == "icuuc.dll"
        or destination.startswith("icudt") and destination.endswith(".dll")
        or destination.startswith("icuin") and destination.endswith(".dll")
    )
    if not is_icu:
        return False
    source = Path(entry[1]).resolve()
    return not source.is_relative_to(runtime_root)


a.binaries = TOC(entry for entry in a.binaries if not is_external_icu(entry))
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
    name="FaceMatching-v2.5.2-windows-x64",
)
