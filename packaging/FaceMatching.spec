from pathlib import Path
from importlib.util import find_spec
import os
import sys
import runpy

from PyInstaller.utils.hooks import collect_all, copy_metadata
from PyInstaller.building.datastruct import TOC
from PyInstaller.utils.win32.versioninfo import (
    FixedFileInfo, StringFileInfo, StringStruct, StringTable, VarFileInfo, VarStruct, VSVersionInfo,
)
from face_matching.models import required_paths, required_person_vehicle_model


project_root = Path(os.environ.get("FACE_MATCHING_PROJECT_ROOT", Path.cwd())).resolve()
model_root_value = os.environ.get("FACE_MATCHING_BUNDLE_MODEL_DIR")
if not model_root_value:
    raise SystemExit("FACE_MATCHING_BUNDLE_MODEL_DIR must point to the verified offline models")
model_root = Path(model_root_value).resolve()
if not model_root.is_dir():
    raise SystemExit(f"Offline model directory does not exist: {model_root}")
version = runpy.run_path(str(project_root / "src/face_matching/__init__.py"))["__version__"]
profiles = os.environ.get("FACE_MATCHING_BUNDLE_PROFILES", "lvface-b,auraface").split(',')
models = {required_person_vehicle_model(model_root)}
for profile in profiles:
    models.update(required_paths(profile, model_root, verify_hash=True))

datas = [
    *[(str(model), "models") for model in sorted(models)],
    (str(project_root / "src/face_matching/resources"), "face_matching/resources"),
    (str(project_root / "README.md"), "."),
    (str(project_root / "THIRD_PARTY_NOTICES.md"), "."),
    (str(project_root / "packaging" / "licenses" / "RT-DETR-LICENSE.txt"), "licenses"),
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
parts = tuple(int(part) for part in version.split('.')) + (0,)
version_info = VSVersionInfo(
    ffi=FixedFileInfo(filevers=parts, prodvers=parts, mask=0x3F, flags=0,
                      OS=0x40004, fileType=1, subtype=0, date=(0, 0)),
    kids=[StringFileInfo([StringTable("040904B0", [
        StringStruct("FileDescription", "Offline Unified Analysis"),
        StringStruct("FileVersion", version), StringStruct("ProductName", "FaceMatching"),
        StringStruct("ProductVersion", version), StringStruct("OriginalFilename", "FaceMatching.exe"),
    ])]), VarFileInfo([VarStruct("Translation", [1033, 1200])])],
)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="FaceMatching",
    version=version_info,
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
    name=f"FaceMatching-v{version}-windows-x64",
)
