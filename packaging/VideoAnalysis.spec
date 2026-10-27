from pathlib import Path
from importlib.util import find_spec
import os
import runpy
import sys

from PyInstaller.utils.hooks import collect_all, copy_metadata
from PyInstaller.utils.win32.versioninfo import (
    FixedFileInfo, StringFileInfo, StringStruct, StringTable, VarFileInfo, VarStruct, VSVersionInfo,
)

root = Path(os.environ["FACE_MATCHING_PROJECT_ROOT"]).resolve()
model_root = Path(os.environ["FACE_MATCHING_BUNDLE_MODEL_DIR"]).resolve()
version = runpy.run_path(str(root / "src/face_matching/__init__.py"))["__version__"]
model = model_root / "rtdetr_v2_r50vd.onnx"
if not model.is_file():
    raise SystemExit("The verified RT-DETRv2 model is required")

datas = [
    (str(model), "models"),
    (str(root / "packaging/licenses/RT-DETR-LICENSE.txt"), "licenses"),
    (str(root / "packaging/VideoAnalysis-NOTICES.txt"), "."),
]
binaries, hiddenimports = [], []
for name in (
    "onnxruntime", "openvino", "nvidia.cuda_runtime", "nvidia.cublas",
    "nvidia.cudnn", "nvidia.cufft", "nvidia.curand", "nvidia.nvjitlink",
):
    if find_spec(name) is not None:
        module_data, module_binaries, module_imports = collect_all(name)
        datas += module_data
        binaries += module_binaries
        hiddenimports += module_imports
for name in (
    "numpy", "opencv-python-headless", "PySide6", "shiboken6", "onnxruntime-gpu", "openvino",
    "nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12", "nvidia-cudnn-cu12",
    "nvidia-cufft-cu12", "nvidia-curand-cu12", "nvidia-nvjitlink-cu12",
):
    try:
        datas += copy_metadata(name, recursive=True)
    except Exception:
        pass

# The standalone release contains only the two analysis tools.
excluded = [
    "face_matching.__main__", "face_matching.config", "face_matching.engine",
    "face_matching.database", "face_matching.enrollment", "face_matching.matching",
    "face_matching.annotations", "face_matching.annotation_export", "face_matching.diagnostics",
    "face_matching.gui.main_window", "face_matching.gui.preview", "face_matching.gui.person_photos",
    "face_matching.gui.annotation_compare", "face_matching.gui.export_dialog", "face_matching.vision",
    "IPython", "matplotlib", "pytest", "tkinter",
]
a = Analysis(
    [str(root / "packaging/video_analysis_launcher.py")],
    pathex=[str(root / "src")], binaries=binaries, datas=datas, hiddenimports=hiddenimports,
    hookspath=[], hooksconfig={}, runtime_hooks=[str(root / "packaging/pyi_runtime_hook.py")],
    excludes=excluded, noarchive=False, optimize=1,
)

runtime_root = Path(sys.prefix).resolve()
def external_icu(entry):
    name = Path(entry[0]).name.lower()
    is_icu = name == "icuuc.dll" or (name.startswith(("icudt", "icuin")) and name.endswith(".dll"))
    return is_icu and not Path(entry[1]).resolve().is_relative_to(runtime_root)
a.binaries = [entry for entry in a.binaries if not external_icu(entry)]

parts = tuple(int(part) for part in version.split(".")) + (0,)
version_info = VSVersionInfo(
    ffi=FixedFileInfo(filevers=parts, prodvers=parts, mask=0x3F, flags=0,
                      OS=0x40004, fileType=1, subtype=0, date=(0, 0)),
    kids=[StringFileInfo([StringTable("040904B0", [
        StringStruct("FileDescription", "Offline Video Analysis"),
        StringStruct("FileVersion", version), StringStruct("ProductName", "VideoAnalysis"),
        StringStruct("ProductVersion", version), StringStruct("OriginalFilename", "VideoAnalysis.exe"),
    ])]), VarFileInfo([VarStruct("Translation", [1033, 1200])])],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="VideoAnalysis", version=version_info,
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False, console=False,
          disable_windowed_traceback=False, argv_emulation=False, target_arch="x86_64")
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False,
               name=f"VideoAnalysis-v{version}-windows-x64")
