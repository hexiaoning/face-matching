# -*- mode: python ; coding: utf-8 -*-
from __future__ import annotations

import os
import site
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all


project_root = Path(SPECPATH).parent
model_root = Path(os.environ["FACE_MATCHING_BUNDLE_MODELS"]).resolve()
model_names = ("scrfd_10g_bnkps.onnx", "LVFace-B_Glint360K.onnx")
missing = [name for name in model_names if not (model_root / name).is_file()]
if missing:
    raise SystemExit(f"Offline bundle models are missing: {', '.join(missing)}")

datas = [(str(model_root / name), "models") for name in model_names]
binaries = []
hiddenimports = []
for package in ("onnxruntime", "cv2", "PySide6"):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

# nvidia-* wheels use a shared namespace package. Copy every runtime DLL and
# its metadata so ORT can preload CUDA/cuDNN without a CUDA Toolkit install.
site_roots = [Path(value) for value in site.getsitepackages()]
site_roots.append(Path(sys.prefix) / "Lib" / "site-packages")
for root in dict.fromkeys(site_roots):
    nvidia_root = root / "nvidia"
    if not nvidia_root.is_dir():
        continue
    for source in nvidia_root.rglob("*"):
        if not source.is_file():
            continue
        relative = source.relative_to(nvidia_root)
        destination = str(Path("nvidia") / relative.parent)
        target = (str(source), destination)
        if source.suffix.lower() in {".dll", ".pyd"}:
            binaries.append(target)
        else:
            datas.append(target)

a = Analysis(
    [str(project_root / "packaging" / "launcher.py")],
    pathex=[str(project_root / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["onnxruntime.tools", "tkinter"],
    noarchive=False,
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
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="FaceMatching",
)
