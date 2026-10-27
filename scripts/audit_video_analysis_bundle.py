"""Verify the standalone release's payload and offline dependencies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PyInstaller.archive.readers import CArchiveReader
from face_matching.models import required_person_vehicle_model


def audit(root: Path) -> dict:
    model_root = root / "_internal/models"
    model = required_person_vehicle_model(model_root)
    if {path.name for path in model_root.iterdir()} != {model.name}:
        raise RuntimeError("Unexpected files in the standalone model directory")
    archive = CArchiveReader(str(root / "VideoAnalysis.exe"))
    modules = archive.open_embedded_archive("PYZ.pyz").toc
    forbidden = {
        "face_matching.__main__", "face_matching.engine", "face_matching.database", "face_matching.enrollment",
        "face_matching.matching", "face_matching.gui.main_window", "face_matching.config", "face_matching.vision",
    }
    unexpected = [name for name in modules if any(name == item or name.startswith(item + ".") for item in forbidden)]
    if unexpected:
        raise RuntimeError(f"Unexpected identity modules: {unexpected}")
    dlls = {path.name.lower() for path in (root / "_internal").rglob("*.dll")}
    required = ["qwindows.dll", "qt6widgets.dll", "onnxruntime_providers_cuda.dll",
                "openvino_intel_gpu_plugin.dll", "cublas64_12.dll", "cublaslt64_12.dll",
                "cudart64_12.dll", "cudnn64_9.dll"]
    missing = [name for name in required if name not in dlls]
    if missing:
        raise RuntimeError(f"Missing offline runtime DLLs: {missing}")
    if "icuuc.dll" in dlls:
        raise RuntimeError("External ICU shim must not shadow the Windows system DLL")
    return {"ok": True, "model": model.name, "identity_modules": unexpected,
            "python_modules": len(modules), "dll_count": len(dlls), "checked_dlls": required}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.directory), indent=2))
