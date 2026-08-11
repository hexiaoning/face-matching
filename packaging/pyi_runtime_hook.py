from __future__ import annotations

import os
import sys
from pathlib import Path


_DLL_HANDLES: list[object] = []

bundle_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
os.environ.setdefault("FACE_MATCHING_BUNDLED_MODELS", str(bundle_root / "models"))

if os.name == "nt":
    candidates = (
        bundle_root / "nvidia" / "cuda_runtime" / "bin",
        bundle_root / "nvidia" / "cublas" / "bin",
        bundle_root / "nvidia" / "cudnn" / "bin",
        bundle_root / "nvidia" / "cufft" / "bin",
        bundle_root / "nvidia" / "curand" / "bin",
        bundle_root / "nvidia" / "nvjitlink" / "bin",
        bundle_root / "onnxruntime" / "capi",
    )
    existing = [str(path) for path in candidates if path.is_dir()]
    if existing:
        os.environ["PATH"] = os.pathsep.join(existing + [os.environ.get("PATH", "")])
        if hasattr(os, "add_dll_directory"):
            for path in existing:
                _DLL_HANDLES.append(os.add_dll_directory(path))
