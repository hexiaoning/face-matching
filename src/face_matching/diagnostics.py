from __future__ import annotations

import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .gpu import assert_gpu_available, create_gpu_session, resolve_gpu_backend
from .models import feature_model_id, profile_spec, required_paths


def _real_inference(session: Any, default_size: int) -> dict[str, Any]:
    input_meta = session.get_inputs()[0]
    raw_shape = list(input_meta.shape)
    shape = [
        int(value) if isinstance(value, int) and value > 0 else (default_size if index >= 2 else 1 if index == 0 else 3)
        for index, value in enumerate(raw_shape)
    ]
    tensor = np.zeros(shape, dtype=np.float32)
    started = time.perf_counter()
    outputs = session.run(None, {input_meta.name: tensor})
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if not outputs:
        raise RuntimeError("模型 GPU 自检没有返回输出")
    for output in outputs:
        array = np.asarray(output)
        if array.size and not np.isfinite(array).all():
            raise RuntimeError("模型 GPU 自检返回 NaN/Inf")
    return {
        "input_shape": shape,
        "output_shapes": [list(np.asarray(output).shape) for output in outputs],
        "elapsed_ms": round(elapsed_ms, 2),
        "provider": session.get_providers()[0],
    }


def run_diagnostics(
    profile: str,
    model_root: Path | None = None,
    backend: str = "auto",
    device_id: int = 0,
    mirror_augmentation: bool = True,
) -> dict[str, Any]:
    """Hash-check both models and execute one real GPU inference per model."""
    selected_backend = resolve_gpu_backend(backend, device_id)
    gpu = assert_gpu_available(selected_backend, device_id)
    spec = profile_spec(profile)
    detector_path, recognizer_path = required_paths(profile, model_root, verify_hash=True)
    detector = create_gpu_session(detector_path, device_id, selected_backend)
    recognizer = create_gpu_session(recognizer_path, device_id, selected_backend)
    return {
        "ok": True,
        "application_version": __version__,
        "platform": platform.platform(),
        "gpu_runtime": gpu,
        "profile": spec.key,
        "backend": selected_backend,
        "model_id": feature_model_id(profile, mirror_augmentation),
        "models": {
            "detector": str(detector_path),
            "recognizer": str(recognizer_path),
        },
        "inference": {
            "detector": _real_inference(detector, 640),
            "recognizer": _real_inference(recognizer, 112),
        },
    }
