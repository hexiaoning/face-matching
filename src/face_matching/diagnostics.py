from __future__ import annotations

import argparse
import json
import platform
import sys

import onnxruntime as ort

from .config import EngineConfig
from .engine import FaceEngine
from .gpu import available_gpu_backends, available_openvino_devices, resolve_gpu_backend
from .model_manager import required_models


def collect() -> dict[str, object]:
    config = EngineConfig()
    providers = ort.get_available_providers()
    openvino_devices = available_openvino_devices()
    backends = available_gpu_backends()
    selected_backend = None
    backend_error = None
    try:
        selected_backend = resolve_gpu_backend(config.gpu_backend)
    except Exception as exc:
        backend_error = str(exc)
    models = {str(path): path.is_file() for path in required_models(config)}
    result: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "onnxruntime": ort.__version__,
        "onnxruntime_providers": providers,
        "openvino_devices": openvino_devices,
        "available_gpu_backends": backends,
        "requested_backend": config.gpu_backend,
        "selected_backend": selected_backend,
        "gpu_ready": selected_backend is not None,
        "models": models,
        "model_id": config.model_id,
        "inference_ready": False,
    }
    if backend_error:
        result["backend_error"] = backend_error
    if result["gpu_ready"] and all(models.values()):
        try:
            engine = FaceEngine(config)
            result["active_provider"] = engine.provider
            result["inference_ready"] = True
        except Exception as exc:
            result["inference_error"] = str(exc)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Face matching installation diagnostics")
    parser.parse_args(argv)
    result = collect()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    models_ok = all(result["models"].values())
    gpu_ok = bool(result["gpu_ready"])
    if not models_ok:
        return 2
    if not gpu_ok or not result["inference_ready"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
