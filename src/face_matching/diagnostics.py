from __future__ import annotations

import argparse
import json
import platform
import sys

import onnxruntime as ort

from .config import EngineConfig
from .engine import FaceEngine
from .gpu import assert_gpu_available, available_gpu_providers
from .model_manager import required_models


def collect() -> dict[str, object]:
    config = EngineConfig()
    providers = ort.get_available_providers()
    gpu = available_gpu_providers()
    models = {str(path): path.is_file() for path in required_models(config)}
    try:
        selected_provider = assert_gpu_available(
            config.gpu_backend, config.prefer_tensorrt
        )
    except Exception as exc:
        selected_provider = None
        gpu_error = str(exc)
    result: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "onnxruntime": ort.__version__,
        "providers": providers,
        "requested_backend": config.gpu_backend,
        "available_gpu_providers": gpu,
        "selected_provider": selected_provider,
        "gpu_ready": selected_provider is not None,
        "models": models,
        "model_id": config.model_id,
        "inference_ready": False,
    }
    if selected_provider is None:
        result["gpu_error"] = gpu_error
    if result["gpu_ready"] and all(models.values()):
        try:
            engine = FaceEngine(config)
            result["active_provider"] = engine.provider
            result["inference_ready"] = True
        except Exception as exc:
            result["inference_error"] = str(exc)
    return result


def exit_code(result: dict[str, object]) -> int:
    if not all(result["models"].values()):
        return 2
    if not result["gpu_ready"] or not result["inference_ready"]:
        return 3
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Face matching installation diagnostics")
    parser.parse_args(argv)
    result = collect()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return exit_code(result)


if __name__ == "__main__":
    raise SystemExit(main())
