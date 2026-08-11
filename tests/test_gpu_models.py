from types import SimpleNamespace

import pytest

from face_matching import gpu
from face_matching.errors import GPUUnavailableError
from face_matching import models
from face_matching.models import ModelFile, ModelProfile, file_sha256, is_valid_model


class FakeOptions:
    def __init__(self):
        self.log_severity_level = None
        self.entries = {}

    def add_session_config_entry(self, name, value):
        self.entries[name] = value


class FakeOrt:
    __version__ = "test"

    def __init__(self):
        self.options = None
        self.providers = None

    def get_available_providers(self):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def SessionOptions(self):
        self.options = FakeOptions()
        return self.options

    def InferenceSession(self, path, sess_options, providers):
        self.providers = providers
        return SimpleNamespace(get_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])


def test_gpu_package_is_a_hard_requirement(monkeypatch):
    def version(name):
        if name == "onnxruntime-gpu":
            raise gpu.importlib.metadata.PackageNotFoundError(name)
        if name == "onnxruntime":
            return "1.24.0"
        raise AssertionError(name)

    monkeypatch.setattr(gpu.importlib.metadata, "version", version)
    with pytest.raises(GPUUnavailableError, match="onnxruntime-gpu"):
        gpu.assert_cuda_available()


def test_session_disables_cpu_execution_fallback(monkeypatch, tmp_path):
    runtime = FakeOrt()
    monkeypatch.setattr(gpu, "_load_ort", lambda: runtime)
    session = gpu.create_gpu_session(tmp_path / "model.onnx", device_id=2)

    assert session.get_providers()[0] == "CUDAExecutionProvider"
    assert runtime.options.entries["session.disable_cpu_ep_fallback"] == "1"
    assert runtime.providers == [
        (
            "CUDAExecutionProvider",
            {
                "device_id": "2",
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_algo_search": "HEURISTIC",
                "do_copy_in_default_stream": "1",
                "use_tf32": "1",
            },
        )
    ]


def test_model_file_size_and_sha256_are_verified(tmp_path):
    target = tmp_path / "model.onnx"
    target.write_bytes(b"known model payload")
    digest = file_sha256(target)
    spec = ModelFile("model.onnx", "https://example.invalid/model", digest, target.stat().st_size)

    assert is_valid_model(target, spec, verify_hash=True)
    target.write_bytes(b"tampered payload")
    assert not is_valid_model(target, spec, verify_hash=True)


def test_required_paths_falls_back_to_packaged_models(monkeypatch, tmp_path):
    user_models = tmp_path / "user-models"
    bundled_models = tmp_path / "bundle" / "models"
    bundled_models.mkdir(parents=True)
    detector = bundled_models / "detector.onnx"
    recognizer = bundled_models / "recognizer.onnx"
    detector.write_bytes(b"detector")
    recognizer.write_bytes(b"recognizer")
    detector_spec = ModelFile(
        detector.name, "https://example.invalid/detector", file_sha256(detector), detector.stat().st_size
    )
    recognizer_spec = ModelFile(
        recognizer.name, "https://example.invalid/recognizer", file_sha256(recognizer), recognizer.stat().st_size
    )
    monkeypatch.setitem(
        models.PROFILES,
        "test-bundle",
        ModelProfile(
            "test-bundle", "test", detector_spec, recognizer_spec, "test-v1", "test", True, "",
        ),
    )
    monkeypatch.setattr(models, "models_dir", lambda: user_models)
    monkeypatch.setattr(models, "packaged_models_dir", lambda: bundled_models)

    paths = models.required_paths("test-bundle", verify_hash=True)

    assert paths == (detector, recognizer)
