from types import SimpleNamespace

import numpy as np
import pytest

from face_matching.diagnostics import _real_inference


class FakeDiagnosticSession:
    def __init__(self, shape, output):
        self.shape = shape
        self.output = output
        self.input = None

    def get_inputs(self):
        return [SimpleNamespace(name="input", shape=self.shape)]

    def get_providers(self):
        return ["CUDAExecutionProvider"]

    def run(self, output_names, inputs):
        self.input = inputs["input"]
        return [self.output]


def test_real_inference_resolves_dynamic_spatial_shape():
    session = FakeDiagnosticSession(["batch", 3, "height", "width"], np.ones((1, 4)))

    result = _real_inference(session, 112)

    assert session.input.shape == (1, 3, 112, 112)
    assert result["provider"] == "CUDAExecutionProvider"
    assert result["output_shapes"] == [[1, 4]]


def test_real_inference_rejects_non_finite_model_output():
    session = FakeDiagnosticSession([1, 3, 112, 112], np.asarray([[np.nan]]))

    with pytest.raises(RuntimeError, match="NaN/Inf"):
        _real_inference(session, 112)
