import pytest

from face_matching.errors import ModelError
from face_matching.model_manager import _safe_member_name, download_research_models


def test_model_download_requires_explicit_license_acceptance(tmp_path):
    with pytest.raises(ModelError, match="非商业研究"):
        download_research_models(tmp_path)


def test_model_archive_paths_are_sanitized():
    with pytest.raises(ModelError, match="不安全路径"):
        _safe_member_name("../det_10g.onnx")
    assert _safe_member_name("buffalo_l/det_10g.onnx") == "det_10g.onnx"
    assert _safe_member_name("buffalo_l/README.txt") is None
