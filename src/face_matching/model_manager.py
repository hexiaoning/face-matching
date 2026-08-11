from __future__ import annotations

import hashlib
import zipfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath

import requests

from face_matching.errors import ModelError

MODEL_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
EXPECTED_MODEL_SHA256 = {
    "det_10g.onnx": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
    "w600k_r50.onnx": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
}
EXPECTED_MODELS = set(EXPECTED_MODEL_SHA256)
MAX_MODEL_ARCHIVE_BYTES = 400 * 1024 * 1024


def locate_model_files(model_dir: Path) -> tuple[Path, Path]:
    detector = model_dir / "det_10g.onnx"
    recognizer = model_dir / "w600k_r50.onnx"
    if not detector.is_file() or not recognizer.is_file():
        custom_detector = model_dir / "detector.onnx"
        custom_recognizer = model_dir / "recognizer.onnx"
        if custom_detector.is_file() and custom_recognizer.is_file():
            return custom_detector, custom_recognizer
        raise ModelError(
            "缺少模型。目录内应有 det_10g.onnx + w600k_r50.onnx，"
            "或自有授权的 detector.onnx + recognizer.onnx。"
        )
    return detector, recognizer


def models_ready(model_dir: Path) -> bool:
    try:
        locate_model_files(model_dir)
    except ModelError:
        return False
    return True


def _safe_member_name(name: str) -> str | None:
    normalized = PurePosixPath(name.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ModelError("模型压缩包包含不安全路径，已停止解压。")
    return normalized.name if normalized.name in EXPECTED_MODELS else None


def download_research_models(
    model_dir: Path,
    progress: Callable[[int, int], None] | None = None,
    *,
    accept_research_license: bool = False,
) -> None:
    if not accept_research_license:
        raise ModelError("必须先确认 InsightFace 预训练模型仅限非商业研究使用。")
    model_dir.mkdir(parents=True, exist_ok=True)
    archive = model_dir.parent / "buffalo_l.zip.part"
    try:
        with requests.get(MODEL_URL, stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", "0"))
            if total > MAX_MODEL_ARCHIVE_BYTES:
                raise ModelError("模型压缩包超过 400 MiB 安全上限，已停止下载。")
            downloaded = 0
            with archive.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    output.write(chunk)
                    downloaded += len(chunk)
                    if downloaded > MAX_MODEL_ARCHIVE_BYTES:
                        raise ModelError("模型压缩包超过 400 MiB 安全上限，已停止下载。")
                    if progress:
                        progress(downloaded, total)
        extracted: set[str] = set()
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                target_name = _safe_member_name(member.filename)
                if target_name is None:
                    continue
                if target_name in extracted:
                    raise ModelError(f"模型压缩包包含重复文件：{target_name}")
                if member.file_size > MAX_MODEL_ARCHIVE_BYTES:
                    raise ModelError(f"模型文件异常过大：{target_name}")
                pending = model_dir / f"{target_name}.tmp"
                model_digest = hashlib.sha256()
                with bundle.open(member) as source, pending.open("wb") as target:
                    while block := source.read(1024 * 1024):
                        target.write(block)
                        model_digest.update(block)
                if model_digest.hexdigest().lower() != EXPECTED_MODEL_SHA256[target_name]:
                    pending.unlink(missing_ok=True)
                    raise ModelError(f"{target_name} 的 SHA-256 校验失败，已拒绝使用。")
                pending.replace(model_dir / target_name)
                extracted.add(target_name)
        if extracted != EXPECTED_MODELS:
            raise ModelError("模型压缩包缺少检测或识别网络。")
    finally:
        archive.unlink(missing_ok=True)
