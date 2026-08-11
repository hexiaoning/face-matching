from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import EngineConfig
from .errors import ModelMissingError
from .paths import is_frozen, model_dir


Progress = Callable[[str, int, int], None]


@dataclass(frozen=True, slots=True)
class DownloadSpec:
    name: str
    url: str
    destination: str
    size: int
    sha256: str | None = None
    destination_sha256: str | None = None
    zip_member_suffix: str | None = None
    destination_min_size: int = 1_000_000


DETECTOR = DownloadSpec(
    name="SCRFD-10G face detector",
    url="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    destination="scrfd_10g_bnkps.onnx",
    size=288_621_354,
    sha256="80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f",
    destination_sha256="5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
    zip_member_suffix="det_10g.onnx",
    destination_min_size=10_000_000,
)

RECOGNIZER = DownloadSpec(
    name="LVFace-B recognizer",
    url=(
        "https://huggingface.co/bytedance-research/LVFace/resolve/main/"
        "LVFace-B_Glint360K/LVFace-B_Glint360K.onnx?download=true"
    ),
    destination="LVFace-B_Glint360K.onnx",
    size=455_533_594,
    sha256="9d834ed8e927fd35b9123b2bf97c40aad05785b1f9ecfb1c4c1f6242d38d1382",
    destination_sha256="9d834ed8e927fd35b9123b2bf97c40aad05785b1f9ecfb1c4c1f6242d38d1382",
    destination_min_size=400_000_000,
)


def required_models(config: EngineConfig | None = None) -> tuple[Path, Path]:
    config = config or EngineConfig()
    return config.detector_model, config.recognizer_model


def assert_models_present(config: EngineConfig | None = None) -> None:
    missing = [str(path) for path in required_models(config) if not path.is_file()]
    if missing:
        joined = "\n".join(f"  - {item}" for item in missing)
        remedy = (
            "离线发布包不完整，请在联网构建机重新运行 build_offline_bundle.ps1。"
            if is_frozen()
            else "请双击 install.bat，或运行 face-matching-download-models。"
        )
        raise ModelMissingError(
            "缺少人脸模型文件：\n"
            f"{joined}\n\n{remedy}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _download(spec: DownloadSpec, temp_path: Path, progress: Progress) -> None:
    request = urllib.request.Request(spec.url, headers={"User-Agent": "FaceMatching/0.1"})
    with urllib.request.urlopen(request, timeout=90) as response, temp_path.open("wb") as output:
        total = int(response.headers.get("Content-Length") or spec.size)
        downloaded = 0
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            downloaded += len(chunk)
            progress(spec.name, downloaded, total)


def _materialize(spec: DownloadSpec, temp_path: Path, destination: Path) -> None:
    if spec.sha256 and _sha256(temp_path) != spec.sha256:
        raise RuntimeError(f"{spec.name} checksum mismatch; delete the partial file and retry")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_suffix(destination.suffix + ".partial")
    if staging.exists():
        staging.unlink()
    if spec.zip_member_suffix:
        with zipfile.ZipFile(temp_path) as archive:
            matches = [name for name in archive.namelist() if name.endswith(spec.zip_member_suffix)]
            if len(matches) != 1:
                raise RuntimeError(f"Could not locate {spec.zip_member_suffix} in model archive")
            with archive.open(matches[0]) as source, staging.open("wb") as output:
                shutil.copyfileobj(source, output)
    else:
        shutil.copy2(temp_path, staging)
    if spec.destination_sha256 and _sha256(staging) != spec.destination_sha256:
        staging.unlink(missing_ok=True)
        raise RuntimeError(f"{spec.name} installed model checksum mismatch")
    os.replace(staging, destination)


def _existing_is_valid(spec: DownloadSpec, destination: Path) -> bool:
    if not destination.is_file() or destination.stat().st_size < spec.destination_min_size:
        return False
    expected_hash = spec.destination_sha256
    if expected_hash is None and spec.zip_member_suffix is None:
        expected_hash = spec.sha256
    if expected_hash:
        return _sha256(destination) == expected_hash
    return True


def download_models(progress: Progress | None = None, force: bool = False) -> tuple[Path, Path]:
    destination_dir = model_dir()
    progress = progress or _console_progress
    completed: list[Path] = []
    for spec in (DETECTOR, RECOGNIZER):
        destination = destination_dir / spec.destination
        if not force and _existing_is_valid(spec, destination):
            progress(spec.name, spec.size, spec.size)
            completed.append(destination)
            continue
        fd, temp_name = tempfile.mkstemp(prefix="face-model-", suffix=".download")
        os.close(fd)
        temp_path = Path(temp_name)
        try:
            _download(spec, temp_path, progress)
            _materialize(spec, temp_path, destination)
        finally:
            temp_path.unlink(missing_ok=True)
        completed.append(destination)
    return completed[0], completed[1]


def _console_progress(name: str, current: int, total: int) -> None:
    percent = 100.0 * current / max(total, 1)
    print(f"\r{name}: {percent:6.2f}% ({current / 1e6:.1f}/{total / 1e6:.1f} MB)", end="")
    if current >= total:
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download CUDA face matching model weights")
    parser.add_argument("--force", action="store_true", help="download files again")
    args = parser.parse_args(argv)
    print("Pretrained weights are provided for non-commercial research only.")
    print("For production, set FACE_MATCHING_*_MODEL to licensed ONNX weights.\n")
    try:
        paths = download_models(force=args.force)
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        print(f"Model download failed: {exc}", file=sys.stderr)
        return 1
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
