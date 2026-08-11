from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .errors import ModelMissingError
from .paths import models_dir, packaged_models_dir

ProgressCallback = Callable[[str, int, int], None]
CancelCallback = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class ModelFile:
    name: str
    url: str
    sha256: str
    size: int


@dataclass(frozen=True, slots=True)
class ModelProfile:
    key: str
    title: str
    detector: ModelFile
    recognizer: ModelFile
    model_id: str
    license_name: str
    commercial_ok: bool
    note: str


SCRFD_10G = ModelFile(
    name="scrfd_10g_bnkps.onnx",
    url=("https://huggingface.co/fal/AuraFace-v1/resolve/"
         "94ee6ceb788a98a88807db884ae1f00d7e070d23/scrfd_10g_bnkps.onnx"),
    sha256="5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
    size=16_923_827,
)

PROFILES: dict[str, ModelProfile] = {
    "lvface-b": ModelProfile(
        key="lvface-b",
        title="LVFace-B 高精度（研究用途）",
        detector=SCRFD_10G,
        recognizer=ModelFile(
            name="LVFace-B_Glint360K.onnx",
            url=("https://huggingface.co/bytedance-research/LVFace/resolve/"
                 "48fb6c10b26367f82cd39704874f45413bdf092b/"
                 "LVFace-B_Glint360K/LVFace-B_Glint360K.onnx"),
            sha256="9d834ed8e927fd35b9123b2bf97c40aad05785b1f9ecfb1c4c1f6242d38d1382",
            size=455_533_594,
        ),
        model_id="lvface-b-glint360k-v1",
        license_name="MIT code / non-commercial research weights",
        commercial_ok=False,
        note="ICCV 2025 Highlight；权重仅限非商业研究，商用需另行取得授权。",
    ),
    "auraface": ModelProfile(
        key="auraface",
        title="AuraFace-R100（Apache-2.0）",
        detector=SCRFD_10G,
        recognizer=ModelFile(
            name="glintr100.onnx",
            url=("https://huggingface.co/fal/AuraFace-v1/resolve/"
                 "94ee6ceb788a98a88807db884ae1f00d7e070d23/glintr100.onnx"),
            sha256="a7933ea5330113b01c9b60351d8f4c33003f145d8470ac5f0e52ee2effe25c60",
            size=260_694_151,
        ),
        model_id="auraface-r100-v1",
        license_name="Apache-2.0",
        commercial_ok=True,
        note="可商用开源权重；准确率低于 LVFace-B，部署前仍需按现场数据校准。",
    ),
}


def profile_spec(profile: str) -> ModelProfile:
    try:
        return PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"未知模型配置: {profile}") from exc


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_valid_model(path: Path, spec: ModelFile, verify_hash: bool = False) -> bool:
    if not path.is_file() or path.stat().st_size != spec.size:
        return False
    return not verify_hash or file_sha256(path) == spec.sha256


def required_paths(profile: str, root: Path | None = None, verify_hash: bool = False) -> tuple[Path, Path]:
    spec = profile_spec(profile)
    directories = [root] if root is not None else [models_dir(), packaged_models_dir()]
    candidates = [directory for directory in directories if directory is not None]

    def locate(item: ModelFile) -> Path | None:
        for directory in candidates:
            path = directory / item.name
            if is_valid_model(path, item, verify_hash=verify_hash):
                return path
        return None

    detector = locate(spec.detector)
    recognizer = locate(spec.recognizer)
    missing = [item.name for item, path in ((spec.detector, detector), (spec.recognizer, recognizer)) if path is None]
    if missing:
        raise ModelMissingError("缺少或损坏的模型文件: " + ", ".join(missing))
    assert detector is not None and recognizer is not None
    return detector, recognizer


def available_profiles(root: Path | None = None) -> list[str]:
    available: list[str] = []
    for profile in PROFILES:
        try:
            required_paths(profile, root=root)
            available.append(profile)
        except ModelMissingError:
            pass
    return available


def _download_one(
    spec: ModelFile,
    directory: Path,
    progress: ProgressCallback | None,
    cancelled: CancelCallback | None,
) -> Path:
    destination = directory / spec.name
    if is_valid_model(destination, spec, verify_hash=True):
        if progress:
            progress(spec.name, spec.size, spec.size)
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    request = urllib.request.Request(spec.url, headers={"User-Agent": "FaceMatching/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            total = int(response.headers.get("Content-Length") or spec.size)
            downloaded = 0
            while True:
                if cancelled and cancelled():
                    raise InterruptedError("模型下载已取消")
                block = response.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
                downloaded += len(block)
                if progress:
                    progress(spec.name, downloaded, total)
        if temporary.stat().st_size != spec.size:
            raise ModelMissingError(
                f"{spec.name} 下载大小不正确: {temporary.stat().st_size} != {spec.size}"
            )
        digest = file_sha256(temporary)
        if digest != spec.sha256:
            raise ModelMissingError(f"{spec.name} SHA-256 校验失败")
        os.replace(temporary, destination)
        return destination
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def download_profile(
    profile: str,
    root: Path | None = None,
    progress: ProgressCallback | None = None,
    cancelled: CancelCallback | None = None,
) -> tuple[Path, Path]:
    spec = profile_spec(profile)
    directory = root or models_dir()
    directory.mkdir(parents=True, exist_ok=True)
    detector = _download_one(spec.detector, directory, progress, cancelled)
    recognizer = _download_one(spec.recognizer, directory, progress, cancelled)
    return detector, recognizer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="下载并校验 Face Matching 模型")
    parser.add_argument("command", choices=["download", "verify"])
    parser.add_argument("--profile", choices=sorted(PROFILES), default="lvface-b")
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args(argv)

    def show(name: str, done: int, total: int) -> None:
        percent = int(done * 100 / max(total, 1))
        print(f"\r{name}: {percent:3d}%", end="", flush=True)
        if done >= total:
            print()

    try:
        if args.command == "download":
            download_profile(args.profile, args.directory, show)
        else:
            required_paths(args.profile, args.directory, verify_hash=True)
        print(f"模型 {args.profile} 已就绪。")
        return 0
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
