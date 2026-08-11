from __future__ import annotations

import hashlib
import shutil
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from face_match.errors import ModelDownloadError

ProgressCallback = Callable[[str, int, int], None]
CancelCallback = Callable[[], bool]


@dataclass(frozen=True)
class ModelAsset:
    name: str
    filename: str
    url: str
    sha256: str
    size: int
    archive_member: str | None = None
    installed_size: int | None = None
    installed_sha256: str | None = None


LVFACE_B = ModelAsset(
    name="LVFace-B Glint360K",
    filename="LVFace-B_Glint360K.onnx",
    url=(
        "https://huggingface.co/bytedance-research/LVFace/resolve/"
        "b12702ab1f5c721748e054a66dc90e1edd1f0724/"
        "LVFace-B_Glint360K/LVFace-B_Glint360K.onnx?download=true"
    ),
    sha256="9d834ed8e927fd35b9123b2bf97c40aad05785b1f9ecfb1c4c1f6242d38d1382",
    size=455_533_594,
)

INSIGHTFACE_DETECTOR = ModelAsset(
    name="InsightFace 10GF 五点检测器",
    filename="det_10g.onnx",
    url="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    sha256="80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f",
    size=288_621_354,
    archive_member="det_10g.onnx",
    installed_size=16_923_827,
    installed_sha256="5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
)

MODEL_LICENSE_NOTICE = (
    "本程序代码采用 MIT 许可证，但自动下载的 LVFace 与 InsightFace 预训练权重仅限非商业研究使用。"
    "商业部署前必须分别向权利方取得模型授权。照片和身份证号仅保存在本机数据目录。"
)


class ModelManager:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

    @property
    def detector_path(self) -> Path:
        return self.model_dir / INSIGHTFACE_DETECTOR.filename

    @property
    def recognizer_path(self) -> Path:
        return self.model_dir / LVFACE_B.filename

    def missing_models(self) -> list[ModelAsset]:
        missing: list[ModelAsset] = []
        for asset in (INSIGHTFACE_DETECTOR, LVFACE_B):
            target = self.model_dir / asset.filename
            marker = target.with_suffix(target.suffix + ".verified")
            expected_size = asset.installed_size or asset.size
            expected_hash = asset.installed_sha256 or asset.sha256
            try:
                marker_valid = marker.read_text(encoding="ascii").strip() == expected_hash
            except OSError:
                marker_valid = False
            if (
                target.is_file()
                and target.stat().st_size == expected_size
                and not marker_valid
                and self._file_sha256(target) == expected_hash
            ):
                marker_valid = True
                try:
                    marker.write_text(expected_hash, encoding="ascii")
                except OSError:
                    pass
            if not target.is_file() or target.stat().st_size != expected_size or not marker_valid:
                missing.append(asset)
        return missing

    def ensure_models(
        self,
        progress: ProgressCallback | None = None,
        cancelled: CancelCallback | None = None,
    ) -> None:
        for asset in self.missing_models():
            self._install_asset(asset, progress or (lambda *_: None), cancelled or (lambda: False))

    def _install_asset(
        self, asset: ModelAsset, progress: ProgressCallback, cancelled: CancelCallback
    ) -> None:
        download_path = self.model_dir / (
            asset.filename + (".zip.part" if asset.archive_member else ".part")
        )
        target = self.model_dir / asset.filename
        marker = target.with_suffix(target.suffix + ".verified")
        for stale in (download_path, marker):
            if stale.exists():
                stale.unlink()
        digest = hashlib.sha256()
        request = urllib.request.Request(asset.url, headers={"User-Agent": "FaceMatch/0.1"})
        try:
            with (
                urllib.request.urlopen(request, timeout=30) as response,
                download_path.open("wb") as out,
            ):
                total = int(response.headers.get("Content-Length") or asset.size)
                downloaded = 0
                while True:
                    if cancelled():
                        raise ModelDownloadError("模型下载已取消")
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    digest.update(chunk)
                    downloaded += len(chunk)
                    progress(asset.name, downloaded, total)
        except ModelDownloadError:
            download_path.unlink(missing_ok=True)
            raise
        except (OSError, urllib.error.URLError) as exc:
            download_path.unlink(missing_ok=True)
            raise ModelDownloadError(f"下载 {asset.name} 失败：{exc}") from exc

        actual_hash = digest.hexdigest()
        if actual_hash != asset.sha256:
            download_path.unlink(missing_ok=True)
            raise ModelDownloadError(
                f"{asset.name} 校验失败，文件可能不完整或被篡改。\n"
                f"期望 SHA-256：{asset.sha256}\n实际 SHA-256：{actual_hash}"
            )

        try:
            if asset.archive_member:
                self._extract_member(download_path, asset.archive_member, target)
                download_path.unlink(missing_ok=True)
            else:
                download_path.replace(target)
            expected_size = asset.installed_size or asset.size
            if target.stat().st_size != expected_size:
                raise OSError(
                    f"安装后大小不正确：期望 {expected_size} 字节，实际 {target.stat().st_size} 字节"
                )
            expected_installed_hash = asset.installed_sha256 or asset.sha256
            actual_installed_hash = (
                self._file_sha256(target) if asset.archive_member else actual_hash
            )
            if actual_installed_hash != expected_installed_hash:
                raise OSError(
                    "解压后的模型校验失败："
                    f"期望 {expected_installed_hash}，实际 {actual_installed_hash}"
                )
            marker.write_text(expected_installed_hash, encoding="ascii")
        except (OSError, zipfile.BadZipFile, KeyError) as exc:
            target.unlink(missing_ok=True)
            marker.unlink(missing_ok=True)
            raise ModelDownloadError(f"安装 {asset.name} 失败：{exc}") from exc

    @staticmethod
    def _extract_member(archive_path: Path, wanted_name: str, target: Path) -> None:
        temporary = target.with_suffix(target.suffix + ".part")
        temporary.unlink(missing_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            matches = [
                member
                for member in archive.infolist()
                if not member.is_dir() and PurePosixPath(member.filename).name == wanted_name
            ]
            if len(matches) != 1:
                raise KeyError(f"压缩包内未唯一找到 {wanted_name}")
            with archive.open(matches[0]) as source, temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
        temporary.replace(target)

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()
