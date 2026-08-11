from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

from .domain import ModelPaths


class ModelDownloadError(RuntimeError):
    pass


class DownloadCancelled(ModelDownloadError):
    pass


ProgressCallback = Callable[[int, int], None]


class ModelManager:
    MODEL_NAME = "antelopev2"
    MODEL_URL = (
        "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip"
    )
    MODEL_SHA256 = "8e182f14fc6e80b3bfa375b33eb6cff7ee05d8ef7633e738d1c89021dcf0c5c5"
    DETECTOR_NAMES = ("scrfd_10g_bnkps.onnx", "det_10g.onnx")
    RECOGNIZER_NAMES = ("glintr100.onnx", "w600k_r50.onnx")

    def __init__(self, models_root: Path) -> None:
        self.models_root = models_root
        self.model_dir = models_root / self.MODEL_NAME

    def locate(self) -> ModelPaths | None:
        detector = next(
            (self.model_dir / name for name in self.DETECTOR_NAMES if (self.model_dir / name).is_file()),
            None,
        )
        recognizer = next(
            (
                self.model_dir / name
                for name in self.RECOGNIZER_NAMES
                if (self.model_dir / name).is_file()
            ),
            None,
        )
        if detector and recognizer:
            return ModelPaths(detector=detector, recognizer=recognizer, model_name=self.MODEL_NAME)
        return None

    def ensure_models(self, progress: ProgressCallback | None = None) -> ModelPaths:
        located = self.locate()
        if located:
            return located
        self.models_root.mkdir(parents=True, exist_ok=True)
        archive = self.models_root / f"{self.MODEL_NAME}.zip"
        partial = archive.with_suffix(".zip.part")
        try:
            self._download(partial, progress)
            digest = self._sha256(partial)
            if digest.lower() != self.MODEL_SHA256.lower():
                raise ModelDownloadError(
                    "模型文件 SHA-256 校验失败，已拒绝使用。\n"
                    f"期望：{self.MODEL_SHA256}\n实际：{digest}"
                )
            os.replace(partial, archive)
            self._extract_required(archive)
            self._write_manifest()
        except Exception:
            partial.unlink(missing_ok=True)
            raise
        finally:
            archive.unlink(missing_ok=True)

        located = self.locate()
        if not located:
            raise ModelDownloadError("模型压缩包中缺少检测或识别 ONNX 文件。")
        return located

    def _download(self, destination: Path, progress: ProgressCallback | None) -> None:
        request = urllib.request.Request(
            self.MODEL_URL,
            headers={"User-Agent": "FaceMatching/0.1 (+local desktop application)"},
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response, destination.open("wb") as out:
                total = int(response.headers.get("Content-Length", "0") or 0)
                downloaded = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    if progress:
                        progress(downloaded, total)
        except DownloadCancelled:
            raise
        except Exception as exc:
            raise ModelDownloadError(
                f"模型下载失败：{exc}\n可手动下载 {self.MODEL_URL} 并解压到 {self.model_dir}"
            ) from exc

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _extract_required(self, archive: Path) -> None:
        wanted = set(self.DETECTOR_NAMES + self.RECOGNIZER_NAMES)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as package:
            for member in package.infolist():
                basename = Path(member.filename).name
                if basename not in wanted or member.is_dir():
                    continue
                target = self.model_dir / basename
                temporary = target.with_suffix(target.suffix + ".tmp")
                with package.open(member) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
                temporary.replace(target)

    def _write_manifest(self) -> None:
        manifest = {
            "name": self.MODEL_NAME,
            "source": self.MODEL_URL,
            "archive_sha256": self.MODEL_SHA256,
            "license_note": (
                "InsightFace provided pretrained models are restricted to non-commercial research; "
                "obtain a separate license before commercial use."
            ),
        }
        (self.model_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

