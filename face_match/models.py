"""模型文件管理：下载 InsightFace buffalo_l 模型包并定位各模型。"""
from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

from . import config

ProgressCb = Callable[[int, int], None]  # (downloaded_bytes, total_bytes)


class ModelNotFoundError(RuntimeError):
    pass


def required_models() -> list[str]:
    return [config.DETECTOR_MODEL, config.RECOGNIZER_MODEL]


def models_ready(mdir: Path | None = None) -> bool:
    mdir = mdir or config.models_dir()
    return all((mdir / m).is_file() for m in required_models())


def model_path(name: str, mdir: Path | None = None) -> Path:
    mdir = mdir or config.models_dir()
    p = mdir / name
    if not p.is_file():
        raise ModelNotFoundError(f"模型缺失: {p}（请先在界面中下载模型，或放入该目录）")
    return p


def download_models(progress: ProgressCb | None = None,
                    mdir: Path | None = None,
                    url: str = config.BUFFALO_L_URL) -> list[Path]:
    """下载并解压 buffalo_l 模型包，返回解压出的模型文件列表。"""
    mdir = mdir or config.models_dir()
    zip_path = mdir / "buffalo_l.zip.tmp"

    req = urllib.request.Request(url, headers={"User-Agent": "facematch/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        with open(zip_path, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)

    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            base = Path(name).name
            if not base.endswith(".onnx"):
                continue
            target = mdir / base
            with zf.open(name) as src, open(target, "wb") as dst:
                dst.write(src.read())
            extracted.append(target)
    zip_path.unlink(missing_ok=True)

    if not models_ready(mdir):
        raise ModelNotFoundError("下载完成但关键模型仍缺失，请检查网络后重试")
    return extracted
