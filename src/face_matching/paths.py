from __future__ import annotations

import os
import sys
from pathlib import Path


APP_NAME = "FaceMatching"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def resource_root() -> Path:
    """Directory containing read-only files bundled by PyInstaller."""
    bundled = getattr(sys, "_MEIPASS", None)
    if bundled:
        return Path(bundled).resolve()
    return Path(__file__).resolve().parents[2]


def bundled_model_dir() -> Path | None:
    candidates = [resource_root() / "models"]
    if is_frozen():
        candidates.append(Path(sys.executable).resolve().parent / "models")
    return next((path for path in candidates if path.is_dir()), None)


def default_model_path(filename: str) -> Path:
    bundled = bundled_model_dir()
    if bundled is not None and (bundled / filename).is_file():
        return bundled / filename
    return model_dir() / filename


def app_home() -> Path:
    override = os.environ.get("FACE_MATCHING_HOME")
    if override:
        return Path(override).expanduser().resolve()
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / APP_NAME


def ensure_app_dirs() -> Path:
    root = app_home()
    for child in (root, root / "models", root / "photos"):
        child.mkdir(parents=True, exist_ok=True)
    return root


def model_dir() -> Path:
    custom = os.environ.get("FACE_MATCHING_MODEL_DIR")
    if custom:
        path = Path(custom).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    return ensure_app_dirs() / "models"


def database_path() -> Path:
    custom = os.environ.get("FACE_MATCHING_DATABASE")
    if custom:
        path = Path(custom).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    return ensure_app_dirs() / "face_matching.sqlite3"


def photo_dir() -> Path:
    return ensure_app_dirs() / "photos"
