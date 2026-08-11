from __future__ import annotations

import os
import sys
from pathlib import Path


APP_NAME = "FaceMatching"


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
    # PyInstaller exposes added data below _MEIPASS. Models in an offline
    # bundle are immutable application assets; user data still lives under
    # LOCALAPPDATA and remains writable across upgrades.
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        return Path(frozen_root) / "models"
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
