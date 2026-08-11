from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

APP_DIR_NAME = "SurveillanceFaceMatch"
MODEL_VERSION = "lvface-b-glint360k@b12702a"


def default_data_dir() -> Path:
    override = os.environ.get("FACE_MATCH_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / APP_DIR_NAME
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "surveillance-face-match"


@dataclass(frozen=True)
class AppPaths:
    root: Path
    database: Path
    photos: Path
    models: Path
    settings: Path

    @classmethod
    def create(cls, root: Path | None = None) -> AppPaths:
        root = (root or default_data_dir()).resolve()
        result = cls(
            root=root,
            database=root / "faces.sqlite3",
            photos=root / "photos",
            models=root / "models",
            settings=root / "settings.json",
        )
        result.root.mkdir(parents=True, exist_ok=True)
        result.photos.mkdir(parents=True, exist_ok=True)
        result.models.mkdir(parents=True, exist_ok=True)
        return result


@dataclass
class AppSettings:
    similarity_threshold: float = 0.48
    ambiguity_margin: float = 0.04
    detection_threshold: float = 0.55
    minimum_quality: float = 0.30
    minimum_track_samples: int = 3
    maximum_track_samples: int = 12
    target_fps: float = 8.0
    detector_size: int = 640
    model_license_accepted: bool = False

    @classmethod
    def load(cls, path: Path) -> AppSettings:
        if not path.exists():
            return cls()
        try:
            raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return cls()
        defaults = asdict(cls())
        values = {key: raw.get(key, value) for key, value in defaults.items()}
        try:
            settings = cls(**values)
            settings.validate()
            return settings
        except (TypeError, ValueError):
            return cls()

    def validate(self) -> None:
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("相似度阈值必须在 0 到 1 之间")
        if not 0.0 <= self.ambiguity_margin <= 0.5:
            raise ValueError("候选间隔必须在 0 到 0.5 之间")
        if not 0.0 <= self.detection_threshold <= 1.0:
            raise ValueError("检测阈值必须在 0 到 1 之间")
        if not 0.0 <= self.minimum_quality <= 1.0:
            raise ValueError("质量阈值必须在 0 到 1 之间")
        if self.minimum_track_samples < 1:
            raise ValueError("轨迹最少帧数不能小于 1")
        if self.maximum_track_samples < self.minimum_track_samples:
            raise ValueError("轨迹最大帧数不能小于最少帧数")
        if not 0.5 <= self.target_fps <= 60.0:
            raise ValueError("处理帧率必须在 0.5 到 60 之间")
        if self.detector_size not in (320, 480, 640, 768, 960):
            raise ValueError("不支持的检测分辨率")

    def save(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)
