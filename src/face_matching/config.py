from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def application_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path.cwd().resolve()


@dataclass(frozen=True, slots=True)
class AppPaths:
    root: Path
    data: Path
    database: Path
    gallery: Path
    models: Path
    settings: Path

    @classmethod
    def create(cls, data_dir: str | Path | None = None) -> "AppPaths":
        root = application_root()
        configured = data_dir or os.getenv("FACE_MATCH_DATA_DIR")
        data = Path(configured).expanduser().resolve() if configured else root / "data"
        return cls(
            root=root,
            data=data,
            database=data / "face_matching.sqlite3",
            gallery=data / "gallery",
            models=data / "models",
            settings=data / "settings.json",
        )

    def ensure(self) -> None:
        self.data.mkdir(parents=True, exist_ok=True)
        self.gallery.mkdir(parents=True, exist_ok=True)
        self.models.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class RecognitionSettings:
    similarity_threshold: float = 0.50
    detection_threshold: float = 0.50
    minimum_quality: float = 0.28
    detector_size: int = 960
    recognition_interval: int = 4
    confirmation_hits: int = 3
    max_track_embeddings: int = 12
    track_ttl_frames: int = 18

    def validate(self) -> None:
        if not 0.0 < self.similarity_threshold < 1.0:
            raise ValueError("相似度阈值必须在 0 和 1 之间")
        if not 0.0 < self.detection_threshold < 1.0:
            raise ValueError("检测阈值必须在 0 和 1 之间")
        if not 0.0 <= self.minimum_quality <= 1.0:
            raise ValueError("最低质量必须在 0 和 1 之间")
        if self.detector_size not in {640, 960, 1280}:
            raise ValueError("检测尺寸只能是 640、960 或 1280")
        if self.recognition_interval < 1:
            raise ValueError("识别间隔必须大于等于 1")
        if self.confirmation_hits < 1:
            raise ValueError("连续确认次数必须大于等于 1")

    @classmethod
    def load(cls, path: Path) -> "RecognitionSettings":
        if not path.exists():
            return cls()
        try:
            raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            allowed = cls.__dataclass_fields__.keys()
            settings = cls(**{key: raw[key] for key in allowed if key in raw})
            settings.validate()
            return settings
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return cls()

    def save(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)

