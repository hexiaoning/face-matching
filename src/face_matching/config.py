from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from platformdirs import user_data_path

APP_NAME = "FaceMatching"
APP_AUTHOR = "FaceMatching"


def default_data_dir() -> Path:
    return Path(user_data_path(APP_NAME, APP_AUTHOR, roaming=False))


@dataclass(slots=True)
class AppConfig:
    data_dir: str = ""
    model_dir: str = ""
    # 960x544 is divisible by SCRFD's maximum stride and preserves substantially
    # more detail from a typical 16:9 surveillance stream than a 640x640 canvas.
    detection_width: int = 960
    detection_height: int = 544
    detection_threshold: float = 0.55
    match_threshold: float = 0.45
    min_top2_margin: float = 0.04
    min_face_size: int = 36
    frame_stride: int = 2
    max_track_age: int = 12
    min_track_hits: int = 3
    min_recognition_quality: float = 0.35
    recognition_flip_tta: bool = True
    gpu_device_id: int = 0

    def __post_init__(self) -> None:
        root = Path(self.data_dir).expanduser() if self.data_dir else default_data_dir()
        self.data_dir = str(root.resolve())
        if not self.model_dir:
            self.model_dir = str((root / "models" / "buffalo_l").resolve())
        else:
            self.model_dir = str(Path(self.model_dir).expanduser().resolve())
        self.validate()

    @property
    def root(self) -> Path:
        return Path(self.data_dir)

    @property
    def config_path(self) -> Path:
        return self.root / "config.json"

    @property
    def database_path(self) -> Path:
        return self.root / "people.sqlite3"

    @property
    def enrollment_dir(self) -> Path:
        return self.root / "enrollment"

    def validate(self) -> None:
        if self.detection_width < 160 or self.detection_height < 160:
            raise ValueError("检测分辨率不得小于 160×160")
        if self.detection_width % 32 or self.detection_height % 32:
            raise ValueError("检测分辨率必须是 32 的整数倍")
        if not 0 < self.detection_threshold < 1:
            raise ValueError("检测阈值必须位于 0 到 1 之间")
        if not -1 < self.match_threshold < 1:
            raise ValueError("比对阈值必须位于 -1 到 1 之间")
        if not 0 <= self.min_top2_margin < 1:
            raise ValueError("次优差值必须位于 0 到 1 之间")
        if self.min_face_size < 16:
            raise ValueError("最小人脸尺寸不得小于 16")
        if self.frame_stride < 1:
            raise ValueError("抽帧间隔不得小于 1")
        if self.max_track_age < 1:
            raise ValueError("轨迹最大丢失帧数不得小于 1")
        if self.min_track_hits < 1:
            raise ValueError("稳定确认帧数不得小于 1")
        if not 0 <= self.min_recognition_quality <= 1:
            raise ValueError("最低识别画质必须位于 0 到 1 之间")
        if self.gpu_device_id < 0:
            raise ValueError("GPU 编号不得小于 0")

    def save(self) -> None:
        self.validate()
        self.root.mkdir(parents=True, exist_ok=True)
        pending = self.config_path.with_suffix(".json.tmp")
        pending.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        pending.replace(self.config_path)

    @classmethod
    def load(cls, data_dir: Path | None = None) -> AppConfig:
        root = (data_dir or default_data_dir()).expanduser().resolve()
        path = root / "config.json"
        if not path.exists():
            config = cls(data_dir=str(root))
            config.save()
            return config
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["data_dir"] = str(root)
        return cls(**raw)
