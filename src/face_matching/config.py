from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .models import feature_model_id
from .paths import config_path


@dataclass(slots=True)
class AppConfig:
    model_profile: str = "lvface-b"
    gpu_backend: str = "auto"
    gpu_device_id: int = 0
    detector_size: int = 960
    detector_threshold: float = 0.40
    nms_threshold: float = 0.40
    match_threshold: float = 0.45
    match_margin: float = 0.06
    # Target-search thresholds label ranked candidates; they never remove a
    # track from the final Top-K list. Automatic confirmation stays disabled
    # until the operator explicitly enables it after field calibration.
    target_match_threshold: float = 0.19
    target_review_threshold: float = 0.12
    target_min_support: int = 3
    target_min_evidence_gap: float = 0.75
    target_top_k: int = 20
    target_auto_confirm: bool = False
    min_face_size: int = 32
    min_quality: float = 0.22
    enrollment_min_quality: float = 0.30
    frame_interval: int = 1
    mirror_augmentation: bool = True
    min_track_observations: int = 3
    track_top_k: int = 10
    track_max_misses: int = 12
    track_consistency_threshold: float = 0.12
    confirmation_matches: int = 2
    fast_file_scan: bool = True

    def validate(self) -> "AppConfig":
        if self.model_profile not in {"lvface-b", "auraface"}:
            raise ValueError(f"未知模型配置: {self.model_profile}")
        if self.gpu_backend not in {"auto", "cuda", "openvino"}:
            raise ValueError("gpu_backend 必须是 auto/cuda/openvino")
        if self.gpu_device_id < 0:
            raise ValueError("GPU 编号不能为负数")
        if self.detector_size not in {320, 480, 640, 768, 960, 1280}:
            raise ValueError("检测尺寸必须为 320/480/640/768/960/1280")
        for name in (
            "detector_threshold",
            "nms_threshold",
            "match_threshold",
            "target_match_threshold",
            "target_review_threshold",
            "min_quality",
            "enrollment_min_quality",
            "track_consistency_threshold",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} 必须在 0 到 1 之间")
        if not 0.0 <= self.match_margin <= 0.5:
            raise ValueError("match_margin 必须在 0 到 0.5 之间")
        if self.target_review_threshold >= self.target_match_threshold:
            raise ValueError("候选复核阈值必须低于目标确认阈值")
        if self.min_face_size < 12:
            raise ValueError("最小人脸尺寸不能小于 12 像素")
        if self.frame_interval < 1:
            raise ValueError("抽帧间隔必须大于 0")
        if not isinstance(self.mirror_augmentation, bool):
            raise ValueError("mirror_augmentation 必须是布尔值")
        if self.min_track_observations < 1 or self.track_top_k < 1:
            raise ValueError("轨迹参数必须大于 0")
        if self.track_max_misses < 1 or self.confirmation_matches < 1:
            raise ValueError("轨迹保留和确认次数必须大于 0")
        if self.target_min_support < 1:
            raise ValueError("目标确认支持帧数必须大于 0")
        if not 0.0 <= self.target_min_evidence_gap <= 30.0:
            raise ValueError("目标证据最小时间间隔必须在 0 到 30 秒之间")
        if not 1 <= self.target_top_k <= 500:
            raise ValueError("目标候选数量必须在 1 到 500 之间")
        if not isinstance(self.target_auto_confirm, bool):
            raise ValueError("target_auto_confirm 必须是布尔值")
        if not isinstance(self.fast_file_scan, bool):
            raise ValueError("fast_file_scan 必须是布尔值")
        return self

    @property
    def feature_model_id(self) -> str:
        return feature_model_id(self.model_profile, self.mirror_augmentation)

    @classmethod
    def load(cls, path: Path | None = None) -> "AppConfig":
        target = path or config_path()
        if not target.exists():
            return cls()
        raw = json.loads(target.read_text(encoding="utf-8"))
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in raw.items() if key in allowed}).validate()

    def save(self, path: Path | None = None) -> None:
        self.validate()
        target = path or config_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)
