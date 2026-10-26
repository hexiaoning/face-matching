"""Offline pixel adjustments, independent of detection and identity matching."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np


@dataclass(frozen=True)
class EnhancementSettings:
    texture: float = 0.0
    denoise: float = 0.0
    gamma: float = 1.0
    contrast: float = 0.0
    sharpen: float = 0.0

    def validate(self) -> None:
        for name, lower, upper in (
            ("texture", 0, 2), ("denoise", 0, 1), ("gamma", 0.5, 2),
            ("contrast", 0, 3), ("sharpen", 0, 1),
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"{name} 必须在 {lower}～{upper} 之间")


PRESETS = {
    "原始参数": EnhancementSettings(),
    "保守增强": EnhancementSettings(denoise=0.25, contrast=0.8, sharpen=0.15),
    "屏幕翻拍": EnhancementSettings(texture=0.9, denoise=0.3, contrast=0.6),
    "暗部提亮": EnhancementSettings(denoise=0.3, gamma=1.35, contrast=0.8),
}


def enhance_image(image: np.ndarray, settings: EnhancementSettings) -> np.ndarray:
    """Preserve dimensions and geometry; never synthesize missing content."""
    settings.validate()
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3 or image.size == 0):
        raise ValueError("需要非空的 8 位 BGR 彩色图片")
    result = image.copy()
    if settings.texture:
        # Low-pass suppression of photographed screen pixels. This also softens
        # real fine detail, so keep it adjustable instead of claiming recovery.
        result = cv2.GaussianBlur(result, (0, 0), settings.texture)
    if settings.denoise:
        smoothed = cv2.bilateralFilter(result, 5, 35, 3)
        result = cv2.addWeighted(result, 1 - settings.denoise, smoothed, settings.denoise, 0)
    if settings.gamma != 1:
        lut = np.round(255 * (np.arange(256, dtype=np.float64) / 255) ** (1 / settings.gamma))
        result = cv2.LUT(result, lut.astype(np.uint8))
    if settings.contrast:
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        lightness = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        amount = settings.contrast / 3
        lab[:, :, 0] = cv2.addWeighted(lab[:, :, 0], 1 - amount, lightness, amount, 0)
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if settings.sharpen:
        blurred = cv2.GaussianBlur(result, (0, 0), 1)
        result = cv2.addWeighted(result, 1 + settings.sharpen, blurred, -settings.sharpen, 0)
    return result


def encode_png(image: np.ndarray) -> bytes:
    ok, payload = cv2.imencode(".png", image)
    if not ok:
        raise OSError("PNG 编码失败")
    return payload.tobytes()


def save_new_png(destination: Path, image: np.ndarray) -> None:
    payload = encode_png(image)
    with destination.open("xb") as stream:
        try:
            stream.write(payload)
        except BaseException:
            stream.close()
            destination.unlink(missing_ok=True)
            raise


def export_enhancement(
    destination: Path, source_name: str, source_bytes: bytes, original: np.ndarray,
    processed: np.ndarray, settings: EnhancementSettings,
    region: tuple[int, int, int, int] | None = None,
) -> None:
    """Export an exclusive ZIP: existing files (including the source) are safe."""
    settings.validate()
    if original.shape != processed.shape:
        raise ValueError("处理图与原图尺寸不一致")
    height, width = original.shape[:2]
    if region is not None:
        x1, y1, x2, y2 = region
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("局部选区超出图片范围")
    extension = Path(source_name).suffix.lower()
    if extension not in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}:
        extension = ".bin"
    files = {f"original/source{extension}": source_bytes, "processed.png": encode_png(processed)}
    if region is not None:
        files["region/original.png"] = encode_png(original[y1:y2, x1:x2])
        files["region/processed.png"] = encode_png(processed[y1:y2, x1:x2])
    manifest = {
        "schema_version": 1,
        "algorithm": "pixel-adjustments-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_name": Path(source_name).name,
        "width": width, "height": height,
        "settings": asdict(settings), "region_xyxy": region,
        "notice": "仅作像素调整；清晰化不等于恢复真实细节，无法确认缺失的人脸细节或车牌字符。",
        "sha256": {name: hashlib.sha256(payload).hexdigest() for name, payload in files.items()},
    }
    # Opening in exclusive mode also protects against a save-dialog overwrite.
    with destination.open("xb") as stream:
        try:
            with ZipFile(stream, "w", compression=ZIP_DEFLATED) as archive:
                for name, payload in files.items():
                    archive.writestr(name, payload)
                archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        except BaseException:
            stream.close()
            destination.unlink(missing_ok=True)
            raise
