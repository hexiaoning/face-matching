from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path) -> np.ndarray | None:
    """Read an image from a Unicode Windows path."""
    try:
        payload = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if payload.size == 0:
        return None
    return cv2.imdecode(payload, cv2.IMREAD_COLOR)


def write_jpeg(path: str | Path, image: np.ndarray, quality: int = 95) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ok, payload = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise OSError(f"图片编码失败: {destination}")
    payload.tofile(str(destination))
