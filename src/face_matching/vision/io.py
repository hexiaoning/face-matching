from __future__ import annotations

import base64
import binascii
import hashlib
from pathlib import Path
from urllib.parse import unquote_to_bytes

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def is_image_data_uri(source: str) -> bool:
    return source.lower().startswith("data:image/")


def is_image_source(source: str) -> bool:
    return is_image_data_uri(source) or Path(source).suffix.lower() in IMAGE_EXTENSIONS


def image_data_payload(source: str) -> bytes:
    header, separator, payload = source.partition(",")
    if not separator or not is_image_data_uri(source):
        raise ValueError("图片数据格式无效")
    encoded = unquote_to_bytes(payload)
    try:
        if "base64" in header.lower().split(";")[1:]:
            return base64.b64decode(b"".join(encoded.split()), validate=True)
        return encoded
    except (ValueError, binascii.Error) as exc:
        raise ValueError("图片 Base64 数据无效或不完整") from exc


def store_image_data_uri(source: str, directory: Path) -> Path:
    """Store a validated image so annotations and exports retain a local source."""
    payload = image_data_payload(source)
    if not payload:
        raise ValueError("图片数据为空")
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("无法解码图片数据，请重新复制图片或选择图片文件")
    # Store as PNG to keep the decoded pixels and support any input image MIME type.
    destination = directory / f"粘贴图片-{hashlib.sha256(payload).hexdigest()[:20]}.png"
    if not destination.is_file():
        directory.mkdir(parents=True, exist_ok=True)
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise OSError("无法保存粘贴图片")
        encoded.tofile(str(destination))
    return destination


def read_image(path: str | Path) -> np.ndarray | None:
    """Read a Unicode Windows image path or an inline image data URI."""
    try:
        source = str(path)
        payload = (np.frombuffer(image_data_payload(source), dtype=np.uint8)
                   if is_image_data_uri(source) else np.fromfile(source, dtype=np.uint8))
    except (OSError, ValueError):
        return None
    if payload.size == 0:
        return None
    try:
        return cv2.imdecode(payload, cv2.IMREAD_COLOR)
    except cv2.error:
        return None


def write_jpeg(path: str | Path, image: np.ndarray, quality: int = 95) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ok, payload = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise OSError(f"图片编码失败: {destination}")
    payload.tofile(str(destination))
