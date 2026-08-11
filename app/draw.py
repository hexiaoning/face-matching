"""画面文字标注：cv2.putText 不支持中文，带中文的标签统一用 PIL 批量渲染。"""
from __future__ import annotations

import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",       # Windows 微软雅黑
    "C:/Windows/Fonts/simhei.ttf",     # Windows 黑体
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]
_font_cache: dict[int, "ImageFont.FreeTypeFont | None"] = {}


def _get_font(px: int) -> "ImageFont.FreeTypeFont | None":
    if px in _font_cache:
        return _font_cache[px]
    font = None
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, px)
                break
            except OSError:
                continue
    _font_cache[px] = font
    return font


def draw_labels(img: np.ndarray, labels: list[tuple[str, int, int, tuple]]) -> None:
    """在 BGR 图像上批量绘制 (文本, x, y, BGR颜色) 标签，支持中文，带黑底色块。

    每帧只调用一次，整体只做一轮 BGR<->RGB 转换。
    """
    if not labels:
        return
    font = _get_font(20)
    if font is None or all(t[0].isascii() for t in labels):
        for text, x, y, color in labels:
            # 无中文字体时把非 ASCII 字符替换为 ?，避免 cv2.putText 抛异常
            safe = text.encode("ascii", "replace").decode("ascii")
            cv2.putText(img, safe, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        return

    h, w = img.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    for text, x, y, color_bgr in labels:
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
        bbox = draw.textbbox((x, y), text, font=font)
        draw.rectangle(
            [max(0, bbox[0] - 2), max(0, bbox[1] - 2), min(w, bbox[2] + 3), min(h, bbox[3] + 3)],
            fill=(0, 0, 0),
        )
        draw.text((x, y), text, font=font, fill=color_rgb)
    result = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    np.copyto(img, result)
