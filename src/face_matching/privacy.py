from __future__ import annotations

import re


def mask_id_card(value: str) -> str:
    """Mask an identifier before it is shown in recognition results."""
    if not value:
        return ""
    if len(value) <= 4:
        return "*" * len(value)
    if len(value) <= 8:
        return value[0] + "*" * (len(value) - 2) + value[-1]
    return value[:4] + "*" * (len(value) - 8) + value[-4:]


def redact_source_credentials(value: str) -> str:
    """Remove camera URL user-info from labels, logs, and error messages."""
    return re.sub(r"(?i)\b(rtsp|rtmp|http|https)://[^/@\s]+@", r"\1://", value, count=1)
