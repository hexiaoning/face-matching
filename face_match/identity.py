from __future__ import annotations

import re

_CHINESE_ID_PATTERN = re.compile(r"^\d{17}[\dX]$")
_CHECKSUM_WEIGHTS = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
_CHECKSUM_CODES = "10X98765432"


def normalize_id_number(value: str) -> str:
    return "".join(value.strip().upper().split())


def is_valid_chinese_id(value: str) -> bool:
    value = normalize_id_number(value)
    if not _CHINESE_ID_PATTERN.fullmatch(value):
        return False
    total = sum(int(digit) * weight for digit, weight in zip(value[:17], _CHECKSUM_WEIGHTS))
    return value[-1] == _CHECKSUM_CODES[total % 11]


def validate_identity(name: str, id_number: str) -> tuple[str, str]:
    name = " ".join(name.strip().split())
    id_number = normalize_id_number(id_number)
    if not name:
        raise ValueError("姓名不能为空")
    if len(name) > 100:
        raise ValueError("姓名不能超过 100 个字符")
    if not id_number:
        raise ValueError("身份证号不能为空")
    if len(id_number) > 64:
        raise ValueError("身份证号不能超过 64 个字符")
    # Chinese resident IDs get checksum validation. Other document formats remain supported.
    if (
        len(id_number) == 18
        and all(char.isdigit() or char == "X" for char in id_number)
        and not is_valid_chinese_id(id_number)
    ):
        raise ValueError("18 位居民身份证校验位不正确")
    return name, id_number


def mask_id_number(value: str) -> str:
    if len(value) >= 10:
        return f"{value[:6]}{'*' * max(4, len(value) - 10)}{value[-4:]}"
    if len(value) >= 4:
        return f"{value[:2]}{'*' * (len(value) - 4)}{value[-2:]}"
    return "*" * len(value)
