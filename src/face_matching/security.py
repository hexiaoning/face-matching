from __future__ import annotations

import base64
import ctypes
import hashlib
import hmac
import os
import stat
import sys
from ctypes import wintypes
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_byte))]


def _blob(data: bytes) -> tuple[_DataBlob, ctypes.Array[ctypes.c_char]]:
    buffer = ctypes.create_string_buffer(data)
    value = _DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte)))
    return value, buffer


def _dpapi_protect(data: bytes) -> bytes:
    source, source_buffer = _blob(data)
    output = _DataBlob()
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    crypt32.CryptProtectData.argtypes = [
        ctypes.POINTER(_DataBlob),
        wintypes.LPCWSTR,
        ctypes.POINTER(_DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(_DataBlob),
    ]
    crypt32.CryptProtectData.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    if not crypt32.CryptProtectData(
        ctypes.byref(source),
        "FaceMatching local key",
        None,
        None,
        None,
        0,
        ctypes.byref(output),
    ):
        raise ctypes.WinError()
    try:
        return ctypes.string_at(output.pbData, output.cbData)
    finally:
        kernel32.LocalFree(output.pbData)
        del source_buffer


def _dpapi_unprotect(data: bytes) -> bytes:
    source, source_buffer = _blob(data)
    output = _DataBlob()
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    crypt32.CryptUnprotectData.argtypes = [
        ctypes.POINTER(_DataBlob),
        ctypes.POINTER(wintypes.LPWSTR),
        ctypes.POINTER(_DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(_DataBlob),
    ]
    crypt32.CryptUnprotectData.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    if not crypt32.CryptUnprotectData(
        ctypes.byref(source), None, None, None, None, 0, ctypes.byref(output)
    ):
        raise ctypes.WinError()
    try:
        return ctypes.string_at(output.pbData, output.cbData)
    finally:
        kernel32.LocalFree(output.pbData)
        del source_buffer


class LocalVault:
    """Encrypt sensitive fields; DPAPI binds the master key to the Windows user."""

    def __init__(self, key_path: Path):
        self.key_path = key_path
        self._key = self._load_or_create_key()
        self._cipher = AESGCM(self._key)

    def _load_or_create_key(self) -> bytes:
        if self.key_path.exists():
            stored = base64.urlsafe_b64decode(self.key_path.read_bytes())
            return _dpapi_unprotect(stored) if sys.platform == "win32" else stored
        key = AESGCM.generate_key(bit_length=256)
        stored = _dpapi_protect(key) if sys.platform == "win32" else key
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        self.key_path.write_bytes(base64.urlsafe_b64encode(stored))
        if sys.platform != "win32":
            os.chmod(self.key_path, stat.S_IRUSR | stat.S_IWUSR)
        return key

    def encrypt_text(self, value: str) -> bytes:
        nonce = os.urandom(12)
        return nonce + self._cipher.encrypt(nonce, value.encode("utf-8"), None)

    def decrypt_text(self, value: bytes) -> str:
        nonce, ciphertext = value[:12], value[12:]
        return self._cipher.decrypt(nonce, ciphertext, None).decode("utf-8")

    def keyed_digest(self, value: str) -> str:
        normalized = "".join(value.split()).upper().encode("utf-8")
        return hmac.new(self._key, normalized, hashlib.sha256).hexdigest()


def mask_government_id(value: str) -> str:
    clean = "".join(value.split())
    if len(clean) <= 6:
        return "*" * len(clean)
    return f"{clean[:3]}{'*' * (len(clean) - 7)}{clean[-4:]}"
