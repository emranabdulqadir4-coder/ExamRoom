"""Encrypted evidence storage.

Clips and thumbnails are encrypted at rest with Fernet (AES-128-CBC + HMAC).
They are only decrypted in memory when an authenticated teacher opens them.

The key lives in ``<data>/vault.key`` unless EXAMGUARD_VAULT_KEY is set. For a real
deployment keep the key somewhere other than the data folder (env var / secret store).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

_SAFE = re.compile(r"^[A-Za-z0-9._-]+$")


class VaultError(Exception):
    pass


class Vault:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._fernet = Fernet(self._load_key())

    def _load_key(self) -> bytes:
        env = os.environ.get("EXAMGUARD_VAULT_KEY", "").strip()
        if env:
            return env.encode()
        key_file = self.root / "vault.key"
        if key_file.exists():
            return key_file.read_bytes().strip()
        key = Fernet.generate_key()
        key_file.write_bytes(key)
        try:
            os.chmod(key_file, 0o600)
        except OSError:
            pass
        return key

    def _path(self, name: str) -> Path:
        if not _SAFE.match(name or ""):
            raise VaultError(f"unsafe file name: {name!r}")
        return self.root / (name + ".enc")

    def put(self, name: str, data: bytes) -> str:
        path = self._path(name)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(self._fernet.encrypt(data))
        os.replace(tmp, path)
        return name

    def get(self, name: str) -> bytes:
        path = self._path(name)
        try:
            return self._fernet.decrypt(path.read_bytes())
        except FileNotFoundError as exc:
            raise VaultError("evidence not found (it may have been purged)") from exc
        except InvalidToken as exc:
            raise VaultError("evidence could not be decrypted (wrong key or tampered file)") from exc

    def exists(self, name: str) -> bool:
        return self._path(name).exists()

    def delete(self, name: str) -> bool:
        path = self._path(name)
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False
