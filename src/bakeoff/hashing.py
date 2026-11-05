"""Canonical JSON and SHA-256 helpers used for every hash in a bundle."""

import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


def canonical_json(obj: Any) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bad_relpath(rel: object) -> bool:
    if not isinstance(rel, str) or not rel:
        return True
    for cls in (PurePosixPath, PureWindowsPath):
        p = cls(rel)
        if p.is_absolute() or p.anchor or ".." in p.parts:
            return True
    return False
