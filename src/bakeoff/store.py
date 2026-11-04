"""Read side of the content-addressed store.

Published bundles keep a few small files in git; every other file is fetched by sha256 from a
public store, cached locally and used only if its hash matches.
"""

import hashlib
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from bakeoff import safejson
from bakeoff.hashing import bad_relpath
from bakeoff.ledger import Withdrawal, load_withdrawn

PUBLIC_URL = "https://data.bakeoff.si"
DEFAULT_CACHE = "~/.cache/bakeoff/store"
BACKOFF = (1, 2, 4)
TIMEOUT = 60
CHUNK = 1 << 20


class StoreError(Exception):
    pass


class Unavailable(StoreError):
    def __init__(self, sha: str, url: str, status: int | str | None):
        super().__init__(f"object {sha} unavailable at {url} (status {status})")
        self.sha, self.url, self.status = sha, url, status


class HashMismatch(StoreError):
    def __init__(self, sha: str, where: str):
        super().__init__(f"object {sha}: content from {where} does not match its sha256")
        self.sha, self.where = sha, where


class Withdrawn(StoreError):
    def __init__(self, sha: str, date: str, reason: str):
        super().__init__(f"object {sha} was withdrawn on {date}: {reason}")
        self.sha, self.date, self.reason = sha, date, reason


def object_key(sha: str) -> str:
    return f"sha256/{sha[:2]}/{sha}"


def _retryable(status: int) -> bool:
    return status == 429 or status >= 500


@dataclass
class Store:
    base_url: str
    cache: Path
    withdrawn: dict[str, Withdrawal]

    @classmethod
    def from_env(cls, root: Path) -> "Store":
        base = os.environ.get("BAKEOFF_STORE_URL") or PUBLIC_URL
        try:
            cache = Path(os.environ.get("BAKEOFF_CACHE") or DEFAULT_CACHE).expanduser()
        except RuntimeError:
            raise StoreError(
                "cannot find a home directory for the default cache; set BAKEOFF_CACHE"
            ) from None
        return cls(base, cache, load_withdrawn(root))

    def _url(self, sha: str) -> str:
        return self.base_url.rstrip("/") + "/" + object_key(sha)

    def _path(self, sha: str) -> Path:
        return self.cache / object_key(sha)

    def cached(self, sha: str) -> Path | None:
        path = self._path(sha)
        return path if path.is_file() else None

    def exists(self, sha: str) -> bool:
        url = self._url(sha)
        for attempt in range(len(BACKOFF) + 1):
            try:
                req = urllib.request.Request(url, method="HEAD")
                with urllib.request.urlopen(req, timeout=TIMEOUT):
                    return True
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return False
                if not _retryable(e.code):
                    raise Unavailable(sha, url, e.code) from None
                status: int | str = e.code
            except OSError as e:
                status = type(e).__name__
            if attempt == len(BACKOFF):
                raise Unavailable(sha, url, status)
            time.sleep(BACKOFF[attempt])
        raise AssertionError("unreachable")

    def resolve(self, sha: str) -> Path:
        if sha in self.withdrawn:
            w = self.withdrawn[sha]
            raise Withdrawn(sha, w.date, w.reason)
        path = self._path(sha)
        if path.is_file():
            if _hash_file(path) != sha:
                raise HashMismatch(sha, str(path))
            return path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise StoreError(f"cannot create cache directory {path.parent}: {e}") from None
        self._download(sha, path)
        return path

    def _download(self, sha: str, path: Path) -> None:
        url = self._url(sha)
        status: int | str = "no attempt"
        for attempt in range(len(BACKOFF) + 1):
            try:
                self._fetch_once(sha, url, path)
                return
            except urllib.error.HTTPError as e:
                if not _retryable(e.code):
                    raise Unavailable(sha, url, e.code) from None
                status = e.code
            except OSError as e:
                if isinstance(e, PermissionError | FileNotFoundError) and not path.parent.exists():
                    raise StoreError(f"cannot write to cache {path.parent}: {e}") from None
                status = type(e).__name__
            if attempt == len(BACKOFF):
                raise Unavailable(sha, url, status)
            time.sleep(BACKOFF[attempt])

    def _fetch_once(self, sha: str, url: str, path: Path) -> None:
        h = hashlib.sha256()
        try:
            tmp = tempfile.NamedTemporaryFile(dir=path.parent, delete=False)
        except OSError as e:
            raise StoreError(f"cannot write to cache {path.parent}: {e}") from None
        try:
            with tmp, urllib.request.urlopen(url, timeout=TIMEOUT) as resp:
                while chunk := resp.read(CHUNK):
                    h.update(chunk)
                    tmp.write(chunk)
            if h.hexdigest() != sha:
                raise HashMismatch(sha, url)
            os.replace(tmp.name, path)
        except BaseException:
            Path(tmp.name).unlink(missing_ok=True)
            raise


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def resolve_file(bundle_dir: Path, rel: str, sha: str, store: Store | None) -> Path:
    local = bundle_dir / rel
    if local.is_file():
        return local
    if store is None:
        raise Unavailable(sha, str(local), "no local file and no store")
    return store.resolve(sha)


def materialize(bundle_dir: Path, dst: Path, store: Store) -> Path:
    """Copy a split bundle to `dst` and fill in every listed file from the store."""
    bundle_dir, dst = Path(bundle_dir), Path(dst)
    try:
        meta = safejson.loads((bundle_dir / "bundle.json").read_bytes().decode("utf-8"))
    except (OSError, ValueError) as e:
        raise StoreError(f"cannot read bundle.json in {bundle_dir.name}: {e}") from e
    listed = meta.get("files") if isinstance(meta, dict) else None
    if not isinstance(listed, dict):
        raise StoreError("bundle.json has no files object")
    for rel in listed:
        if bad_relpath(rel):
            raise StoreError(f"unsafe path in bundle files: {rel!r}")
    shutil.copytree(bundle_dir, dst, symlinks=True)
    for rel, sha in listed.items():
        target = dst / rel
        if target.is_file() and not target.is_symlink():
            continue
        cur = dst
        for part in PurePosixPath(rel).parts[:-1]:
            cur = cur / part
            if cur.is_symlink():
                raise StoreError(f"refusing to write through a symlink: {rel!r}")
        if target.is_symlink():
            raise StoreError(f"refusing to write through a symlink: {rel!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(store.resolve(sha), target)
    return dst
