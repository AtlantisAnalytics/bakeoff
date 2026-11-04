"""Read side of the content-addressed store.

Published bundles keep a few small files in git; every other file is fetched by sha256 from a
public store, cached locally and used only if its hash matches.
"""

import hashlib
import http.client
import os
import re
import shutil
import tarfile
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


class ShortBody(OSError):
    """The server closed the connection before the announced number of bytes arrived."""


class Withdrawn(StoreError):
    def __init__(self, sha: str, date: str, reason: str):
        super().__init__(f"object {sha} was withdrawn on {date}: {reason}")
        self.sha, self.date, self.reason = sha, date, reason


MANIFEST = "MANIFEST.sha256"
_SHA = re.compile(r"[0-9a-f]{64}")


NETWORK_ERRORS = (OSError, http.client.HTTPException)


def valid_sha(sha: object) -> bool:
    return isinstance(sha, str) and _SHA.fullmatch(sha) is not None


def object_key(sha: str) -> str:
    if not valid_sha(sha):
        raise StoreError(f"not a sha256: {str(sha)[:80]!r}")
    return f"sha256/{sha[:2]}/{sha}"


def _retryable(status: int) -> bool:
    return status == 429 or status >= 500


class GaveUp(Exception):
    def __init__(self, last: int | str):
        super().__init__(last)
        self.last = last


def retry_call(call):
    """Run `call`, retrying network errors, 429 and 5xx with BACKOFF; return its result.

    Another HTTP status is raised as is. When the retries run out, raise GaveUp."""
    last: int | str = "no attempt"
    for attempt in range(len(BACKOFF) + 1):
        if attempt:
            time.sleep(BACKOFF[attempt - 1])
        try:
            return call()
        except urllib.error.HTTPError as e:
            if not _retryable(e.code):
                raise
            last = e.code
        except NETWORK_ERRORS as e:
            last = type(e).__name__
    raise GaveUp(last)


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
        object_key(sha)
        path = self._path(sha)
        return path if path.is_file() else None

    def exists(self, sha: str) -> bool:
        url = self._url(sha)

        def head() -> bool:
            req = urllib.request.Request(url, method="HEAD")
            try:
                with urllib.request.urlopen(req, timeout=TIMEOUT):
                    return True
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return False
                raise

        try:
            return retry_call(head)
        except urllib.error.HTTPError as e:
            raise Unavailable(sha, url, e.code) from None
        except GaveUp as e:
            raise Unavailable(sha, url, e.last) from None

    def resolve(self, sha: str) -> Path:
        object_key(sha)
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

        def once() -> None:
            try:
                self._fetch_once(sha, url, path)
            except (PermissionError, FileNotFoundError) as e:
                if not path.parent.exists():
                    raise StoreError(f"cannot write to cache {path.parent}: {e}") from None
                raise

        try:
            retry_call(once)
        except urllib.error.HTTPError as e:
            raise Unavailable(sha, url, e.code) from None
        except GaveUp as e:
            raise Unavailable(sha, url, e.last) from None

    def load_archive(self, archive_dir: Path) -> int:
        """Copy the objects of an unpacked yearly archive into the cache; return how many.

        Every tar member must be listed in the manifest and match its sha256 and size."""
        archive_dir = Path(archive_dir)
        sizes = read_manifest(archive_dir / MANIFEST)
        count = 0
        for tar_path in sorted(archive_dir.glob("*.tar")):
            with tarfile.open(tar_path) as tar:
                for member in tar:
                    sha = member.name.rsplit("/", 1)[-1]
                    if not member.isreg() or member.name != object_key(sha) or sha not in sizes:
                        raise StoreError(f"{tar_path.name}: {member.name!r} is not in the manifest")
                    if member.size != sizes[sha]:
                        raise HashMismatch(sha, f"{tar_path.name} (size)")
                    if sha in self.withdrawn:
                        continue
                    src = tar.extractfile(member)
                    if src is None:
                        raise StoreError(f"{tar_path.name}: cannot read {member.name!r}")
                    with src:
                        self._admit(sha, src, tar_path.name)
                    count += 1
        return count

    def _admit(self, sha: str, src, where: str) -> None:
        path = self._path(sha)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = tempfile.NamedTemporaryFile(dir=path.parent, delete=False)
        except OSError as e:
            raise StoreError(f"cannot write to cache {path.parent}: {e}") from None
        h = hashlib.sha256()
        try:
            with tmp:
                while chunk := src.read(CHUNK):
                    h.update(chunk)
                    tmp.write(chunk)
            if h.hexdigest() != sha:
                raise HashMismatch(sha, where)
            os.replace(tmp.name, path)
        except BaseException:
            Path(tmp.name).unlink(missing_ok=True)
            raise

    def _fetch_once(self, sha: str, url: str, path: Path) -> None:
        h = hashlib.sha256()
        try:
            tmp = tempfile.NamedTemporaryFile(dir=path.parent, delete=False)
        except OSError as e:
            raise StoreError(f"cannot write to cache {path.parent}: {e}") from None
        try:
            with tmp, urllib.request.urlopen(url, timeout=TIMEOUT) as resp:
                expected = resp.headers.get("Content-Length")
                got = 0
                while chunk := resp.read(CHUNK):
                    got += len(chunk)
                    h.update(chunk)
                    tmp.write(chunk)
            if expected is not None and expected.isdigit() and got != int(expected):
                raise ShortBody(f"got {got} of {expected} bytes")
            if h.hexdigest() != sha:
                raise HashMismatch(sha, url)
            os.replace(tmp.name, path)
        except BaseException:
            Path(tmp.name).unlink(missing_ok=True)
            raise


def read_manifest(path: Path) -> dict[str, int]:
    """Parse MANIFEST.sha256 into {sha256: size}."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise StoreError(f"cannot read {MANIFEST}: {type(e).__name__}") from None
    out: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split("  ")
        if len(parts) != 3 or not _SHA.fullmatch(parts[0]) or not parts[1].isdigit():
            raise StoreError(f"{MANIFEST}: bad line {line[:80]!r}")
        out[parts[0]] = int(parts[1])
    return out


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


def materialize(bundle_dir: Path, dst: Path, store: Store, *, skip_withdrawn: bool = False) -> Path:
    """Copy a split bundle to `dst` and fill in every listed file from the store.

    A withdrawn file is an error unless `skip_withdrawn`, which leaves it out."""
    bundle_dir, dst = Path(bundle_dir), Path(dst)
    try:
        meta = safejson.loads((bundle_dir / "bundle.json").read_bytes().decode("utf-8"))
    except (OSError, ValueError) as e:
        raise StoreError(f"cannot read bundle.json in {bundle_dir.name}: {e}") from e
    listed = meta.get("files") if isinstance(meta, dict) else None
    if not isinstance(listed, dict):
        raise StoreError("bundle.json has no files object")
    for rel, sha in listed.items():
        if bad_relpath(rel):
            raise StoreError(f"unsafe path in bundle files: {rel!r}")
        if not valid_sha(sha):
            raise StoreError(f"bad sha256 in bundle files: {rel!r}")
    shutil.copytree(bundle_dir, dst, symlinks=True)
    for rel, sha in listed.items():
        target = dst / rel
        if target.is_file() and not target.is_symlink():
            continue
        if skip_withdrawn and sha in store.withdrawn:
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
