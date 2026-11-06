"""The yearly archive: a manifest and one tar per month of every stored object, for Zenodo."""

import json
import os
import re
import shutil
import tarfile
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from bakeoff.ledger import LedgerError, bundle_kind, load_layout
from bakeoff.ledgercheck import bundle_dirs
from bakeoff.store import (
    MANIFEST,
    NETWORK_ERRORS,
    USER_AGENT,
    Store,
    StoreError,
    object_key,
    read_manifest,
    valid_sha,
)

COMMUNITY = "bakeoff-si"
LICENSE = "cc-by-4.0"
DEFAULT_ZENODO = "https://zenodo.org"
MAX_FILES = 100
TIMEOUT = 600

README = """# bakeoff.si raw data, {year}

Every raw file listed by the weekly bundles published in {year} whose files live in the
content-addressed store rather than in git. License: CC BY 4.0.

- `{year}-MM.tar` holds the objects of the weeks starting in month MM. Each member is named by its
  object key, `sha256/<aa>/<hash>`, where `<hash>` is the SHA-256 of the member's bytes.
- `MANIFEST.sha256` has one line per object: `<sha256>  <size>  <bundle path>[,<bundle path>...]`.
  The paths are bundle directories in the repository and `bundle.json` in each lists the file.

To verify, hash every member and compare it with its name and with the manifest, then compare
the hashes with the `files` object of `bundle.json` in the bundle directories named in the
manifest. `bakeoff store fetch --archive {year} --archive-dir DIR BUNDLE...` does this.
Objects withdrawn before this record was made are not included.
"""


class ArchiveError(Exception):
    pass


@dataclass(frozen=True)
class ArchiveRow:
    year: int
    doi: str
    manifest_sha256: str


def load_archive_rows(root: Path) -> dict[int, ArchiveRow]:
    path = Path(root) / "results" / "archive.toml"
    if not path.exists():
        return {}
    try:
        rows = tomllib.loads(path.read_text()).get("archive", [])
    except tomllib.TOMLDecodeError as e:
        raise ArchiveError(f"archive.toml: {e}") from None
    out: dict[int, ArchiveRow] = {}
    for row in rows:
        year, doi, sha = row.get("year"), row.get("doi"), row.get("manifest_sha256")
        ok = isinstance(year, int) and isinstance(doi, str) and doi.strip()
        ok = ok and isinstance(sha, str) and len(sha) == 64 and set(sha) <= set("0123456789abcdef")
        if not ok or year in out:
            raise ArchiveError(f"archive.toml: bad or repeated row for year {year!r}")
        out[year] = ArchiveRow(year, doi, sha)
    return out


def _split_objects(root: Path, year: int) -> dict[str, tuple[str, list[str]]]:
    """{sha: (first week, bundle paths)} over the year's split weeks."""
    layout = load_layout(root)
    found: dict[str, tuple[str, list[str]]] = {}
    for bundle, week in bundle_dirs(root):
        if week is None or not week.startswith(f"{year:04d}-"):
            continue
        try:
            meta = json.loads((bundle / "bundle.json").read_bytes())
        except (OSError, ValueError) as e:
            raise ArchiveError(f"{bundle.name}: cannot read bundle.json: {e}") from None
        files = meta.get("files") if isinstance(meta, dict) else None
        if not isinstance(files, dict):
            raise ArchiveError(f"{bundle.name}: bundle.json has no files object")
        if layout.kept(week, bundle_kind(meta)) is None:
            continue
        rel = bundle.relative_to(root).as_posix()
        for sha in files.values():
            if not valid_sha(sha):
                raise ArchiveError(f"{bundle.name}: bad sha256 in bundle.json files")
            first, paths = found.setdefault(sha, (week, []))
            if rel not in paths:
                paths.append(rel)
            if week < first:
                found[sha] = (week, paths)
    return found


def build_archive(root: Path, year: int, out: Path, store: Store) -> Path:
    root, out = Path(root), Path(out)
    try:
        objects = _split_objects(root, year)
    except LedgerError as e:
        raise ArchiveError(str(e)) from None
    objects = {s: v for s, v in objects.items() if s not in store.withdrawn}
    if not objects:
        raise ArchiveError(f"no split weeks (or no objects) in {year}; nothing to archive")
    months: dict[str, list[str]] = {}
    for sha, (week, _) in objects.items():
        months.setdefault(week[5:7], []).append(sha)
    if len(months) + 2 > MAX_FILES:
        raise ArchiveError("too many files for one Zenodo record")
    sizes: dict[str, int] = {}
    try:
        for sha in sorted(objects):
            sizes[sha] = store.resolve(sha).stat().st_size
    except StoreError as e:
        raise ArchiveError(str(e)) from None
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ArchiveError(f"{out} exists and is not empty; use a new directory")
    out.mkdir(parents=True, exist_ok=True)
    for month, shas in sorted(months.items()):
        with tarfile.open(out / f"{year}-{month}.tar", "w", format=tarfile.USTAR_FORMAT) as tar:
            for sha in sorted(shas, key=object_key):
                info = tarfile.TarInfo(object_key(sha))
                info.size, info.mode = sizes[sha], 0o644
                with store.resolve(sha).open("rb") as f:
                    tar.addfile(info, f)
    lines = [f"{sha}  {sizes[sha]}  {','.join(sorted(objects[sha][1]))}" for sha in sorted(objects)]
    (out / MANIFEST).write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "README.md").write_text(README.format(year=year), encoding="utf-8")
    return out


def _call(method: str, url: str, token: str, body=None, *, raw: Path | None = None) -> dict:
    headers = {"Authorization": f"Bearer {token}", "User-Agent": USER_AGENT}
    data = None
    if raw is not None:
        headers["Content-Type"] = "application/octet-stream"
        headers["Content-Length"] = str(raw.stat().st_size)
        data = raw.open("rb")
    elif body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode()
    try:
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            text = resp.read()
    except urllib.error.HTTPError as e:
        raise ArchiveError(
            f"Zenodo {method} {urllib.parse.urlsplit(url).path}: HTTP {e.code}"
        ) from None
    except NETWORK_ERRORS as e:
        raise ArchiveError(
            f"Zenodo {method} {urllib.parse.urlsplit(url).path}: {type(e).__name__}"
        ) from None
    finally:
        if raw is not None and data is not None:
            data.close()
    try:
        return json.loads(text) if text else {}
    except ValueError:
        raise ArchiveError(f"Zenodo {method}: reply is not JSON") from None


def _expected_files(out: Path, year: int) -> list[str]:
    """The files of a built archive; anything else in `out`, or a tar that disagrees with the
    manifest, is refused because a published Zenodo record can never be changed."""
    if not out.is_dir():
        raise ArchiveError(f"{out} is not a directory")
    tar_name = re.compile(rf"{year}-(0[1-9]|1[0-2])\.tar")
    names = sorted(p.name for p in out.iterdir())
    for n in names:
        if n not in (MANIFEST, "README.md") and not tar_name.fullmatch(n):
            raise ArchiveError(f"unexpected file in the archive directory: {n}")
    tars = [n for n in names if n.endswith(".tar")]
    if MANIFEST not in names or "README.md" not in names or not tars:
        raise ArchiveError("the archive directory is incomplete")
    try:
        listed = set(read_manifest(out / MANIFEST))
    except StoreError as e:
        raise ArchiveError(str(e)) from None
    members: list[str] = []
    for n in tars:
        try:
            with tarfile.open(out / n) as tar:
                members += [m.name.rsplit("/", 1)[-1] for m in tar]
        except (tarfile.TarError, OSError):
            raise ArchiveError(f"{n} is not a readable tar file") from None
    if sorted(members) != sorted(listed):
        raise ArchiveError("the tar files do not match MANIFEST.sha256; rebuild the archive")
    return names


def zenodo_publish(out: Path, year: int, prev_doi: str | None, base_url: str, token: str) -> str:
    """Create a deposition, upload every file of `out`, set the metadata, publish; return the DOI.

    The token is sent only to the base URL's own host."""
    base = base_url.rstrip("/")
    names = _expected_files(Path(out), year)
    dep = _call("POST", f"{base}/api/deposit/depositions", token, {})
    try:
        dep_id, bucket = dep["id"], dep["links"]["bucket"].rstrip("/")
    except (KeyError, TypeError):
        raise ArchiveError("Zenodo deposition reply has no id or bucket link") from None
    if urllib.parse.urlsplit(bucket)[:2] != urllib.parse.urlsplit(base)[:2]:
        raise ArchiveError("Zenodo bucket link is on a different host; refusing to send the token")
    for name in names:
        _call("PUT", f"{bucket}/{urllib.parse.quote(name)}", token, raw=Path(out) / name)
    meta: dict = {
        "title": f"bakeoff.si raw data, {year}",
        "upload_type": "dataset",
        "description": (
            f"<p>Raw files of the bakeoff.si weekly open-model benchmark for {year}, "
            "named by SHA-256. See README.md and MANIFEST.sha256.</p>"
        ),
        "creators": [{"name": "bakeoff.si contributors"}],
        "access_right": "open",
        "license": LICENSE,
        "communities": [{"identifier": COMMUNITY}],
    }
    if prev_doi:
        meta["related_identifiers"] = [{"identifier": prev_doi, "relation": "continues"}]
    _call("PUT", f"{base}/api/deposit/depositions/{dep_id}", token, {"metadata": meta})
    done = _call("POST", f"{base}/api/deposit/depositions/{dep_id}/actions/publish", token, {})
    doi = done.get("doi")
    if not isinstance(doi, str) or not doi:
        raise ArchiveError("Zenodo published the record but returned no DOI")
    return doi


def download_record(doi: str, dest: Path) -> Path:
    """Download the files of the Zenodo record behind `doi` into `dest` (public files API)."""
    base = (os.environ.get("ZENODO_URL") or DEFAULT_ZENODO).rstrip("/")
    record = doi.rsplit(".", 1)[-1]
    if not record.isdigit():
        raise ArchiveError(f"cannot find a Zenodo record number in {doi!r}")
    try:
        listing_req = urllib.request.Request(
            f"{base}/api/records/{record}", headers={"User-Agent": USER_AGENT}
        )
        with urllib.request.urlopen(listing_req, timeout=60) as resp:
            listing = json.loads(resp.read())
        for f in listing["files"]:
            name = f["key"]
            if "/" in name or name.startswith("."):
                raise ArchiveError(f"unsafe file name in record: {name!r}")
            file_req = urllib.request.Request(
                f["links"]["self"], headers={"User-Agent": USER_AGENT}
            )
            with urllib.request.urlopen(file_req, timeout=TIMEOUT) as r:
                with (dest / name).open("wb") as out:
                    shutil.copyfileobj(r, out)
    except (*NETWORK_ERRORS, ValueError, KeyError, TypeError) as e:
        raise ArchiveError(f"cannot download record {record}: {type(e).__name__}") from None
    return dest
