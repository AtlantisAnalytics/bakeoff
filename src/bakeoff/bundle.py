"""Bundle writing, verification and the scrub scan."""

import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath

from bakeoff.hashing import canonical_json, sha256_file, sha256_hex


def slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")[:40].strip("-")


def run_id(started: datetime, label: str, salt: bytes) -> str:
    started = started.astimezone(UTC)
    return f"{started:%Y%m%dT%H%M%SZ}-{slug(label)}-{sha256_hex(salt)[:8]}"


def _bundle_hash(meta: dict) -> str:
    return sha256_hex(canonical_json({k: v for k, v in meta.items() if k != "bundle_sha256"}))


def _bad_relpath(rel: object) -> bool:
    if not isinstance(rel, str) or not rel:
        return True
    for cls in (PurePosixPath, PureWindowsPath):
        p = cls(rel)
        if p.is_absolute() or p.anchor or ".." in p.parts:
            return True
    return False


def write_bundle(run_dir: Path, meta: dict, files: dict[str, list[dict]]) -> str:
    run_dir = Path(run_dir)
    for rel in files:
        if _bad_relpath(rel):
            raise ValueError(f"unsafe bundle file path: {rel!r}")
    run_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for rel, records in files.items():
        path = run_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        text = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in records)
        path.write_bytes(text.encode("utf-8"))
        hashes[rel] = sha256_file(path)
    meta["files"] = hashes
    meta.pop("bundle_sha256", None)
    meta["bundle_sha256"] = _bundle_hash(meta)
    (run_dir / "bundle.json").write_bytes(
        (json.dumps(meta, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    )
    return meta["bundle_sha256"]


@dataclass(frozen=True)
class Finding:
    file: str
    line: int
    kind: str
    detail: str


_OCT = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
_PRIVATE_IPV4 = re.compile(
    r"(?<![\d.])(?:"
    rf"10\.{_OCT}\.{_OCT}\.{_OCT}"
    rf"|172\.(?:1[6-9]|2\d|3[01])\.{_OCT}\.{_OCT}"
    rf"|192\.168\.{_OCT}\.{_OCT}"
    rf"|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.{_OCT}\.{_OCT}"
    rf"|169\.254\.{_OCT}\.{_OCT}"
    rf"|127\.{_OCT}\.{_OCT}\.{_OCT}"
    r")(?![\d])"
)
_PRIVATE_IPV6 = re.compile(r"(?<![0-9a-f:])(?:f[cd][0-9a-f]{2}|fe[89ab][0-9a-f]):", re.I)
_HOST_PORT = re.compile(
    r"(?i)\b(?:localhost|(?:\d{1,3}\.){3}\d{1,3}|[a-z0-9-]+(?:\.[a-z0-9-]+)+):\d{2,5}\b"
)
_HOME_PATH = re.compile(
    r"(?<![\w.\-])(?:/Users/|/home/)[A-Za-z0-9._-]+"
    r"|(?<![A-Za-z0-9])(?i:[a-z]:\\Users\\)[A-Za-z0-9._-]+"
)
_PATTERNS = (
    ("private_ip", _PRIVATE_IPV4),
    ("private_ip", _PRIVATE_IPV6),
    ("host_port", _HOST_PORT),
    ("home_path", _HOME_PATH),
)


def _word(term: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", re.I)


def scan_text(
    text: str, terms: list[str], username: str | None = None
) -> list[tuple[int, str, str]]:
    word_checks = [("denylist", _word(t)) for t in terms if len(t) >= 3]
    if username and len(username) >= 3:
        word_checks.append(("username", _word(username)))
    hits: list[tuple[int, str, str]] = []
    for no, line in enumerate(text.split("\n"), 1):
        for kind, pat in (*_PATTERNS, *word_checks):
            hits.extend((no, kind, m.group(0)) for m in pat.finditer(line))
    return hits


def _load_terms(denylist: Path | None) -> list[str]:
    if denylist is None or not Path(denylist).is_file():
        return []
    lines = Path(denylist).read_text(encoding="utf-8").splitlines()
    return [s for s in (ln.strip() for ln in lines) if s and not s.startswith("#")]


def _strings(obj: object):
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, str):
            yield cur
        elif isinstance(cur, dict):
            for k, v in cur.items():
                stack.append(k)
                stack.append(v)
        elif isinstance(cur, list):
            stack.extend(cur)


def _walk(run_dir: Path, errors: list[str] | None = None):
    """Yield (relpath, path, is_symlink) for every file and symlink, never following links."""

    def onerror(e: OSError) -> None:
        if errors is not None:
            errors.append(str(e.filename))

    for root, dirs, names in os.walk(run_dir, followlinks=False, onerror=onerror):
        base = Path(root)
        for name in sorted(dirs + names):
            path = base / name
            if path.is_symlink():
                yield path.relative_to(run_dir).as_posix(), path, True
            elif name in names:
                yield path.relative_to(run_dir).as_posix(), path, False


def _has_symlink(run_dir: Path, rel: str) -> bool:
    cur = run_dir
    for part in PurePosixPath(rel).parts:
        cur = cur / part
        if cur.is_symlink():
            return True
    return False


def verify_bundle(run_dir: Path, username: str | None, denylist: Path | None) -> list[Finding]:
    run_dir = Path(run_dir)
    meta_path = run_dir / "bundle.json"
    if meta_path.is_symlink():
        return [Finding("bundle.json", 0, "unreadable", "bundle.json is a symlink")]
    if not meta_path.is_file():
        return [Finding("bundle.json", 0, "missing_file", "bundle.json not found")]
    try:
        meta = json.loads(meta_path.read_bytes().decode("utf-8"))
    except (ValueError, OSError, RecursionError) as e:
        return [Finding("bundle.json", 0, "unreadable", f"cannot parse bundle.json: {e}")]
    if not isinstance(meta, dict):
        return [Finding("bundle.json", 0, "unreadable", "bundle.json is not an object")]
    findings: list[Finding] = []
    listed = meta.get("files", {})
    if not isinstance(listed, dict):
        findings.append(Finding("bundle.json", 0, "unreadable", "files is not an object"))
        listed = {}
    walk_errors: list[str] = []
    entries = list(_walk(run_dir, walk_errors))
    for bad in walk_errors:
        findings.append(Finding(bad, 0, "unreadable", "directory cannot be read"))
    for rel, expected in listed.items():
        if _bad_relpath(rel):
            findings.append(Finding(str(rel), 0, "unreadable", "unsafe path in files"))
            continue
        path = run_dir / rel
        if _has_symlink(run_dir, rel):
            findings.append(Finding(rel, 0, "unreadable", "listed path goes through a symlink"))
        elif not path.is_file():
            findings.append(Finding(rel, 0, "missing_file", "listed file is missing"))
        elif sha256_file(path) != expected:
            findings.append(Finding(rel, 0, "hash_mismatch", "file hash differs from bundle.json"))
    for rel, _, link in entries:
        if link:
            if rel not in listed:
                findings.append(Finding(rel, 0, "unreadable", "symlink in bundle"))
        elif rel != "bundle.json" and rel not in listed:
            findings.append(Finding(rel, 0, "unlisted_file", "file is not listed in bundle.json"))
    if meta.get("bundle_sha256") != _bundle_hash(meta):
        findings.append(Finding("bundle.json", 0, "bundle_hash", "bundle hash does not match"))
    if meta.get("status") != "complete":
        findings.append(
            Finding("bundle.json", 0, "incomplete", f"status is {meta.get('status')!r}")
        )
    terms = _load_terms(denylist)
    seen: set[Finding] = set()

    def add(f: Finding) -> None:
        if f not in seen:
            seen.add(f)
            findings.append(f)

    for rel, path, link in entries:
        if link or _has_symlink(run_dir, rel):
            continue
        try:
            data = path.read_bytes()
        except OSError as e:
            add(Finding(rel, 0, "unreadable", f"cannot read file: {e}"))
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            add(Finding(rel, 0, "unreadable", "file is not valid UTF-8"))
            text = data.decode("utf-8", errors="replace")
        for no, kind, match in scan_text(text, terms, username):
            add(Finding(rel, no, kind, match))
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            for no, line in enumerate(text.split("\n"), 1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except (ValueError, RecursionError):
                    add(Finding(rel, no, "unreadable", "line is not valid JSON"))
                    continue
                for st in _strings(obj):
                    for _, kind, match in scan_text(st, terms, username):
                        add(Finding(rel, no, kind, match))
        elif suffix == ".json":
            try:
                obj = json.loads(text)
            except (ValueError, RecursionError):
                add(Finding(rel, 0, "unreadable", "file is not valid JSON"))
                continue
            for st in _strings(obj):
                for _, kind, match in scan_text(st, terms, username):
                    add(Finding(rel, 0, kind, match))
    return findings
