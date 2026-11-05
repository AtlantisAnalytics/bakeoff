"""Loaders for results/layout.toml and results/withdrawn.toml.

The layout says which files of a published bundle stay in git; the rest live in the store.
Both files are public, so the loaders fail closed with a message that names the problem.
"""

import re
import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

_WEEK = re.compile(r"\d{4}-\d{2}-\d{2}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FIRST = "0000-01-01"


class LedgerError(ValueError):
    pass


@dataclass(frozen=True)
class Range:
    start: str
    keep: dict[str, frozenset[str]] | None  # None: keep = "all"


@dataclass(frozen=True)
class Layout:
    ranges: tuple[Range, ...]

    def kept(self, week: str, kind: str) -> frozenset[str] | None:
        found = None
        for r in self.ranges:
            if r.start <= week:
                found = r
        if found is None:
            raise LedgerError(f"layout.toml: week {week} is before the first range")
        if found.keep is None:
            return None
        if kind not in found.keep:
            raise LedgerError(f"layout.toml: range from {found.start} does not name {kind!r}")
        return found.keep[kind]


@dataclass(frozen=True)
class Withdrawal:
    sha256: str
    date: str
    reason: str
    bundles: tuple[str, ...]


def _iso_date(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _WEEK.fullmatch(value):
        raise LedgerError(f"{field} must be an ISO YYYY-MM-DD date")
    try:
        date.fromisoformat(value)
    except ValueError:
        raise LedgerError(f"{field} is not a real date: {value!r}") from None
    return value


def _read(path: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as e:
        raise LedgerError(f"{path.name}: {e}") from None


def load_layout(root: Path) -> Layout:
    path = root / "results" / "layout.toml"
    if not path.exists():
        return Layout((Range(_FIRST, None),))
    data = _read(path)
    rows = data.get("layout")
    if not isinstance(rows, list) or not rows:
        raise LedgerError("layout.toml: needs at least one [[layout]] range")
    ranges: list[Range] = []
    for row in rows:
        start = _iso_date(row.get("from"), "layout.toml: from")
        if ranges and start == ranges[-1].start:
            raise LedgerError(f"layout.toml: week {start} is repeated")
        if ranges and start < ranges[-1].start:
            raise LedgerError(f"layout.toml: ranges must be sorted by from ({start})")
        rest = {k: v for k, v in row.items() if k != "from"}
        if "keep" in rest:
            if rest != {"keep": "all"}:
                raise LedgerError(f'layout.toml: keep must be "all" and stand alone ({start})')
            ranges.append(Range(start, None))
            continue
        keep: dict[str, frozenset[str]] = {}
        for kind, names in rest.items():
            ok = isinstance(names, list) and names and all(isinstance(n, str) for n in names)
            if not ok:
                raise LedgerError(f"layout.toml: {kind} must be a list of file names ({start})")
            keep[kind] = frozenset(names)
        if "run" not in keep or "judge" not in keep:
            raise LedgerError(f"layout.toml: a split range needs run and judge ({start})")
        ranges.append(Range(start, keep))
    return Layout(tuple(ranges))


def load_withdrawn(root: Path) -> dict[str, Withdrawal]:
    path = root / "results" / "withdrawn.toml"
    if not path.exists():
        return {}
    rows = _read(path).get("withdrawn", [])
    out: dict[str, Withdrawal] = {}
    for row in rows:
        sha = row.get("sha256")
        if not isinstance(sha, str) or not _SHA256.fullmatch(sha):
            raise LedgerError(f"withdrawn.toml: sha256 must be 64 lowercase hex ({sha!r})")
        when = _iso_date(row.get("date"), f"withdrawn.toml: date ({sha[:12]})")
        reason = row.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise LedgerError(f"withdrawn.toml: reason is empty ({sha[:12]})")
        bundles = row.get("bundles")
        if (
            not isinstance(bundles, list)
            or not bundles
            or not all(isinstance(b, str) and b for b in bundles)
        ):
            raise LedgerError(f"withdrawn.toml: bundles must be a non-empty list ({sha[:12]})")
        if sha in out:
            raise LedgerError(f"withdrawn.toml: {sha[:12]} listed twice")
        out[sha] = Withdrawal(sha, when, reason, tuple(bundles))
    return out


def bundle_kind(meta: dict) -> str:
    return "judge" if meta.get("bundle_format") == "judge-m1" else "run"


def week_of(root: Path, bundle_dir: Path) -> str | None:
    try:
        parts = bundle_dir.relative_to(root).parts
    except ValueError:
        return None
    if len(parts) == 4 and parts[0] == "results" and parts[2] in ("runs", "judge"):
        if _WEEK.fullmatch(parts[1]):
            return parts[1]
    return None
