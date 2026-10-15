"""Loaders for tiers.toml, week.toml and site.toml.

These files end up on a public website, so every loader fails closed: unknown keys, missing
keys, wrong types and non-https URLs are all rejected with a message that names the field.
"""

import re
import tomllib
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

VERDICTS = ("replace", "hold", "watch")

_SLUG = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
_HOST = re.compile(r"[a-z0-9.-]+")
_LICENSE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+\-() ]*")
_PUBLISHED_AT = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


class WeekInputError(ValueError):
    pass


@dataclass(frozen=True)
class Tier:
    id: str
    name: str
    envelope: str


@dataclass(frozen=True)
class RunEntry:
    bundle: Path
    tier: str
    license: str
    source_url: str
    license_url: str | None


@dataclass(frozen=True)
class Override:
    tier: str
    track: str
    run: str
    verdict: str
    reason: str


@dataclass(frozen=True)
class WeekInput:
    week: str
    published_at: str
    runs: list[RunEntry]
    judges: list[Path]
    overrides: list[Override]
    correction: str | None
    source_bytes: bytes


@dataclass(frozen=True)
class SiteConfig:
    launched: bool
    base_url: str
    repo_url: str
    newsletter_action: str
    email_field: str


def _read(path: Path) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = Path(path).read_bytes()
    except FileNotFoundError as e:
        raise WeekInputError(f"file not found: {path}") from e
    try:
        return raw, tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
        raise WeekInputError(f"invalid TOML in {path}: {e}") from e


def _table(where: str, v: Any) -> dict[str, Any]:
    if not isinstance(v, dict):
        raise WeekInputError(f"{where} must be a table")
    return v


def _table_list(where: str, v: Any) -> list[dict[str, Any]]:
    if not isinstance(v, list) or not all(isinstance(x, dict) for x in v):
        raise WeekInputError(f"{where} must be an array of tables")
    return v


def _check_keys(where: str, table: dict[str, Any], required: tuple, optional: tuple = ()) -> None:
    unknown = sorted(set(table) - set(required) - set(optional))
    if unknown:
        raise WeekInputError(f"unknown key(s) in {where}: {', '.join(unknown)}")
    for key in required:
        if key not in table:
            raise WeekInputError(f"missing required key: {where}.{key}")


def _text(where: str, v: Any, *, allow_empty: bool = False) -> str:
    if not isinstance(v, str):
        raise WeekInputError(f"{where} must be a string")
    if not allow_empty and not v.strip():
        raise WeekInputError(f"{where} must not be empty")
    return v


def _https(where: str, v: Any, *, allow_empty: bool = False) -> str:
    s = _text(where, v, allow_empty=allow_empty)
    if s == "" and allow_empty:
        return s
    bad = WeekInputError(f"{where} must be an https:// URL with a plain host, got {s!r}")
    if any(c.isspace() or ord(c) < 32 or ord(c) == 127 or c == "\\" for c in s):
        raise bad
    try:
        parts = urlsplit(s)
        parts.port  # noqa: B018 - raises ValueError on a malformed port
    except ValueError as e:
        raise bad from e
    host = parts.hostname or ""
    if parts.scheme != "https" or "@" in parts.netloc or not _HOST.fullmatch(host):
        raise bad
    return s


def load_tiers(path: Path) -> dict[str, Tier]:
    _, data = _read(path)
    _check_keys("tiers.toml", data, ("tier",))
    entries = _table_list("tier", data["tier"])
    if not entries:
        raise WeekInputError("tier must contain at least one [[tier]]")
    tiers: dict[str, Tier] = {}
    envelopes: set[str] = set()
    for i, t in enumerate(entries):
        where = f"tier[{i}]"
        _check_keys(where, t, ("id", "name", "envelope"))
        tid = _text(f"{where}.id", t["id"])
        if not _SLUG.fullmatch(tid):
            raise WeekInputError(f"{where}.id must be a slug (lowercase a-z, 0-9, '-'): {tid!r}")
        name = _text(f"{where}.name", t["name"])
        env = _text(f"{where}.envelope", t["envelope"])
        if tid in tiers:
            raise WeekInputError(f"duplicate tier id: {tid!r}")
        if env in envelopes:
            raise WeekInputError(f"duplicate envelope: {env!r}")
        tiers[tid] = Tier(tid, name, env)
        envelopes.add(env)
    return tiers


def _monday(v: Any) -> str:
    s = _text("week", v)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        raise WeekInputError(f"week must be an ISO date YYYY-MM-DD, got {s!r}")
    try:
        d = date.fromisoformat(s)
    except ValueError as e:
        raise WeekInputError(f"week is not a real date: {s!r}") from e
    if d.weekday() != 0:
        raise WeekInputError(f"week must be a Monday, got {s!r}")
    return s


def _published_at(v: Any) -> str:
    s = _text("published_at", v)
    if not _PUBLISHED_AT.fullmatch(s):
        raise WeekInputError(f"published_at must match YYYY-MM-DDTHH:MM:SSZ, got {s!r}")
    try:
        datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as e:
        raise WeekInputError(f"published_at is not a real date-time: {s!r}") from e
    return s


def _run(i: int, t: dict[str, Any], base: Path, tiers: dict[str, Tier]) -> RunEntry:
    where = f"run[{i}]"
    _check_keys(where, t, ("bundle", "tier", "license", "source_url"), ("license_url",))
    bundle = base / _text(f"{where}.bundle", t["bundle"])
    tier = _text(f"{where}.tier", t["tier"])
    if tier not in tiers:
        raise WeekInputError(f"{where}.tier {tier!r} is not in tiers.toml")
    lic = _text(f"{where}.license", t["license"]).strip()
    if lic != "other" and not _LICENSE.fullmatch(lic):
        raise WeekInputError(f"{where}.license must be an SPDX id or 'other', got {lic!r}")
    license_url = None
    if "license_url" in t:
        license_url = _https(f"{where}.license_url", t["license_url"])
    if lic == "other" and license_url is None:
        raise WeekInputError(f"{where}.license_url is required when license is 'other'")
    source_url = _https(f"{where}.source_url", t["source_url"])
    return RunEntry(bundle, tier, lic, source_url, license_url)


def _override(
    i: int, t: dict[str, Any], tiers: dict[str, Tier], run_tiers: dict[str, str]
) -> Override:
    where = f"override[{i}]"
    _check_keys(where, t, ("tier", "track", "run", "verdict", "reason"))
    tier = _text(f"{where}.tier", t["tier"])
    if tier not in tiers:
        raise WeekInputError(f"{where}.tier {tier!r} is not in tiers.toml")
    track = _text(f"{where}.track", t["track"])
    run = _text(f"{where}.run", t["run"])
    if run not in run_tiers:
        raise WeekInputError(f"{where}.run {run!r} is not a run in this week")
    if run_tiers[run] != tier:
        raise WeekInputError(
            f"{where}.tier {tier!r} does not match the tier of run {run!r} ({run_tiers[run]!r})"
        )
    verdict = _text(f"{where}.verdict", t["verdict"])
    if verdict not in VERDICTS:
        raise WeekInputError(f"{where}.verdict must be one of {', '.join(VERDICTS)}: {verdict!r}")
    reason = _text(f"{where}.reason", t["reason"]).strip()
    return Override(tier, track, run, verdict, reason)


def load_week(path: Path, tiers: dict[str, Tier]) -> WeekInput:
    raw, data = _read(path)
    base = Path(path).parent
    _check_keys(
        "week.toml",
        data,
        ("week", "published_at", "run"),
        ("judge", "override", "correction"),
    )
    week = _monday(data["week"])
    published_at = _published_at(data["published_at"])
    run_tables = _table_list("run", data["run"])
    if not run_tables:
        raise WeekInputError("run must contain at least one [[run]]")
    runs = [_run(i, t, base, tiers) for i, t in enumerate(run_tables)]
    run_ids = [r.bundle.name for r in runs]
    if len(set(run_ids)) != len(run_ids):
        raise WeekInputError("run: two [[run]] entries share a bundle directory name")
    judges = []
    for i, t in enumerate(_table_list("judge", data.get("judge", []))):
        _check_keys(f"judge[{i}]", t, ("bundle",))
        judges.append(base / _text(f"judge[{i}].bundle", t["bundle"]))
    overrides = [
        _override(i, t, tiers, {r.bundle.name: r.tier for r in runs})
        for i, t in enumerate(_table_list("override", data.get("override", [])))
    ]
    correction = None
    if "correction" in data:
        c = _table("correction", data["correction"])
        _check_keys("correction", c, ("note",))
        correction = _text("correction.note", c["note"]).strip()
    return WeekInput(week, published_at, runs, judges, overrides, correction, raw)


def load_site(path: Path) -> SiteConfig:
    _, data = _read(path)
    _check_keys("site.toml", data, ("launched", "base_url", "repo_url", "newsletter"))
    if not isinstance(data["launched"], bool):
        raise WeekInputError("launched must be true or false")
    base_url = _https("base_url", data["base_url"])
    repo_url = _https("repo_url", data["repo_url"])
    nl = _table("newsletter", data["newsletter"])
    _check_keys("newsletter", nl, ("action", "email_field"))
    action = _https("newsletter.action", nl["action"], allow_empty=True)
    email_field = _text("newsletter.email_field", nl["email_field"])
    return SiteConfig(data["launched"], base_url, repo_url, action, email_field)
