"""Offline check of every published bundle against the ledger, and the changed-bundle finder."""

import json
import re
import subprocess
from pathlib import Path

from bakeoff.bundle import Finding, verify_bundle, week_findings
from bakeoff.ledger import LedgerError, bundle_kind, load_layout, load_withdrawn
from bakeoff.store import valid_sha

_WEEK = re.compile(r"\d{4}-\d{2}-\d{2}")


def bundle_dirs(root: Path) -> list[tuple[Path, str | None]]:
    """(directory, week) for every published bundle; week is None for experiments."""
    results = root / "results"
    out: list[tuple[Path, str | None]] = []
    if not results.is_dir():
        return out
    for week_dir in sorted(results.iterdir()):
        if not week_dir.is_dir():
            continue
        if week_dir.name == "experiments":
            for exp in sorted(week_dir.iterdir()):
                out += [(d, None) for d in _children(exp / "runs")]
        elif _WEEK.fullmatch(week_dir.name):
            for sub in ("runs", "judge"):
                out += [(d, week_dir.name) for d in _children(week_dir / sub)]
    return out


def _children(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return [d for d in sorted(path.iterdir()) if d.is_dir()]


def _rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _ledger_finding(path: str, detail: str) -> tuple[str, Finding]:
    return path, Finding(Path(path).name, 0, "ledger", detail)


def _listed(bundle: Path) -> dict | None:
    try:
        meta = json.loads((bundle / "bundle.json").read_bytes())
    except (OSError, ValueError):
        return None
    files = meta.get("files") if isinstance(meta, dict) else None
    if not isinstance(files, dict):
        return None
    return {k: v for k, v in files.items() if valid_sha(v)}


def check_ledger(
    root: Path, *, username: str | None, denylist: Path | None
) -> list[tuple[str, Finding]]:
    """Every finding for every published bundle, offline, paired with the bundle path."""
    root = Path(root)
    found: list[tuple[str, Finding]] = []
    try:
        layout = load_layout(root)
    except LedgerError as e:
        return [_ledger_finding("results/layout.toml", str(e))]
    try:
        withdrawn = load_withdrawn(root)
    except LedgerError as e:
        return [_ledger_finding("results/withdrawn.toml", str(e))]
    for bundle, week in bundle_dirs(root):
        name = _rel(root, bundle)
        kept = None
        if week is not None:
            try:
                meta = json.loads((bundle / "bundle.json").read_bytes())
                kind = bundle_kind(meta if isinstance(meta, dict) else {})
            except (OSError, ValueError):
                kind = "run"
            try:
                kept = layout.kept(week, kind)
            except LedgerError as e:
                found.append(_ledger_finding("results/layout.toml", str(e)))
                continue
        findings = verify_bundle(bundle, username, denylist, kept=kept, offline=True)
        for f in week_findings(bundle, findings):
            found.append((name, f))
    for sha, row in withdrawn.items():
        for rel in row.bundles:
            listed = _listed(root / rel) if (root / rel / "bundle.json").is_file() else None
            if listed is None:
                detail = f"{sha[:12]}: no such bundle {rel}"
            elif sha not in listed.values():
                detail = f"{sha[:12]}: {rel} does not list this object"
            else:
                continue
            found.append(
                ("results/withdrawn.toml", Finding("withdrawn.toml", 0, "withdrawn_row", detail))
            )
    return found


def _git(root: Path, *args: str) -> str:
    try:
        done = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
        )
    except OSError as e:
        raise LedgerError(f"cannot run git: {e}") from None
    if done.returncode != 0:
        raise LedgerError(done.stderr.strip() or f"git {args[0]} failed")
    return done.stdout


def changed_bundles(root: Path, base: str) -> list[Path]:
    """Bundle directories that contain a path changed since `base` (three-dot diff)."""
    root = Path(root)
    try:
        _git(root, "rev-parse", "--verify", "--quiet", f"{base}^{{commit}}")
        names = _git(root, "diff", "--name-only", f"{base}...HEAD", "--", "results/")
    except LedgerError as e:
        raise LedgerError(f"cannot compare against {base!r} (shallow clone?): {e}") from None
    published = {d for d, _ in bundle_dirs(root)}
    out: set[Path] = set()
    for line in names.splitlines():
        for parent in (root / line).parents:
            if parent in published:
                out.add(parent)
                break
    return sorted(out)
