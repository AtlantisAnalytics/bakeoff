"""``bakeoff publish``: turn one week's run bundles into ``results/<week>/``.

This is the last gate before publication, so it fails closed. Every bundle is verified in place
(hashes plus the scrub scan), loaded, assessed and written to a staging directory; each copy is
re-verified and the staged ``week.toml`` and ``week.json`` are scrub-scanned before the staging
directory is renamed into place. Any problem raises ``PublishError`` and leaves ``results/``
as it was: no half-written week, no staging directory.

Results layout::

    results/picks.json            the latest week's picks_after (derived, never hand state)
    results/<week>/week.toml      the input, byte for byte
    results/<week>/week.json
    results/<week>/runs/<run_id>/ byte-identical bundle copies
    results/<week>/judge/<id>/    judge bundles; <id> is the bundle's directory name

Rules enforced here (beyond the loaders, the verdict code and the week schema):

* A run or judge bundle must have ``status`` ``complete`` or ``incomplete``. Its only tolerated
  verify finding is ``incomplete`` on an incomplete bundle; incomplete runs are published but
  ineligible.
* Each run's ``config.model.envelope`` equals its tier's envelope exactly; a run's directory name
  equals its ``run_id`` (overrides name runs by directory name), so run_ids are unique.
* A judge bundle's ``a`` and ``b`` are two different runs of this week.
* Bundle paths in ``week.toml`` are relative (the file is published as given).
* Published bundles are immutable: a run_id already published in any week, or a judge bundle
  already in this week, must carry the same ``bundle_sha256``.
* A new week must be later than every published week (an earlier week would change the picks
  that later weeks were assessed against). An existing week is re-published only with
  ``[correction]``; that raises ``revision``, keeps the original ``published_at``, appends to
  ``corrections`` and is refused when a later week exists and the week's ``picks_after`` changes.
* The username scan of the staged ``week.toml`` and ``week.json`` does not honour
  ``username_skip_output`` (it fails closed); pass ``username=None`` (``--no-username``) instead.
* ``picks_before`` is the ``picks_after`` of the latest published week before this one. A tier
  with no runs this week still carries its picks forward (as ``not_re-measured``).
"""

import json
import os
import re
import shutil
import tomllib
from pathlib import Path, PurePosixPath, PureWindowsPath

from bakeoff.bundle import Finding, iter_strings, load_terms, scan_text, verify_bundle
from bakeoff.runview import RunView, RunViewError, load_run
from bakeoff.verdicts import RULES_V1, TierResult, VerdictError, assess_tier
from bakeoff.week import build_week, validate_week
from bakeoff.weekcfg import RunEntry, WeekInputError, load_tiers, load_week

RULES = RULES_V1
STATUSES = ("complete", "incomplete")
JUDGE_FORMAT = "judge-m1"

_WEEK_DIR = re.compile(r"\d{4}-\d{2}-\d{2}")
_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SHOWN = 20


class PublishError(Exception):
    pass


def _dumps(obj: object) -> bytes:
    text = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return (text + "\n").encode("utf-8")


def _no_constants(name: str) -> float:
    raise ValueError(f"{name} is not valid JSON")


def _read_json(path: Path, what: str) -> object:
    try:
        return json.loads(path.read_bytes().decode("utf-8"), parse_constant=_no_constants)
    except (OSError, ValueError, RecursionError) as e:
        raise PublishError(f"cannot read {what}: {e}") from e


def week_docs(root: Path) -> list[dict]:
    """Every ``results/<week>/week.json``, validated, sorted by week."""
    results = Path(root) / "results"
    if not results.is_dir():
        return []
    docs = []
    for d in sorted(results.iterdir()):
        if not _WEEK_DIR.fullmatch(d.name):  # skips picks.json and dot-named work directories
            continue
        path = d / "week.json"
        if not path.is_file():
            raise PublishError(f"results/{d.name} has no week.json")
        doc = _read_json(path, f"results/{d.name}/week.json")
        problems = validate_week(doc)
        if problems:
            raise PublishError(f"results/{d.name}/week.json is invalid: {'; '.join(problems)}")
        if doc["week"] != d.name:
            raise PublishError(f"results/{d.name}/week.json says week {doc['week']!r}")
        docs.append(doc)
    return sorted(docs, key=lambda doc: doc["week"])


def write_picks(root: Path) -> None:
    """``results/picks.json`` becomes the latest week's ``picks_after``, or ``{}``."""
    docs = week_docs(root)
    picks = docs[-1]["picks_after"] if docs else {}
    path = Path(root) / "results" / "picks.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(".picks.json.tmp")
    try:
        tmp.write_bytes(_dumps(picks))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _findings_text(findings: list[Finding]) -> str:
    shown = "; ".join(f"{f.file}:{f.line} {f.kind} {f.detail}" for f in findings[:_SHOWN])
    more = len(findings) - _SHOWN
    return shown + (f"; and {more} more" if more > 0 else "")


def _verify(path: Path, what: str, username, denylist, skip_output: bool) -> dict:
    """Verify one bundle; return its bundle.json. Refuses on any finding or a bad status."""
    if path.is_symlink():
        raise PublishError(f"{what}: bundle directory is a symlink: {path}")
    if not path.is_dir():
        raise PublishError(f"{what}: bundle directory not found: {path}")
    findings = verify_bundle(path, username, denylist, skip_output)
    real = [f for f in findings if f.kind != "incomplete"]
    if real:
        raise PublishError(f"{what} {path.name}: {len(real)} finding(s): {_findings_text(real)}")
    meta = _read_json(path / "bundle.json", f"{what} {path.name}/bundle.json")
    if not isinstance(meta, dict):
        raise PublishError(f"{what} {path.name}: bundle.json is not an object")
    status = meta.get("status")
    if status not in STATUSES:
        raise PublishError(
            f"{what} {path.name}: status must be complete or incomplete, got {status!r}"
        )
    return meta


def _check_paths(source: bytes) -> None:
    data = tomllib.loads(source.decode("utf-8"))
    for kind in ("run", "judge"):
        for i, t in enumerate(data.get(kind, [])):
            s = t["bundle"]
            if PurePosixPath(s).is_absolute() or PureWindowsPath(s).anchor or s.startswith("~"):
                raise PublishError(
                    f"{kind}[{i}].bundle is an absolute path; week.toml is published as given, "
                    "so bundle paths must be relative to it"
                )


def _segment(name: str, what: str) -> str:
    if not _SEGMENT.fullmatch(name):
        raise PublishError(f"{what} {name!r} is not a safe directory name")
    return name


def _count(meta: dict, key: str, where: str) -> int:
    v = meta.get(key)
    if not isinstance(v, int) or isinstance(v, bool) or v < 0:
        raise PublishError(f"judge bundle {where}: {key} must be a non-negative integer")
    return v


def _judge_entry(meta: dict, name: str, run_by_sha: dict[str, str]) -> dict:
    if meta.get("bundle_format") != JUDGE_FORMAT:
        raise PublishError(f"judge bundle {name}: bundle_format must be {JUDGE_FORMAT!r}")
    for side in ("a", "b"):
        if not isinstance(meta.get(side), str):
            raise PublishError(f"judge bundle {name}: {side} must be a bundle_sha256 string")
        if meta[side] not in run_by_sha:
            raise PublishError(
                f"judge bundle {name}: side {side} ({meta[side]!r}) is not in this week"
            )
    if meta["a"] == meta["b"]:
        raise PublishError(f"judge bundle {name}: a and b are the same run")
    counts = meta.get("counts")
    if not isinstance(counts, dict):
        raise PublishError(f"judge bundle {name}: counts must be an object")
    model = meta.get("judge_model")
    if not isinstance(model, str) or not model:
        raise PublishError(f"judge bundle {name}: judge_model must be a non-empty string")
    return {
        "a": run_by_sha[meta["a"]],
        "b": run_by_sha[meta["b"]],
        "wins": _count(counts, "a", name),
        "ties": _count(counts, "tie", name),
        "losses": _count(counts, "b", name),
        "errors": _count(meta, "call_errors", name),
        "judge_model": model,
        "bundle_sha256": meta["bundle_sha256"],
    }


def _published_copy(path: Path, sha: str, what: str) -> Path | None:
    """The already-published copy at ``path``, if any; refuses when its hash differs."""
    if not path.exists() and not path.is_symlink():
        return None
    meta = _read_json(path / "bundle.json", f"published {what}")
    if not isinstance(meta, dict) or meta.get("bundle_sha256") != sha:
        raise PublishError(
            f"{what} is already published with a different bundle_sha256; "
            "published bundles are immutable"
        )
    return path


def _scan(path: Path, strings, terms: list[str], username: str | None) -> None:
    text = path.read_bytes().decode("utf-8")
    hits = [(no, kind, m) for no, kind, m in scan_text(text, terms, username)]
    for s in strings:  # parsed values too, so escapes cannot hide a match
        hits.extend((0, kind, m) for _, kind, m in scan_text(s, terms, username))
    if hits:
        found = [Finding(path.name, no, kind, m) for no, kind, m in sorted(set(hits))]
        raise PublishError(f"staged {path.name}: scrub finding(s): {_findings_text(found)}")


def _stage_bundle(src: Path, dst: Path, sha: str, what: str, username, denylist, skip) -> None:
    shutil.copytree(src, dst, symlinks=True)
    meta = _verify(dst, f"copied {what}", username, denylist, skip)
    if meta.get("bundle_sha256") != sha:
        raise PublishError(f"copied {what} {dst.name}: bundle_sha256 changed while copying")


def publish(
    week_toml: Path,
    root: Path,
    *,
    username: str | None,
    denylist: Path | None,
    username_skip_output: bool = False,
) -> Path:
    """Publish one week into ``root/results/<week>`` and update ``results/picks.json``."""
    root = Path(root)
    results = root / "results"
    try:
        tiers = load_tiers(root / "tiers.toml")
        inp = load_week(Path(week_toml), tiers)
    except WeekInputError as e:
        raise PublishError(f"week input: {e}") from e
    _check_paths(inp.source_bytes)
    if denylist is not None and not Path(denylist).is_file():
        raise PublishError(f"denylist not found: {denylist}")
    terms = load_terms(denylist)
    vargs = (username, denylist, username_skip_output)

    final = results / inp.week
    staging = results / f".{inp.week}.staging"
    old = results / f".{inp.week}.old"
    for leftover in (staging, old):
        if leftover.exists() or leftover.is_symlink():
            raise PublishError(
                f"leftover results/{leftover.name} from an earlier publish; restore or remove it"
            )

    # Week state: new or correction, and what came before.
    docs = week_docs(root)
    prev = next((d for d in docs if d["week"] == inp.week), None)
    exists = final.exists() or final.is_symlink()
    if inp.correction is None and exists:
        raise PublishError(
            f"week {inp.week} is already published; add a [correction] table to re-publish it"
        )
    if inp.correction is not None and prev is None:
        raise PublishError(f"[correction] given, but week {inp.week} does not exist")
    later = [d["week"] for d in docs if d["week"] > inp.week]
    if inp.correction is None and later:
        raise PublishError(
            f"week {inp.week} is earlier than published later week(s) {later}; "
            "publish a new week instead"
        )
    if prev is not None and prev["rules"] != RULES:
        raise PublishError(
            f"week {inp.week} was published under other rules ({prev['rules']}); "
            "a correction must use the same rules"
        )
    earlier = [d for d in docs if d["week"] < inp.week]
    picks_prev = earlier[-1]["picks_after"] if earlier else {}
    unknown = sorted(t for t in picks_prev if t not in tiers)
    if unknown:
        raise PublishError(f"the previous week's picks name tiers not in tiers.toml: {unknown}")
    published: dict[str, str] = {}
    for d in docs:
        for t in d["tiers"]:
            for r in t["runs"]:
                published.setdefault(r["run_id"], r["bundle_sha256"])

    # Verify every bundle in place, then load the runs.
    runs: dict[str, tuple[RunView, RunEntry]] = {}
    sources: dict[str, Path] = {}
    for entry in inp.runs:
        _verify(entry.bundle, "run", *vargs)
        try:
            view = load_run(entry.bundle)
        except RunViewError as e:
            raise PublishError(f"run {entry.bundle.name}: {e}") from e
        rid = _segment(view.run_id, "run_id")
        if entry.bundle.name != rid:  # overrides name runs by directory name
            raise PublishError(
                f"run directory {entry.bundle.name!r} does not match its bundle.json run_id {rid!r}"
            )
        if rid in runs:  # defensive: load_week already refuses repeated directory names
            raise PublishError(f"duplicate run_id {rid}")
        want = tiers[entry.tier].envelope
        if view.envelope != want:
            raise PublishError(
                f"run {rid}: envelope {view.envelope!r} does not match tier {entry.tier} ({want!r})"
            )
        if rid in published and published[rid] != view.bundle_sha256:
            raise PublishError(
                f"run {rid} is already published with a different bundle_sha256; "
                "published bundles are immutable"
            )
        runs[rid] = (view, entry)
        sources[rid] = (
            _published_copy(final / "runs" / rid, view.bundle_sha256, f"run {rid}") or entry.bundle
        )
    run_by_sha = {view.bundle_sha256: rid for rid, (view, _) in runs.items()}
    judges: list[dict] = []
    judge_src: dict[str, tuple[Path, str]] = {}
    for path in inp.judges:
        meta = _verify(path, "judge bundle", *vargs)
        jid = _segment(path.name, "judge bundle directory name")
        if jid in judge_src:
            raise PublishError(f"duplicate judge bundle directory name {jid}")
        judges.append(_judge_entry(meta, jid, run_by_sha))
        sha = meta["bundle_sha256"]
        judge_src[jid] = (_published_copy(final / "judge" / jid, sha, f"judge {jid}") or path, sha)

    # Assess every tier with runs this week or a pick to carry forward.
    by_tier: dict[str, list[RunView]] = {}
    for view, entry in runs.values():
        by_tier.setdefault(entry.tier, []).append(view)
    assessed: dict[str, TierResult] = {}
    try:
        for tid in tiers:
            before = picks_prev.get(tid, {})
            if tid in by_tier or before:
                assessed[tid] = assess_tier(
                    inp.week, tid, by_tier.get(tid, []), before, inp.overrides, RULES
                )
    except VerdictError as e:
        raise PublishError(f"verdicts: {e}") from e

    if prev is None:
        revision, published_at, corrections = 1, inp.published_at, []
    else:
        revision, published_at = prev["revision"] + 1, prev["published_at"]
        corrections = [
            *prev["corrections"],
            {"revision": revision, "published_at": inp.published_at, "note": inp.correction},
        ]
    try:
        doc = build_week(
            inp,
            tiers,
            runs,
            assessed,
            judges,
            revision=revision,
            published_at=published_at,
            corrections=corrections,
            rules=RULES,
        )
    except ValueError as e:
        raise PublishError(f"week document: {e}") from e
    problems = validate_week(doc)
    if problems:
        raise PublishError(f"week document is invalid: {'; '.join(problems)}")
    if prev is not None and later and doc["picks_after"] != prev["picks_after"]:
        raise PublishError(
            f"this correction changes week {inp.week}'s picks, which later week(s) {later} "
            "depend on; publish the fix as a new week instead"
        )

    # Stage, re-verify, scan, swap.
    created = False
    try:
        results.mkdir(parents=True, exist_ok=True)
        staging.mkdir()
        created = True
        for rid in sorted(runs):
            view = runs[rid][0]
            dst = staging / "runs" / rid
            _stage_bundle(sources[rid], dst, view.bundle_sha256, "run", *vargs)
        for jid in sorted(judge_src):
            src, sha = judge_src[jid]
            _stage_bundle(src, staging / "judge" / jid, sha, "judge bundle", *vargs)
        (staging / "week.toml").write_bytes(inp.source_bytes)
        (staging / "week.json").write_bytes(_dumps(doc))
        toml_strings = list(iter_strings(tomllib.loads(inp.source_bytes.decode("utf-8"))))
        _scan(staging / "week.toml", toml_strings, terms, username)
        _scan(staging / "week.json", list(iter_strings(doc)), terms, username)

        try:
            if exists:
                os.rename(final, old)
            os.rename(staging, final)
        except BaseException:
            # Restore on any failure, even one between the two renames: the old week is only
            # ever in one of the two places.
            if exists and old.exists() and not final.exists():
                os.rename(old, final)
            raise
        if exists:
            shutil.rmtree(old, ignore_errors=True)
    except OSError as e:
        raise PublishError(f"cannot write results: {e}") from e
    finally:
        if created and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    try:
        write_picks(root)
    except (OSError, PublishError) as e:
        raise PublishError(
            f"week {inp.week} published; picks.json not updated — run `bakeoff site` "
            f"to regenerate (cause: {e})"
        ) from e
    return final
