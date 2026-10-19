"""The week document (schema ``bakeoff.week.v1``): built from computed verdicts, then validated.

``build_week`` is pure: it reads no clock and no environment, sorts everything it emits, and writes
floats exactly as the bundles and the verdict code produced them (rounding is for display only).

Judge input (``judges``): a list of already-mapped entries, one per judge bundle, each exactly::

    {"a": run_id, "b": run_id, "wins": int, "ties": int, "losses": int, "errors": int,
     "judge_model": str, "bundle_sha256": str}

``a`` and ``b`` are run_ids (the judge bundle's ``a``/``b`` bundle hashes mapped to this week's
runs); ``wins`` is the count for ``a``, ``losses`` the count for ``b``, ``errors`` the call errors.
Both runs must be in this week and in the same tier; the entry is filed under that tier.
Judge counts are informational and never reach a verdict.
"""

import json
from importlib import resources

from bakeoff.graders import validate_schema
from bakeoff.runview import RunView, model_key
from bakeoff.verdicts import TierResult, Verdict
from bakeoff.weekcfg import RunEntry, Tier, WeekInput

SCHEMA_ID = "bakeoff.week.v1"
_JUDGE_KEYS = ("a", "b", "wins", "ties", "losses", "errors", "judge_model", "bundle_sha256")
_SPEED_METRICS = ("ttft_s", "prefill_tps", "decode_tps", "aggregate_tps")


def load_week_schema() -> dict:
    text = (resources.files("bakeoff") / "schemas" / "week.v1.json").read_text(encoding="utf-8")
    return json.loads(text)


def validate_week(doc: object) -> list[str]:
    return validate_schema(doc, load_week_schema())


def _stat(v: object) -> dict:
    v = v if isinstance(v, dict) else {}
    return {k: v.get(k) for k in ("median", "min", "max", "n")}


def _speed(view: RunView) -> dict:
    out = {}
    for cell in sorted(view.speed):
        data = view.speed[cell]
        if not isinstance(data, dict):
            continue
        metrics = {m: _stat(data[m]) for m in _SPEED_METRICS if isinstance(data.get(m), dict)}
        if metrics:
            out[cell] = metrics
    return out


def _suite(s: dict) -> dict:
    out = {k: s.get(k) for k in ("key", "version", "hash", "n", "passed", "accuracy", "ci95")}
    for k in ("extractable_accuracy", "extractable_ci95"):
        if k in s:
            out[k] = s[k]
    return out


def _quality(view: RunView) -> dict:
    names = sorted(set(view.tracks) | {s.get("track") for s in view.suites.values()})
    return {
        "tracks": {
            t: {
                "score": view.tracks.get(t),
                "suites": [
                    _suite(view.suites[k])
                    for k in sorted(view.suites)
                    if view.suites[k].get("track") == t
                ],
            }
            for t in names
        }
    }


def _fit(fit: dict) -> dict:
    out = {
        "max_context_passed": fit.get("max_context_passed"),
        "server_context": fit.get("server_context"),
    }
    kv = fit.get("kv")
    if isinstance(kv, dict):  # bundles before KV capacity keep the two fields above, unchanged
        step = (fit.get("steps") or {}).get(str(out["max_context_passed"])) or {}
        out["kv_capacity_tokens"] = (kv.get("capacity_tokens") or {}).get("value")
        out["kv_cache_dtype"] = (kv.get("cache_dtype") or {}).get("value")
        out["kv_fraction_at_max"] = step.get("kv_fraction")
    return out


def _verdict(v: Verdict) -> dict:
    out = {
        "computed": v.computed,
        "final": v.final,
        "diff": v.diff,
        "diff_ci95": v.diff_ci95,
        "reasons": list(v.reasons),
        "new_pick": v.new_pick,
    }
    if v.override_reason is not None:
        out["override_reason"] = v.override_reason
    if v.outranked_by is not None:
        out["outranked_by"] = v.outranked_by
    return out


def _run(week: str, view: RunView, entry: RunEntry, res: TierResult | None) -> dict:
    model = {**view.model, "license": entry.license, "source_url": entry.source_url}
    if entry.license_url is not None:
        model["license_url"] = entry.license_url
    reasons = res.ineligible.get(view.run_id, []) if res else []
    verdicts = res.verdicts.get(view.run_id, {}) if res else {}
    pick_tracks = res.pick_tracks.get(view.run_id, []) if res else []
    return {
        "run_id": view.run_id,
        "bundle_sha256": view.bundle_sha256,
        "bundle_path": f"results/{week}/runs/{view.run_id}",
        "model": model,
        "role": "pick" if pick_tracks else "candidate",
        "pick_tracks": sorted(pick_tracks),
        "eligible": not reasons,
        "ineligible_reasons": list(reasons),
        "quality": _quality(view),
        "speed": _speed(view),
        "fit": _fit(view.fit),
        "memory": {} if view.size_vram is None else {"size_vram": view.size_vram},
        "verdicts": {t: _verdict(verdicts[t]) for t in sorted(verdicts)},
    }


def _judges(judges: list[dict], tier_of: dict[str, str]) -> dict[str, list[dict]]:
    by_tier: dict[str, list[dict]] = {}
    for j in judges:
        if not isinstance(j, dict) or set(j) != set(_JUDGE_KEYS):
            raise ValueError(f"judge entry must have exactly the keys {list(_JUDGE_KEYS)}")
        for side in ("a", "b"):
            if j[side] not in tier_of:
                raise ValueError(f"judge entry names run {j[side]!r}, which is not in this week")
        if tier_of[j["a"]] != tier_of[j["b"]]:
            raise ValueError(f"judge entry {j['a']} vs {j['b']} spans two tiers")
        by_tier.setdefault(tier_of[j["a"]], []).append({k: j[k] for k in _JUDGE_KEYS})
    for entries in by_tier.values():
        entries.sort(key=lambda j: (j["a"], j["b"], j["bundle_sha256"]))
    return by_tier


def quality_agreement(runs: dict[str, tuple[RunView, RunEntry]]) -> list[dict]:
    """How often the same weights, quant and engine agreed across tiers (one run per tier, its
    lowest run_id), per public suite with item records in every run, over their shared item ids."""
    groups: dict[tuple[str, str, str], dict[str, RunView]] = {}
    for rid in sorted(runs):
        view, entry = runs[rid]
        groups.setdefault(model_key(view), {}).setdefault(entry.tier, view)
    out = []
    for key in sorted(groups):
        views = sorted(groups[key].values(), key=lambda v: v.run_id)
        if len(views) < 2:
            continue
        suites = {}
        for sk in sorted(set.intersection(*(set(v.items) for v in views))):
            if any(v.suites[sk].get("split") != "public" for v in views):
                continue
            ids = set.intersection(*(set(v.items[sk]) for v in views))
            suites[sk] = {
                "n": len(ids),
                "same_pass": sum(len({v.items[sk][i] for v in views}) == 1 for i in ids),
                "same_output": sum(len({v.outputs[sk].get(i) for v in views}) == 1 for i in ids),
            }
        weights, quant, runtime = key
        out.append(
            {
                "weights": weights,
                "quant": quant,
                "runtime": runtime,
                "runs": [v.run_id for v in views],
                "suites": suites,
            }
        )
    return out


def build_week(
    inp: WeekInput,
    tiers: dict[str, Tier],
    runs: dict[str, tuple[RunView, RunEntry]],
    results: dict[str, TierResult],
    judges: list[dict],
    *,
    revision: int,
    published_at: str,
    corrections: list[dict],
    rules: dict,
) -> dict:
    """Build the week document. ``tiers`` is in tiers.toml order; every tier is listed."""
    tier_of = {rid: entry.tier for rid, (_, entry) in runs.items()}
    unknown = sorted({t for t in tier_of.values() if t not in tiers})
    if unknown:
        raise ValueError(f"runs name tiers that do not exist: {unknown}")
    judge_by_tier = _judges(judges, tier_of)
    harness = {(v.harness.get("version"), v.harness.get("git_commit")) for v, _ in runs.values()}
    out_tiers = []
    picks_after = {}
    for tid, tier in tiers.items():
        res = results.get(tid)
        out_tiers.append(
            {
                "id": tier.id,
                "name": tier.name,
                "envelope": tier.envelope,
                "pick_status": dict(sorted(res.pick_status.items())) if res else {},
                "runs": [
                    _run(inp.week, *runs[rid], res) for rid in sorted(runs) if tier_of[rid] == tid
                ],
                "judge": judge_by_tier.get(tid, []),
            }
        )
        if res:
            picks_after[tid] = {t: dict(res.picks_after[t]) for t in sorted(res.picks_after)}
    doc = {
        "schema": SCHEMA_ID,
        "week": inp.week,
        "revision": revision,
        "published_at": published_at,
        "corrections": [dict(c) for c in corrections],
        "rules": dict(rules),
        "harness": [
            {"version": ver, "git_commit": commit}
            for ver, commit in sorted(harness, key=lambda h: (str(h[0]), str(h[1])))
        ],
        "tiers": out_tiers,
        "picks_after": picks_after,
    }
    agreement = quality_agreement(runs)
    if agreement:  # absent rather than empty, so weeks without it re-derive byte for byte
        doc["quality_agreement"] = agreement
    return doc
