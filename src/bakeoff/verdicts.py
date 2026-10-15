"""Verdict rules v1: a pure function from one tier's loaded runs to verdicts and picks.

Contract (see the spec, "Verdict rules (rules v1)"):

* Eligibility (rule 1): ``ineligible_reasons``; an ineligible run gets no verdict and appears only
  in ``TierResult.ineligible``. Passing ``bakeoff verify`` is checked by publish, before this runs.
* A run "has a score" in a track when its bundle's ``quality.tracks[track]`` is not null. The
  comparison inputs for a track are that track's suites with non-null accuracy, sorted by key.
* Control run (rule 2): the eligible run with the pick's (weights, quant, runtime) and a score in
  the track; the lowest run_id if there are several. Without one, the pick is ``not_re-measured``
  and every candidate is ``watch`` with ``pick_not_remeasured``.
* Not comparable (rule 3): differing suite sets, suite hashes or item ids give ``watch`` with
  ``diff`` and ``diff_ci95`` null (rule 6 lists not_comparable under watch, even for a worse run).
* Reasons record what decided the verdict: ``ci_includes_zero`` / ``not_better`` for the score,
  and ``slower_than_ratio`` / ``less_context`` only when the CI's lower bound is above 0.
* No current pick (rule 7): the best eligible run is ``initial_pick``; the others are compared
  against it. Overrides there are refused, so the initial pick is always computed.
* Ranking (rules 7 and 8): higher track score, then higher short@1 decode, then lower run_id
  (Python ``str`` order, which equals UTF-8 byte order).
* Every output is built in sorted order, so results never depend on input order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bakeoff.resample import paired_diff_ci, verdict_seed
from bakeoff.runview import RunView, ineligible_reasons, model_key
from bakeoff.weekcfg import Override

RULES_V1 = {
    "version": 1,
    "min_context": 16384,
    "min_decode_tps": 10.0,
    "speed_ratio": 0.8,
    "context_cap": 32768,
    "resamples": 10000,
    "ci": 0.95,
}

OVERRIDABLE = ("replace", "hold", "watch")
_PICK_FIELDS = (
    "tier",
    "track",
    "label",
    "weights",
    "quant",
    "runtime",
    "since_week",
    "bundle_sha256",
    "run_id",
)


class VerdictError(ValueError):
    pass


@dataclass
class Verdict:
    computed: str
    final: str
    override_reason: str | None = None
    diff: float | None = None
    diff_ci95: list[float] | None = None
    reasons: list[str] = field(default_factory=list)
    outranked_by: str | None = None
    new_pick: bool = False


@dataclass
class TierResult:
    ineligible: dict[str, list[str]]
    pick_tracks: dict[str, list[str]]
    verdicts: dict[str, dict[str, Verdict]]
    pick_status: dict[str, str]
    picks_after: dict[str, dict]


def _has_score(run: RunView, track: str) -> bool:
    return run.tracks.get(track) is not None


def _track_suites(run: RunView, track: str) -> list[str]:
    keys = sorted(
        k
        for k, s in run.suites.items()
        if s.get("track") == track and s.get("accuracy") is not None
    )
    if not keys or any(not run.items.get(k) for k in keys):
        raise VerdictError(f"{run.run_id}: track {track} has a score but no per-item results")
    return keys


def _best(runs: list[RunView], track: str) -> RunView:
    """Higher track score, then higher short@1 decode, then lower run_id (rules 7 and 8)."""
    return min(runs, key=lambda r: (-r.tracks[track], -r.short_decode, r.run_id))


def _comparability(cand: RunView, ctrl: RunView, track: str) -> tuple[list[str], list[str]]:
    """Return (reasons, suite keys); reasons is empty when the pair is comparable."""
    ck, pk = _track_suites(cand, track), _track_suites(ctrl, track)
    reasons: list[str] = []
    if ck != pk:
        reasons.append("suite_set_mismatch")
    pset = set(pk)
    common = [k for k in ck if k in pset]
    if any(cand.suites[k].get("hash") != ctrl.suites[k].get("hash") for k in common):
        reasons.append("suite_hash_mismatch")
    if any(set(cand.items[k]) != set(ctrl.items[k]) for k in common):
        reasons.append("item_ids_mismatch")
    return reasons, ck


def _compare(
    week: str, tier: str, track: str, cand: RunView, ctrl: RunView, rules: dict
) -> Verdict:
    reasons, keys = _comparability(cand, ctrl, track)
    if reasons:
        return Verdict(computed="watch", final="watch", reasons=reasons)
    pairs = []
    for k in keys:
        ids = sorted(cand.items[k])
        pairs.append(([cand.items[k][i] for i in ids], [ctrl.items[k][i] for i in ids]))
    seed = verdict_seed(week, tier, track, cand.bundle_sha256, ctrl.bundle_sha256)
    diff, lo, hi = paired_diff_ci(pairs, rules["resamples"], rules["ci"], seed)
    if lo > 0:
        if cand.short_decode < rules["speed_ratio"] * ctrl.short_decode:
            reasons.append("slower_than_ratio")
        if cand.max_context < min(ctrl.max_context, rules["context_cap"]):
            reasons.append("less_context")
        verdict = "watch" if reasons else "replace"
    elif diff > 0:
        verdict, reasons = "watch", ["ci_includes_zero"]
    else:
        verdict, reasons = "hold", ["not_better"]
    return Verdict(computed=verdict, final=verdict, diff=diff, diff_ci95=[lo, hi], reasons=reasons)


def _new_pick(tier: str, track: str, run: RunView, week: str) -> dict:
    m = run.model
    return {
        "tier": tier,
        "track": track,
        "label": m["label"],
        "weights": m["weights"],
        "quant": m["quant"],
        "runtime": m["runtime"],
        "since_week": week,
        "bundle_sha256": run.bundle_sha256,
        "run_id": run.run_id,
    }


def _check_pick(tier: str, track: str, p: object) -> dict:
    if not isinstance(p, dict) or any(f not in p for f in _PICK_FIELDS):
        raise VerdictError(f"current pick for track {track} lacks fields {list(_PICK_FIELDS)}")
    if p["tier"] != tier or p["track"] != track:
        raise VerdictError(f"current pick filed under {tier} / {track} names another tier or track")
    return p


def assess_tier(
    week: str,
    tier: str,
    runs: list[RunView],
    picks_before: dict[str, dict],
    overrides: list[Override],
    rules: dict,
) -> TierResult:
    by_id: dict[str, RunView] = {}
    for r in runs:
        if r.run_id in by_id:
            raise VerdictError(f"duplicate run_id {r.run_id}")
        by_id[r.run_id] = r
    ids = sorted(by_id)

    ineligible: dict[str, list[str]] = {}
    eligible: list[RunView] = []
    for rid in ids:
        why = ineligible_reasons(by_id[rid], rules)
        if why:
            ineligible[rid] = why
        else:
            eligible.append(by_id[rid])

    tracks = sorted(set(picks_before) | {t for r in eligible for t in r.tracks if _has_score(r, t)})
    picks = {t: _check_pick(tier, t, picks_before[t]) for t in sorted(picks_before)}

    controls: dict[str, RunView | None] = {}
    for t, p in picks.items():
        key = (p["weights"], p["quant"], p["runtime"])
        matches = [r for r in eligible if model_key(r) == key and _has_score(r, t)]
        controls[t] = matches[0] if matches else None  # eligible is in run_id order

    # Overrides for this tier, keyed by (run_id, track); validated before anything is computed.
    ov_by: dict[tuple[str, str], Override] = {}
    for o in overrides:
        if o.tier != tier:
            continue
        where = f"override {o.run} / {o.track}"
        if (o.run, o.track) in ov_by:
            raise VerdictError(f"{where}: given twice")
        if o.verdict not in OVERRIDABLE:
            raise VerdictError(f"{where}: verdict must be one of {list(OVERRIDABLE)}")
        if o.run not in by_id:
            raise VerdictError(f"{where}: run is not in tier {tier}")
        if o.run in ineligible:
            raise VerdictError(f"{where}: run is ineligible")
        if not _has_score(by_id[o.run], o.track):
            raise VerdictError(f"{where}: run has no score in this track")
        if o.track not in picks:
            raise VerdictError(f"{where}: no current pick (the initial pick is always computed)")
        ctrl = controls[o.track]
        if ctrl is not None and ctrl.run_id == o.run:
            raise VerdictError(f"{where}: run is the pick's control run")
        ov_by[(o.run, o.track)] = o

    verdicts: dict[str, dict[str, Verdict]] = {r.run_id: {} for r in eligible}
    pick_tracks: dict[str, list[str]] = {}
    pick_status: dict[str, str] = {}
    picks_after: dict[str, dict] = {}

    for t in tracks:
        scored = [r for r in eligible if _has_score(r, t)]
        if t not in picks:
            pick_status[t] = "none"
            if not scored:
                continue
            first = _best(scored, t)
            verdicts[first.run_id][t] = Verdict(
                computed="initial_pick", final="initial_pick", new_pick=True
            )
            for r in scored:
                if r is not first:
                    verdicts[r.run_id][t] = _compare(week, tier, t, r, first, rules)
            picks_after[t] = _new_pick(tier, t, first, week)
            continue

        ctrl = controls[t]
        pick_status[t] = "not_re-measured" if ctrl is None else "re-measured"
        if ctrl is not None:
            pick_tracks.setdefault(ctrl.run_id, []).append(t)
        for r in scored:
            if r is ctrl:
                continue
            if ctrl is None:
                v = Verdict(computed="watch", final="watch", reasons=["pick_not_remeasured"])
            else:
                v = _compare(week, tier, t, r, ctrl, rules)
            o = ov_by.get((r.run_id, t))
            if o is not None:
                v.final, v.override_reason = o.verdict, o.reason
            verdicts[r.run_id][t] = v

        winners = [r for r in scored if r is not ctrl and verdicts[r.run_id][t].final == "replace"]
        if winners:
            best = _best(winners, t)
            for r in winners:
                if r is best:
                    verdicts[r.run_id][t].new_pick = True
                else:
                    verdicts[r.run_id][t].outranked_by = best.run_id
            picks_after[t] = _new_pick(tier, t, best, week)
        elif ctrl is not None:
            picks_after[t] = {
                **picks[t],
                "bundle_sha256": ctrl.bundle_sha256,
                "run_id": ctrl.run_id,
            }
        else:
            picks_after[t] = dict(picks[t])

    return TierResult(
        ineligible=ineligible,
        pick_tracks={rid: pick_tracks[rid] for rid in sorted(pick_tracks)},
        verdicts=verdicts,
        pick_status=pick_status,
        picks_after=picks_after,
    )
