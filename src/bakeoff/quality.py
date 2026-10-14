"""Quality pass: run every suite item once at temperature 0 and grade the output."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field

from bakeoff.client import EndpointClient
from bakeoff.errors import AbortRun, ErrorBudget
from bakeoff.graders import grade
from bakeoff.hashing import canonical_json, sha256_hex
from bakeoff.stats import wilson
from bakeoff.suites import Suite


@dataclass
class QualityResult:
    records: dict[str, list[dict]] = field(default_factory=dict)
    suites: list[dict] = field(default_factory=list)
    tracks: dict[str, float | None] = field(default_factory=dict)


def _summary(suite: Suite, recs: list[dict]) -> dict:
    n = len(recs)
    errors = sum(1 for r in recs if r["error"] is not None)
    truncated = sum(1 for r in recs if r["reason"] == "truncated")
    s = {
        "key": suite.key,
        "track": suite.track,
        "name": suite.name,
        "version": suite.version,
        "split": suite.split,
        "hash": suite.hash,
        "n": n,
        "passed": None,
        "errors": errors,
        "truncated": truncated,
        "accuracy": None,
        "ci95": None,
    }
    if suite.grader != "judge":
        passed = sum(1 for r in recs if r["pass"] is True)
        s["passed"] = passed
        if n:
            s["accuracy"] = passed / n
            s["ci95"] = list(wilson(passed, n))
    return s


def _build(suites: list[Suite], records: dict[str, list[dict]]) -> QualityResult:
    summaries = [_summary(s, records[s.key]) for s in suites if s.key in records]
    tracks: dict[str, float | None] = {}
    for s in summaries:
        tracks.setdefault(s["track"], None)
    for track in tracks:
        accs = [
            s["accuracy"] for s in summaries if s["track"] == track and s["accuracy"] is not None
        ]
        tracks[track] = sum(accs) / len(accs) if accs else None
    return QualityResult(records=records, suites=summaries, tracks=tracks)


async def run_quality(
    client: EndpointClient,
    suites: list[Suite],
    on_progress: Callable[[str, int, int], None] | None = None,
) -> QualityResult:
    """Run each suite's items sequentially, in suite order.

    An errored request is a fail (counted in `n`); so are truncated answers. One ErrorBudget is
    shared across all suites. On AbortRun, the exception gains a `partial` attribute holding the
    QualityResult for the items recorded so far (including the item that tripped the budget), so
    the caller can write an incomplete bundle, and is re-raised.
    """
    budget = ErrorBudget("quality")
    records: dict[str, list[dict]] = {}
    try:
        for suite in suites:
            recs = records.setdefault(suite.key, [])
            total = len(suite.items)
            for item in suite.items:
                r = await client.chat(item.messages, suite.max_tokens, temperature=0.0, seed=0)
                if r.error is not None:
                    passed, score, reason = False, 0.0, "error"
                else:
                    g = grade(suite.grader, r.content, item.expected, suite.grader_options)
                    passed, score, reason = g.passed, g.score, g.reason
                    if r.finish_reason == "length":
                        passed, score, reason = False, 0.0, "truncated"
                recs.append(
                    {
                        "id": item.id,
                        "prompt_sha256": sha256_hex(canonical_json(item.messages)),
                        "output": r.content,
                        "reasoning_tokens": r.reasoning_tokens,
                        "finish_reason": r.finish_reason,
                        "latency_s": r.total_s,
                        "pass": passed,
                        "score": score,
                        "reason": reason,
                        "error": r.error,
                    }
                )
                try:
                    budget.record(r.error is None)
                except AbortRun as exc:
                    exc.partial = _build(suites, records)
                    raise
                if on_progress:
                    on_progress(suite.key, len(recs), total)
    except (asyncio.CancelledError, KeyboardInterrupt) as exc:
        exc.partial = _build(suites, records)  # type: ignore[attr-defined]
        raise
    return _build(suites, records)
