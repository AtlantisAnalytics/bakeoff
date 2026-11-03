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

# Requests in flight at once (2026-10-27: one at a time, a 306-item pass took 2.5 h on a 3B model).
# Records, grading, the error budget and progress still go in item order.
QUALITY_CONCURRENCY = 8


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
        "items_total": len(suite.items),  # n is less when the run stopped part-way
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
    if suite.grader == "json_schema":
        ext = sum(1 for r in recs if r.get("extractable") is True)
        s["extractable_accuracy"] = ext / n if n else None
        s["extractable_ci95"] = list(wilson(ext, n)) if n else None
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
    """Run each suite's items, up to QUALITY_CONCURRENCY requests at once, recording them in
    suite and item order.

    An errored request is a fail (counted in `n`); so are truncated answers. One ErrorBudget is
    shared across all suites. On AbortRun, the exception gains a `partial` attribute holding the
    QualityResult for the items recorded so far (including the item that tripped the budget), so
    the caller can write an incomplete bundle, and is re-raised.
    """
    budget = ErrorBudget("quality")
    records: dict[str, list[dict]] = {}
    gate = asyncio.Semaphore(QUALITY_CONCURRENCY)

    async def ask(suite: Suite, item):
        async with gate:
            return await client.chat(item.messages, suite.max_tokens, temperature=0.0, seed=0)

    pending: list[asyncio.Task] = []
    try:
        for suite in suites:
            recs = records.setdefault(suite.key, [])
            total = len(suite.items)
            pending = [asyncio.create_task(ask(suite, item)) for item in suite.items]
            for item, task in zip(suite.items, pending, strict=True):
                r = await task
                if r.error is not None:
                    passed, score, reason = False, 0.0, "error"
                    extractable = False
                else:
                    try:
                        g = grade(suite.grader, r.content, item.expected, suite.grader_options)
                        passed, score, reason = g.passed, g.score, g.reason
                        extractable = g.extractable
                    except Exception as e:  # noqa: BLE001 - one bad item must not kill the run
                        passed, score, reason = False, 0.0, f"grader error: {type(e).__name__}"
                        extractable = False
                    if r.finish_reason == "length":
                        passed, score, reason = False, 0.0, "truncated"
                        extractable = False
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
                        "extractable": extractable,
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
    finally:  # an abort or a cancel sends nothing more
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    return _build(suites, records)
