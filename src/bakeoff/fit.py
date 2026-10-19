"""Fit pass: the largest context that really works (accepted, untruncated, retrievable)."""

import asyncio
import math

from bakeoff.client import ChatResult, EndpointClient
from bakeoff.config import FitConfig
from bakeoff.env import field
from bakeoff.errors import AbortRun, ErrorBudget
from bakeoff.needle import needle_messages
from bakeoff.sampler import Sampler

DEPTHS = (0.1, 0.5, 0.9)
MAX_TOKENS = 2048  # reasoning models burn tight output budgets
HEADROOM = 2304  # the 2048 output tokens plus a margin; the intercept covers the question
DEFAULT_TOKENS_PER_LINE = 25.0
DEFAULT_INTERCEPT = 50.0
TOLERANCE = 0.05
CONTEXT_ERROR_MARKERS = (
    "context length",
    "maximum context",
    "context_length",
    "too long",
    "exceeds",
)


async def calibrate(client: EndpointClient) -> tuple[float, float] | None:
    """Return (tokens_per_line, intercept) from two probe prompts, or None without usage."""
    counts = []
    for n_lines in (50, 200):
        messages, _ = needle_messages(n_lines, 0.5, 11)
        r = await client.chat(messages, max_tokens=1)
        if r.error is not None or r.prompt_tokens is None:
            return None
        counts.append(r.prompt_tokens)
    per_line = (counts[1] - counts[0]) / 150
    if per_line <= 0:
        return None
    return per_line, counts[0] - 50 * per_line


def _expected_failure(r: ChatResult) -> bool:
    """A 4xx that says the prompt is too big: the answer we are looking for, not an error."""
    if r.error is None or r.status is None or not 400 <= r.status < 500:
        return False
    text = r.error.lower()
    return any(marker in text for marker in CONTEXT_ERROR_MARKERS)


def _positive_int(value: str | None) -> int | None:
    try:
        n = int(value) if value is not None else None
    except ValueError:
        return None
    return n if n is not None and n > 0 else None


def kv_summary(labels: dict[str, str] | None) -> dict:
    """KV cache capacity from vLLM's cache_config_info labels; not_captured when unusable."""
    labels = labels or {}
    blocks = _positive_int(labels.get("num_gpu_blocks"))
    size = _positive_int(labels.get("block_size"))
    return {
        "capacity_tokens": field(blocks * size if blocks and size else None, "reported"),
        "block_size": field(size, "reported"),
        "cache_dtype": field(labels.get("cache_dtype") or None, "reported"),
    }


def kv_fraction(step: int, block_size: int, num_gpu_blocks: int) -> float:
    """Share of the KV pool one request of `step` tokens needs (full attention on every layer)."""
    return round(math.ceil(step / block_size) / num_gpu_blocks, 4)


def _summary(
    server_context: int | None,
    probe: str | None,
    source: str,
    calibrated: bool,
    steps: dict[str, dict],
    incomplete: bool,
    kv: dict,
) -> dict:
    passed = [int(s) for s, v in steps.items() if v["status"] == "pass"]
    return {
        "max_context_passed": max(passed) if passed else None,
        "server_context": server_context,
        "server_context_probe": probe,
        "limit_source": source,
        "truncation_check": "measured" if calibrated else "not_captured",
        "incomplete": incomplete,
        "steps": steps,
        "kv": kv,
    }


async def run_fit(
    client: EndpointClient,
    cfg: FitConfig,
    runtime: str,
    server_context: int | None,
    sampler: Sampler | None = None,
    probe: str | None = None,
) -> tuple[list[dict], dict]:
    """Run the fit pass. On AbortRun the exception gains `partial = (records, summary)`.

    `probe` records which runtime probe produced `server_context` (it goes in the summary).
    """
    if cfg.max_context > 0:
        limit, source = cfg.max_context, "operator"
    elif server_context:
        limit, source = server_context, "reported"
    else:
        limit, source = None, "none"
    if sampler is not None:
        sampler.mark("fit:calibrate")
    cal = await calibrate(client)
    per_line, intercept = cal if cal else (DEFAULT_TOKENS_PER_LINE, DEFAULT_INTERCEPT)
    budget = ErrorBudget("fit")
    records: list[dict] = []
    steps: dict[str, dict] = {}
    kv = kv_summary(None)

    def summary(incomplete: bool = False) -> dict:
        return _summary(server_context, probe, source, cal is not None, steps, incomplete, kv)

    ordered = sorted(cfg.steps)
    failed = False
    try:
        for idx, step in enumerate(ordered):
            key = str(step)
            if failed:
                steps[key] = {"status": "not_run"}
                continue
            if limit is not None and step > limit:
                steps[key] = {
                    "status": "skipped",
                    "reason": "above context limit",
                }
                continue
            lines = math.floor((step - intercept - HEADROOM) / per_line)
            if lines < 1:
                steps[key] = {
                    "status": "skipped",
                    "reason": "step too small for the prompt overhead",
                }
                continue
            if sampler is not None:
                sampler.mark(f"fit:{step}")
            expected = round(intercept + lines * per_line) if cal else None
            step_ok = True
            step_error: str | None = None
            tripped: AbortRun | None = None
            for depth in DEPTHS:
                messages, needle = needle_messages(lines, depth, seed=step)
                r = await client.chat(messages, MAX_TOKENS, temperature=0.0, seed=0)
                accepted = r.error is None
                within = None
                if cal and accepted:
                    if r.prompt_tokens is None:
                        if step_error is None:
                            step_error = "usage missing"
                    else:
                        within = abs(r.prompt_tokens - expected) / expected <= TOLERANCE
                retrieved = accepted and needle in r.content
                records.append(
                    {
                        "step": step,
                        "depth": depth,
                        "lines": lines,
                        "expected_prompt_tokens": expected,
                        "prompt_tokens": r.prompt_tokens,
                        "accepted": accepted,
                        "within_5pct": within,
                        "retrieved": retrieved,
                        "finish_reason": r.finish_reason,
                        "ttft_s": r.ttft_s,
                        "error": r.error,
                    }
                )
                if not accepted and not _expected_failure(r) and step_error is None:
                    step_error = (r.error or "unknown error")[:200]
                if (
                    accepted
                    and not retrieved
                    and r.finish_reason == "length"
                    and step_error is None
                ):
                    step_error = "truncated output"  # inconclusive, never a fail
                if not (accepted and within is not False and retrieved):
                    step_ok = False
                try:
                    budget.record(accepted or _expected_failure(r))
                except AbortRun as exc:
                    tripped = tripped or exc
            if step_error is not None:
                steps[key] = {"status": "error", "reason": step_error}
                for later in ordered[idx + 1 :]:
                    steps[str(later)] = {"status": "not_run"}
                exc = tripped or AbortRun(f"fit: step {step} error: {step_error}")
                exc.partial = (records, summary(True))
                raise exc
            steps[key] = {"status": "pass" if step_ok else "fail", "kv_fraction": None}
            if not step_ok:
                failed = True
        if runtime == "vllm":  # inside the try: an interrupt here still keeps the steps
            labels = await client.cache_config()
            kv = kv_summary(labels)
            blocks = _positive_int((labels or {}).get("num_gpu_blocks"))
            size = kv["block_size"]["value"]
            if blocks and size:
                for key, step in steps.items():
                    if "kv_fraction" in step:
                        step["kv_fraction"] = kv_fraction(int(key), size, blocks)
    except (asyncio.CancelledError, KeyboardInterrupt) as exc:
        exc.partial = (records, summary(True))  # type: ignore[attr-defined]
        raise
    return records, summary()
