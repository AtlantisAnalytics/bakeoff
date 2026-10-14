"""Speed pass: TTFT, prefill, decode and aggregate throughput per (shape x concurrency) cell."""

import asyncio

from bakeoff.client import ChatResult, EndpointClient
from bakeoff.config import SpeedConfig
from bakeoff.errors import AbortRun, ErrorBudget
from bakeoff.needle import build_haystack
from bakeoff.sampler import Sampler
from bakeoff.stats import summarize

MAX_TOKENS = 256
IGNORE_EOS_RUNTIMES = ("vllm", "llamacpp")

_SHORT_INSTRUCTION = (
    "Write a numbered list of 60 practical tips for keeping these depots organised. "
    "Number every tip and keep going until you reach 60."
)
_LONG_INSTRUCTION = "Summarise the records above as a numbered list of at least 40 observations."


def _messages(n_lines: int, seed: int, instruction: str) -> list[dict]:
    text = "\n".join(build_haystack(n_lines, seed)) + "\n\n" + instruction
    return [{"role": "user", "content": text}]


SHAPES: dict[str, list[dict]] = {
    "short": _messages(4, 1, _SHORT_INSTRUCTION),
    "long": _messages(80, 2, _LONG_INSTRUCTION),
}


def _decode_s(r: ChatResult) -> float | None:
    if r.t_first is None or r.t_last is None:
        return None
    return r.t_last - r.t_first


def _record(shape: str, c: int, rnd: int, r: ChatResult, ignore_eos: bool) -> dict:
    return {
        "shape": shape,
        "concurrency": c,
        "round": rnd,
        "prompt_tokens": r.prompt_tokens,
        "completion_tokens": r.completion_tokens,
        "usage_source": r.usage_source,
        "ttft_s": r.ttft_s,
        "decode_s": _decode_s(r),
        "total_s": r.total_s,
        "ignore_eos": ignore_eos,
        "error": r.error,
    }


def _cell_summary(
    items: list[tuple[dict, ChatResult]], rounds: list[list[ChatResult]], ignore_eos: bool
) -> dict:
    ok = [(rec, r) for rec, r in items if rec["error"] is None]
    ttft = [rec["ttft_s"] for rec, _ in ok if rec["ttft_s"] is not None and rec["ttft_s"] > 0]
    prefill = [
        rec["prompt_tokens"] / rec["ttft_s"]
        for rec, _ in ok
        if rec["prompt_tokens"] is not None and rec["ttft_s"] is not None and rec["ttft_s"] > 0
    ]
    decode = [
        (rec["completion_tokens"] - 1) / rec["decode_s"]
        for rec, _ in ok
        if rec["completion_tokens"] is not None
        and rec["completion_tokens"] >= 2
        and rec["decode_s"] is not None
        and rec["decode_s"] > 0
    ]
    aggregate = []
    for rnd in rounds:
        good = [
            r
            for r in rnd
            if r.error is None and r.t_last is not None and r.completion_tokens is not None
        ]
        if not good:
            continue
        span = max(r.t_last for r in good) - min(r.t_start for r in good)
        if span <= 0:
            continue
        aggregate.append(sum(r.completion_tokens for r in good) / span)
    usage_source = None
    if ok:
        estimated = any(rec["usage_source"] == "estimated" for rec, _ in ok)
        usage_source = "estimated" if estimated else "reported"
    return {
        "ttft_s": summarize(ttft),
        "prefill_tps": summarize(prefill),
        "decode_tps": summarize(decode),
        "aggregate_tps": summarize(aggregate),
        "usage_source": usage_source,
        "ignore_eos": ignore_eos,
    }


async def run_speed(
    client: EndpointClient,
    cfg: SpeedConfig,
    runtime: str,
    sampler: Sampler | None = None,
) -> tuple[list[dict], dict]:
    """Run the speed pass. On AbortRun the exception gains `partial = (records, summary)`."""
    if sampler is not None:
        sampler.mark("speed")
    ignore_eos = runtime in IGNORE_EOS_RUNTIMES
    extra = {"ignore_eos": True} if ignore_eos else None
    budget = ErrorBudget("speed")
    records: list[dict] = []
    summary: dict = {}

    async def fire(messages: list[dict], c: int) -> list[ChatResult]:
        return list(
            await asyncio.gather(
                *(
                    client.chat(messages, MAX_TOKENS, temperature=0.0, seed=0, extra=extra)
                    for _ in range(c)
                )
            )
        )

    for shape, messages in SHAPES.items():
        for c in cfg.concurrency:
            for _ in range(cfg.warmup):
                await fire(messages, c)
            items: list[tuple[dict, ChatResult]] = []
            rounds: list[list[ChatResult]] = []
            key = f"{shape}@{c}"
            for rnd in range(cfg.repeats):
                results = await fire(messages, c)
                rounds.append(results)
                tripped: AbortRun | None = None
                for r in results:
                    rec = _record(shape, c, rnd, r, ignore_eos)
                    records.append(rec)
                    items.append((rec, r))
                    if tripped is None:
                        try:
                            budget.record(r.error is None)
                        except AbortRun as exc:
                            tripped = exc
                if tripped is not None:
                    summary[key] = _cell_summary(items, rounds, ignore_eos)
                    tripped.partial = (records, summary)
                    raise tripped
            summary[key] = _cell_summary(items, rounds, ignore_eos)
    return records, summary
