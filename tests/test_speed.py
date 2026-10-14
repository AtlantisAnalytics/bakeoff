import asyncio

import pytest
from fake_openai import FakeOpenAI, FakeReply

from bakeoff.client import EndpointClient
from bakeoff.config import SpeedConfig
from bakeoff.errors import AbortRun
from bakeoff.speed import SHAPES, run_speed

WORDS = "w " * 11


def reply(**kw):
    return lambda msgs: FakeReply(content=WORDS, completion_tokens=11, **kw)


def run_s(fake, cfg, runtime="vllm", sampler=None):
    async def go():
        async with EndpointClient(
            "http://h/v1", "m", transport=fake.transport, backoff_s=(0.0, 0.0)
        ) as c:
            return await run_speed(c, cfg, runtime, sampler)

    return asyncio.run(go())


def test_shapes():
    assert set(SHAPES) == {"short", "long"}
    short = SHAPES["short"][0]["content"]
    assert short.endswith(
        "\n\nWrite a numbered list of 60 practical tips for keeping these depots organised. "
        "Number every tip and keep going until you reach 60."
    )
    assert short.count("Record ") == 4
    long = SHAPES["long"][0]["content"]
    assert long.endswith(
        "\n\nSummarise the records above as a numbered list of at least 40 observations."
    )
    assert long.count("Record ") == 80


def test_cells_and_record_counts():
    fake = FakeOpenAI(responder=reply())
    recs, summ = run_s(fake, SpeedConfig(concurrency=[1, 2], repeats=3, warmup=1))
    assert len(recs) == 2 * (1 * 3 + 2 * 3)
    assert set(summ) == {"short@1", "short@2", "long@1", "long@2"}
    assert summ["short@2"]["decode_tps"]["n"] == 6
    assert summ["short@2"]["aggregate_tps"]["n"] == 3
    assert {r["round"] for r in recs} == {0, 1, 2}
    assert set(recs[0]) == {
        "shape", "concurrency", "round", "prompt_tokens", "completion_tokens",
        "usage_source", "ttft_s", "decode_s", "total_s", "ignore_eos", "error",
    }  # fmt: skip
    assert [r["shape"] for r in recs][:1] == ["short"] and recs[-1]["shape"] == "long"
    assert len(fake.requests) == 2 * (1 * 4 + 2 * 4)


def test_decode_rate_math():
    fake = FakeOpenAI(responder=reply(chunk_delay_s=0.01))
    _, summ = run_s(fake, SpeedConfig(concurrency=[1], repeats=3, warmup=0))
    cell = summ["short@1"]
    assert cell["decode_tps"]["median"] == pytest.approx(100, rel=0.3)
    assert cell["usage_source"] == "reported"
    assert cell["ttft_s"]["n"] == 3 and cell["prefill_tps"]["n"] == 3


def test_concurrent_requests_overlap():
    fake = FakeOpenAI(responder=reply(chunk_delay_s=0.01))
    _, summ = run_s(fake, SpeedConfig(concurrency=[4], repeats=2, warmup=0))
    cell = summ["short@4"]
    assert cell["aggregate_tps"]["median"] > 2 * cell["decode_tps"]["median"]


def test_ignore_eos_only_for_vllm_llamacpp():
    for runtime, expected in [("vllm", True), ("llamacpp", True), ("ollama", False)]:
        fake = FakeOpenAI(responder=reply())
        recs, summ = run_s(fake, SpeedConfig(concurrency=[1], repeats=1, warmup=1), runtime)
        assert all(("ignore_eos" in b) == expected for b in fake.requests)
        assert all(r["ignore_eos"] is expected for r in recs)
        assert summ["short@1"]["ignore_eos"] is expected
        assert all(b["max_tokens"] == 256 for b in fake.requests)


def test_estimated_when_no_usage():
    fake = FakeOpenAI(responder=lambda m: FakeReply(content=WORDS), support_stream_options=False)
    recs, summ = run_s(fake, SpeedConfig(concurrency=[1], repeats=2, warmup=0))
    assert summ["short@1"]["usage_source"] == "estimated"
    assert all(r["usage_source"] == "estimated" for r in recs)


def test_failed_request_excluded_from_stats():
    state = {"n": 0}

    def responder(msgs):
        state["n"] += 1
        if state["n"] == 2:
            return FakeReply(status=400, body="bad")
        return FakeReply(content=WORDS, completion_tokens=11)

    fake = FakeOpenAI(responder=responder)
    recs, summ = run_s(fake, SpeedConfig(concurrency=[1], repeats=4, warmup=0))
    bad = [r for r in recs if r["error"]]
    assert len(bad) == 1
    cell = summ["short@1"]
    assert cell["decode_tps"]["n"] == 3 and cell["ttft_s"]["n"] == 3
    assert cell["aggregate_tps"]["n"] == 3


def test_abort_carries_partial():
    fake = FakeOpenAI(responder=lambda m: FakeReply(status=400, body="bad"))
    with pytest.raises(AbortRun) as ei:
        run_s(fake, SpeedConfig(concurrency=[1], repeats=20, warmup=0))
    recs, summ = ei.value.partial
    assert len(recs) == 10 and summ["short@1"]["decode_tps"]["n"] == 0


def test_sampler_marked():
    class S:
        marks: list[str] = []

        def mark(self, name):
            self.marks.append(name)

    s = S()
    run_s(FakeOpenAI(responder=reply()), SpeedConfig(concurrency=[1], repeats=1, warmup=0), "x", s)
    assert s.marks == ["speed"]
