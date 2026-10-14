import asyncio

import pytest
from fake_openai import FakeOpenAI, FakeReply

from bakeoff.client import ChatResult, EndpointClient
from bakeoff.config import SpeedConfig
from bakeoff.errors import AbortRun
from bakeoff.speed import SHAPES, _cell_summary, _record, run_speed

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


def _res(t_start=0.0, t_first=0.5, t_last=1.5, prompt=100, comp=11, error=None, src="reported"):
    return ChatResult(
        prompt_tokens=prompt,
        completion_tokens=comp,
        usage_source=src,
        t_start=t_start,
        t_first=t_first,
        t_last=t_last,
        error=error,
    )


def _summ(rounds):
    items = [(_record("short", 1, i, r, False), r) for i, rnd in enumerate(rounds) for r in rnd]
    return _cell_summary(items, rounds, False)


def test_exact_arithmetic():
    cell = _summ([[_res()]])
    assert cell["ttft_s"]["median"] == pytest.approx(0.5)
    assert cell["prefill_tps"]["median"] == pytest.approx(200.0)
    assert cell["decode_tps"]["median"] == pytest.approx(10.0)  # (11-1)/1.0, not 11/1.0
    assert cell["aggregate_tps"]["median"] == pytest.approx(11 / 1.5)


def test_exact_aggregate_two_requests():
    a = _res(t_start=1.0, t_first=1.5, t_last=3.0, comp=20)
    b = _res(t_start=1.25, t_first=1.75, t_last=4.0, comp=30)
    cell = _summ([[a, b]])
    assert cell["aggregate_tps"]["median"] == pytest.approx(50 / (4.0 - 1.0))
    assert cell["aggregate_tps"]["n"] == 1


def test_exclusions():
    cell = _summ([[_res(t_first=0.0)]])  # ttft 0
    assert cell["ttft_s"]["n"] == 0 and cell["prefill_tps"]["n"] == 0
    assert cell["decode_tps"]["n"] == 1
    cell = _summ([[_res(comp=1)]])
    assert cell["decode_tps"]["n"] == 0 and cell["prefill_tps"]["n"] == 1
    cell = _summ([[_res(t_first=1.5, t_last=1.5)]])  # decode_s 0
    assert cell["decode_tps"]["n"] == 0 and cell["ttft_s"]["n"] == 1
    cell = _summ([[_res(prompt=None)]])
    assert cell["prefill_tps"]["n"] == 0 and cell["ttft_s"]["n"] == 1
    cell = _summ([[_res(error="boom")]])
    assert all(cell[k]["n"] == 0 for k in ("ttft_s", "prefill_tps", "decode_tps", "aggregate_tps"))
    assert cell["usage_source"] is None
    cell = _summ([[_res(comp=None)]])
    assert cell["decode_tps"]["n"] == 0 and cell["aggregate_tps"]["n"] == 0


def test_none_t_last_excluded_from_sum_and_span():
    ok = _res(t_start=0.0, t_first=0.5, t_last=2.0, comp=10)
    cut = _res(t_start=-5.0, t_first=None, t_last=None, comp=99)
    cell = _summ([[ok, cut]])
    assert cell["aggregate_tps"]["median"] == pytest.approx(10 / 2.0)
    assert cell["ttft_s"]["n"] == 1


def test_errored_excluded_from_aggregate_and_usage():
    ok = _res(t_start=0.0, t_last=2.0, comp=10)
    bad = _res(t_start=-1.0, t_last=9.0, comp=50, error="x", src="estimated")
    cell = _summ([[ok, bad]])
    assert cell["aggregate_tps"]["median"] == pytest.approx(5.0)
    assert cell["usage_source"] == "reported"


def test_estimated_has_decode_but_no_prefill():
    fake = FakeOpenAI(responder=lambda m: FakeReply(content=WORDS), support_stream_options=False)
    _, summ = run_s(fake, SpeedConfig(concurrency=[1], repeats=2, warmup=0))
    cell = summ["short@1"]
    assert cell["prefill_tps"]["n"] == 0
    assert cell["decode_tps"]["n"] > 0


def test_warmup_errors_not_recorded_or_budgeted():
    state = {"n": 0}

    def responder(msgs):
        state["n"] += 1
        if state["n"] <= 15:  # all warmup requests fail; would trip the budget if counted
            return FakeReply(status=400, body="bad")
        return FakeReply(content=WORDS, completion_tokens=11)

    fake = FakeOpenAI(responder=responder)
    recs, summ = run_s(fake, SpeedConfig(concurrency=[1], repeats=3, warmup=15))
    assert len(recs) == 6 and all(r["error"] is None for r in recs[:3])
    assert summ["short@1"]["decode_tps"]["n"] == 3
