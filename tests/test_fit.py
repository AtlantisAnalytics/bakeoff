import asyncio

import pytest
from fake_openai import FakeOpenAI

from bakeoff.client import EndpointClient
from bakeoff.config import FitConfig
from bakeoff.errors import AbortRun
from bakeoff.fit import calibrate, run_fit

STEPS = [4096, 8192, 16384, 32768, 65536]


def go(fake, cfg, runtime="llamacpp", server_context=None, sampler=None):
    async def main():
        async with EndpointClient(
            "http://fake/v1", "m", transport=fake.transport, backoff_s=()
        ) as c:
            return await run_fit(c, cfg, runtime, server_context, sampler)

    return asyncio.run(main())


def statuses(summary):
    return {k: v["status"] for k, v in summary["steps"].items()}


def test_calibrate_is_linear():
    async def main():
        async with EndpointClient("http://fake/v1", "m", transport=FakeOpenAI().transport) as c:
            return await calibrate(c)

    per_line, intercept = asyncio.run(main())
    assert 15 < per_line < 30
    assert intercept == pytest.approx(0, abs=40)


def test_passes_up_to_server_limit():
    fake = FakeOpenAI(context_limit=20000)
    records, s = go(fake, FitConfig(steps=STEPS), server_context=16384)
    assert s["max_context_passed"] == 16384
    assert s["limit_source"] == "reported"
    assert s["truncation_check"] == "measured"
    assert statuses(s) == {
        "4096": "pass",
        "8192": "pass",
        "16384": "pass",
        "32768": "skipped",
        "65536": "skipped",
    }
    assert {r["step"] for r in records} == {4096, 8192, 16384}
    assert all(r["accepted"] and r["within_5pct"] and r["retrieved"] for r in records)


def test_silent_truncation_detected():
    fake = FakeOpenAI(context_limit=9000, truncate_silently=True)
    records, s = go(fake, FitConfig(steps=STEPS))
    assert statuses(s) == {
        "4096": "pass",
        "8192": "pass",
        "16384": "fail",
        "32768": "not_run",
        "65536": "not_run",
    }
    assert s["max_context_passed"] == 8192
    assert s["limit_source"] == "none"
    bad = [r for r in records if r["step"] == 16384]
    assert bad and all(r["accepted"] and r["within_5pct"] is False for r in bad)


def test_rejected_overflow_is_fail_not_error():
    fake = FakeOpenAI(context_limit=9000)
    records, s = go(fake, FitConfig(steps=STEPS))
    assert statuses(s)["16384"] == "fail"
    assert s["max_context_passed"] == 8192
    bad = [r for r in records if r["step"] == 16384]
    assert all(r["accepted"] is False and "maximum context" in r["error"] for r in bad)


def test_unexpected_errors_abort_with_partial():
    import httpx

    class Unhappy(FakeOpenAI):
        def _chat(self, body):
            if len(body["messages"][0]["content"]) > 22000:
                return httpx.Response(422, text="bad request")
            return super()._chat(body)

    # 9 good requests, then 3 unexpected 4xx: 3/12 is over the 20% budget
    with pytest.raises(AbortRun) as ei:
        go(Unhappy(), FitConfig(steps=[3000, 4096, 6000, 8192, 16384]))
    records, summary = ei.value.partial
    assert len(records) == 12
    assert summary["steps"]["8192"]["status"] == "fail"
    assert summary["max_context_passed"] == 6000


def test_operator_cap_wins():
    fake = FakeOpenAI()
    _, s = go(fake, FitConfig(steps=[4096, 8192, 16384], max_context=8192), server_context=131072)
    assert s["limit_source"] == "operator"
    assert s["max_context_passed"] == 8192
    assert statuses(s)["16384"] == "skipped"


def test_no_usage_falls_back():
    fake = FakeOpenAI(support_stream_options=False)
    records, s = go(fake, FitConfig(steps=[4096, 8192]))
    assert s["truncation_check"] == "not_captured"
    assert s["max_context_passed"] == 8192
    assert all(r["within_5pct"] is None and r["retrieved"] for r in records)


def test_max_tokens_is_1024_for_fit_requests():
    fake = FakeOpenAI()
    go(fake, FitConfig(steps=[4096]))
    fit_bodies = [b for b in fake.requests if b["max_tokens"] != 1]
    assert len(fit_bodies) == 3
    assert all(b["max_tokens"] == 1024 for b in fit_bodies)


def test_small_step_is_skipped_with_reason():
    fake = FakeOpenAI()
    records, s = go(fake, FitConfig(steps=[1024, 4096]))
    assert s["steps"]["1024"]["status"] == "skipped"
    assert s["steps"]["1024"]["reason"]
    assert s["steps"]["4096"]["status"] == "pass"
    assert all(r["step"] != 1024 for r in records)


def test_steps_sorted_and_marked_and_kv():
    marks = []

    class S:
        def mark(self, name):
            marks.append(name)

    fake = FakeOpenAI(extras={"/metrics": '# HELP x\nvllm:gpu_cache_usage_perc{model="m"} 0.25\n'})
    _, s = go(fake, FitConfig(steps=[8192, 4096]), runtime="vllm", sampler=S())
    assert marks == ["fit:4096", "fit:8192"]
    assert s["steps"]["4096"]["kv_cache_usage"] == 0.25
    _, s2 = go(fake, FitConfig(steps=[4096]), runtime="ollama")
    assert s2["steps"]["4096"]["kv_cache_usage"] is None
