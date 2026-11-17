import asyncio

import pytest
from fake_openai import FakeOpenAI, FakeReply, default_responder

from bakeoff.client import EndpointClient
from bakeoff.config import FitConfig
from bakeoff.errors import AbortRun
from bakeoff.fit import calibrate, kv_fraction, kv_summary, run_fit

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
    assert per_line == pytest.approx(18.38)
    assert intercept == pytest.approx(29.0)


def test_degenerate_calibration_is_none():
    def flat(messages):
        return FakeReply(prompt_tokens=100)

    async def main():
        fake = FakeOpenAI(responder=flat)
        async with EndpointClient("http://fake/v1", "m", transport=fake.transport) as c:
            return await calibrate(c)

    assert asyncio.run(main()) is None


def test_passes_up_to_server_limit():
    fake = FakeOpenAI(context_limit=20000)
    records, s = go(fake, FitConfig(steps=STEPS), server_context=16384)
    assert s["max_context_passed"] == 16384
    assert s["limit_source"] == "reported"
    assert s["truncation_check"] == "measured"
    assert statuses(s) == {
        "4096": "skipped",
        "8192": "pass",
        "16384": "pass",
        "32768": "skipped",
        "65536": "skipped",
    }
    assert {r["step"] for r in records} == {8192, 16384}
    assert all(r["accepted"] and r["within_5pct"] and r["retrieved"] for r in records)


def test_silent_truncation_detected():
    fake = FakeOpenAI(context_limit=9000, truncate_silently=True)
    records, s = go(fake, FitConfig(steps=STEPS))
    assert statuses(s) == {
        "4096": "skipped",
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
        go(Unhappy(), FitConfig(steps=[6000, 8192, 9000, 12288, 16384]))
    records, summary = ei.value.partial
    assert len(records) == 12
    assert summary["steps"]["12288"]["status"] == "error"
    assert summary["steps"]["16384"]["status"] == "not_run"
    assert summary["max_context_passed"] == 9000


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


def test_max_tokens_is_4096_for_fit_requests():
    fake = FakeOpenAI()
    go(fake, FitConfig(steps=[8192]))
    fit_bodies = [b for b in fake.requests if b["max_tokens"] != 1]
    assert len(fit_bodies) == 3
    assert all(b["max_tokens"] == 4096 for b in fit_bodies)


def test_long_reasoning_within_the_needle_budget_passes():
    def reply(messages, body):
        if body["max_tokens"] < 3000:  # needs ~3000 tokens of reasoning before it answers
            return FakeReply(content="thinking...", finish_reason="length")
        return default_responder(messages)

    class Reasons(FakeOpenAI):
        def _chat(self, body):
            self.responder = lambda messages: reply(messages, body)
            return super()._chat(body)

    _, s = go(Reasons(), FitConfig(steps=[8192, 16384]))
    assert statuses(s) == {"8192": "pass", "16384": "pass"}


def test_4096_step_has_no_room_for_the_answer_budget():
    records, s = go(FakeOpenAI(), FitConfig(steps=[4096, 8192]))
    assert s["steps"]["4096"] == {
        "status": "skipped",
        "reason": "step too small for the prompt overhead",
    }
    assert {r["step"] for r in records} == {8192}


def test_small_step_is_skipped_with_reason():
    fake = FakeOpenAI()
    records, s = go(fake, FitConfig(steps=[1024, 8192]))
    assert s["steps"]["1024"]["status"] == "skipped"
    assert s["steps"]["1024"]["reason"]
    assert s["steps"]["8192"]["status"] == "pass"
    assert all(r["step"] != 1024 for r in records)


CACHE = (
    'vllm:cache_config_info{block_size="16",cache_dtype="auto",engine="0",'
    'num_gpu_blocks="1000"} 1.0\n'
)


def test_steps_sorted_and_marked_and_kv():
    marks = []

    class S:
        def mark(self, name):
            marks.append(name)

    fake = FakeOpenAI(extras={"/metrics": "# HELP x\n" + CACHE})
    _, s = go(fake, FitConfig(steps=[16384, 8192]), runtime="vllm", sampler=S())
    assert marks == ["fit:calibrate", "fit:8192", "fit:16384"]
    assert s["kv"]["capacity_tokens"] == {"value": 16000, "source": "reported"}
    assert s["kv"]["block_size"] == {"value": 16, "source": "reported"}
    assert s["kv"]["cache_dtype"] == {"value": "auto", "source": "reported"}
    assert s["steps"]["8192"]["kv_fraction"] == 0.512
    assert all("kv_cache_usage" not in step for step in s["steps"].values())
    _, s2 = go(fake, FitConfig(steps=[8192]), runtime="ollama")
    assert s2["kv"]["capacity_tokens"]["source"] == "not_captured"
    assert s2["steps"]["8192"]["kv_fraction"] is None


def test_kv_fraction_math():
    assert kv_fraction(4096, 16, 1000) == 0.256
    assert kv_fraction(4097, 16, 1000) == 0.2570
    assert kv_fraction(16, 16, 3) == 0.3333


def test_kv_summary_unusable_blocks():
    for blocks in ("None", "0", "x"):
        kv = kv_summary({"num_gpu_blocks": blocks, "block_size": "16", "cache_dtype": "auto"})
        assert kv["capacity_tokens"]["source"] == "not_captured"
        assert kv["cache_dtype"] == {"value": "auto", "source": "reported"}
    assert all(f["source"] == "not_captured" for f in kv_summary(None).values())


def test_fit_abort_keeps_kv_shape():
    class Drops(FakeOpenAI):
        def _chat(self, body):
            if body["max_tokens"] == 4096:
                body = {k: v for k, v in body.items() if k != "stream_options"}
            return super()._chat(body)

    with pytest.raises(AbortRun) as ei:
        go(Drops(extras={"/metrics": CACHE}), FitConfig(steps=[8192]), runtime="vllm")
    _, s = ei.value.partial
    assert s["kv"]["capacity_tokens"]["source"] == "not_captured"


def _is_big(body):
    return len(body["messages"][0]["content"]) > 40000  # the 16k step and above


def test_503_at_step_is_error_and_aborts():
    import httpx

    class Flaky(FakeOpenAI):
        def _chat(self, body):
            if _is_big(body):
                return httpx.Response(503, text="unavailable")
            return super()._chat(body)

    with pytest.raises(AbortRun) as ei:
        go(Flaky(), FitConfig(steps=STEPS))
    records, s = ei.value.partial
    assert "step 16384 error" in ei.value.reason
    assert s["incomplete"] is True
    assert statuses(s) == {
        "4096": "skipped",
        "8192": "pass",
        "16384": "error",
        "32768": "not_run",
        "65536": "not_run",
    }
    assert s["steps"]["16384"]["reason"] == "unavailable"
    assert s["max_context_passed"] == 8192
    assert len([r for r in records if r["step"] == 16384]) == 3


def test_connection_error_is_error_and_aborts():
    import httpx

    class Broken(FakeOpenAI):
        async def _handle(self, request):
            if request.url.path.endswith("/chat/completions"):
                import json

                if _is_big(json.loads(request.content)):
                    raise httpx.ConnectError("refused")
            return await super()._handle(request)

    with pytest.raises(AbortRun) as ei:
        go(Broken(), FitConfig(steps=STEPS))
    _, s = ei.value.partial
    assert s["steps"]["16384"]["status"] == "error"
    assert "transport error" in s["steps"]["16384"]["reason"]
    assert s["incomplete"] is True


def test_context_400_stays_fail_without_abort():
    _, s = go(FakeOpenAI(context_limit=9000), FitConfig(steps=STEPS))
    assert s["steps"]["16384"]["status"] == "fail"
    assert s["incomplete"] is False


def test_usage_missing_after_calibration_is_error():
    class Drops(FakeOpenAI):
        def _chat(self, body):
            if body["max_tokens"] == 4096:
                body = {k: v for k, v in body.items() if k != "stream_options"}
            return super()._chat(body)

    with pytest.raises(AbortRun) as ei:
        go(Drops(), FitConfig(steps=[8192, 16384]))
    records, s = ei.value.partial
    assert s["steps"]["8192"] == {"status": "error", "reason": "usage missing"}
    assert s["steps"]["16384"]["status"] == "not_run"
    assert all(r["within_5pct"] is None for r in records)


def test_fit_records_carry_finish_reason():
    records, _ = go(FakeOpenAI(), FitConfig(steps=[8192]))
    assert {r["finish_reason"] for r in records} == {"stop"}


def test_length_truncated_without_key_is_error_not_fail():
    def reply(messages):
        text = "\n".join(str(m.get("content", "")) for m in messages)
        if len(text) > 20000:  # the 16k step: a reasoning model out of tokens mid-thought
            return FakeReply(content="thinking...", finish_reason="length")
        return default_responder(messages)

    with pytest.raises(AbortRun) as ei:
        go(FakeOpenAI(responder=reply), FitConfig(steps=[8192, 16384, 32768]))
    _, s = ei.value.partial
    assert s["steps"]["16384"]["status"] == "error"
    assert s["steps"]["16384"]["reason"] == "truncated output"
    assert s["steps"]["32768"]["status"] == "not_run"
    assert s["max_context_passed"] == 8192


def test_length_but_key_found_is_a_pass():
    def reply(messages):
        r = default_responder(messages)
        return FakeReply(content=r.content, finish_reason="length")

    _, s = go(FakeOpenAI(responder=reply), FitConfig(steps=[8192]))
    assert s["steps"]["8192"]["status"] == "pass"


def test_calibrate_runs_under_its_own_mark_and_probe_is_recorded():
    marks = []

    class S:
        def mark(self, name):
            marks.append(name)

    _, s = go(FakeOpenAI(), FitConfig(steps=[8192]), sampler=S(), server_context=9999)
    assert marks[0] == "fit:calibrate" and marks[1] == "fit:8192"

    async def main():
        async with EndpointClient("http://fake/v1", "m", transport=FakeOpenAI().transport) as c:
            return await run_fit(c, FitConfig(steps=[8192]), "ollama", 8192, None, "pre_fit")

    _, s = asyncio.run(main())
    assert s["server_context_probe"] == "pre_fit" and s["server_context"] == 8192


def test_cancel_during_cache_config_keeps_partial():
    class CancelMetrics(FakeOpenAI):
        async def _handle(self, request):
            if request.url.path == "/metrics":
                raise asyncio.CancelledError
            return await super()._handle(request)

    with pytest.raises(asyncio.CancelledError) as ei:
        go(CancelMetrics(), FitConfig(steps=[8192]), runtime="vllm")
    records, s = ei.value.partial
    assert records and s["steps"]["8192"]["status"] == "pass"
    assert s["incomplete"] is True
    assert s["kv"]["capacity_tokens"]["source"] == "not_captured"
