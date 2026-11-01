import asyncio
import json
from dataclasses import replace

import pytest
from fake_openai import FakeOpenAI, FakeReply

from bakeoff.client import EndpointClient
from bakeoff.errors import AbortRun
from bakeoff.quality import run_quality
from bakeoff.stats import wilson
from bakeoff.suites import Item, Suite


def make_suite(name="s", track="general", grader="exact", n=4, max_tokens=16):
    items = [
        Item(f"i{k}", [{"role": "user", "content": f"PROMPT-{name}-{k}"}], "OK-42")
        for k in range(n)
    ]
    return Suite(
        path=None,
        track=track,
        name=name,
        version=1,
        split="public",
        license="CC0-1.0",
        source={},
        grader=grader,
        grader_options={},
        max_tokens=max_tokens,
        system="",
        items=items,
        hash=f"hash-{name}",
    )


def run_q(fake, suites, **kw):
    async def go():
        async with EndpointClient(
            "http://h/v1", "m", transport=fake.transport, backoff_s=(0.0, 0.0)
        ) as c:
            return await run_quality(c, suites, **kw)

    return asyncio.run(go())


def reply_for(fn):
    return lambda msgs: fn(msgs[-1]["content"])


def test_scores_and_ci():
    fake = FakeOpenAI(
        responder=reply_for(lambda p: FakeReply("wrong" if p.endswith("-3") else "OK-42"))
    )
    res = run_q(fake, [make_suite()])
    (s,) = res.suites
    assert s["accuracy"] == 0.75 and s["passed"] == 3 and s["n"] == 4
    assert s["ci95"] == list(wilson(3, 4))
    assert set(s) == {
        "key", "track", "name", "version", "split", "hash",
        "n", "items_total", "passed", "errors", "truncated", "accuracy", "ci95",
    }  # fmt: skip
    assert s["key"] == "general/s" and s["hash"] == "hash-s"
    rec = res.records["general/s"][0]
    assert set(rec) == {
        "id", "prompt_sha256", "output", "reasoning_tokens", "finish_reason",
        "latency_s", "pass", "score", "reason", "error", "extractable",
    }  # fmt: skip
    assert rec["pass"] is True and rec["error"] is None
    assert fake.requests[0]["temperature"] == 0.0 and fake.requests[0]["seed"] == 0
    assert fake.requests[0]["max_tokens"] == 16


def test_truncated_counts_as_fail():
    fake = FakeOpenAI(responder=lambda m: FakeReply("OK-42", finish_reason="length"))
    res = run_q(fake, [make_suite(n=2)])
    (s,) = res.suites
    assert s["truncated"] == 2 and s["passed"] == 0 and s["accuracy"] == 0.0 and s["errors"] == 0
    rec = res.records["general/s"][0]
    assert rec["pass"] is False and rec["reason"] == "truncated"


def test_reasoning_not_graded():
    fake = FakeOpenAI(
        responder=lambda m: FakeReply("wrong", reasoning="OK-42", completion_tokens=5)
    )
    res = run_q(fake, [make_suite(n=1)])
    rec = res.records["general/s"][0]
    assert rec["pass"] is False and rec["output"] == "wrong"
    fake2 = FakeOpenAI(
        responder=lambda m: FakeReply(
            "OK-42",
            chunks=[
                {"choices": [{"index": 0, "delta": {"content": "OK-42"}, "finish_reason": "stop"}]},
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 9,
                        "completion_tokens_details": {"reasoning_tokens": 7},
                    },
                },
            ],
        )
    )
    rec2 = run_q(fake2, [make_suite(n=1)]).records["general/s"][0]
    assert rec2["reasoning_tokens"] == 7 and rec2["pass"] is True


def test_inline_think_stripped():
    fake = FakeOpenAI(responder=lambda m: FakeReply("<think>x</think>OK-42"))
    res = run_q(fake, [make_suite(n=1)])
    assert res.records["general/s"][0]["pass"] is True


def test_judge_suite_deferred():
    fake = FakeOpenAI(responder=lambda m: FakeReply("OK-42"))
    res = run_q(fake, [make_suite("j", grader="judge", n=2), make_suite("e", n=2)])
    j, e = res.suites
    assert j["accuracy"] is None and j["ci95"] is None and j["passed"] is None and j["n"] == 2
    rec = res.records["general/j"][0]
    assert rec["pass"] is None and rec["score"] == 0.0 and rec["reason"]
    assert res.tracks["general"] == 1.0


def test_track_all_judge_is_none():
    fake = FakeOpenAI()
    res = run_q(fake, [make_suite("j", track="t", grader="judge", n=1)])
    assert res.tracks == {"t": None}


def test_errors_separate_and_abort():
    fake = FakeOpenAI(fail_first=1000)
    seen = []
    with pytest.raises(AbortRun) as ei:
        run_q(
            fake,
            [make_suite(n=30)],
            on_progress=lambda k, d, t: seen.append((k, d, t)),
        )
    assert len(fake.requests) >= 10
    partial = ei.value.partial
    recs = partial.records["general/s"]
    assert len(recs) == 10  # the aborting item is recorded
    assert all(r["error"] and r["pass"] is False and r["reason"] == "error" for r in recs)
    assert partial.suites[0]["errors"] == 10 and partial.suites[0]["n"] == 10
    assert partial.suites[0]["items_total"] == 30  # a partial summary says how much it covers


def test_errors_below_budget_count_as_fails():
    def r(p):
        return FakeReply("x", status=400, body="bad") if p.endswith("-0") else FakeReply("OK-42")

    res = run_q(FakeOpenAI(responder=reply_for(r)), [make_suite(n=10)])
    (s,) = res.suites
    assert s["errors"] == 1 and s["n"] == 10 and s["passed"] == 9 and s["accuracy"] == 0.9
    assert s["items_total"] == 10
    rec = res.records["general/s"][0]
    assert rec["error"] and rec["pass"] is False and rec["reason"] == "error"


def test_track_mean():
    def r(p):
        return FakeReply("OK-42" if ("-a-" in p and not p.endswith("-1")) or "-b-" in p else "no")

    fake = FakeOpenAI(responder=reply_for(r))
    res = run_q(fake, [make_suite("a", n=2), make_suite("b", n=2)])
    assert [s["accuracy"] for s in res.suites] == [0.5, 1.0]
    assert res.tracks["general"] == 0.75


def test_records_have_no_prompt_text_and_progress():
    fake = FakeOpenAI(responder=lambda m: FakeReply("OK-42"))
    seen = []
    res = run_q(fake, [make_suite(n=2)], on_progress=lambda k, d, t: seen.append((k, d, t)))
    blob = json.dumps(res.records)
    assert "PROMPT-" not in blob
    assert seen == [("general/s", 1, 2), ("general/s", 2, 2)]


def test_grader_crash_is_a_fail_not_a_dead_run(monkeypatch):
    from bakeoff import graders

    def boom(output, expected, options):
        raise RuntimeError("secret detail")

    monkeypatch.setitem(graders.GRADERS, "boom", boom)
    fake = FakeOpenAI(responder=reply_for(lambda p: FakeReply("OK-42")))
    res = run_q(fake, [make_suite(grader="boom", n=3)])
    recs = res.records["general/s"]
    assert len(recs) == 3
    assert all(r["pass"] is False and r["score"] == 0.0 for r in recs)
    assert all(r["reason"] == "grader error: RuntimeError" for r in recs)
    assert res.suites[0]["accuracy"] == 0.0


def test_json_suite_extractable_summary_and_error_records():
    fake = FakeOpenAI(
        responder=reply_for(
            lambda p: FakeReply('<json>{"a":1}</json>' if p.endswith("-0") else "nope")
        )
    )
    suite = make_suite(name="j", grader="json_schema", n=2)
    schema = {"type": "object", "required": ["a"]}
    suite = replace(suite, items=[replace(it, expected=schema) for it in suite.items])
    res = run_q(fake, [suite])
    (s,) = res.suites
    assert s["extractable_accuracy"] == 0.5 and s["extractable_ci95"] == list(wilson(1, 2))
    assert [r["extractable"] for r in res.records["general/j"]] == [True, False]
    (plain,) = run_q(FakeOpenAI(responder=lambda m: FakeReply("OK-42")), [make_suite()]).suites
    assert "extractable_accuracy" not in plain and "extractable_ci95" not in plain


def test_truncated_json_not_extractable():
    fake = FakeOpenAI(responder=lambda m: FakeReply('{"a":1}', finish_reason="length"))
    suite = make_suite(name="j", grader="json_schema", n=1)
    suite = replace(suite, items=[replace(suite.items[0], expected={"type": "object"})])
    res = run_q(fake, [suite])
    assert res.records["general/j"][0]["extractable"] is False


# --- 2026-10-27: real models took ~25 s an answer one at a time, so a 306-item quality pass
# outran Athena's 90-minute cap. Requests now overlap; grading, budget and order do not change.


class OverlapClient:
    """Answers OK-42 (fails the items in `fail`) after a short wait, counting requests in flight."""

    def __init__(self, fail=(), slow_first=True):
        self.fail, self.slow_first = set(fail), slow_first
        self.in_flight, self.peak, self.calls = 0, 0, []

    async def chat(self, messages, max_tokens, temperature=None, seed=None):
        from bakeoff.client import ChatResult

        prompt = messages[-1]["content"]
        self.calls.append(prompt)
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        slow = self.slow_first and prompt.endswith("-0")  # a slow first answer holds up nothing
        await asyncio.sleep(0.01 if slow else 0.001)
        self.in_flight -= 1
        if prompt in self.fail:
            return ChatResult(status=500, error="HTTP 500")
        return ChatResult(content="OK-42", finish_reason="stop")


def test_requests_overlap_up_to_the_limit_and_records_keep_item_order():
    from bakeoff.quality import QUALITY_CONCURRENCY

    client = OverlapClient()
    seen = []
    res = asyncio.run(
        run_quality(
            client,
            [make_suite("a", n=20), make_suite("b", n=3)],
            on_progress=lambda k, d, t: seen.append((k, d)),
        )
    )
    assert QUALITY_CONCURRENCY == 8 and client.peak == QUALITY_CONCURRENCY
    assert [r["id"] for r in res.records["general/a"]] == [f"i{k}" for k in range(20)]
    assert seen[:3] == [("general/a", 1), ("general/a", 2), ("general/a", 3)]
    assert seen[-1] == ("general/b", 3) and [s["passed"] for s in res.suites] == [20, 3]


def test_abort_records_items_in_order_and_sends_no_more_requests():
    client = OverlapClient(fail={f"PROMPT-s-{k}" for k in range(40)}, slow_first=False)
    with pytest.raises(AbortRun) as ei:
        asyncio.run(run_quality(client, [make_suite(n=40)]))
    recs = ei.value.partial.records["general/s"]
    assert [r["id"] for r in recs] == [f"i{k}" for k in range(10)]  # as one at a time
    assert len(client.calls) < 40  # the rest were never sent
