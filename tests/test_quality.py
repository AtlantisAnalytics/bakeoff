import asyncio
import json

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
        "n", "passed", "errors", "truncated", "accuracy", "ci95",
    }  # fmt: skip
    assert s["key"] == "general/s" and s["hash"] == "hash-s"
    rec = res.records["general/s"][0]
    assert set(rec) == {
        "id", "prompt_sha256", "output", "reasoning_tokens", "finish_reason",
        "latency_s", "pass", "score", "reason", "error",
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


def test_errors_below_budget_count_as_fails():
    def r(p):
        return FakeReply("x", status=400, body="bad") if p.endswith("-0") else FakeReply("OK-42")

    res = run_q(FakeOpenAI(responder=reply_for(r)), [make_suite(n=10)])
    (s,) = res.suites
    assert s["errors"] == 1 and s["n"] == 10 and s["passed"] == 9 and s["accuracy"] == 0.9
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
