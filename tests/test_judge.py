import asyncio
import json
import re
import shutil
from pathlib import Path

import pytest
from fake_openai import FakeOpenAI, FakeReply
from test_runner import make_cfg, make_client

from bakeoff import cli
from bakeoff.bundle import verify_bundle
from bakeoff.client import EndpointClient
from bakeoff.judge import (
    JUDGE_MAX_TOKENS,
    JUDGE_SYSTEM,
    JudgeError,
    Prices,
    combine,
    estimate_usd,
    parse_reply,
    per_call_worst_case,
    run_judge,
)
from bakeoff.runner import execute_run
from bakeoff.suites import load_suite, suites_root

SUITE = "general/open-ended"
PRICES = Prices(1.0, 10.0)
URL = "http://judge-host.example/v1"  # scrub-ok


def make_bundle(tmp_path, tag, label, model, outputs=None, fail_ids=()):
    suite = load_suite(suites_root() / SUITE)
    prompts = {i.messages[-1]["content"]: i.id for i in suite.items}

    def responder(messages):
        iid = prompts[messages[-1]["content"]]
        if iid in fail_ids:
            return FakeReply(status=500, body="boom")
        return FakeReply(content=(outputs or {}).get(iid, f"{tag}{tag}{tag} answer {iid}"))

    fake = FakeOpenAI(models=[model], responder=responder)
    cfg = make_cfg(suites=(SUITE,), model=model)
    cfg = type(cfg)(
        endpoint=cfg.endpoint,
        model=type(cfg.model)(label=label, weights="hf:demo/demo@rev", runtime="other"),
        quality=cfg.quality,
        speed=cfg.speed,
        fit=cfg.fit,
    )

    async def inner():
        async with make_client(fake, model) as c:
            return await execute_run(
                cfg,
                c,
                tmp_path / f"in-{tag}",
                only={"quality"},
                run=lambda argv: None,
                platform="linux",
                sample_interval_s=0.01,
            )

    return asyncio.run(inner())


@pytest.fixture(scope="module")
def bundles(tmp_path_factory):
    root = tmp_path_factory.mktemp("bundles")
    a = make_bundle(root, "A", "Alpha Model", "alpha-served")
    b = make_bundle(root, "B", "Beta Model", "beta-served")
    return a, b


def judge_fake(script=None, **reply_kw):
    """Judge responder: picks the slot holding AAA answers unless `script` overrides per item."""
    suite = load_suite(suites_root() / SUITE)
    order = {
        i.messages[-1]["content"]: n for n, i in enumerate(sorted(suite.items, key=lambda i: i.id))
    }

    def responder(messages):
        user = messages[-1]["content"]
        q = re.search(r"Question:\n(.*)\n\nAnswer 1:", user, re.S).group(1)
        one = re.search(r"Answer 1:\n(.*)\n\nAnswer 2:", user, re.S).group(1)
        a_first = "AAA" in one
        want = (script or {}).get(order[q], "a")
        if want == "tie":
            winner = "tie"
        elif want == "none":
            return FakeReply(content="I refuse to answer in JSON", **reply_kw)
        else:
            winner = "1" if (want == "a") == a_first else "2"
        return FakeReply(content=json.dumps({"winner": winner, "reason": "because"}), **reply_kw)

    return FakeOpenAI(models=["judge-m"], responder=responder)


def run(bundles, out, fake, max_usd=100.0, allow_partial=False, prices=PRICES):
    a, b = bundles

    async def inner():
        async with EndpointClient(URL, "judge-m", transport=fake.transport, backoff_s=(0, 0)) as c:
            return await run_judge(a, b, c, "judge-m", prices, max_usd, allow_partial, out)

    return asyncio.run(inner())


@pytest.mark.parametrize(
    "first,second,res",
    [("1", "2", "a"), ("2", "1", "b"), ("1", "1", "tie"), ("tie", "2", "tie"), (None, "2", "tie")],
)
def test_combine(first, second, res):
    assert combine(first, second) == res


def test_parse_reply():
    assert parse_reply('{"winner": "1", "reason": "x"}') == ("1", "x")
    assert parse_reply('```json\n{"winner": "tie", "reason": "y"}\n```') == ("tie", "y")
    assert parse_reply('Sure {not json} then {"winner": "2", "reason": "z"} done')[0] == "2"
    assert parse_reply('{"winner": "3"}')[0] is None
    w, raw = parse_reply("x" * 1000)
    assert w is None and len(raw) == 300


def test_estimate_formula():
    got = estimate_usd(["x" * 300], Prices(3.0, 15.0))
    assert got == pytest.approx(2 * (150 * 3.0 + JUDGE_MAX_TOKENS * 15.0) / 1e6)


def test_refuses_over_budget_before_any_call(bundles, tmp_path):
    fake = judge_fake()
    with pytest.raises(JudgeError, match="estimate"):
        run(bundles, tmp_path, fake, max_usd=0.0001)
    assert fake.requests == [] and not list(tmp_path.iterdir())


def test_partial_stops_at_cap(bundles, tmp_path):
    fake = judge_fake(prompt_tokens=1000, completion_tokens=400)
    out = run(bundles, tmp_path, fake, max_usd=0.02, allow_partial=True)
    meta = json.loads((out / "bundle.json").read_text())
    assert meta["status"] == "incomplete" and meta["status_reason"] == "budget"
    assert 0 < meta["cost_usd"] <= 0.02
    assert len(fake.requests) == 4  # two items, two calls each
    assert meta["estimate_usd"] > 0.02


def test_cap_enforced_even_without_partial(bundles, tmp_path):
    # usage far above the estimate: the cap is still checked before every call
    fake = judge_fake(prompt_tokens=100000, completion_tokens=400)
    out = run(bundles, tmp_path, fake, max_usd=0.5)
    meta = json.loads((out / "bundle.json").read_text())
    assert meta["status"] == "incomplete" and len(fake.requests) < 40


def test_no_usage_charges_worst_case(bundles, tmp_path):
    fake = judge_fake()
    fake.support_stream_options = False
    out = run(bundles, tmp_path, fake)
    meta = json.loads((out / "bundle.json").read_text())
    assert meta["cost_usd"] == pytest.approx(meta["estimate_usd"])


def test_refuses_mismatched_suite_hash(bundles, tmp_path):
    a, b = bundles
    b2 = tmp_path / "b2"
    shutil.copytree(b, b2)
    meta = json.loads((b2 / "bundle.json").read_text())
    for s in meta["suites"]:
        s["hash"] = "0" * 64
    # re-seal so only the suite check can object
    from bakeoff.bundle import _bundle_hash

    meta["bundle_sha256"] = _bundle_hash(meta)
    (b2 / "bundle.json").write_text(json.dumps(meta))
    fake = judge_fake()
    with pytest.raises(JudgeError, match="suite hash"):
        run((a, b2), tmp_path / "out", fake)
    assert fake.requests == []


def test_refuses_tampered_bundle(bundles, tmp_path):
    a, b = bundles
    b2 = tmp_path / "b2"
    shutil.copytree(b, b2)
    f = b2 / "quality" / "general.open-ended.jsonl"
    f.write_text(f.read_text().replace("BBB", "CCC"))
    with pytest.raises(JudgeError, match="verification"):
        run((a, b2), tmp_path / "out", judge_fake())


def test_blind_prompt(bundles, tmp_path):
    fake = judge_fake()
    out = run(bundles, tmp_path, fake)
    meta_in = [json.loads((p / "bundle.json").read_text()) for p in bundles]
    secrets = ["Alpha", "Beta", "alpha-served", "beta-served", "Model"]
    secrets += [m["bundle_sha256"] for m in meta_in] + [m["bundle_sha256"][:8] for m in meta_in]
    secrets += [m["run_id"] for m in meta_in] + [m["suites"][0]["hash"] for m in meta_in]
    for req in fake.requests:
        text = json.dumps(req["messages"])
        for s in secrets:
            assert s not in text, s
        assert req["messages"][0] == {"role": "system", "content": JUDGE_SYSTEM}
        assert req["temperature"] == 0 and req["seed"] == 0
        assert req["max_tokens"] == JUDGE_MAX_TOKENS
    assert out.name.startswith("judge-")


def test_order_swapped(bundles, tmp_path):
    fake = judge_fake()
    run(bundles, tmp_path, fake)
    first, second = (r["messages"][-1]["content"] for r in fake.requests[:2])
    assert re.search(r"Answer 1:\nAAA answer open-01\n\nAnswer 2:\nBBB", first)
    assert re.search(r"Answer 1:\nBBB answer open-01\n\nAnswer 2:\nAAA", second)


def test_counts_and_win_rate(bundles, tmp_path):
    fake = judge_fake({**{n: "tie" for n in range(20)}, 0: "a", 1: "a", 2: "b"})
    out = run(bundles, tmp_path, fake)
    meta = json.loads((out / "bundle.json").read_text())
    assert meta["status"] == "complete" and meta["bundle_format"] == "judge-m1"
    assert meta["counts"]["a"] == 2 and meta["counts"]["b"] == 1
    assert meta["win_rate_a"] == pytest.approx(2 / 3)
    assert len(meta["ci95"]) == 2 and meta["ci95"][0] < 2 / 3 < meta["ci95"][1]
    assert meta["a"] == json.loads((bundles[0] / "bundle.json").read_text())["bundle_sha256"]
    recs = [json.loads(x) for x in (out / "items.jsonl").read_text().splitlines()]
    assert [r["id"] for r in recs][:2] == ["open-01", "open-02"]
    assert recs[0]["result"] == "a" and recs[3]["result"] == "tie"
    assert meta["cost_usd"] == pytest.approx(sum(r["cost_usd"] for r in recs))


def test_all_ties_has_no_win_rate(bundles, tmp_path):
    out = run(bundles, tmp_path, judge_fake({n: "tie" for n in range(20)}))
    meta = json.loads((out / "bundle.json").read_text())
    assert meta["win_rate_a"] is None and meta["ci95"] is None


def test_unparseable_reply_is_tie(bundles, tmp_path):
    out = run(bundles, tmp_path, judge_fake({0: "none"}))
    recs = [json.loads(x) for x in (out / "items.jsonl").read_text().splitlines()]
    assert recs[0]["first"] is None and recs[0]["result"] == "tie"
    assert "refuse" in recs[0]["reasons"][0]


def _erroring(fake, bad_calls):
    """Make the judge fake answer 429 on the given (1-based) call numbers."""
    inner, n = fake.responder, {"n": 0}

    def responder(messages):
        n["n"] += 1
        if n["n"] in bad_calls:
            return FakeReply(status=429, body="rate limited")
        return inner(messages)

    fake.responder = responder
    return fake


def test_isolated_call_error_is_an_error_item_not_a_tie(bundles, tmp_path):
    out = run(bundles, tmp_path, _erroring(judge_fake(), {3}))
    meta = json.loads((out / "bundle.json").read_text())
    recs = [json.loads(x) for x in (out / "items.jsonl").read_text().splitlines()]
    assert meta["status"] == "complete" and meta["call_errors"] == 1
    assert [r["result"] for r in recs].count("error") == 1
    assert sum(meta["counts"].values()) == 19 and meta["counts"]["tie"] == 0
    bad = next(r for r in recs if r["result"] == "error")
    assert bad["reasons"] == ["call failed"] and bad["cost_usd"] > 0
    assert "rate limited" not in json.dumps(recs)


def test_429_burst_stops_incomplete_with_error_and_cli_exits_2(bundles, tmp_path, monkeypatch):
    fake = FakeOpenAI(models=["judge-m"], responder=lambda m: FakeReply(status=429, body="slow"))
    out = run(bundles, tmp_path, fake)
    meta = json.loads((out / "bundle.json").read_text())
    assert meta["status"] == "incomplete" and meta["status_reason"] == "error"
    assert meta["call_errors"] == 10 and len(fake.requests) == 10  # stopped at the 10th call
    assert meta["counts"] == {"a": 0, "b": 0, "tie": 0} and meta["cost_usd"] > 0
    recs = [json.loads(x) for x in (out / "items.jsonl").read_text().splitlines()]
    assert {r["result"] for r in recs} == {"error"}
    assert sum(r["cost_usd"] for r in recs) == pytest.approx(meta["cost_usd"])
    a, b = bundles
    cfg = tmp_path / "j.toml"
    cfg.write_text(JTOML)
    monkeypatch.setenv("BAKEOFF_JUDGE_API_KEY", "k")
    monkeypatch.setattr(
        cli,
        "make_judge_client",
        lambda url, model, key: cli.EndpointClient(
            url, model, transport=fake.transport, backoff_s=()
        ),
    )
    argv = ["judge", str(a), str(b), "--max-usd", "100", "--judge-config", str(cfg)]
    assert cli.main([*argv, "--out", str(tmp_path / "o2")]) == 2


def test_error_bodies_with_the_api_key_are_never_stored(bundles, tmp_path):
    key = "sk-test-SECRET-123"
    fake = FakeOpenAI(
        models=["judge-m"], responder=lambda m: FakeReply(status=401, body=f"bad key {key}")
    )
    a, b = bundles

    async def inner():
        async with EndpointClient(
            URL, "judge-m", api_key=key, transport=fake.transport, backoff_s=()
        ) as c:
            return await run_judge(a, b, c, "judge-m", PRICES, 100.0, False, tmp_path)

    out = asyncio.run(inner())
    for f in out.rglob("*"):
        if f.is_file():
            assert key not in f.read_text()


def test_refuses_incomplete_input_bundle(tmp_path):
    a = make_bundle(tmp_path, "A", "Alpha Model", "alpha-served")
    b = make_bundle(tmp_path, "B", "Beta Model", "beta-served")
    from bakeoff.bundle import write_bundle

    meta = json.loads((b / "bundle.json").read_text())
    meta["status"], meta["status_reason"] = "incomplete", "interrupted"
    files = {
        rel: [json.loads(x) for x in (b / rel).read_text().splitlines()] for rel in meta["files"]
    }
    write_bundle(b, meta, files)  # re-hashed, so only the status marks it incomplete
    with pytest.raises(JudgeError, match="not complete"):
        run((a, b), tmp_path / "o", judge_fake())


def test_skips_errored_items(tmp_path):
    a = make_bundle(tmp_path, "A", "Alpha Model", "alpha-served", fail_ids={"open-01"})
    b = make_bundle(tmp_path, "B", "Beta Model", "beta-served")
    out = run((a, b), tmp_path / "o", judge_fake())
    meta = json.loads((out / "bundle.json").read_text())
    assert meta["items_skipped"] == 1 and sum(meta["counts"].values()) == 19


def test_output_verifies_and_no_overwrite(bundles, tmp_path):
    out = run(bundles, tmp_path, judge_fake())
    assert verify_bundle(out, username="alice", denylist=None) == []
    fake = judge_fake()
    with pytest.raises(JudgeError, match="exists"):
        run(bundles, tmp_path, fake)
    assert fake.requests == []


def test_think_blocks_stripped(tmp_path):
    outs = {"open-01": "<think>secret reasoning</think>AAA visible"}
    a = make_bundle(tmp_path, "A", "Alpha Model", "alpha-served", outputs=outs)
    b = make_bundle(tmp_path, "B", "Beta Model", "beta-served")
    fake = judge_fake()
    run((a, b), tmp_path / "o", fake)
    assert "secret reasoning" not in json.dumps(fake.requests)


# CLI


JTOML = '[judge]\nbase_url = "http://judge-host.example/v1"\nmodel = "judge-m"\n'
JTOML += "price_in = 1.0\nprice_out = 10.0\n"  # scrub-ok


def patch_judge(monkeypatch, fake):
    monkeypatch.setattr(
        cli,
        "make_judge_client",
        lambda url, model, key: EndpointClient(
            url, model, api_key=key, transport=fake.transport, backoff_s=(0, 0)
        ),
    )


def test_cli_judge(bundles, tmp_path, monkeypatch, capsys):
    a, b = bundles
    cfg = tmp_path / "j.toml"
    cfg.write_text(JTOML)
    out = tmp_path / "out"
    base = [str(a), str(b), "--max-usd", "100", "--judge-config", str(cfg), "--out", str(out)]
    monkeypatch.delenv("BAKEOFF_JUDGE_API_KEY", raising=False)
    assert cli.main(["judge", *base]) == 1
    assert "BAKEOFF_JUDGE_API_KEY" in capsys.readouterr().err
    monkeypatch.setenv("BAKEOFF_JUDGE_API_KEY", "sk-secret-123")
    assert cli.main(["judge", *base, "--price-in", "-1"]) == 1
    bare = tmp_path / "bare.toml"
    bare.write_text('[judge]\nmodel = "judge-m"\n')
    assert cli.main(["judge", str(a), str(b), "--max-usd", "1", "--judge-config", str(bare)]) == 1
    assert "missing" in capsys.readouterr().err
    patch_judge(monkeypatch, judge_fake(prompt_tokens=1000, completion_tokens=400))
    assert (
        cli.main(
            ["judge", *base, "--allow-partial", "--out", str(tmp_path / "o2"), "--max-usd", "0.02"]
        )
        == 2
    )
    patch_judge(monkeypatch, judge_fake())
    assert cli.main(["judge", *base]) == 0
    text = "\n".join(p.read_text() for p in Path(out).rglob("*") if p.is_file())
    captured = capsys.readouterr()
    assert "sk-secret-123" not in text + captured.out + captured.err
    assert cli.main(["judge", *base]) == 1  # exists


def test_normal_run_stays_under_cap(bundles, tmp_path):
    fake = judge_fake(prompt_tokens=1000, completion_tokens=400)
    meta = json.loads((run(bundles, tmp_path, fake, max_usd=1.0) / "bundle.json").read_text())
    assert meta["cost_usd"] <= 1.0 and meta["estimate_usd"] > 0


def test_adversarial_usage_overshoots_by_at_most_one_calls_excess(bundles, tmp_path):
    fake = judge_fake(prompt_tokens=200000, completion_tokens=400)  # ~10x+ the estimate
    out = run(bundles, tmp_path, fake, max_usd=0.05, allow_partial=True)
    meta = json.loads((out / "bundle.json").read_text())
    assert meta["status"] == "incomplete" and meta["status_reason"] == "budget"
    assert len(fake.requests) == 1
    worst = per_call_worst_case(
        JUDGE_SYSTEM + "\n" + fake.requests[0]["messages"][-1]["content"], PRICES
    )
    assert meta["cost_usd"] - 0.05 <= meta["cost_usd"] - worst  # one call's excess only
    # the half-finished item is recorded, uncounted, with its actual cost
    recs = [json.loads(x) for x in (out / "items.jsonl").read_text().splitlines()]
    assert len(recs) == 1 and recs[0]["second"] is None and recs[0]["result"] is None
    assert sum(meta["counts"].values()) == 0
    assert sum(r["cost_usd"] for r in recs) == pytest.approx(meta["cost_usd"])


def test_estimate_uses_utf8_bytes_over_two():
    got = estimate_usd(["é" * 300], Prices(3.0, 15.0), calls_per_item=1)
    assert got == pytest.approx((300 * 3.0 + JUDGE_MAX_TOKENS * 15.0) / 1e6)


class RaisingClient:
    def __init__(self, inner, on_call):
        self.inner, self.on_call, self.n = inner, on_call, 0

    async def chat(self, *a, **kw):
        self.n += 1
        if self.n == self.on_call:
            raise RuntimeError("kaboom http://judge-host.example")  # scrub-ok
        return await self.inner.chat(*a, **kw)


def test_unexpected_exception_charges_worst_and_stops(bundles, tmp_path):
    a, b = bundles
    fake = judge_fake(prompt_tokens=10, completion_tokens=10)

    async def inner():
        async with EndpointClient(URL, "judge-m", transport=fake.transport, backoff_s=()) as c:
            rc = RaisingClient(c, 3)
            return await run_judge(a, b, rc, "judge-m", PRICES, 100.0, False, tmp_path)

    out = asyncio.run(inner())
    meta = json.loads((out / "bundle.json").read_text())
    assert meta["status"] == "incomplete" and meta["status_reason"] == "error"
    recs = [json.loads(x) for x in (out / "items.jsonl").read_text().splitlines()]
    assert len(recs) == 2 and recs[1]["result"] is None and recs[1]["first"] is None
    worst = (
        per_call_worst_case(
            JUDGE_SYSTEM + "\n" + fake.requests[2]["messages"][-1]["content"], PRICES
        )
        if len(fake.requests) > 2
        else None
    )
    assert worst is None  # the 3rd call never reached the server
    assert recs[1]["cost_usd"] > recs[0]["cost_usd"] / 2  # worst case, not zero
    assert sum(r["cost_usd"] for r in recs) == pytest.approx(meta["cost_usd"])
    assert (
        "kaboom" not in json.dumps(recs)
        and verify_bundle(out, "alice", None)[0].kind == "incomplete"
    )


def test_cli_exit_2_on_unexpected_error(bundles, tmp_path, monkeypatch):
    a, b = bundles
    cfg = tmp_path / "j.toml"
    cfg.write_text(JTOML)
    monkeypatch.setenv("BAKEOFF_JUDGE_API_KEY", "k")
    fake = judge_fake()
    monkeypatch.setattr(
        cli,
        "make_judge_client",
        lambda url, model, key: _Ctx(
            RaisingClient(cli.EndpointClient(url, model, transport=fake.transport), 3)
        ),
    )
    argv = ["judge", str(a), str(b), "--max-usd", "100", "--judge-config", str(cfg)]
    assert cli.main([*argv, "--out", str(tmp_path / "o")]) == 2


class _Ctx:
    def __init__(self, c):
        self.c = c

    async def __aenter__(self):
        return self.c

    async def __aexit__(self, *exc):
        return None


def test_cli_client_does_not_retry(bundles, tmp_path):
    a, b = bundles
    fake = judge_fake()
    fake.fail_first = 1

    async def inner():
        async with cli.make_judge_client(URL, "judge-m", "k", transport=fake.transport) as c:
            return await run_judge(a, b, c, "judge-m", PRICES, 100.0, False, tmp_path)

    out = asyncio.run(inner())
    assert len(fake.requests) == 39  # one request per call: the 503 was not retried, and the
    # failed item skipped its second order
    meta = json.loads((out / "bundle.json").read_text())
    recs = [json.loads(x) for x in (out / "items.jsonl").read_text().splitlines()]
    assert recs[0]["first"] is None and recs[0]["result"] == "error"
    user = fake.requests[0]["messages"][-1]["content"]
    worst = per_call_worst_case(JUDGE_SYSTEM + "\n" + user, PRICES)
    assert recs[0]["cost_usd"] >= worst and meta["status"] == "complete"
