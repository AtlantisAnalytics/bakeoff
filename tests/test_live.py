"""Live checks against a real server. Skipped unless BAKEOFF_LIVE_ENDPOINT and BAKEOFF_LIVE_MODEL
are set. The endpoint is read from the environment only and is never written to a tracked file."""

import asyncio
import getpass
import os
import statistics

import httpx
import pytest

from bakeoff import cli, speed
from bakeoff.bundle import verify_bundle
from bakeoff.client import EndpointClient, normalize_base_url

ENDPOINT = os.environ.get("BAKEOFF_LIVE_ENDPOINT")
MODEL = os.environ.get("BAKEOFF_LIVE_MODEL")

pytestmark = pytest.mark.skipif(
    not (ENDPOINT and MODEL), reason="set BAKEOFF_LIVE_ENDPOINT and BAKEOFF_LIVE_MODEL"
)


def test_live_smoke(tmp_path):
    cfg = tmp_path / "run.toml"
    cfg.write_text(
        f"""[endpoint]
base_url = "{ENDPOINT}"
model = "{MODEL}"
timeout_s = 600

[model]
label = "live smoke"
weights = "operator-supplied"
quant = "unknown"
runtime = "ollama"
envelope = "live smoke envelope"

[quality]
suites = ["format/exact-reply", "longctx/needle"]

[speed]
concurrency = [1]
repeats = 1
warmup = 0

[fit]
steps = [4096, 8192]
""",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    assert cli.main(["run", str(cfg), "--out", str(out)]) == 0
    (run_dir,) = [p for p in out.iterdir() if p.is_dir()]
    assert verify_bundle(run_dir, getpass.getuser(), None) == []


def test_live_decode_matches_runtime_counters(capsys):
    """Our client-side decode tok/s should agree with the runtime's own counters (Ollama)."""
    api_base, root = normalize_base_url(ENDPOINT)
    messages = speed.SHAPES["short"]
    n = 3

    async def ours():
        out = []
        async with EndpointClient(api_base, MODEL, timeout_s=600) as client:
            await client.chat(messages, 256, temperature=0.0, seed=0)  # warm the model
            for _ in range(n):
                r = await client.chat(messages, 256, temperature=0.0, seed=0)
                assert r.error is None, r.error
                out.append(r)
        return out

    results = asyncio.run(ours())
    our_decode = [(r.completion_tokens - 1) / (r.t_last - r.t_first) for r in results]
    our_ttft = [r.ttft_s for r in results]

    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0, "seed": 0, "num_predict": 256},
    }
    native = []
    for _ in range(n):
        resp = httpx.post(f"{root}/api/chat", json=payload, timeout=600)
        resp.raise_for_status()
        native.append(resp.json())
    rt_decode = [d["eval_count"] / (d["eval_duration"] / 1e9) for d in native]
    rt_prefill_s = [d["prompt_eval_duration"] / 1e9 for d in native]
    rt_prefill_tps = [
        d["prompt_eval_count"] / s for d, s in zip(native, rt_prefill_s, strict=True) if s > 0
    ]

    ours_med = statistics.median(our_decode)
    rt_med = statistics.median(rt_decode)
    with capsys.disabled():
        print(
            f"\nLIVE decode tok/s ours={ours_med:.2f} runtime={rt_med:.2f} "
            f"ratio={ours_med / rt_med:.3f} | ttft ours={statistics.median(our_ttft):.3f}s "
            f"runtime prompt_eval(median)={statistics.median(rt_prefill_s):.3f}s | "
            f"runtime prefill tok/s={[round(x, 1) for x in rt_prefill_tps]}"
        )
    assert abs(ours_med - rt_med) / rt_med <= 0.15
