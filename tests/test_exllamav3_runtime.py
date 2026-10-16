"""The exllamav3 runtime (ExLlamaV3 served by TabbyAPI)."""

import asyncio
import dataclasses

import httpx
from fake_openai import FakeOpenAI
from test_config import MINIMAL, write
from test_env import CFG, LINUX_FILES, fake_read
from test_speed import reply, run_s

from bakeoff.client import EndpointClient
from bakeoff.config import RUNTIMES, SpeedConfig, load_config
from bakeoff.env import capture_env
from bakeoff.speed import FORCED_LENGTH

VERSION = "exllamav3 1.6.2 (TabbyAPI 63860e6)"


def test_runtime_exllamav3_is_allowed(tmp_path):
    assert "exllamav3" in RUNTIMES
    text = MINIMAL.replace(
        'runtime = "vllm"', f'runtime = "exllamav3"\nruntime_version = "{VERSION}"'
    )
    cfg = load_config(write(tmp_path, text))
    assert cfg.model.runtime == "exllamav3" and cfg.model.runtime_version == VERSION


def test_runtime_version_defaults_empty(tmp_path):
    assert load_config(write(tmp_path)).model.runtime_version == ""


def test_forced_length_per_runtime():
    assert FORCED_LENGTH == {
        "vllm": {"ignore_eos": True},
        "llamacpp": {"ignore_eos": True},
        "exllamav3": {"ban_eos_token": True},
    }
    fake = FakeOpenAI(responder=reply())
    recs, summ = run_s(fake, SpeedConfig(concurrency=[1], repeats=1, warmup=1), "exllamav3")
    assert fake.requests and all(b.get("ban_eos_token") is True for b in fake.requests)
    assert all("ignore_eos" not in b for b in fake.requests)
    assert all(r["ignore_eos"] is True for r in recs)
    assert summ["short@1"]["ignore_eos"] is True


def test_runtime_info_exllamav3():
    def handler(request):
        if request.url.path == "/v1/model":
            return httpx.Response(
                200, json={"id": "m", "parameters": {"max_seq_len": 32768, "cache_mode": "FP16"}}
            )
        return httpx.Response(404)

    async def go():
        async with EndpointClient("http://h/v1", "m", transport=httpx.MockTransport(handler)) as c:
            return await c.runtime_info("exllamav3")

    assert asyncio.run(go()) == {"version": None, "context_length": 32768, "size_vram": None}


def _cfg(runtime_version):
    return dataclasses.replace(
        CFG,
        model=dataclasses.replace(CFG.model, runtime="exllamav3", runtime_version=runtime_version),
    )


def _env(cfg, info):
    return capture_env(
        cfg, info, "m", run=lambda argv: None, read=fake_read(LINUX_FILES), platform="linux"
    )


def test_env_runtime_version_falls_back_to_operator():
    env = _env(_cfg(VERSION), {"version": None})
    assert env["runtime"]["version"] == {"value": VERSION, "source": "operator"}


def test_env_reported_runtime_version_wins():
    env = _env(_cfg(VERSION), {"version": "9.9"})
    assert env["runtime"]["version"] == {"value": "9.9", "source": "reported"}


def test_env_without_any_version_stays_reported_none():
    env = _env(_cfg(""), {"version": None})
    assert env["runtime"]["version"]["value"] is None
