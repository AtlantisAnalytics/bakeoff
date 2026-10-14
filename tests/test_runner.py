import asyncio
import json
from pathlib import Path

import pytest
from fake_openai import FakeOpenAI, FakeReply

from bakeoff.bundle import verify_bundle
from bakeoff.client import EndpointClient
from bakeoff.config import (
    EndpointConfig,
    FitConfig,
    ModelConfig,
    QualityConfig,
    RunConfig,
    SpeedConfig,
)
from bakeoff.env import NVIDIA_QUERY_ARGV
from bakeoff.runner import PreflightError, execute_run, make_redactor, preflight
from bakeoff.sampler import SAMPLER_NVIDIA_ARGV

URL = "http://lab-box-7.lan:11434/v1"  # scrub-ok


def fake_nvidia(argv):
    if argv == ["nvidia-smi", "-L"]:
        return "GPU 0: Fake Card\n"
    if argv == NVIDIA_QUERY_ARGV:
        return "0, Fake Card, 24564, 550.54.14, 450.00, 1000, 100.5\n"
    if argv == ["nvidia-smi"]:
        return "CUDA Version: 12.4\n"
    if argv == SAMPLER_NVIDIA_ARGV:
        return "0, 12000, 200.0\n"
    return None


def make_cfg(
    suites=("format/exact-reply", "longctx/needle"), private=(), model="served-model-name"
):
    return RunConfig(
        endpoint=EndpointConfig(base_url=URL, model=model),
        model=ModelConfig(label="Demo Model", weights="hf:demo/demo@rev", runtime="other"),
        quality=QualityConfig(suites=list(suites), private_suites=list(private)),
        speed=SpeedConfig(concurrency=[1], repeats=1, warmup=0),
        fit=FitConfig(steps=[2048]),
    )


def make_client(fake, model="served-model-name"):
    return EndpointClient(URL, model, transport=fake.transport, backoff_s=(0, 0))


def go(cfg, fake, tmp_path, **kw):
    async def inner():
        async with make_client(fake, cfg.endpoint.model) as c:
            return await execute_run(cfg, c, tmp_path, sample_interval_s=0.01, **kw)

    return asyncio.run(inner())


def all_text(path):
    return "\n".join(p.read_text() for p in path.rglob("*") if p.is_file())


def test_end_to_end_bundle_verifies(tmp_path):
    path = go(make_cfg(), FakeOpenAI(), tmp_path, run=fake_nvidia, platform="linux")
    meta = json.loads((path / "bundle.json").read_text())
    assert meta["status"] == "complete" and meta["bundle_format"] == "m1"
    assert verify_bundle(path, username="alice", denylist=None) == []
    assert meta["memory"]["peaks"]["value"]
    assert meta["memory"]["platform"]["value"] == "nvidia"
    assert {"quality/format.exact-reply.jsonl", "speed.jsonl", "fit.jsonl", "samples.jsonl"} <= set(
        meta["files"]
    )
    assert {s["key"] for s in meta["suites"]} == {"format/exact-reply", "longctx/needle"}


def test_endpoint_url_never_written(tmp_path):
    def responder(messages):
        body = f"cannot reach {URL}/chat via lab-box-7.lan:11434"  # scrub-ok
        return FakeReply(status=400, body=body)

    fake = FakeOpenAI(responder=responder)
    path = go(make_cfg(), fake, tmp_path, run=fake_nvidia, platform="linux")
    text = all_text(path)
    assert "<endpoint>" in text  # the error text was recorded, redacted
    assert "lab-box-7" not in text and ":11434" not in text
    assert json.loads((path / "bundle.json").read_text())["status"] == "incomplete"


def test_redactor_variants():
    r = make_redactor(URL)
    assert r({"a": [f"x {URL}/y", "lab-box-7.lan:11434", "LAB-BOX-7.LAN"]}) == {  # scrub-ok
        "a": ["x <endpoint>/y", "<endpoint>", "<endpoint>"]
    }
    assert make_redactor("http://box:80")("box here") == "box here"  # host < 4 chars kept


def test_private_suite_routed_and_text_absent(tmp_path):
    suite = tmp_path / "priv"
    suite.mkdir()
    (suite / "manifest.toml").write_text(
        'track = "demo"\nname = "secret"\nversion = 1\nsplit = "private"\n'
        'license = "CC0-1.0"\ngrader = "exact"\nmax_tokens = 16\n'
    )
    (suite / "items.jsonl").write_text(
        '{"id": "a", "prompt": "zebra-quantum-marmalade please", "expected": "ok"}\n'
    )
    out = tmp_path / "out"
    cfg = make_cfg(suites=(), private=(str(suite),))
    path = go(cfg, FakeOpenAI(), out, run=fake_nvidia, platform="linux")
    meta = json.loads((path / "bundle.json").read_text())
    assert "private/demo.secret.jsonl" in meta["files"]
    text = all_text(path)
    assert "zebra-quantum-marmalade" not in text and str(suite) not in text
    assert "prompt_sha256" in (path / "private/demo.secret.jsonl").read_text()


def test_extra_suite_with_private_split_is_routed(tmp_path):
    suite = tmp_path / "priv"
    suite.mkdir()
    (suite / "manifest.toml").write_text(
        'track = "demo"\nname = "secret"\nversion = 1\nsplit = "private"\n'
        'license = "CC0-1.0"\ngrader = "exact"\nmax_tokens = 16\n'
    )
    (suite / "items.jsonl").write_text('{"id": "a", "prompt": "hi", "expected": "ok"}\n')
    path = go(
        make_cfg(suites=()), FakeOpenAI(), tmp_path / "o", extra_suites=[str(suite)],
        run=fake_nvidia, platform="linux",
    )  # fmt: skip
    assert (path / "private/demo.secret.jsonl").is_file()


def test_abort_writes_incomplete(tmp_path):
    path = go(make_cfg(), FakeOpenAI(fail_first=1000), tmp_path, run=fake_nvidia, platform="linux")
    meta = json.loads((path / "bundle.json").read_text())
    assert meta["status"] == "incomplete" and meta["status_reason"].startswith("quality:")
    assert not (path / "speed.jsonl").exists()
    assert (path / "quality/format.exact-reply.jsonl").is_file()


def test_interrupt_writes_bundle_and_reraises(tmp_path, monkeypatch):
    import bakeoff.runner as runner

    async def boom(*a, **k):
        raise asyncio.CancelledError

    monkeypatch.setattr(runner, "run_speed", boom)
    with pytest.raises(asyncio.CancelledError):
        go(make_cfg(), FakeOpenAI(), tmp_path, run=fake_nvidia, platform="linux")
    meta = json.loads(next(tmp_path.glob("*/bundle.json")).read_text())
    assert meta["status"] == "incomplete" and meta["status_reason"] == "interrupted"
    assert "quality/format.exact-reply.jsonl" in meta["files"]


def test_preflight_latest_tag():
    async def inner():
        async with make_client(FakeOpenAI(models=["llama3.1:latest"]), "llama3.1") as c:
            return await preflight(c, "llama3.1")

    assert asyncio.run(inner()) == "llama3.1:latest"


def test_preflight_mismatch_lists_models():
    async def inner():
        async with make_client(FakeOpenAI(models=["other-model"]), "x") as c:
            return await preflight(c, "served-model-name")

    with pytest.raises(PreflightError, match="served-model-name") as ei:
        asyncio.run(inner())
    assert "other-model" in str(ei.value)


def test_preflight_connection_failure():
    class Broken:
        async def list_models(self):
            raise OSError("down")

    with pytest.raises(PreflightError):
        asyncio.run(preflight(Broken(), "m"))


def test_no_gpu_host_runs(tmp_path):
    path = go(make_cfg(), FakeOpenAI(), tmp_path, run=lambda argv: None, platform="linux")
    meta = json.loads((path / "bundle.json").read_text())
    assert meta["status"] == "complete"
    assert meta["memory"]["platform"]["value"] == "none"
    assert meta["memory"]["peaks"]["source"] == "not_captured"
    assert meta["env"]["gpus"] == []
    assert verify_bundle(path, username="alice", denylist=None) == []


def test_only_speed(tmp_path):
    path = go(make_cfg(), FakeOpenAI(), tmp_path, only={"speed"}, run=fake_nvidia, platform="linux")
    meta = json.loads((path / "bundle.json").read_text())
    assert meta["quality"] == {"suites": [], "tracks": {}}
    assert not (path / "quality").exists() and (path / "speed.jsonl").is_file()
    assert "fit.jsonl" not in meta["files"]


def test_example_config_suites_resolve():
    from bakeoff.config import load_config
    from bakeoff.suites import resolve_suites

    cfg = load_config(Path(__file__).resolve().parents[1] / "examples" / "run.toml")
    assert [s.key for s in resolve_suites(cfg.quality.suites)] == [
        "format/exact-reply",
        "general/gsm8k",
        "general/ifeval",
        "general/open-ended",
        "grounding/context-only",
        "longctx/needle",
        "structured/json",
    ]


def test_redactor_keeps_undotted_hostnames_intact():
    r = make_redactor("http://llama:8000/v1")  # scrub-ok
    assert r("llama says hello; runtime ollama") == "llama says hello; runtime ollama"
    assert r("see http://llama:8000 and llama:8000") == "see <endpoint> and <endpoint>"  # scrub-ok


def test_redactor_bare_fqdn_at_token_boundaries():
    r = make_redactor(URL)
    assert r("host lab-box-7.lan down") == "host <endpoint> down"
    assert r("xlab-box-7.lan and lab-box-7.lan.evil") == "xlab-box-7.lan and lab-box-7.lan.evil"


def test_redactor_bare_ip_and_ipv6():
    r = make_redactor("http://192.0.2.7:9000/v1")  # scrub-ok
    assert r("connect to 192.0.2.7 failed") == "connect to <endpoint> failed"  # scrub-ok
    assert r("192.0.2.77 and 1192.0.2.7") == "192.0.2.77 and 1192.0.2.7"  # scrub-ok
    r6 = make_redactor("http://[2001:db8::7]:9000/v1")  # scrub-ok
    assert r6("at 2001:db8::7 or [2001:db8::7]") == "at <endpoint> or <endpoint>"  # scrub-ok


def _cancel_when(marker, after=0):
    seen = {"n": 0}

    def responder(messages):
        text = "\n".join(str(m.get("content", "")) for m in messages)
        if marker in text:
            if seen["n"] >= after:
                raise asyncio.CancelledError
            seen["n"] += 1
        from fake_openai import default_responder

        return default_responder(messages)

    return responder


def test_interrupt_in_quality_keeps_partial_records(tmp_path):
    fake = FakeOpenAI(responder=_cancel_when("access code", after=2))
    with pytest.raises(asyncio.CancelledError):
        go(make_cfg(), fake, tmp_path, run=fake_nvidia, platform="linux")
    path = next(tmp_path.glob("*"))
    meta = json.loads((path / "bundle.json").read_text())
    assert meta["status_reason"] == "interrupted"
    assert "quality/format.exact-reply.jsonl" in meta["files"]
    needle = (path / "quality/longctx.needle.jsonl").read_text().splitlines()
    assert len(needle) == 2
    assert meta["quality"]["suites"]
    assert verify_bundle(path, username="alice", denylist=None)[0].kind == "incomplete"


def test_interrupt_in_speed_keeps_partial_records(tmp_path):
    fake = FakeOpenAI(responder=_cancel_when("Summarise the records above"))
    with pytest.raises(asyncio.CancelledError):
        go(make_cfg(), fake, tmp_path, run=fake_nvidia, platform="linux")
    path = next(tmp_path.glob("*"))
    meta = json.loads((path / "bundle.json").read_text())
    assert "quality/format.exact-reply.jsonl" in meta["files"]
    assert len((path / "speed.jsonl").read_text().splitlines()) == 1
    assert "fit.jsonl" not in meta["files"]


def test_interrupt_in_fit_keeps_partial_records(tmp_path):
    fake = FakeOpenAI(responder=_cancel_when("access code", after=6 + 2 + 1))
    with pytest.raises(asyncio.CancelledError):
        go(make_cfg(), fake, tmp_path, run=fake_nvidia, platform="linux")
    path = next(tmp_path.glob("*"))
    meta = json.loads((path / "bundle.json").read_text())
    assert meta["fit"]["incomplete"] is True
    assert "fit.jsonl" in meta["files"]


def test_preflight_accepts_latest_in_config_when_server_lists_bare():
    async def inner():
        async with make_client(FakeOpenAI(models=["llama3.1"]), "x") as c:
            return await preflight(c, "llama3.1:latest")

    assert asyncio.run(inner()) == "llama3.1"


def test_dead_endpoint_does_not_stall_teardown(tmp_path, monkeypatch):
    import bakeoff.runner as runner

    monkeypatch.setattr(runner, "REPORT_TIMEOUT_S", 0.05)

    async def hang(self, runtime):
        if getattr(self, "_started", False):
            await asyncio.sleep(30)
        self._started = True
        return {"version": None, "context_length": None, "size_vram": None}

    monkeypatch.setattr(EndpointClient, "runtime_info", hang)
    path = go(make_cfg(), FakeOpenAI(), tmp_path, run=fake_nvidia, platform="linux")
    meta = json.loads((path / "bundle.json").read_text())
    assert meta["memory"]["runtime_reported"]["size_vram"]["source"] == "not_captured"
