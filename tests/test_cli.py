import json

import pytest
from fake_openai import FakeOpenAI

from bakeoff import cli
from bakeoff.client import EndpointClient

CONFIG = """
[endpoint]
base_url = "http://lab-box-7.lan:11434/v1"  # scrub-ok
model = "served-model-name"

[model]
label = "Demo"
weights = "hf:demo/demo@rev"
runtime = "other"

[quality]
suites = ["format/exact-reply"]

[speed]
concurrency = [1]
repeats = 1
warmup = 0

[fit]
steps = [2048]
"""  # scrub-ok


@pytest.fixture
def cfg_file(tmp_path):
    p = tmp_path / "run.toml"
    p.write_text(CONFIG)
    return p


def patch_client(monkeypatch, fake):
    monkeypatch.setattr(
        cli,
        "make_client",
        lambda cfg: EndpointClient(
            cfg.endpoint.base_url, cfg.endpoint.model, transport=fake.transport, backoff_s=(0, 0)
        ),
    )


def test_cli_run_exit_codes(tmp_path, monkeypatch, cfg_file, capsys):
    out = tmp_path / "out"
    patch_client(monkeypatch, FakeOpenAI())
    assert cli.main(["run", str(cfg_file), "--out", str(out), "--only", "speed"]) == 0
    patch_client(monkeypatch, FakeOpenAI(fail_first=1000))
    assert cli.main(["run", str(cfg_file), "--out", str(out)]) == 2
    bad = tmp_path / "bad.toml"
    bad.write_text("[endpoint]\n")
    assert cli.main(["run", str(bad), "--out", str(out)]) == 1
    patch_client(monkeypatch, FakeOpenAI(models=["nope"]))
    assert cli.main(["run", str(cfg_file), "--out", str(out)]) == 1


def test_cli_run_interrupt_is_130(tmp_path, monkeypatch, cfg_file):
    patch_client(monkeypatch, FakeOpenAI())

    async def boom(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "execute_run", boom)
    assert cli.main(["run", str(cfg_file), "--out", str(tmp_path)]) == 130


def test_cli_verify(tmp_path, monkeypatch, cfg_file, capsys):
    patch_client(monkeypatch, FakeOpenAI())
    out = tmp_path / "out"
    assert cli.main(["run", str(cfg_file), "--out", str(out)]) == 0
    run_dir = next(out.iterdir())
    assert cli.main(["verify", str(run_dir)]) == 0
    capsys.readouterr()
    with (run_dir / "speed.jsonl").open("a") as f:
        f.write("{}\n")
    assert cli.main(["verify", str(run_dir)]) == 1
    assert "hash_mismatch" in capsys.readouterr().out


def test_cli_env_prints_json(capsys):
    assert cli.main(["env"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert {"host", "gpus", "runtime", "model", "harness"} <= set(data)


def test_cli_env_with_config(capsys, monkeypatch, cfg_file):
    patch_client(monkeypatch, FakeOpenAI())
    assert cli.main(["env", "--config", str(cfg_file)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["runtime"]["served_model"]["value"] == "served-model-name"
    assert "lab-box-7" not in json.dumps(data)


def test_make_client_reads_key_from_env(monkeypatch, cfg_file):
    from bakeoff.config import load_config

    monkeypatch.setenv("MY_KEY", "sekret")
    cfg = load_config(cfg_file, ["endpoint.api_key_env=MY_KEY"])
    client = cli.make_client(cfg)
    assert client._http.headers["authorization"] == "Bearer sekret"
