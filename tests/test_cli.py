import json

import pytest
from fake_openai import FakeOpenAI
from weekfix import make_bundle

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
steps = [4096]
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


def test_cli_unexpected_error_is_one_line(tmp_path, monkeypatch, cfg_file, capsys):
    patch_client(monkeypatch, FakeOpenAI())

    async def boom(*a, **k):
        raise RuntimeError("kaput")

    monkeypatch.setattr(cli, "execute_run", boom)
    monkeypatch.delenv("BAKEOFF_DEBUG", raising=False)
    assert cli.main(["run", str(cfg_file), "--out", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "kaput" in err and "Traceback" not in err and len(err.strip().splitlines()) == 1
    monkeypatch.setenv("BAKEOFF_DEBUG", "1")
    with pytest.raises(RuntimeError):
        cli.main(["run", str(cfg_file), "--out", str(tmp_path)])


def test_cli_env_without_config_never_makes_client(monkeypatch, capsys):
    def boom(cfg):
        raise AssertionError("make_client must not be called")

    monkeypatch.setattr(cli, "make_client", boom)
    assert cli.main(["env"]) == 0


def _planted_bundle(tmp_path, text):
    from bakeoff.bundle import write_bundle

    run = tmp_path / "planted"
    write_bundle(run, {"status": "complete"}, {"speed.jsonl": [{"a": text}]})
    return run


@pytest.mark.parametrize(
    "name", ["root", "ROOT", "ubuntu", "ec2-user", "nobody", "pi", "bob", "ab"]
)
def test_generic_and_short_usernames_are_skipped(monkeypatch, name):
    monkeypatch.setattr(cli.getpass, "getuser", lambda: name)
    assert cli.pick_username(False, lambda: "") == (None, None, False)


def test_getuser_failure_skips_the_check(monkeypatch):
    def boom():
        raise KeyError("uid not found")

    monkeypatch.setattr(cli.getpass, "getuser", boom)
    assert cli.pick_username(False, lambda: "") == (None, None, False)


def test_pick_username_returns_skip_output(monkeypatch):
    monkeypatch.setattr(cli.getpass, "getuser", lambda: "natalia")
    name, notice, skip = cli.pick_username(False, lambda: "Natalia sold clips to 48 of her friends")
    assert name == "natalia" and skip is True
    assert "natalia" in notice and "public suites" in notice


def test_distinctive_username_is_scanned(monkeypatch):
    monkeypatch.setattr(cli.getpass, "getuser", lambda: "alice-dev")
    assert cli.pick_username(False, lambda: "square root of Natalia") == ("alice-dev", None, False)
    assert cli.pick_username(True) == (None, None, False)


def test_real_suites_text_is_consulted(monkeypatch):
    import re

    text = cli._public_suite_text()
    word = next(w for w in re.findall(r"[A-Za-z]{6,}", text))
    monkeypatch.setattr(cli.getpass, "getuser", lambda: word)
    name, notice, skip = cli.pick_username(False)
    assert name == word and notice and skip is True


def test_verify_username_flag_and_notice(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli.getpass, "getuser", lambda: "alice-dev")  # scrub-ok
    run = _planted_bundle(tmp_path, "built by alice-dev")  # scrub-ok
    assert cli.main(["verify", str(run), "--denylist", str(tmp_path / "none")]) == 1
    assert "username" in capsys.readouterr().out
    argv = ["verify", str(run), "--denylist", str(tmp_path / "none"), "--no-username"]
    assert cli.main(argv) == 0


def test_verify_notes_private_outputs(tmp_path, monkeypatch, capsys):
    from bakeoff.bundle import write_bundle

    monkeypatch.setattr(cli.getpass, "getuser", lambda: "root")
    run = tmp_path / "r"
    write_bundle(run, {"status": "complete"}, {"private/x.y.jsonl": [{"a": 1}]})
    assert cli.main(["verify", str(run), "--denylist", str(tmp_path / "none")]) == 0
    assert "private-suite outputs; do not share" in capsys.readouterr().err
    clean = _planted_bundle(tmp_path, "fine")
    cli.main(["verify", str(clean), "--denylist", str(tmp_path / "none")])
    assert "private" not in capsys.readouterr().err


def test_verify_does_not_fail_on_root_user_and_ordinary_text(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.getpass, "getuser", lambda: "root")
    run = _planted_bundle(tmp_path, "the square root of 144 is 12")
    assert cli.main(["verify", str(run), "--denylist", str(tmp_path / "none")]) == 0


def test_interrupt_message_claims_a_bundle_only_if_one_was_written(
    tmp_path, monkeypatch, cfg_file, capsys
):
    patch_client(monkeypatch, FakeOpenAI())

    async def boom(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "execute_run", boom)
    assert cli.main(["run", str(cfg_file), "--out", str(tmp_path)]) == 130
    err = capsys.readouterr().err
    assert "no bundle was written" in err and "incomplete bundle was written" not in err

    async def wrote(*a, **k):
        e = KeyboardInterrupt()
        e.bundle_written = True
        raise e

    monkeypatch.setattr(cli, "execute_run", wrote)
    assert cli.main(["run", str(cfg_file), "--out", str(tmp_path)]) == 130
    assert "an incomplete bundle was written" in capsys.readouterr().err


def test_real_interrupt_mid_run_writes_a_bundle_and_says_so(
    tmp_path, monkeypatch, cfg_file, capsys
):
    import asyncio

    def responder(messages):
        raise asyncio.CancelledError

    patch_client(monkeypatch, FakeOpenAI(responder=responder))
    assert cli.main(["run", str(cfg_file), "--out", str(tmp_path / "o")]) == 130
    assert "an incomplete bundle was written" in capsys.readouterr().err
    assert len(list((tmp_path / "o").iterdir())) == 1


def test_existing_run_dir_is_a_clean_cli_error(tmp_path, monkeypatch, cfg_file, capsys):
    patch_client(monkeypatch, FakeOpenAI())

    async def exists(*a, **k):
        raise FileExistsError("refusing to write into an existing, non-empty run")

    monkeypatch.setattr(cli, "execute_run", exists)
    assert cli.main(["run", str(cfg_file), "--out", str(tmp_path)]) == 1
    assert "refusing to write" in capsys.readouterr().err


def test_public_suite_text_survives_any_error(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "resolve_suites", boom)
    assert cli._public_suite_text() == ""


# --- publish and site ---------------------------------------------------------------------


def _site_root(tmp_path, launched=False):
    import re
    import shutil
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    r = tmp_path / "root"
    r.mkdir()
    for name in ("tiers.toml", "site.toml"):
        shutil.copy(repo / name, r / name)
    shutil.copytree(repo / "site_src", r / "site_src")
    shutil.copytree(repo / "site", r / "site")
    t = (r / "site.toml").read_text(encoding="utf-8")
    t = re.sub(r"(?m)^launched = (true|false)$", f"launched = {str(launched).lower()}", t)
    (r / "site.toml").write_text(t, encoding="utf-8")
    return r


def test_cli_site_write_check_preview(tmp_path, capsys):
    r = _site_root(tmp_path)
    assert cli.main(["site", "--root", str(r)]) == 0
    assert cli.main(["site", "--root", str(r), "--check"]) == 0
    (r / "site" / "index.html").write_text("edited", encoding="utf-8")
    assert cli.main(["site", "--root", str(r), "--check"]) == 1
    assert "changed: index.html" in capsys.readouterr().out
    prev = tmp_path / "prev"
    assert cli.main(["site", "--root", str(r), "--preview", str(prev)]) == 0
    assert (prev / "feed.xml").is_file() and (prev / "api/v1/picks.json").is_file()
    for asset in ("style.css", "logo.webp", "favicon-32.png", "apple-touch-icon.png", "og.jpg"):
        assert (prev / asset).is_file()
    assert cli.main(["site", "--root", str(r), "--preview", str(r / "site")]) == 2
    assert cli.main(["site", "--root", str(r), "--preview", str(r / "site" / "sub")]) == 2
    assert not (r / "site" / "sub").exists()


def test_cli_site_bad_source_is_one_line(tmp_path, capsys):
    r = _site_root(tmp_path)
    (r / "site_src" / "method.html").write_bytes(b"\xff\xfe bad")
    assert cli.main(["site", "--root", str(r)]) == 1
    assert capsys.readouterr().err.startswith("error:")


def test_cli_site_usage_errors(tmp_path):
    with pytest.raises(SystemExit) as e:
        cli.main(["site", "--check", "--preview", str(tmp_path)])
    assert e.value.code == 2


def test_cli_publish_end_to_end(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "DEFAULT_DENYLIST", tmp_path / "no-such-denylist")
    r = _site_root(tmp_path, launched=True)
    g = [True] * 6 + [False] * 4
    b = make_bundle(
        r / "runs",
        label="m",
        weights="m",
        envelope="1× RTX 3090 Ti 24 GB",
        grades={"general/gsm8k": g},
    )
    toml = r / "w.toml"
    toml.write_text(
        'week = "2026-10-19"\npublished_at = "2026-10-19T07:00:00Z"\n\n[[run]]\n'
        f'bundle = "{b.relative_to(r).as_posix()}"\ntier = "24gb"\nlicense = "Apache-2.0"\n'
        'source_url = "https://example.org/m"\n',
        encoding="utf-8",
    )
    assert cli.main(["publish", str(toml), "--root", str(r), "--no-username"]) == 0
    assert (r / "site" / "weeks" / "2026-10-19" / "index.html").is_file()
    assert cli.main(["site", "--root", str(r), "--check"]) == 0
    # a second publish of the same week is refused with exit 1
    assert cli.main(["publish", str(toml), "--root", str(r), "--no-username"]) == 1
    assert "already published" in capsys.readouterr().err
    # an explicit missing denylist is an error, not silently ignored
    assert (
        cli.main(["publish", str(toml), "--root", str(r), "--denylist", str(tmp_path / "nope")])
        == 1
    )


def test_run_to_publish_end_to_end(tmp_path, monkeypatch):
    """A bundle the runner writes against a fake vLLM server publishes, and its KV pool and
    verdict reach week.json and the week page."""
    import dataclasses
    import json

    from fake_openai import FakeOpenAI, FakeReply, default_responder
    from test_runner import fake_nvidia, go, make_cfg

    from bakeoff.config import FitConfig

    monkeypatch.setattr(cli, "DEFAULT_DENYLIST", tmp_path / "no-such-denylist")
    r = _site_root(tmp_path, launched=True)
    cfg = make_cfg()
    cfg = dataclasses.replace(
        cfg,
        model=dataclasses.replace(cfg.model, runtime="vllm", envelope="1× RTX 3090 Ti 24 GB"),
        fit=FitConfig(steps=[16384, 32768]),
    )
    cache = (
        'vllm:cache_config_info{block_size="16",cache_dtype="auto",engine="0",'
        'num_gpu_blocks="10000"} 1.0\n'
    )

    def responder(messages):  # needles answered; otherwise a few streamed words, so decode is timed
        base = default_responder(messages)
        words = base.content if base.content != "ok" else "one two three four five six"
        return FakeReply(content=words, chunk_delay_s=0.002)

    fake = FakeOpenAI(responder=responder, extras={"/metrics": cache})
    bundle = go(cfg, fake, r / "runs", run=fake_nvidia, platform="linux")
    toml = r / "w.toml"
    toml.write_text(
        'week = "2026-10-19"\npublished_at = "2026-10-19T07:00:00Z"\n\n[[run]]\n'
        f'bundle = "{bundle.relative_to(r).as_posix()}"\ntier = "24gb"\nlicense = "Apache-2.0"\n'
        'source_url = "https://example.org/m"\n',
        encoding="utf-8",
    )
    assert cli.main(["publish", str(toml), "--root", str(r), "--no-username"]) == 0
    doc = json.loads((r / "results" / "2026-10-19" / "week.json").read_text(encoding="utf-8"))
    [run] = next(t for t in doc["tiers"] if t["id"] == "24gb")["runs"]
    assert run["fit"]["kv_capacity_tokens"] == 160000
    assert run["fit"]["kv_fraction_at_max"] == 0.2048
    assert run["eligible"] and run["verdicts"], run["ineligible_reasons"]
    page = (r / "site" / "weeks" / "2026-10-19" / "index.html").read_text(encoding="utf-8")
    assert "KV pool 156K · 20.5% at max" in page
    assert cli.main(["site", "--root", str(r), "--check"]) == 0
