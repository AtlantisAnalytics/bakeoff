from pathlib import Path

import pytest

from bakeoff.config import RUNTIMES, ConfigError, load_config, public_config

REPO = Path(__file__).resolve().parents[1]

MINIMAL = """
[endpoint]
base_url = "http://192.0.2.10:8000/v1"  # scrub-ok
model = "served-model-name"
api_key_env = "SOME_KEY_ENV"

[model]
label = "Example"
weights = "hf:example-org/Example@abc"
runtime = "vllm"
"""


def write(tmp_path, text=MINIMAL):
    p = tmp_path / "run.toml"
    p.write_text(text)
    return p


def test_load_example_config():
    cfg = load_config(REPO / "examples" / "run.toml")
    assert cfg.model.runtime in RUNTIMES and cfg.speed.concurrency == [1, 4]
    assert cfg.fit.max_context == 0 and cfg.endpoint.timeout_s == 600.0


def test_defaults(tmp_path):
    cfg = load_config(write(tmp_path))
    assert cfg.speed.repeats == 5 and cfg.speed.warmup == 1
    assert cfg.fit.steps[0] == 4096 and cfg.quality.suites == []


def test_missing_required_key_raises(tmp_path):
    p = write(tmp_path, MINIMAL.replace('model = "served-model-name"\n', ""))
    with pytest.raises(ConfigError, match=r"endpoint\.model"):
        load_config(p)


def test_bad_runtime_raises(tmp_path):
    p = write(tmp_path, MINIMAL.replace('"vllm"', '"tgi"'))
    with pytest.raises(ConfigError, match="vllm") as ei:
        load_config(p)
    assert "tgi" in str(ei.value)


def test_unknown_key_raises(tmp_path):
    with pytest.raises(ConfigError, match="bogus"):
        load_config(write(tmp_path, MINIMAL + "bogus = 1\n"))


def test_overrides(tmp_path):
    p = write(tmp_path)
    cfg = load_config(p, ["speed.repeats=2", "endpoint.model=other", "fit.steps=[4096]"])
    assert cfg.speed.repeats == 2 and cfg.endpoint.model == "other" and cfg.fit.steps == [4096]


def test_bad_override_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_config(write(tmp_path), ["nonsense"])


def test_public_config_strips_endpoint_secrets(tmp_path):
    pub = public_config(load_config(write(tmp_path)))
    assert "base_url" not in pub["endpoint"] and "api_key_env" not in pub["endpoint"]
    assert pub["endpoint"]["model"] == "served-model-name"


@pytest.mark.parametrize(
    ("override", "key"),
    [
        ("fit.steps=[]", "fit.steps"),
        ("fit.steps=[0]", "fit.steps"),
        ("fit.steps=[true]", "fit.steps"),
        ("fit.steps=[1.5]", "fit.steps"),
        ("speed.concurrency=[]", "speed.concurrency"),
        ("speed.concurrency=[0]", "speed.concurrency"),
        ("speed.concurrency=[true]", "speed.concurrency"),
        ("speed.repeats=0", "speed.repeats"),
        ("speed.repeats=true", "speed.repeats"),
        ("speed.warmup=-1", "speed.warmup"),
        ("fit.max_context=-5", "fit.max_context"),
        ("endpoint.timeout_s=0", "endpoint.timeout_s"),
        ("endpoint.timeout_s=-3", "endpoint.timeout_s"),
        ("endpoint.timeout_s=nan", "endpoint.timeout_s"),
        ("speed.repeats=lots", "speed.repeats"),
    ],
)
def test_out_of_range_values_name_the_key(tmp_path, override, key):
    with pytest.raises(ConfigError, match=key.replace(".", r"\.")):
        load_config(write(tmp_path), [override])


def test_valid_edge_values_are_accepted(tmp_path):
    cfg = load_config(
        write(tmp_path), ["speed.warmup=0", "fit.max_context=0", "speed.concurrency=[1]"]
    )
    assert cfg.speed.warmup == 0 and cfg.speed.concurrency == [1]


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")


def test_invalid_toml_raises(tmp_path):
    with pytest.raises(ConfigError, match="invalid TOML"):
        load_config(write(tmp_path, "[endpoint\nbase_url = "))
