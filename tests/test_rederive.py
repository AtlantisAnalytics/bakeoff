import json
import shutil
from pathlib import Path

import pytest
from weekfix import make_bundle

from bakeoff import cli
from bakeoff.bundle import write_bundle
from bakeoff.publish import publish
from bakeoff.rederive import rederive

REPO = Path(__file__).resolve().parents[1]
W1, W2 = "2026-10-19", "2026-10-26"
AT1, AT2 = "2026-10-19T07:00:00Z", "2026-10-26T07:00:00Z"
ENV24 = "1× RTX 3090 Ti 24 GB"
GSM = "general/gsm8k"


def g(k, n=10):
    return [True] * k + [False] * (n - k)


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "root"
    r.mkdir()
    for name in ("tiers.toml", "site.toml"):
        shutil.copy(REPO / name, r / name)
    return r


def bundle(root, weights, k, sub):
    return make_bundle(
        root / sub, label=weights, weights=weights, envelope=ENV24, grades={GSM: g(k)}
    )


def sha(path):
    return json.loads((path / "bundle.json").read_text(encoding="utf-8"))["bundle_sha256"]


def judge_bundle(root, name, a, b):
    meta = {
        "bundle_format": "judge-m1",
        "status": "complete",
        "status_reason": None,
        "a": a,
        "b": b,
        "suite": {"key": "general/chat", "hash": "c" * 64},
        "judge_model": "judge-x",
        "counts": {"a": 3, "b": 1, "tie": 1},
        "call_errors": 0,
    }
    out = root / "judges" / name
    write_bundle(out, meta, {"items.jsonl": [{"id": "chat-0", "result": "a"}]})
    return out


def write_week(root, name, week, at, runs, *, judges=(), extra="", tier_of=None):
    lines = [f'week = "{week}"', f'published_at = "{at}"', ""]
    for path in runs:
        lines += [
            "[[run]]",
            f'bundle = "{path.relative_to(root).as_posix()}"',
            f'tier = "{(tier_of or {}).get(path, "24gb")}"',
            'license = "Apache-2.0"',
            'source_url = "https://example.org/model"',
            "",
        ]
    for path in judges:
        lines += ["[[judge]]", f'bundle = "{path.relative_to(root).as_posix()}"', ""]
    out = root / name
    out.write_text("\n".join(lines) + extra, encoding="utf-8")
    return out


def pub(toml, root):
    publish(toml, root, username=None, denylist=None)


def two_weeks(root):
    """Week 1: two runs. Week 2: a re-run of the pick, a challenger with an override, a judge."""
    a, b = bundle(root, "base", 6, "runs"), bundle(root, "small", 4, "runs")
    pub(write_week(root, "w1.toml", W1, AT1, [a, b]), root)
    rerun, worse = bundle(root, "base", 6, "runs2"), bundle(root, "worse", 3, "runs2")
    j = judge_bundle(root, "judge-chat-1", sha(rerun), sha(worse))
    override = (
        f'[[override]]\ntier = "24gb"\ntrack = "general"\nrun = "{worse.name}"\n'
        'verdict = "watch"\nreason = "Owner wants another look \\"soon\\"."\n'
    )
    pub(write_week(root, "w2.toml", W2, AT2, [rerun, worse], judges=[j], extra=override), root)
    for raw in ("runs", "runs2", "judges"):  # rederive must only need results/
        shutil.rmtree(root / raw)
    return a, b


def test_rederive_two_weeks_identical(root, capsys):
    two_weeks(root)
    assert rederive(root, W1) == []
    assert rederive(root, W2) == []
    assert cli.main(["rederive", W1, "--root", str(root)]) == 0
    assert cli.main(["rederive", W2, "--root", str(root)]) == 0
    assert capsys.readouterr().out.splitlines() == ["identical", "identical"]
    doc = json.loads((root / "results" / W2 / "week.json").read_text(encoding="utf-8"))
    tier = next(t for t in doc["tiers"] if t["id"] == "24gb")
    assert tier["judge"] and any(
        v.get("override_reason") for r in tier["runs"] for v in r["verdicts"].values()
    )


def test_rederive_leaves_results_untouched(root):
    two_weeks(root)
    before = {p: p.read_bytes() for p in (root / "results").rglob("*") if p.is_file()}
    rederive(root, W2)
    after = {p: p.read_bytes() for p in (root / "results").rglob("*") if p.is_file()}
    assert after == before


def test_rederive_detects_tampered_week_json(root, capsys):
    two_weeks(root)
    path = root / "results" / W2 / "week.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["tiers"][1]["runs"][0]["model"]["label"] = "tampered"
    path.write_text(
        json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    problems = rederive(root, W2)
    assert any("label" in p for p in problems)
    assert cli.main(["rederive", W2, "--root", str(root)]) == 1
    out = capsys.readouterr().out
    assert "differs" in out and "label" in out


def test_rederive_detects_byte_level_change(root, capsys):
    two_weeks(root)
    path = root / "results" / W1 / "week.json"
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    assert rederive(root, W1) != []


def test_rederive_with_a_tier_added_later(root):
    two_weeks(root)
    with (root / "tiers.toml").open("a", encoding="utf-8") as f:
        f.write('\n[[tier]]\nid = "48gb"\nname = "48 GB"\nenvelope = "2× RTX 3090 Ti 24 GB"\n')
    assert rederive(root, W1) == []
    assert rederive(root, W2) == []


def test_rederive_after_a_coming_tier_launches(root):
    """`coming` is a fact of the week it was published in, not of today's tiers.toml."""
    two_weeks(root)
    doc = json.loads((root / "results" / W1 / "week.json").read_text(encoding="utf-8"))
    assert next(t for t in doc["tiers"] if t["id"] == "36gb-unified")["coming"] is True
    path = root / "tiers.toml"
    text = path.read_text(encoding="utf-8")
    assert "coming = true\n" in text
    path.write_text(text.replace("coming = true\n", ""), encoding="utf-8")
    assert rederive(root, W1) == []
    assert rederive(root, W2) == []


def test_rederive_correction(root):
    a, b = bundle(root, "base", 6, "runs"), bundle(root, "small", 4, "runs")
    pub(write_week(root, "w1.toml", W1, AT1, [a, b]), root)
    rerun = bundle(root, "base", 6, "runs2")
    pub(write_week(root, "w2.toml", W2, AT2, [rerun]), root)
    note = '\n[correction]\nnote = "Note only."\n'
    pub(write_week(root, "w1c.toml", W1, "2026-10-27T08:00:00Z", [a, b], extra=note), root)
    note2 = '\n[correction]\nnote = "Second note."\n'
    pub(write_week(root, "w1d.toml", W1, "2026-10-28T08:00:00Z", [a, b], extra=note2), root)
    doc = json.loads((root / "results" / W1 / "week.json").read_text(encoding="utf-8"))
    assert doc["revision"] == 3 and len(doc["corrections"]) == 2
    assert rederive(root, W1) == []
    assert rederive(root, W2) == []


def test_rederive_correction_detects_tampered_note(root):
    a, b = bundle(root, "base", 6, "runs"), bundle(root, "small", 4, "runs")
    pub(write_week(root, "w1.toml", W1, AT1, [a, b]), root)
    note = '\n[correction]\nnote = "Note only."\n'
    pub(write_week(root, "w1c.toml", W1, "2026-10-20T08:00:00Z", [a, b], extra=note), root)
    path = root / "results" / W1 / "week.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["corrections"][-1]["note"] = "Something else."
    path.write_text(
        json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    assert any("note" in p for p in rederive(root, W1))


def test_rederive_usage_and_missing(root, capsys):
    assert cli.main(["rederive", "not-a-week", "--root", str(root)]) == 2
    assert cli.main(["rederive", W1, "--root", str(root)]) == 1
    assert "error:" in capsys.readouterr().err


def test_rederive_missing_bundle_is_an_error(root, capsys):
    two_weeks(root)
    shutil.rmtree(next((root / "results" / W1 / "runs").iterdir()))
    assert cli.main(["rederive", W1, "--root", str(root)]) == 1
    assert "error:" in capsys.readouterr().err


def test_rederive_with_agreement(root):
    env12 = "1× RTX 3060 12 GB"
    a = make_bundle(root / "runs", label="m", weights="m", envelope=env12, grades={GSM: g(6)},
                    outputs={GSM: [f"o{i}" for i in range(10)]}, run_id="r-12")  # fmt: skip
    b = make_bundle(root / "runs", label="m", weights="m", envelope=ENV24, grades={GSM: g(7)},
                    outputs={GSM: [f"o{i}" for i in range(10)]}, run_id="r-24")  # fmt: skip
    pub(write_week(root, "w.toml", W1, AT1, [a, b], tier_of={a: "12gb", b: "24gb"}), root)
    doc = json.loads((root / "results" / W1 / "week.json").read_text(encoding="utf-8"))
    counts = {"n": 10, "same_pass": 9, "same_output": 10}
    assert doc["quality_agreement"][0]["suites"][GSM] == counts
    shutil.rmtree(root / "runs")
    assert rederive(root, W1) == []
