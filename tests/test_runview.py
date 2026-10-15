import pytest
from weekfix import make_bundle

from bakeoff.bundle import verify_bundle
from bakeoff.runview import RunViewError, ineligible_reasons, load_run, model_key

RULES = {"min_context": 16384, "min_decode_tps": 10.0}
G = {"general/gsm8k": [True, True, False, True], "structured/json": [True, False]}


def _mk(tmp_path, **kw):
    kw.setdefault("label", "Test-Model-A")
    kw.setdefault("weights", "Test-Model-A")
    kw.setdefault("envelope", "lab-box-7 24 GB")
    kw.setdefault("grades", G)
    return make_bundle(tmp_path, **kw)


def test_fixture_verifies(tmp_path):
    assert verify_bundle(_mk(tmp_path), None, None) == []


def test_load_run_basic(tmp_path):
    run = load_run(_mk(tmp_path))
    assert run.tracks["general"] == 0.75
    assert run.items["general/gsm8k"] == {
        "gsm8k-0": True,
        "gsm8k-1": True,
        "gsm8k-2": False,
        "gsm8k-3": True,
    }
    assert run.short_decode == 40.0 and run.max_context == 32768
    assert run.size_vram == 8_000_000_000
    assert run.harness["version"] == "0.1.0" and run.harness["git_commit"]
    assert run.suites["general/gsm8k"]["n"] == 4
    assert run.status == "complete" and run.bundle_format == "m2"
    assert model_key(run) == ("Test-Model-A", "Q4_K_M", "ollama")
    assert run.model["label"] == "Test-Model-A" and run.envelope == "lab-box-7 24 GB"
    assert ineligible_reasons(run, RULES) == []


def test_missing_speed_and_fit(tmp_path):
    run = load_run(_mk(tmp_path, speed=False, fit=False))
    assert run.short_decode is None and run.max_context is None
    assert ineligible_reasons(run, RULES) == ["no_fit", "no_speed"]


def test_incomplete_first(tmp_path):
    run = load_run(_mk(tmp_path, status="incomplete", speed=False))
    assert ineligible_reasons(run, RULES)[0] == "status_incomplete"


def test_thresholds(tmp_path):
    assert ineligible_reasons(load_run(_mk(tmp_path, decode=9.9)), RULES) == ["decode_below_min"]
    run = load_run(_mk(tmp_path, max_context=8192, label="B", weights="B"))
    assert ineligible_reasons(run, RULES) == ["context_below_min"]


def test_private_suite_from_private_dir(tmp_path):
    p = _mk(tmp_path, private=("general/gsm8k",))
    assert (p / "private" / "general.gsm8k.jsonl").is_file()
    run = load_run(p)
    assert run.suites["general/gsm8k"]["split"] == "private"
    assert len(run.items["general/gsm8k"]) == 4


def test_judge_suite_left_out(tmp_path):
    run = load_run(_mk(tmp_path, judge=("general/open-ended",)))
    assert "general/open-ended" in run.suites
    assert "general/open-ended" not in run.items


def test_m1_bundle(tmp_path):
    run = load_run(_mk(tmp_path, fmt="m1"))
    assert run.bundle_format == "m1"
    assert "extractable_accuracy" not in run.suites["structured/json"]


def test_m2_has_extractable(tmp_path):
    run = load_run(_mk(tmp_path))
    assert "extractable_accuracy" in run.suites["structured/json"]


def test_distinct_ids_and_hashes(tmp_path):
    a = load_run(_mk(tmp_path))
    b = load_run(_mk(tmp_path, label="Test-Model-B", weights="Test-Model-B"))
    assert a.run_id != b.run_id and a.bundle_sha256 != b.bundle_sha256


def test_missing_and_corrupt_bundle(tmp_path):
    with pytest.raises(RunViewError):
        load_run(tmp_path / "nope")
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "bundle.json").write_text("{not json")
    with pytest.raises(RunViewError):
        load_run(tmp_path / "bad")


def test_missing_item_file(tmp_path):
    p = _mk(tmp_path)
    (p / "quality" / "general.gsm8k.jsonl").unlink()
    with pytest.raises(RunViewError):
        load_run(p)
