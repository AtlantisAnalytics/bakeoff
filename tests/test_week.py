import copy
import json

import pytest
from weekfix import make_bundle

from bakeoff.graders import schema_problems
from bakeoff.runview import load_run
from bakeoff.verdicts import RULES_V1, assess_tier
from bakeoff.week import build_week, load_week_schema, validate_week
from bakeoff.weekcfg import RunEntry, Tier, WeekInput

RULES = {**RULES_V1, "resamples": 500}
WEEK = "2026-10-19"
AT = "2026-10-19T07:00:00Z"
ENV_A = "lab-box-7 24 GB"
ENV_B = "lab-box-7 36 GB unified"
GSM = "general/gsm8k"
JSN = "structured/json"


def g(k, n=50):
    return [True] * k + [False] * (n - k)


def pick(track, weights):
    return {
        "tier": "24gb",
        "track": track,
        "label": weights,
        "weights": weights,
        "quant": "Q4_K_M",
        "runtime": "ollama",
        "since_week": "2026-10-12",
        "bundle_sha256": "0" * 64,
        "run_id": "old-run",
    }


@pytest.fixture
def week(tmp_path):
    def mk(rid, weights, env, tier, grades, **kw):
        path = make_bundle(
            tmp_path, label=weights, weights=weights, envelope=env, grades=grades, run_id=rid, **kw
        )
        entry = RunEntry(
            bundle=path,
            tier=tier,
            license="Apache-2.0",
            source_url="https://example.org/" + weights,
            license_url=None,
        )
        return load_run(path), entry

    both = {GSM: g(20), JSN: g(30)}
    made = [
        mk("r-ctrl", "base", ENV_A, "24gb", {GSM: g(20)}),
        mk("r-cand", "big", ENV_A, "24gb", {GSM: g(48)}, harness_version="0.2.0"),
        mk("r-slow", "slow", ENV_A, "24gb", {GSM: g(30)}, decode=3.0),
        mk("r-solo", "solo", ENV_B, "36gb-unified", both),
        mk("r-bare", "bare", ENV_B, "36gb-unified", both, speed=False, fit=False, size_vram=None),
    ]
    made[3] = (
        made[3][0],
        RunEntry(
            bundle=made[3][1].bundle,
            tier="36gb-unified",
            license="other",
            source_url="https://example.org/solo",
            license_url="https://example.org/solo/license",
        ),
    )
    runs = {v.run_id: (v, e) for v, e in made}
    tiers = {
        "24gb": Tier("24gb", "24 GB", ENV_A),
        "36gb-unified": Tier("36gb-unified", "36 GB unified", ENV_B),
    }
    results = {}
    for tid, picks in (("24gb", {"general": pick("general", "base")}), ("36gb-unified", {})):
        rs = [v for v, e in runs.values() if e.tier == tid]
        results[tid] = assess_tier(WEEK, tid, rs, picks, [], RULES)
    inp = WeekInput(WEEK, AT, [e for _, e in made], [], [], None, b"")
    judges = [
        {
            "a": "r-ctrl",
            "b": "r-cand",
            "wins": 3,
            "ties": 1,
            "losses": 6,
            "errors": 0,
            "judge_model": "judge-x",
            "bundle_sha256": "a" * 64,
        }
    ]
    return inp, tiers, runs, results, judges


def build(week, **kw):
    inp, tiers, runs, results, judges = week
    args = dict(revision=1, published_at=AT, corrections=[], rules=RULES)
    args.update(kw)
    return build_week(inp, tiers, runs, results, judges, **args)


def test_build_validates_and_has_core_fields(week):
    doc = build(week)
    assert validate_week(doc) == []
    assert doc["schema"] == "bakeoff.week.v1"
    assert (doc["week"], doc["revision"], doc["published_at"]) == (WEEK, 1, AT)
    assert doc["rules"] == RULES
    assert [t["id"] for t in doc["tiers"]] == ["24gb", "36gb-unified"]
    assert [r["run_id"] for r in doc["tiers"][0]["runs"]] == ["r-cand", "r-ctrl", "r-slow"]


def test_corrections_copied(week):
    c = [{"revision": 2, "published_at": AT, "note": "regraded"}]
    doc = build(week, revision=2, corrections=c)
    assert doc["corrections"] == c and doc["revision"] == 2
    assert validate_week(doc) == []


def test_harness_distinct_sorted(week):
    doc = build(week)
    commit = "0123456789abcdef0123456789abcdef01234567"
    assert doc["harness"] == [
        {"version": "0.1.0", "git_commit": commit},
        {"version": "0.2.0", "git_commit": commit},
    ]


def test_roles_verdicts_and_pick_fields(week):
    doc = build(week)
    t = {r["run_id"]: r for r in doc["tiers"][0]["runs"]}
    assert t["r-ctrl"]["role"] == "pick" and t["r-ctrl"]["pick_tracks"] == ["general"]
    assert t["r-ctrl"]["verdicts"] == {}
    assert t["r-cand"]["role"] == "candidate" and t["r-cand"]["pick_tracks"] == []
    v = t["r-cand"]["verdicts"]["general"]
    assert v["computed"] == v["final"] == "replace" and v["new_pick"] is True
    assert "override_reason" not in v and "outranked_by" not in v
    assert t["r-slow"]["eligible"] is False
    assert t["r-slow"]["ineligible_reasons"] == ["decode_below_min"]
    assert t["r-slow"]["verdicts"] == {}
    assert doc["tiers"][0]["pick_status"] == {"general": "re-measured"}
    assert doc["picks_after"]["24gb"]["general"]["run_id"] == "r-cand"
    assert doc["picks_after"]["36gb-unified"]["general"]["run_id"] == "r-solo"
    assert doc["tiers"][1]["pick_status"] == {"general": "none", "structured": "none"}
    solo = doc["tiers"][1]["runs"][1]
    assert solo["verdicts"]["general"]["computed"] == "initial_pick"


def test_run_details(week):
    inp, tiers, runs, results, judges = week
    doc = build(week)
    cand = {r["run_id"]: r for r in doc["tiers"][0]["runs"]}["r-cand"]
    view = runs["r-cand"][0]
    assert cand["bundle_path"] == f"results/{WEEK}/runs/r-cand"
    assert cand["bundle_sha256"] == view.bundle_sha256
    assert cand["model"] == {
        "label": "big",
        "weights": "big",
        "quant": "Q4_K_M",
        "runtime": "ollama",
        "license": "Apache-2.0",
        "source_url": "https://example.org/big",
    }
    assert cand["fit"] == {"max_context_passed": 32768, "server_context": 65536}
    assert cand["memory"] == {"size_vram": 8_000_000_000}
    q = cand["quality"]["tracks"]["general"]
    assert q["score"] == view.tracks["general"] == 0.96  # unrounded, straight from the bundle
    s = q["suites"][0]
    assert (s["key"], s["version"], s["n"], s["passed"]) == (GSM, 1, 50, 48)
    assert s["hash"] == view.suites[GSM]["hash"] and "extractable_accuracy" not in s
    assert cand["speed"]["short@1"]["decode_tps"] == {
        "median": 40.0,
        "min": 40.0,
        "max": 40.0,
        "n": 5,
    }
    bare, solo = doc["tiers"][1]["runs"]
    assert bare["memory"] == {} and bare["speed"] == {}
    assert bare["fit"] == {"max_context_passed": None, "server_context": None}
    assert bare["ineligible_reasons"] == ["no_fit", "no_speed"]
    assert solo["model"]["license_url"] == "https://example.org/solo/license"
    js = {s["key"]: s for s in solo["quality"]["tracks"]["structured"]["suites"]}[JSN]
    assert js["extractable_accuracy"] == 0.6 and len(js["extractable_ci95"]) == 2


def test_floats_unrounded(week):
    inp, tiers, runs, results, judges = week
    doc = build(week)
    v = results["24gb"].verdicts["r-cand"]["general"]
    out = {r["run_id"]: r for r in doc["tiers"][0]["runs"]}["r-cand"]["verdicts"]["general"]
    assert out["diff"] == v.diff and out["diff_ci95"] == v.diff_ci95
    assert json.loads(json.dumps(doc)) == doc


def test_judges_attach_to_tier(week):
    doc = build(week)
    assert doc["tiers"][0]["judge"] == week[4]
    assert doc["tiers"][1]["judge"] == []


def test_judge_with_unknown_or_cross_tier_run_rejected(week):
    bad = copy.deepcopy(week[4])
    bad[0]["b"] = "r-solo"
    with pytest.raises(ValueError, match="tier"):
        build_week(*week[:4], bad, revision=1, published_at=AT, corrections=[], rules=RULES)
    bad[0]["b"] = "nope"
    with pytest.raises(ValueError, match="nope"):
        build_week(*week[:4], bad, revision=1, published_at=AT, corrections=[], rules=RULES)


def test_deterministic(week):
    assert json.dumps(build(week), sort_keys=True) == json.dumps(build(week), sort_keys=True)


def test_wrong_verdict_enum_fails(week):
    doc = build(week)
    doc["tiers"][0]["runs"][0]["verdicts"]["general"]["final"] = "maybe"
    errs = validate_week(doc)
    assert errs and "maybe" in errs[0]


def test_extra_and_missing_keys_fail(week):
    doc = build(week)
    doc["surprise"] = 1
    assert any("surprise" in e for e in validate_week(doc))
    doc = build(week)
    del doc["tiers"][0]["runs"][0]["eligible"]
    assert any("eligible" in e for e in validate_week(doc))


def test_schema_is_inside_the_checker_subset():
    schema = load_week_schema()
    assert schema_problems(schema) == []
    assert schema["$id"] == "https://bakeoff.si/api/v1/schema/week.v1.json"


def _objects(node, path="$"):
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            yield path, node
        for k, v in node.items():
            if k == "properties":
                for pk, pv in v.items():
                    yield from _objects(pv, f"{path}.{pk}")
            elif k in ("items", "additionalProperties"):
                yield from _objects(v, f"{path}.{k}")


def test_every_fixed_key_object_is_closed():
    nodes = list(_objects(load_week_schema()))
    assert len(nodes) > 10
    for path, node in nodes:
        if "properties" in node:
            assert node.get("additionalProperties") is False, path
            assert "required" in node, path
        else:  # a map: values must be described
            assert isinstance(node.get("additionalProperties"), dict), path


# --- KV fit fields ---------------------------------------------------------------------------

KV = {
    "capacity_tokens": {"value": 16000, "source": "reported"},
    "block_size": {"value": 16, "source": "reported"},
    "cache_dtype": {"value": "auto", "source": "reported"},
}


def _one_run(tmp_path, fit_extra):
    path = make_bundle(
        tmp_path, label="m", weights="m", envelope=ENV_A, grades={GSM: g(20)}, run_id="r-kv",
        max_context=4096, fit_extra=fit_extra,
    )  # fmt: skip
    entry = RunEntry(bundle=path, tier="24gb", license="MIT", source_url="https://example.org/m",
                     license_url=None)  # fmt: skip
    view = load_run(path)
    tiers = {"24gb": Tier("24gb", "24 GB", ENV_A)}
    results = {"24gb": assess_tier(WEEK, "24gb", [view], {}, [], RULES)}
    inp = WeekInput(WEEK, AT, [entry], [], [], None, b"")
    doc = build_week(inp, tiers, {view.run_id: (view, entry)}, results, [],
                     revision=1, published_at=AT, corrections=[], rules=RULES)  # fmt: skip
    return doc, doc["tiers"][0]["runs"][0]


def test_fit_kv_fields_present(tmp_path):
    steps = {"4096": {"status": "pass", "kv_fraction": 0.256}}
    doc, run = _one_run(tmp_path, {"kv": KV, "steps": steps})
    assert run["fit"] == {
        "max_context_passed": 4096,
        "server_context": 65536,
        "kv_capacity_tokens": 16000,
        "kv_cache_dtype": "auto",
        "kv_fraction_at_max": 0.256,
    }
    assert validate_week(doc) == []


def test_fit_kv_fields_absent_for_old_bundles(tmp_path):
    _, run = _one_run(tmp_path, None)
    assert run["fit"] == {"max_context_passed": 4096, "server_context": 65536}


def test_fit_kv_not_captured(tmp_path):
    none = {k: {"value": None, "source": "not_captured"} for k in KV}
    doc, run = _one_run(tmp_path, {"kv": none, "steps": {"4096": {"status": "pass"}}})
    assert run["fit"]["kv_capacity_tokens"] is None
    assert run["fit"]["kv_cache_dtype"] is None
    assert run["fit"]["kv_fraction_at_max"] is None
    assert validate_week(doc) == []


# --- quality_agreement -----------------------------------------------------------------------

from bakeoff.week import quality_agreement  # noqa: E402

IFE = "general/ifeval"
JUDGED = "general/open-ended"


def _runs(tmp_path, specs):
    """specs: (run_id, tier, weights, grades, outputs, judge)."""
    runs = {}
    for rid, tier, weights, grades, outputs, judge in specs:
        path = make_bundle(
            tmp_path, label=weights, weights=weights, envelope=ENV_A, grades=grades,
            outputs=outputs, judge=judge, run_id=rid,
        )  # fmt: skip
        entry = RunEntry(bundle=path, tier=tier, license="MIT",
                         source_url="https://example.org/m", license_url=None)  # fmt: skip
        runs[rid] = (load_run(path), entry)
    return runs


def _doc(runs):
    tiers = {t: Tier(t, t, ENV_A) for t in ("12gb", "24gb", "36gb")}
    results = {
        t: assess_tier(WEEK, t, [v for v, e in runs.values() if e.tier == t], {}, [], RULES)
        for t in tiers
    }
    inp = WeekInput(WEEK, AT, [e for _, e in runs.values()], [], [], None, b"")
    return build_week(inp, tiers, runs, results, [],
                      revision=1, published_at=AT, corrections=[], rules=RULES)  # fmt: skip


def test_agreement_two_tiers(tmp_path):
    runs = _runs(tmp_path, [
        ("r-a", "12gb", "m", {GSM: [True, True, False, True]}, {GSM: ["a", "b", "c", "d"]}, ()),
        ("r-b", "24gb", "m", {GSM: [True, True, True, True]}, {GSM: ["a", "B", "c", "D"]}, ()),
    ])  # fmt: skip
    doc = _doc(runs)
    assert doc["quality_agreement"] == [
        {
            "weights": "m",
            "quant": "Q4_K_M",
            "runtime": "ollama",
            "runs": ["r-a", "r-b"],
            "suites": {GSM: {"n": 4, "same_pass": 3, "same_output": 2}},
        }
    ]
    assert validate_week(doc) == []


def test_agreement_three_tiers_needs_all(tmp_path):
    runs = _runs(tmp_path, [
        ("r-a", "12gb", "m", {GSM: [True, False]}, {GSM: ["x", "y"]}, ()),
        ("r-b", "24gb", "m", {GSM: [True, False]}, {GSM: ["x", "y"]}, ()),
        ("r-c", "36gb", "m", {GSM: [True, True]}, {GSM: ["x", "z"]}, ()),
    ])  # fmt: skip
    [entry] = quality_agreement(runs)
    assert entry["runs"] == ["r-a", "r-b", "r-c"]
    assert entry["suites"][GSM] == {"n": 2, "same_pass": 1, "same_output": 1}


def test_agreement_uses_shared_items_only(tmp_path):
    runs = _runs(tmp_path, [
        ("r-a", "12gb", "m", {GSM: [True, True, True], IFE: [True]}, None, ()),
        ("r-b", "24gb", "m", {GSM: [True, True]}, None, ()),
    ])  # fmt: skip
    [entry] = quality_agreement(runs)
    assert entry["suites"] == {GSM: {"n": 2, "same_pass": 2, "same_output": 2}}


def test_agreement_skips_judged_and_single_tier(tmp_path):
    runs = _runs(tmp_path, [
        ("r-a", "12gb", "m", {GSM: [True]}, None, (JUDGED,)),
        ("r-b", "24gb", "m", {GSM: [True]}, None, (JUDGED,)),
        ("r-c", "24gb", "solo", {GSM: [True]}, None, ()),
        ("r-d", "24gb", "solo", {GSM: [False]}, None, ()),
    ])  # fmt: skip
    [entry] = quality_agreement(runs)
    assert entry["weights"] == "m" and set(entry["suites"]) == {GSM}
    only_one_tier = {k: v for k, v in runs.items() if k in ("r-c", "r-d")}
    assert quality_agreement(only_one_tier) == []
    assert "quality_agreement" not in _doc(only_one_tier)


def test_agreement_one_run_per_tier_lowest_id(tmp_path):
    runs = _runs(tmp_path, [
        ("r-a", "12gb", "m", {GSM: [True]}, None, ()),
        ("r-b", "24gb", "m", {GSM: [True]}, None, ()),
        ("r-c", "24gb", "m", {GSM: [False]}, None, ()),
    ])  # fmt: skip
    [entry] = quality_agreement(runs)
    assert entry["runs"] == ["r-a", "r-b"]
    assert entry["suites"][GSM]["same_pass"] == 1
