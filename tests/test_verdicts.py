import pytest
from weekfix import make_bundle

from bakeoff.runview import load_run
from bakeoff.verdicts import RULES_V1, TierResult, Verdict, VerdictError, assess_tier
from bakeoff.weekcfg import Override

RULES = {**RULES_V1, "resamples": 2000}
WEEK = "2026-10-19"
TIER = "24gb"
ENV = "lab-box-7 24 GB"
GSM = "general/gsm8k"


def g(k: int, n: int = 100) -> list[bool]:
    return [True] * k + [False] * (n - k)


def run(tmp_path, rid, weights, grades=None, **kw):
    kw.setdefault("envelope", ENV)
    grades = {GSM: g(60)} if grades is None else grades
    return load_run(
        make_bundle(tmp_path, label=weights, weights=weights, grades=grades, run_id=rid, **kw)
    )


def pick(track, weights, since="2026-10-12", quant="Q4_K_M", runtime="ollama"):
    return {
        "tier": TIER,
        "track": track,
        "label": weights,
        "weights": weights,
        "quant": quant,
        "runtime": runtime,
        "since_week": since,
        "bundle_sha256": "0" * 64,
        "run_id": "old-run",
    }


def assess(runs, picks, overrides=()):
    return assess_tier(WEEK, TIER, runs, picks, list(overrides), RULES)


def test_rules_v1_values():
    assert RULES_V1 == {
        "version": 1,
        "min_context": 16384,
        "min_decode_tps": 10.0,
        "speed_ratio": 0.8,
        "context_cap": 32768,
        "resamples": 10000,
        "ci": 0.95,
    }
    assert issubclass(VerdictError, ValueError)


def test_clear_win_replace(tmp_path):
    p = run(tmp_path, "run-p", "Pick", {GSM: g(60)})
    c = run(tmp_path, "run-c", "Cand", {GSM: g(90)}, decode=38.0)
    res = assess([p, c], {"general": pick("general", "Pick")})
    assert isinstance(res, TierResult)
    v = res.verdicts["run-c"]["general"]
    assert isinstance(v, Verdict)
    assert (v.computed, v.final, v.new_pick, v.outranked_by) == ("replace", "replace", True, None)
    assert v.override_reason is None
    assert v.reasons == []
    assert v.diff == pytest.approx(0.3)
    assert isinstance(v.diff_ci95, list) and len(v.diff_ci95) == 2
    assert 0 < v.diff_ci95[0] < 0.3 < v.diff_ci95[1]
    assert res.pick_tracks == {"run-p": ["general"]}
    assert res.pick_status == {"general": "re-measured"}
    assert res.verdicts["run-p"] == {}
    after = res.picks_after["general"]
    assert after == {
        "tier": TIER,
        "track": "general",
        "label": "Cand",
        "weights": "Cand",
        "quant": "Q4_K_M",
        "runtime": "ollama",
        "since_week": WEEK,
        "bundle_sha256": c.bundle_sha256,
        "run_id": "run-c",
    }


# (case id, candidate kwargs, pick kwargs, final verdict, reason, diff is None)
CASES = [
    ("small_edge", {"grades": {GSM: g(62)}}, {}, "watch", "ci_includes_zero", False),
    ("worse", {"grades": {GSM: g(55)}}, {}, "hold", "not_better", False),
    ("equal", {"grades": {GSM: g(60)}}, {}, "hold", "not_better", False),
    ("slower", {"decode": 20.0}, {}, "watch", "slower_than_ratio", False),
    ("speed_at_ratio", {"decode": 32.0}, {}, "replace", None, False),
    ("less_context", {"max_context": 16384}, {}, "watch", "less_context", False),
    ("context_cap", {"max_context": 32768}, {"max_context": 65536}, "replace", None, False),
    ("context_short_of_cap", {"max_context": 24576}, {"max_context": 65536}, "watch",
     "less_context", False),
    ("suite_hash", {"suite_hashes": {GSM: "f" * 64}}, {}, "watch", "suite_hash_mismatch", True),
    ("suite_set", {}, {"grades": {GSM: g(60), "general/mmlu": g(5, 10)}}, "watch",
     "suite_set_mismatch", True),
    ("item_ids", {"grades": {GSM: g(90, 99)}}, {}, "watch", "item_ids_mismatch", True),
    ("judge_suite_ignored", {"judge": ("general/open-ended",)}, {}, "replace", None, False),
    ("worse_and_mismatched", {"grades": {GSM: g(10)}, "suite_hashes": {GSM: "f" * 64}}, {},
     "watch", "suite_hash_mismatch", True),
]  # fmt: skip


@pytest.mark.parametrize(
    "cand_kw,pick_kw,final,reason,diff_none", [c[1:] for c in CASES], ids=[c[0] for c in CASES]
)
def test_single_candidate_table(tmp_path, cand_kw, pick_kw, final, reason, diff_none):
    pk = {"grades": {GSM: g(60)}, "decode": 40.0, **pick_kw}
    ck = {"grades": {GSM: g(90)}, "decode": 40.0, **cand_kw}
    p = run(tmp_path, "run-p", "Pick", **pk)
    c = run(tmp_path, "run-c", "Cand", **ck)
    res = assess([p, c], {"general": pick("general", "Pick")})
    v = res.verdicts["run-c"]["general"]
    assert v.computed == v.final == final
    assert v.reasons == ([] if reason is None else [reason])
    assert v.new_pick is (final == "replace")
    if diff_none:
        assert v.diff is None and v.diff_ci95 is None
    else:
        assert isinstance(v.diff, float) and len(v.diff_ci95) == 2
    expected_pick = "run-c" if final == "replace" else "run-p"
    assert res.picks_after["general"]["run_id"] == expected_pick


def test_pick_not_remeasured(tmp_path):
    a = run(tmp_path, "run-a", "A", {GSM: g(90)})
    b = run(tmp_path, "run-b", "B", {GSM: g(30)})
    before = {"general": pick("general", "Pick")}
    res = assess([a, b], before)
    assert res.pick_status == {"general": "not_re-measured"}
    assert res.pick_tracks == {}
    for rid in ("run-a", "run-b"):
        v = res.verdicts[rid]["general"]
        assert (v.computed, v.final) == ("watch", "watch")
        assert v.reasons == ["pick_not_remeasured"]
        assert v.diff is None and v.diff_ci95 is None and v.new_pick is False
    assert res.picks_after == before


def test_control_ineligible_is_not_remeasured(tmp_path):
    p = run(tmp_path, "run-p", "Pick", decode=5.0)
    c = run(tmp_path, "run-c", "Cand", {GSM: g(90)})
    res = assess([p, c], {"general": pick("general", "Pick")})
    assert res.ineligible == {"run-p": ["decode_below_min"]}
    assert res.pick_status == {"general": "not_re-measured"}
    assert res.verdicts["run-c"]["general"].final == "watch"


def test_control_without_track_score_is_not_remeasured(tmp_path):
    p = run(tmp_path, "run-p", "Pick", {"structured/json": g(5, 10)})
    c = run(tmp_path, "run-c", "Cand", {GSM: g(90)})
    res = assess([p, c], {"general": pick("general", "Pick")})
    assert res.pick_status["general"] == "not_re-measured"
    assert res.verdicts["run-c"]["general"].reasons == ["pick_not_remeasured"]


def test_control_lowest_run_id_and_since_week(tmp_path):
    p2 = run(tmp_path, "run-p2", "Pick", {GSM: g(60)})
    p1 = run(tmp_path, "run-p1", "Pick", {GSM: g(60)})
    c = run(tmp_path, "run-c", "Cand", {GSM: g(55)})
    before = {"general": pick("general", "Pick", since="2026-09-07")}
    res = assess([p2, c, p1], before)
    assert res.pick_tracks == {"run-p1": ["general"]}
    # the duplicate configuration is assessed as a candidate against the control
    assert res.verdicts["run-p2"]["general"].final == "hold"
    after = res.picks_after["general"]
    assert after == {**before["general"], "bundle_sha256": p1.bundle_sha256, "run_id": "run-p1"}
    assert after["since_week"] == "2026-09-07"
    assert before["general"]["run_id"] == "old-run"  # input not mutated


def test_pick_matched_on_quant_and_runtime(tmp_path):
    p = run(tmp_path, "run-p", "Pick", quant="Q8_0")
    res = assess([p], {"general": pick("general", "Pick")})
    assert res.pick_status == {"general": "not_re-measured"}
    assert res.verdicts["run-p"]["general"].reasons == ["pick_not_remeasured"]


def test_initial_pick(tmp_path):
    a = run(tmp_path, "run-a", "A", {GSM: g(70)})
    b = run(tmp_path, "run-b", "B", {GSM: g(80)})
    c = run(tmp_path, "run-c", "C", {GSM: g(60)})
    res = assess([a, b, c], {})
    vb = res.verdicts["run-b"]["general"]
    assert (vb.computed, vb.final, vb.new_pick) == ("initial_pick", "initial_pick", True)
    assert vb.diff is None and vb.diff_ci95 is None and vb.reasons == []
    for rid in ("run-a", "run-c"):
        v = res.verdicts[rid]["general"]
        assert (v.final, v.reasons, v.new_pick) == ("hold", ["not_better"], False)
        assert v.diff < 0
    assert res.pick_status == {"general": "none"}
    assert res.pick_tracks == {}
    assert res.picks_after["general"]["run_id"] == "run-b"
    assert res.picks_after["general"]["since_week"] == WEEK
    assert res.picks_after["general"]["bundle_sha256"] == b.bundle_sha256


@pytest.mark.parametrize(
    "decodes,winner",
    [((30.0, 50.0, 40.0), "run-b"), ((40.0, 40.0, 40.0), "run-a"), ((40.0, 40.0, 45.0), "run-c")],
    ids=["higher_decode", "lower_run_id", "decode_before_run_id"],
)
def test_initial_pick_tiebreak(tmp_path, decodes, winner):
    runs = [
        run(tmp_path, rid, rid.upper(), {GSM: g(70)}, decode=d)
        for rid, d in zip(("run-a", "run-b", "run-c"), decodes, strict=True)
    ]
    res = assess(list(reversed(runs)), {})
    assert res.picks_after["general"]["run_id"] == winner
    assert res.verdicts[winner]["general"].final == "initial_pick"


def test_two_clear_wins(tmp_path):
    p = run(tmp_path, "run-p", "Pick", {GSM: g(60)})
    a = run(tmp_path, "run-a", "A", {GSM: g(90)})
    b = run(tmp_path, "run-b", "B", {GSM: g(95)})
    res = assess([p, a, b], {"general": pick("general", "Pick")})
    va, vb = res.verdicts["run-a"]["general"], res.verdicts["run-b"]["general"]
    assert (vb.final, vb.new_pick, vb.outranked_by) == ("replace", True, None)
    assert (va.final, va.new_pick, va.outranked_by) == ("replace", False, "run-b")
    assert res.picks_after["general"]["run_id"] == "run-b"


def test_two_clear_wins_tie_on_score_goes_to_decode(tmp_path):
    p = run(tmp_path, "run-p", "Pick", {GSM: g(60)})
    a = run(tmp_path, "run-a", "A", {GSM: g(90)}, decode=40.0)
    b = run(tmp_path, "run-b", "B", {GSM: g(90)}, decode=45.0)
    res = assess([p, a, b], {"general": pick("general", "Pick")})
    assert res.picks_after["general"]["run_id"] == "run-b"
    assert res.verdicts["run-a"]["general"].outranked_by == "run-b"


def test_override_hold_to_replace(tmp_path):
    p = run(tmp_path, "run-p", "Pick", {GSM: g(60)})
    c = run(tmp_path, "run-c", "Cand", {GSM: g(55)})
    ov = Override(TIER, "general", "run-c", "replace", "Upstream fix confirmed.")
    res = assess([p, c], {"general": pick("general", "Pick")}, [ov])
    v = res.verdicts["run-c"]["general"]
    assert (v.computed, v.final, v.new_pick) == ("hold", "replace", True)
    assert v.override_reason == "Upstream fix confirmed."
    assert v.reasons == ["not_better"]
    assert res.picks_after["general"]["run_id"] == "run-c"
    assert res.picks_after["general"]["since_week"] == WEEK


def test_override_replace_to_hold_removes_from_selection(tmp_path):
    p = run(tmp_path, "run-p", "Pick", {GSM: g(60)})
    a = run(tmp_path, "run-a", "A", {GSM: g(90)})
    b = run(tmp_path, "run-b", "B", {GSM: g(95)})
    ov = Override(TIER, "general", "run-b", "hold", "Known tokenizer bug in this quant.")
    res = assess([p, a, b], {"general": pick("general", "Pick")}, [ov])
    vb = res.verdicts["run-b"]["general"]
    assert (vb.computed, vb.final, vb.new_pick, vb.outranked_by) == ("replace", "hold", False, None)
    va = res.verdicts["run-a"]["general"]
    assert (va.final, va.new_pick, va.outranked_by) == ("replace", True, None)
    assert res.picks_after["general"]["run_id"] == "run-a"


def test_override_only_replace_to_hold_keeps_pick(tmp_path):
    p = run(tmp_path, "run-p", "Pick", {GSM: g(60)})
    c = run(tmp_path, "run-c", "Cand", {GSM: g(90)})
    ov = Override(TIER, "general", "run-c", "hold", "Wait for the re-upload.")
    before = {"general": pick("general", "Pick")}
    res = assess([p, c], before, [ov])
    assert res.verdicts["run-c"]["general"].new_pick is False
    assert res.picks_after["general"]["run_id"] == "run-p"
    assert res.picks_after["general"]["weights"] == "Pick"


def test_override_other_tier_ignored(tmp_path):
    p = run(tmp_path, "run-p", "Pick", {GSM: g(60)})
    c = run(tmp_path, "run-c", "Cand", {GSM: g(90)})
    ov = Override("12gb", "general", "run-c", "hold", "Other tier.")
    res = assess([p, c], {"general": pick("general", "Pick")}, [ov])
    assert res.verdicts["run-c"]["general"].final == "replace"


OVERRIDE_ERRORS = [
    ("ineligible_run", Override(TIER, "general", "run-slow", "replace", "r"), "ineligible"),
    ("pick_control", Override(TIER, "general", "run-p", "hold", "r"), "control"),
    ("no_score_in_track", Override(TIER, "structured", "run-c", "hold", "r"), "no score"),
    ("no_current_pick", Override(TIER, "structured", "run-s", "hold", "r"), "no current pick"),
    ("unknown_run", Override(TIER, "general", "run-zzz", "hold", "r"), "not in tier"),
    ("bad_verdict", Override(TIER, "general", "run-c", "initial_pick", "r"), "verdict"),
]


@pytest.mark.parametrize(
    "ov,match", [c[1:] for c in OVERRIDE_ERRORS], ids=[c[0] for c in OVERRIDE_ERRORS]
)
def test_override_errors(tmp_path, ov, match):
    runs = [
        run(tmp_path, "run-p", "Pick", {GSM: g(60)}),
        run(tmp_path, "run-c", "Cand", {GSM: g(90)}),
        run(tmp_path, "run-slow", "Slow", {GSM: g(90)}, decode=5.0),
        run(tmp_path, "run-s", "Struct", {"structured/json": g(5, 10)}),
    ]
    with pytest.raises(VerdictError, match=match):
        assess(runs, {"general": pick("general", "Pick")}, [ov])


def test_duplicate_override_rejected(tmp_path):
    p = run(tmp_path, "run-p", "Pick", {GSM: g(60)})
    c = run(tmp_path, "run-c", "Cand", {GSM: g(90)})
    ovs = [
        Override(TIER, "general", "run-c", "hold", "a"),
        Override(TIER, "general", "run-c", "watch", "b"),
    ]
    with pytest.raises(VerdictError, match="twice"):
        assess([p, c], {"general": pick("general", "Pick")}, ovs)


def test_duplicate_run_id_rejected(tmp_path):
    p = run(tmp_path, "run-p", "Pick", {GSM: g(60)})
    with pytest.raises(VerdictError, match="duplicate"):
        assess([p, p], {"general": pick("general", "Pick")})


def test_ineligible_only_in_ineligible(tmp_path):
    p = run(tmp_path, "run-p", "Pick", {GSM: g(60)})
    bad = run(tmp_path, "run-bad", "Bad", {GSM: g(99)}, max_context=8192, status="incomplete")
    res = assess([p, bad], {"general": pick("general", "Pick")})
    assert res.ineligible == {"run-bad": ["status_incomplete", "context_below_min"]}
    assert "run-bad" not in res.verdicts
    assert "run-bad" not in res.pick_tracks
    assert res.picks_after["general"]["run_id"] == "run-p"


def test_ineligible_never_initial_pick(tmp_path):
    a = run(tmp_path, "run-a", "A", {GSM: g(99)}, decode=5.0)
    b = run(tmp_path, "run-b", "B", {GSM: g(50)})
    res = assess([a, b], {})
    assert res.picks_after["general"]["run_id"] == "run-b"
    assert list(res.verdicts) == ["run-b"]


def test_multi_track(tmp_path):
    js = "structured/json"
    p = run(tmp_path, "run-p", "Pick", {GSM: g(60), js: g(50)})
    c = run(tmp_path, "run-c", "Cand", {GSM: g(90), js: g(40)})
    res = assess([p, c], {"general": pick("general", "Pick")})
    assert res.pick_status == {"general": "re-measured", "structured": "none"}
    assert res.pick_tracks == {"run-p": ["general"]}
    assert res.verdicts["run-c"]["general"].final == "replace"
    # no structured pick before: the higher structured score is the initial pick
    assert res.verdicts["run-p"]["structured"].final == "initial_pick"
    assert res.verdicts["run-c"]["structured"].final == "hold"
    assert res.picks_after["structured"]["run_id"] == "run-p"
    assert list(res.picks_after) == ["general", "structured"]


def test_multi_suite_track_uses_all_suites(tmp_path):
    mm = "general/mmlu"
    p = run(tmp_path, "run-p", "Pick", {GSM: g(60), mm: g(60)})
    c = run(tmp_path, "run-c", "Cand", {GSM: g(90), mm: g(60)})
    res = assess([p, c], {"general": pick("general", "Pick")})
    assert res.verdicts["run-c"]["general"].diff == pytest.approx(0.15)


def test_pick_kept_when_no_candidates(tmp_path):
    p = run(tmp_path, "run-p", "Pick", {GSM: g(60)})
    before = {"general": pick("general", "Pick"), "code": pick("code", "Coder")}
    res = assess([p], before)
    assert res.pick_status == {"code": "not_re-measured", "general": "re-measured"}
    assert res.picks_after["code"] == before["code"]
    assert res.picks_after["general"]["run_id"] == "run-p"


def test_deterministic(tmp_path):
    runs = [
        run(tmp_path, "run-p", "Pick", {GSM: g(60)}),
        run(tmp_path, "run-a", "A", {GSM: g(62)}),
        run(tmp_path, "run-b", "B", {GSM: g(90)}),
        run(tmp_path, "run-c", "C", {GSM: g(95)}, decode=20.0),
    ]
    before = {"general": pick("general", "Pick")}
    first = assess(runs, before)
    second = assess(list(reversed(runs)), before)
    assert first == second
    assert list(second.verdicts) == sorted(second.verdicts)
    assert (
        first.verdicts["run-a"]["general"].diff_ci95
        == second.verdicts["run-a"]["general"].diff_ci95
    )


def test_override_to_replace_when_pick_not_remeasured(tmp_path):
    c = run(tmp_path, "run-c", "Cand", {GSM: g(90)})
    ov = Override(TIER, "general", "run-c", "replace", "Old pick withdrawn upstream.")
    res = assess([c], {"general": pick("general", "Pick")}, [ov])
    v = res.verdicts["run-c"]["general"]
    assert (v.computed, v.final, v.new_pick) == ("watch", "replace", True)
    assert v.reasons == ["pick_not_remeasured"]
    assert res.picks_after["general"]["run_id"] == "run-c"


def test_pick_filed_under_wrong_track_rejected(tmp_path):
    p = run(tmp_path, "run-p", "Pick", {GSM: g(60)})
    with pytest.raises(VerdictError, match="another tier or track"):
        assess([p], {"general": pick("code", "Pick")})
