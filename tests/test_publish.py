import json
import os
import shutil
from pathlib import Path

import pytest
from weekfix import make_bundle

import bakeoff.publish as pub
from bakeoff.bundle import verify_bundle, write_bundle
from bakeoff.publish import PublishError, publish, week_docs, write_picks
from bakeoff.week import validate_week

REPO = Path(__file__).resolve().parents[1]
W1 = "2026-10-19"
W2 = "2026-10-26"
AT1 = "2026-10-19T07:00:00Z"
AT2 = "2026-10-26T07:00:00Z"
ENV24 = "1× RTX 3090 Ti 24 GB"
ENV12 = "1× RTX 3060 12 GB"
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


def bundle(root, weights, k=6, *, sub="runs", envelope=ENV24, **kw):
    return make_bundle(
        root / sub, label=weights, weights=weights, envelope=envelope, grades={GSM: g(k)}, **kw
    )


def judge_bundle(root, name, a, b, counts=(3, 1, 1)):
    meta = {
        "bundle_format": "judge-m1",
        "status": "complete",
        "status_reason": None,
        "a": a,
        "b": b,
        "suite": {"key": "general/chat", "hash": "c" * 64},
        "judge_model": "judge-x",
        "counts": {"a": counts[0], "b": counts[1], "tie": counts[2]},
        "call_errors": 0,
    }
    out = root / "judges" / name
    write_bundle(out, meta, {"items.jsonl": [{"id": "chat-0", "result": "a"}]})
    return out


def sha(path):
    return json.loads((path / "bundle.json").read_text(encoding="utf-8"))["bundle_sha256"]


def week_toml(root, runs, *, week=W1, at=AT1, judges=(), correction=None, name=None, raw=None):
    """runs: list of (bundle path or raw string, tier)."""
    lines = [f'week = "{week}"', f'published_at = "{at}"', ""]
    for path, tier in runs:
        rel = path if isinstance(path, str) else path.relative_to(root).as_posix()
        lines += [
            "[[run]]",
            f'bundle = "{rel}"',
            f'tier = "{tier}"',
            'license = "Apache-2.0"',
            'source_url = "https://example.org/model"',
            "",
        ]
    for path in judges:
        lines += ["[[judge]]", f'bundle = "{path.relative_to(root).as_posix()}"', ""]
    if correction is not None:
        lines += ["[correction]", f'note = "{correction}"', ""]
    out = root / (name or f"{week}.toml")
    out.write_text(raw if raw is not None else "\n".join(lines), encoding="utf-8")
    return out


def run(toml, root):
    return publish(toml, root, username="alice", denylist=None)


def no_week(root, week=W1):
    results = root / "results"
    assert not (results / week).exists()
    if results.exists():
        assert not [p.name for p in results.iterdir() if p.name.startswith(".")]


def tree(path):
    return {
        p.relative_to(path).as_posix(): p.read_bytes()
        for p in sorted(path.rglob("*"))
        if p.is_file()
    }


def doc_of(root, week=W1):
    return json.loads((root / "results" / week / "week.json").read_text(encoding="utf-8"))


# --- happy path ---------------------------------------------------------------------------------


def test_happy_path(root):
    a, b = bundle(root, "base", 6), bundle(root, "small", 4)
    toml = week_toml(root, [(a, "24gb"), (b, "24gb")])
    out = run(toml, root)
    assert out == root / "results" / W1
    assert (out / "week.toml").read_bytes() == toml.read_bytes()
    doc = doc_of(root)
    assert validate_week(doc) == []
    assert doc["revision"] == 1 and doc["published_at"] == AT1 and doc["corrections"] == []
    for src in (a, b):
        copy = out / "runs" / src.name
        assert verify_bundle(copy, "alice", None) == []
        assert sha(copy) == sha(src)
        assert tree(copy) == tree(src)
    picks = json.loads((root / "results" / "picks.json").read_text(encoding="utf-8"))
    assert picks == doc["picks_after"]
    assert picks["24gb"]["general"]["run_id"] == a.name
    raw = (out / "week.json").read_bytes()
    assert raw == (
        json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    assert "×" in raw.decode("utf-8")  # ensure_ascii=False
    assert [d["week"] for d in week_docs(root)] == [W1]
    no_week(root, "never")  # no staging or .old leftovers


def test_judge_bundle_is_copied_and_mapped(root):
    a, b = bundle(root, "base", 6), bundle(root, "small", 4)
    j = judge_bundle(root, "judge-chat-1", sha(a), sha(b))
    out = run(week_toml(root, [(a, "24gb"), (b, "24gb")], judges=[j]), root)
    assert sha(out / "judge" / "judge-chat-1") == sha(j)
    tier = next(t for t in doc_of(root)["tiers"] if t["id"] == "24gb")
    assert tier["judge"] == [
        {
            "a": a.name,
            "b": b.name,
            "wins": 3,
            "ties": 1,
            "losses": 1,
            "errors": 0,
            "judge_model": "judge-x",
            "bundle_sha256": sha(j),
        }
    ]


def test_incomplete_run_is_published_but_ineligible(root):
    a, b = bundle(root, "base", 6), bundle(root, "partial", 9, status="incomplete")
    out = run(week_toml(root, [(a, "24gb"), (b, "24gb")]), root)
    assert (out / "runs" / b.name / "bundle.json").is_file()
    runs = {r["run_id"]: r for t in doc_of(root)["tiers"] for r in t["runs"]}
    assert runs[b.name]["eligible"] is False
    assert "status_incomplete" in runs[b.name]["ineligible_reasons"]


def test_reproducible(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    a, b = bundle(src, "base", 6), bundle(src, "small", 4)
    trees = []
    for name in ("one", "two"):
        r = tmp_path / name
        r.mkdir()
        for f in ("tiers.toml", "site.toml"):
            shutil.copy(REPO / f, r / f)
        shutil.copytree(src / "runs", r / "runs")
        toml = week_toml(r, [(r / "runs" / a.name, "24gb"), (r / "runs" / b.name, "24gb")])
        run(toml, r)
        trees.append(tree(r / "results"))
    assert trees[0] == trees[1]
    assert f"{W1}/week.json" in trees[0] and "picks.json" in trees[0]


def test_week_docs_and_picks_when_empty(root):
    assert week_docs(root) == []
    write_picks(root)
    assert (root / "results" / "picks.json").read_bytes() == b"{}\n"


# --- refusals -----------------------------------------------------------------------------------


def refused(root, toml, match, week=W1):
    with pytest.raises(PublishError, match=match):
        run(toml, root)
    no_week(root, week)


def test_refuses_corrupted_bundle(root):
    a, b = bundle(root, "base", 6), bundle(root, "small", 4)
    with (b / "quality" / "general.gsm8k.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"id": "gsm8k-99", "pass": true, "error": null}\n')
    refused(root, week_toml(root, [(a, "24gb"), (b, "24gb")]), "hash_mismatch")


def test_refuses_scrub_finding_in_label(root):
    a = bundle(root, "base", 6)
    b = make_bundle(
        root / "runs",
        label="/home/alice/models/small",  # scrub-ok
        weights="small",
        envelope=ENV24,
        grades={GSM: g(4)},
    )
    refused(root, week_toml(root, [(a, "24gb"), (b, "24gb")]), "home_path")


def test_refuses_username_in_week_toml(root):
    a = bundle(root, "base", 6)
    text = week_toml(root, [(a, "24gb")]).read_text(encoding="utf-8")
    raw = text.replace("https://example.org/model", "https://example.org/alice")
    refused(root, week_toml(root, [], raw=raw), "username")


def test_refuses_scrub_finding_only_in_week_toml_comment(root):
    a = bundle(root, "base", 6)
    text = week_toml(root, [(a, "24gb")]).read_text(encoding="utf-8")
    raw = "# copied from /Users/alice/weeks\n" + text  # scrub-ok
    refused(root, week_toml(root, [], raw=raw), "week.toml.*home_path")


def test_refuses_absolute_path_in_week_toml(root):
    a = bundle(root, "base", 6)
    refused(root, week_toml(root, [(a.as_posix(), "24gb")]), "absolute")


def test_refuses_duplicate_run_id(root):
    a = bundle(root, "base", 6)
    dup = root / "runs" / "copy-of-base"
    shutil.copytree(a, dup)
    refused(root, week_toml(root, [(a, "24gb"), (dup, "24gb")]), "duplicate run_id")


def test_refuses_envelope_not_matching_tier(root):
    a = bundle(root, "base", 6)
    refused(root, week_toml(root, [(a, "12gb")]), "envelope")


def test_refuses_bad_status(root):
    a, b = bundle(root, "base", 6), bundle(root, "odd", 4, status="aborted")
    refused(root, week_toml(root, [(a, "24gb"), (b, "24gb")]), "status")


def test_refuses_judge_naming_run_outside_week(root):
    a, b = bundle(root, "base", 6), bundle(root, "small", 4)
    j = judge_bundle(root, "judge-chat-1", sha(a), "f" * 64)
    refused(root, week_toml(root, [(a, "24gb"), (b, "24gb")], judges=[j]), "not in this week")


def test_refuses_existing_week_without_correction(root):
    a = bundle(root, "base", 6)
    toml = week_toml(root, [(a, "24gb")])
    run(toml, root)
    before = tree(root / "results")
    with pytest.raises(PublishError, match=r"\[correction\]"):
        run(toml, root)
    assert tree(root / "results") == before


def test_refuses_correction_for_missing_week(root):
    a = bundle(root, "base", 6)
    refused(root, week_toml(root, [(a, "24gb")], correction="Re-graded."), "does not exist")


def test_refuses_new_week_older_than_published(root):
    a = bundle(root, "base", 6)
    run(week_toml(root, [(a, "24gb")], week=W2, at=AT2), root)
    refused(root, week_toml(root, [(a, "24gb")]), "later week")


def test_refuses_missing_denylist(root):
    a = bundle(root, "base", 6)
    toml = week_toml(root, [(a, "24gb")])
    with pytest.raises(PublishError, match="denylist"):
        publish(toml, root, username=None, denylist=root / "nope.txt")
    no_week(root)


def test_denylist_term_in_bundle_refused(root):
    a = bundle(root, "lab-box-7 base", 6)
    deny = root / "deny.txt"
    deny.write_text("lab-box-7\n", encoding="utf-8")
    with pytest.raises(PublishError, match="denylist"):
        publish(week_toml(root, [(a, "24gb")]), root, username=None, denylist=deny)
    no_week(root)


def test_refuses_leftover_old_week(root):
    a = bundle(root, "base", 6)
    (root / "results" / f".{W1}.old").mkdir(parents=True)
    with pytest.raises(PublishError, match="leftover"):
        run(week_toml(root, [(a, "24gb")]), root)
    assert (root / "results" / f".{W1}.old").is_dir()


# --- week 2 and corrections ---------------------------------------------------------------------


def week1(root):
    a, b = bundle(root, "base", 6), bundle(root, "small", 4)
    run(week_toml(root, [(a, "24gb"), (b, "24gb")]), root)
    return a, b


def test_week2_picks_before_from_week1(root):
    a, _ = week1(root)
    # A re-run of the pick (same weights/quant/runtime, new run_id) and a weaker challenger.
    rerun = bundle(root, "base", 6, sub="runs2")
    worse = bundle(root, "worse", 3, sub="runs2")
    run(week_toml(root, [(rerun, "24gb"), (worse, "24gb")], week=W2, at=AT2), root)
    doc = doc_of(root, W2)
    tier = next(t for t in doc["tiers"] if t["id"] == "24gb")
    assert tier["pick_status"] == {"general": "re-measured"}
    pick = doc["picks_after"]["24gb"]["general"]
    assert pick["since_week"] == W1 and pick["run_id"] == rerun.name
    runs = {r["run_id"]: r for r in tier["runs"]}
    assert runs[rerun.name]["role"] == "pick"
    assert runs[worse.name]["verdicts"]["general"]["final"] == "hold"
    picks = json.loads((root / "results" / "picks.json").read_text(encoding="utf-8"))
    assert picks == doc["picks_after"]
    assert [d["week"] for d in week_docs(root)] == [W1, W2]


def test_tier_without_runs_keeps_its_pick(root):
    week1(root)
    other = bundle(root, "tiny", 6, sub="runs2", envelope=ENV12)
    run(week_toml(root, [(other, "12gb")], week=W2, at=AT2), root)
    doc = doc_of(root, W2)
    assert doc["picks_after"]["24gb"] == doc_of(root)["picks_after"]["24gb"]
    tier = next(t for t in doc["tiers"] if t["id"] == "24gb")
    assert tier["pick_status"] == {"general": "not_re-measured"}


def test_correction_raises_revision(root):
    a, b = week1(root)
    toml = week_toml(
        root, [(a, "24gb"), (b, "24gb")], at="2026-10-20T08:00:00Z", correction="Re-graded."
    )
    out = run(toml, root)
    doc = doc_of(root)
    assert doc["revision"] == 2
    assert doc["published_at"] == AT1
    assert doc["corrections"] == [
        {"revision": 2, "published_at": "2026-10-20T08:00:00Z", "note": "Re-graded."}
    ]
    assert (out / "week.toml").read_bytes() == toml.read_bytes()
    assert validate_week(doc) == []
    no_week(root, "never")


def test_correction_may_change_runs_without_later_week(root):
    a, b = week1(root)
    run(week_toml(root, [(b, "24gb")], at=AT1, correction="Dropped a run."), root)
    doc = doc_of(root)
    assert doc["picks_after"]["24gb"]["general"]["run_id"] == b.name
    assert not (root / "results" / W1 / "runs" / a.name).exists()
    picks = json.loads((root / "results" / "picks.json").read_text(encoding="utf-8"))
    assert picks["24gb"]["general"]["run_id"] == b.name


def test_correction_refused_when_later_week_depends_on_picks(root):
    a, b = week1(root)
    rerun = bundle(root, "base", 6, sub="runs2")
    run(week_toml(root, [(rerun, "24gb")], week=W2, at=AT2), root)
    before = tree(root / "results")
    with pytest.raises(PublishError, match="later week"):
        run(week_toml(root, [(b, "24gb")], correction="Dropped a run."), root)
    assert tree(root / "results") == before
    no_week(root, "never")


def test_correction_allowed_with_later_week_if_picks_unchanged(root):
    a, b = week1(root)
    rerun = bundle(root, "base", 6, sub="runs2")
    run(week_toml(root, [(rerun, "24gb")], week=W2, at=AT2), root)
    run(week_toml(root, [(a, "24gb"), (b, "24gb")], correction="Note only."), root)
    assert doc_of(root)["revision"] == 2
    picks = json.loads((root / "results" / "picks.json").read_text(encoding="utf-8"))
    assert picks == doc_of(root, W2)["picks_after"]


def test_correction_reuses_published_bundle_paths(root):
    a, b = week1(root)
    pub_a = root / "results" / W1 / "runs" / a.name
    pub_b = root / "results" / W1 / "runs" / b.name
    shutil.rmtree(a.parent)  # the raw runs are gone; only the published copies remain
    run(week_toml(root, [(pub_a, "24gb"), (pub_b, "24gb")], correction="Re-graded."), root)
    assert doc_of(root)["revision"] == 2
    assert sha(root / "results" / W1 / "runs" / a.name) == sha(pub_a)


def test_correction_refuses_changed_bundle_for_same_run_id(root):
    a, b = week1(root)
    changed = bundle(root, "base", 9, sub="runs2", run_id=a.name)
    assert sha(changed) != sha(a)
    before = tree(root / "results")
    with pytest.raises(PublishError, match="immutable"):
        run(week_toml(root, [(changed, "24gb"), (b, "24gb")], correction="Swap."), root)
    assert tree(root / "results") == before


def test_correction_refuses_changed_judge_bundle(root):
    a, b = bundle(root, "base", 6), bundle(root, "small", 4)
    j = judge_bundle(root, "judge-chat-1", sha(a), sha(b))
    run(week_toml(root, [(a, "24gb"), (b, "24gb")], judges=[j]), root)
    shutil.rmtree(j)
    j2 = judge_bundle(root, "judge-chat-1", sha(a), sha(b), counts=(1, 3, 1))
    before = tree(root / "results")
    with pytest.raises(PublishError, match="immutable"):
        run(week_toml(root, [(a, "24gb"), (b, "24gb")], judges=[j2], correction="Re-judged."), root)
    assert tree(root / "results") == before


def test_new_week_refuses_changed_bundle_for_published_run_id(root):
    a, _ = week1(root)
    changed = bundle(root, "base", 9, sub="runs2", run_id=a.name)
    refused(root, week_toml(root, [(changed, "24gb")], week=W2, at=AT2), "immutable", week=W2)


def test_correction_refuses_other_rules(root):
    a, b = week1(root)
    path = root / "results" / W1 / "week.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["rules"]["resamples"] = 500
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(PublishError, match="rules"):
        run(week_toml(root, [(a, "24gb"), (b, "24gb")], correction="Re-graded."), root)


# --- interruption and failed swaps --------------------------------------------------------------


def test_interrupted_copy_cleans_staging_and_rerun_succeeds(root, monkeypatch):
    a, b = bundle(root, "base", 6), bundle(root, "small", 4)
    toml = week_toml(root, [(a, "24gb"), (b, "24gb")])
    real = shutil.copytree
    calls = []

    def flaky(src, dst, *args, **kw):
        calls.append(src)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real(src, dst, *args, **kw)

    monkeypatch.setattr(pub.shutil, "copytree", flaky)
    with pytest.raises(KeyboardInterrupt):
        run(toml, root)
    assert len(calls) == 2
    no_week(root)
    monkeypatch.setattr(pub.shutil, "copytree", real)
    assert run(toml, root) == root / "results" / W1


def test_failed_correction_swap_keeps_old_week(root, monkeypatch):
    a, b = week1(root)
    before = tree(root / "results")
    real = os.rename
    calls = []

    def flaky(src, dst):
        calls.append((Path(src).name, Path(dst).name))
        if len(calls) == 2:  # old -> .old succeeded; staging -> week fails
            raise KeyboardInterrupt
        return real(src, dst)

    monkeypatch.setattr(pub.os, "rename", flaky)
    with pytest.raises(KeyboardInterrupt):
        run(week_toml(root, [(b, "24gb")], correction="Dropped a run."), root)
    monkeypatch.setattr(pub.os, "rename", real)
    assert calls[0] == (W1, f".{W1}.old")
    assert tree(root / "results") == before
    no_week(root, "never")


def test_week_docs_refuses_week_dir_without_week_json(root):
    (root / "results" / W1).mkdir(parents=True)
    with pytest.raises(PublishError, match="week.json"):
        week_docs(root)
