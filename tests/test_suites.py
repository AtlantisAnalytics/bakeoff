import shutil
from pathlib import Path

import pytest

from bakeoff.suites import (
    SuiteError,
    check_lock,
    load_suite,
    resolve_suites,
    suites_root,
)

DATA = Path(__file__).resolve().parent / "data" / "suites"
DEMO = DATA / "demo"


def _copy(tmp_path):
    dst = tmp_path / "demo"
    shutil.copytree(DEMO, dst)
    return dst


def test_load_items_suite():
    s = load_suite(DEMO)
    assert s.key == "demo/exact"
    assert len(s.items) == 2
    assert s.items[0].messages == [
        {"role": "system", "content": "Follow the instructions exactly."},
        {"role": "user", "content": "Reply with exactly: ok"},
    ]
    assert s.items[1].messages[0]["role"] == "system"
    assert s.items[1].expected == "yes"
    assert len(s.hash) == 64


def test_load_generated_suite(tmp_path):
    d = tmp_path / "gen"
    d.mkdir()
    (d / "manifest.toml").write_text(
        'track = "longctx"\nname = "needle"\nversion = 1\nsplit = "public"\n'
        'grader = "needle"\nmax_tokens = 32\n'
        '[generator]\nname = "needle"\nlines = [10, 20]\ndepths = [0.1, 0.9]\nseed = 5\n'
    )
    s = load_suite(d)
    assert [i.id for i in s.items] == [
        "needle-10-10",
        "needle-10-90",
        "needle-20-10",
        "needle-20-90",
    ]
    assert s.items[0].expected in s.items[0].messages[-1]["content"]
    assert s.generator["seed"] == 5
    assert load_suite(d).hash == s.hash


def test_hash_changes_with_item(tmp_path):
    d = _copy(tmp_path)
    before = load_suite(d).hash
    (d / "items.jsonl").write_text(
        '{"id": "a", "prompt": "Reply with exactly: OK", "expected": "OK"}\n'
        '{"id": "b", "prompt": "x", "expected": "yes"}\n'
    )
    assert load_suite(d).hash != before


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ('split = "public"', 'split = "secret"'),
        ('grader = "exact"', 'grader = ""'),
        ("max_tokens = 16", "max_tokens = 0"),
        ("version = 1", "version = 0"),
        ("version = 1", 'version = "1"'),
        ("max_tokens = 16", "max_tokens = 1.5"),
    ],
)
def test_bad_manifest_raises(tmp_path, old, new):
    d = _copy(tmp_path)
    m = d / "manifest.toml"
    m.write_text(m.read_text().replace(old, new))
    with pytest.raises(SuiteError):
        load_suite(d)


def test_item_errors(tmp_path):
    d = _copy(tmp_path)
    (d / "items.jsonl").write_text('{"id": "a", "expected": "x"}\n')
    with pytest.raises(SuiteError):
        load_suite(d)
    (d / "items.jsonl").write_text("not json\n")
    with pytest.raises(SuiteError):
        load_suite(d)


def test_resolve_by_name_and_path(tmp_path):
    assert [s.key for s in resolve_suites([str(DEMO)])] == ["demo/exact"]
    assert [s.key for s in resolve_suites([str(DATA)])] == ["demo/exact"]
    with pytest.raises(SuiteError):
        resolve_suites(["no/such/suite"])
    # names resolve under suites_root(); the demo fixtures never live there
    assert "tests" not in suites_root().parts[-2:]


def test_check_lock_detects_unbumped_edit():
    s = load_suite(DEMO)
    problems = check_lock([s], {s.key: {str(s.version): "0" * 64}})
    assert problems
    assert "bump" in problems[0]
    assert check_lock([s], {s.key: {str(s.version): s.hash}}) == []
    assert "lock_suites" in check_lock([s], {})[0]
    assert "lock_suites" in check_lock([s], {s.key: {"99": s.hash}})[0]
