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
        'track = "longctx"\nlicense = "CC0-1.0"\nname = "needle"\nversion = 1\nsplit = "public"\n'
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
    assert [s.key for s in resolve_suites([str(DATA)])] == ["demo/exact", "demo/needle"]
    with pytest.raises(SuiteError):
        resolve_suites(["no/such/suite"])
    assert suites_root().name == "suites"
    assert "tests" not in suites_root().parts


def test_check_lock_detects_unbumped_edit():
    s = load_suite(DEMO)
    problems = check_lock([s], {s.key: {str(s.version): "0" * 64}})
    assert problems
    assert "bump" in problems[0]
    assert check_lock([s], {s.key: {str(s.version): s.hash}}) == []
    assert "lock_suites" in check_lock([s], {})[0]
    assert "lock_suites" in check_lock([s], {s.key: {"99": s.hash}})[0]


DEMO_SHA = "1ff408296a48a634081c1936e88ccfc5174bf85e0ef42a5d647b915b5e4d2940"
GEN_SHA = "40477e5e9bf900678a76a5ed72696ecf61be37f471c983ad155a7ba7f3f35e7f"


def test_golden_hashes():
    assert load_suite(DEMO).hash == DEMO_SHA
    assert load_suite(DATA / "needle-demo").hash == GEN_SHA


def _edit(tmp_path, name, old, new):
    d = tmp_path / name
    shutil.copytree(DATA / name, d)
    m = d / "manifest.toml"
    text = m.read_text()
    assert old in text
    m.write_text(text.replace(old, new))
    return d


def test_manifest_value_changes_hash(tmp_path):
    d = _edit(tmp_path, "demo", "max_tokens = 16", "max_tokens = 17")
    assert load_suite(d).hash != DEMO_SHA


@pytest.mark.parametrize(
    ("old", "new"),
    [("seed = 11", "seed = 12"), ("lines = [5, 8]", "lines = [5, 9]"), ("0.5]", "0.6]")],
)
def test_generator_param_changes_hash(tmp_path, old, new):
    d = _edit(tmp_path, "needle-demo", old, new)
    assert load_suite(d).hash != GEN_SHA


def test_generated_hash_covers_materialised_items(tmp_path, monkeypatch):
    import bakeoff.needle as needle

    monkeypatch.setattr(needle, "QUESTION", "\n\nCode?")
    assert load_suite(DATA / "needle-demo").hash != GEN_SHA


def test_generated_id_rounding(tmp_path):
    d = _edit(tmp_path, "needle-demo", "depths = [0.0, 0.5]", "depths = [0.29]")
    assert [i.id for i in load_suite(d).items] == ["needle-5-29", "needle-8-29"]


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ('name = "needle"\nlines', 'name = "bogus"\nlines'),
        ("lines = [5, 8]", "lines = []"),
        ("lines = [5, 8]", "lines = [0]"),
        ("lines = [5, 8]", "lines = [1.5]"),
        ("lines = [5, 8]", "lines = [true]"),
        ("depths = [0.0, 0.5]", "depths = [1.5]"),
        ("depths = [0.0, 0.5]", "depths = [-0.1]"),
        ("depths = [0.0, 0.5]", 'depths = ["a"]'),
        ("depths = [0.0, 0.5]", "depths = []"),
        ("seed = 11", "seed = 1.5"),
        ("seed = 11", 'seed = "x"'),
        ("seed = 11\n", ""),
    ],
)
def test_bad_generator_raises(tmp_path, old, new):
    d = _edit(tmp_path, "needle-demo", old, new)
    with pytest.raises(SuiteError):
        load_suite(d)


def test_generator_plus_items_raises(tmp_path):
    d = _edit(tmp_path, "needle-demo", "seed = 11", "seed = 11")
    (d / "items.jsonl").write_text("")
    with pytest.raises(SuiteError):
        load_suite(d)


def test_duplicate_item_ids_raise(tmp_path):
    d = _copy(tmp_path)
    (d / "items.jsonl").write_text(
        '{"id": "a", "prompt": "x", "expected": "x"}\n{"id": "a", "prompt": "y", "expected": "y"}\n'
    )
    with pytest.raises(SuiteError, match="duplicate"):
        load_suite(d)


def test_duplicate_suite_keys_raise(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    shutil.copytree(DEMO, a)
    shutil.copytree(DEMO, b)
    with pytest.raises(SuiteError, match="duplicate"):
        resolve_suites([str(a), str(b)])


def test_unicode_line_separator_in_item(tmp_path):
    d = _copy(tmp_path)
    (d / "items.jsonl").write_bytes(
        '{"id": "a", "prompt": "x\u2028y", "expected": "e"}\r\n'.encode().replace(
            b"\\u2028", "\u2028".encode()
        )
    )
    s = load_suite(d)
    assert len(s.items) == 1
    assert s.items[0].messages[-1]["content"] == "x\u2028y"


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("[source]", "source = 3\n#"),
        ('license = "CC0-1.0"\n', ""),
        ('license = "CC0-1.0"', 'license = ""'),
        ("\n[source]", 'grader_options = "x"\n[source]'),
    ],
)
def test_bad_tables_and_license(tmp_path, old, new):
    d = _copy(tmp_path)
    m = d / "manifest.toml"
    m.write_text(m.read_text().replace(old, new))
    with pytest.raises(SuiteError):
        load_suite(d)


@pytest.mark.parametrize(
    "msgs",
    ["[]", '["x"]', '[{"role": "user"}]', '[{"role": 1, "content": "x"}]', '"x"'],
)
def test_bad_messages_raise(tmp_path, msgs):
    d = _copy(tmp_path)
    (d / "items.jsonl").write_text(f'{{"id": "a", "messages": {msgs}, "expected": "x"}}\n')
    with pytest.raises(SuiteError):
        load_suite(d)


def test_system_not_duplicated(tmp_path):
    d = _copy(tmp_path)
    (d / "items.jsonl").write_text(
        '{"id": "a", "messages": [{"role": "system", "content": "mine"},'
        ' {"role": "user", "content": "q"}], "expected": "x"}\n'
    )
    msgs = load_suite(d).items[0].messages
    assert [m["content"] for m in msgs] == ["mine", "q"]


def test_unknown_grader_raises_suite_error(tmp_path):
    d = _copy(tmp_path)
    m = d / "manifest.toml"
    m.write_text(m.read_text().replace('grader = "exact"', 'grader = "nonsense"'))
    with pytest.raises(SuiteError, match="unknown grader"):
        load_suite(d)


def test_ifeval_grader_is_registered_on_import():
    root = suites_root()
    assert load_suite(root / "general" / "ifeval").grader == "ifeval"


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ('track = "demo"', 'track = "../x"'),
        ('track = "demo"', 'track = "Demo"'),
        ('track = "demo"', 'track = "a b"'),
        ('name = "exact"', 'name = "-exact"'),
        ('name = "exact"', 'name = "ex/act"'),
        ('name = "exact"', 'name = "ex.act"'),
    ],
)
def test_track_and_name_must_be_slugs(tmp_path, old, new):
    d = _copy(tmp_path)
    m = d / "manifest.toml"
    m.write_text(m.read_text().replace(old, new))
    with pytest.raises(SuiteError, match="track|name"):
        load_suite(d)
