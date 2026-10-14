import json

from bakeoff.bundle import scan_text
from bakeoff.graders import grade
from bakeoff.suites import check_lock, resolve_suites, suites_root

EXPECTED = {
    "format/exact-reply": 20,
    "structured/json": 30,
    "grounding/context-only": 30,
    "general/open-ended": 20,
    "longctx/needle": 6,
}
LOCK = suites_root() / "SUITES.lock.json"


def _suites():
    return {s.key: s for s in resolve_suites([str(suites_root())])}


def raw_items(s):
    f = s.path / "items.jsonl"
    if not f.is_file():
        return []
    return [json.loads(ln) for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_counts():
    found = {k: len(s.items) for k, s in _suites().items()}
    for key, n in EXPECTED.items():
        assert found[key] == n


def test_common_manifest_fields():
    for key in EXPECTED:
        s = _suites()[key]
        assert s.version == 1 and s.split == "public" and s.license == "CC-BY-4.0"


def test_reference_outputs_pass():
    n = 0
    for s in _suites().values():
        for raw in raw_items(s):
            if "reference" in raw:
                n += 1
                r = grade(s.grader, raw["reference"], raw["expected"], s.grader_options)
                assert r.passed, (s.key, raw["id"], r)
    assert n == 80


def test_wrong_outputs_fail():
    s = _suites()["grounding/context-only"]
    for raw in raw_items(s):
        e = raw["expected"]
        assert not grade(s.grader, "I do not know", e, s.grader_options).passed, raw["id"]
        for bad in e["must_not_include"]:
            out = ("NOT IN CONTEXT " if not e["answerable"] else raw["reference"] + " ") + bad
            assert not grade(s.grader, out, e, s.grader_options).passed, (raw["id"], bad)


def test_explained_refusals_pass():
    s = _suites()["grounding/context-only"]
    for raw in raw_items(s):
        if not raw["expected"]["answerable"]:
            r = grade(s.grader, raw["reference_explained"], raw["expected"], s.grader_options)
            assert r.passed, (raw["id"], r)


def test_json_values_are_checked():
    s = _suites()["structured/json"]
    raw = next(r for r in raw_items(s) if r["id"] == "json-05")
    assert not grade(s.grader, '{"sentiment": "positive"}', raw["expected"], {}).passed
    for r in raw_items(s):
        assert r["expected"]["values"], r["id"]


def test_needle_and_open_ended_settings():
    assert _suites()["longctx/needle"].max_tokens == 1024
    assert all(i.expected is None for i in _suites()["general/open-ended"].items)


def test_all_suite_text_scrub_clean():
    for p in suites_root().rglob("*"):
        if p.suffix in {".toml", ".jsonl", ".md"}:
            assert scan_text(p.read_text(encoding="utf-8"), []) == [], p


def test_lock_matches():
    found = list(_suites().values())
    assert check_lock(found, json.loads(LOCK.read_text(encoding="utf-8"))) == []
