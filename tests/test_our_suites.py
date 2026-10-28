import json

from bakeoff.bundle import scan_text
from bakeoff.graders import grade
from bakeoff.suites import check_lock, resolve_suites, suites_root

EXPECTED = {  # key: (items, version)
    "format/exact-reply": (30, 2),
    "structured/json": (30, 2),
    "grounding/context-only": (40, 2),
    "general/open-ended": (20, 1),
    "longctx/needle": (27, 2),
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
    for key, (n, _) in EXPECTED.items():
        assert found[key] == n


def test_common_manifest_fields():
    for key, (_, version) in EXPECTED.items():
        s = _suites()[key]
        assert s.version == version and s.split == "public" and s.license == "CC-BY-4.0"


def test_reference_outputs_pass():
    n = 0
    for s in _suites().values():
        for raw in raw_items(s):
            if "reference" in raw:
                n += 1
                r = grade(s.grader, raw["reference"], raw["expected"], s.grader_options)
                assert r.passed, (s.key, raw["id"], r)
    assert n == 100


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
    raw = next(r for r in raw_items(s) if r["id"] == "json-21")  # the schema holds, a value not
    wrong = dict(raw["expected"]["values"], sentiment="positive")
    assert not grade(s.grader, json.dumps(wrong), raw["expected"], {}).passed
    for r in raw_items(s):
        assert r["expected"]["values"], r["id"]


def test_needle_and_open_ended_settings():
    assert _suites()["longctx/needle"].max_tokens == 4096
    assert all(i.expected is None for i in _suites()["general/open-ended"].items)


def test_grounding_balance():
    s = _suites()["grounding/context-only"]
    items = raw_items(s)
    assert sum(1 for r in items if r["expected"]["answerable"] is False) == 14
    assert sum(1 for r in items if r["expected"]["answerable"] is True) == 26
    for r in items:
        e = r["expected"]
        assert e["answerable"] is False or e["must_include"], r["id"]
        for a in e["must_include"]:
            for b in e["must_not_include"]:
                assert a.lower() not in b.lower() and b.lower() not in a.lower(), r["id"]


def test_must_not_include_rules():
    s = _suites()["grounding/context-only"]
    for r in raw_items(s):
        doc = r["prompt"].split("Context:\n")[1].split("\n\nQuestion:")[0].lower()
        e = r["expected"]
        for b in e["must_not_include"]:
            if not e["answerable"]:
                assert b.lower() not in doc, (r["id"], b)  # rule (a)
            assert len(b.split()) >= 2 or any(c.isdigit() for c in b), (r["id"], b)  # rule (b)


def test_all_suite_text_scrub_clean():
    for p in suites_root().rglob("*"):
        if p.suffix in {".toml", ".jsonl", ".md"}:
            assert scan_text(p.read_text(encoding="utf-8"), []) == [], p


def test_lock_matches():
    found = list(_suites().values())
    assert check_lock(found, json.loads(LOCK.read_text(encoding="utf-8"))) == []


def test_output_caps_leave_room_for_reasoning_models():
    caps = {k: s.max_tokens for k, s in _suites().items()}
    assert caps == {
        "general/gsm8k": 4096,
        "general/ifeval": 4096,
        "structured/json": 4096,
        "grounding/context-only": 4096,
        "format/exact-reply": 2048,
        "longctx/needle": 4096,
        "general/open-ended": 2048,
    }
