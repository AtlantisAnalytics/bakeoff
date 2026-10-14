import re
from pathlib import Path

from bakeoff.graders import grade
from bakeoff.ifeval import SUPPORTED
from bakeoff.suites import load_suite, suites_root

ROOT = Path(__file__).resolve().parents[1]


def test_gsm8k_slice():
    s = load_suite(suites_root() / "general/gsm8k")
    assert len(s.items) == 100 and s.grader == "numeric" and s.license == "MIT"
    assert all(re.fullmatch(r"-?\d+(\.\d+)?", str(i.expected)) for i in s.items)


def test_ifeval_slice_only_supported():
    s = load_suite(suites_root() / "general/ifeval")
    assert 50 <= len(s.items) <= 100 and s.license == "Apache-2.0"
    assert all(set(i.expected["instruction_id_list"]) <= SUPPORTED for i in s.items)


def test_slice_sources_pinned():
    for name in ("gsm8k", "ifeval"):
        src = load_suite(suites_root() / "general" / name).source
        assert re.fullmatch(r"[0-9a-f]{40}", src["revision"]) and len(src["sha256"]) == 64
        assert src["revision"] in src["url"] and src["revision"] in src["license_url"]
        assert src["dropped_scrub"] >= 0 and src["pool_size"] >= 50


def test_slices_have_attribution():
    for name, author in (("gsm8k", "Cobbe"), ("ifeval", "Zhou")):
        text = (suites_root() / "general" / name / "ATTRIBUTION.md").read_text(encoding="utf-8")
        assert author in text and "unmodified" in text


def test_ifeval_expected_has_no_none_and_matching_lengths():
    s = load_suite(suites_root() / "general/ifeval")
    for i in s.items:
        e = i.expected
        assert len(e["instruction_id_list"]) == len(e["kwargs"])
        assert all(v is not None for kw in e["kwargs"] for v in kw.values())


def test_gsm8k_expected_grades_itself():
    s = load_suite(suites_root() / "general/gsm8k")
    for i in s.items[:10]:
        r = grade(s.grader, f"Answer: {i.expected}", i.expected, s.grader_options)
        assert r.passed, (i.id, r)


def test_make_slices_script_present():
    assert (ROOT / "scripts" / "make_slices.py").is_file()
