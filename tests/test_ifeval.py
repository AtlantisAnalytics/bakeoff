import pytest

from bakeoff.graders import grade
from bakeoff.ifeval import SUPPORTED, check

CASES = [
    ("punctuation:no_comma", {}, "no commas here.", "one, two"),
    ("change_case:english_lowercase", {}, "all lower", "Not all lower"),
    ("change_case:english_capital", {}, "ALL UPPER", "ALL UPPER but Not"),
    ("detectable_format:json_format", {}, '```json\n{"a": 1}\n```', "{not json"),
    ("detectable_format:title", {}, "<<A Title>>\nbody", "no title\n<< >>"),
    ("detectable_format:number_bullet_lists", {"num_bullets": 2}, "* a\n* b\n", "* a\n* b\n- c"),
    (
        "startend:end_checker",
        {"end_phrase": "Is there anything else?"},
        "x. Is there anything else?",
        "x.",
    ),
    ("startend:quotation", {}, '  "wrapped"  ', '"only start'),
    ("keywords:existence", {"keywords": ["alpha", "beta"]}, "Alpha and BETA", "alpha only"),
    ("keywords:forbidden_words", {"forbidden_words": ["bad"]}, "all good, badge ok", "so Bad."),
    (
        "keywords:frequency",
        {"keyword": "cat", "frequency": 2, "relation": "at least"},
        "cat Cat",
        "cat",
    ),
    (
        "keywords:letter_frequency",
        {"letter": "z", "let_frequency": 3, "let_relation": "less than"},
        "zz",
        "zzz",
    ),
    ("detectable_content:number_placeholders", {"num_placeholders": 2}, "[a] and [b]", "[a] only"),
    ("detectable_content:postscript", {"postscript_marker": "P.S."}, "body\nP.S. hi", "body"),
    (
        "length_constraints:number_words",
        {"num_words": 5, "relation": "less than"},
        "one two three",
        "one two three four five",
    ),
    (
        "length_constraints:number_paragraphs",
        {"num_paragraphs": 2},
        "first\n***\nsecond",
        "first\n***\n***\nsecond",
    ),
]


@pytest.mark.parametrize("iid,kwargs,good,bad", CASES, ids=[c[0] for c in CASES])
def test_checker_pass_and_fail(iid, kwargs, good, bad):
    assert check(iid, kwargs, good) is True
    assert check(iid, kwargs, bad) is False


def test_supported_count():
    assert len(SUPPORTED) == 16
    assert SUPPORTED == {c[0] for c in CASES}


def test_ifeval_grader_all_must_pass():
    exp = {
        "instruction_id_list": ["punctuation:no_comma", "change_case:english_lowercase"],
        "kwargs": [{}, {}],
    }
    g = grade("ifeval", "all lower, but a comma", exp, {})
    assert g.passed is False and g.score == 0.5 and "no_comma" in g.reason


def test_ifeval_grader_all_pass():
    exp = {"instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]}
    g = grade("ifeval", "fine", exp, {})
    assert g.passed is True and g.score == 1.0


def test_paragraph_edges_and_bullets():
    kw = {"num_paragraphs": 2}
    assert check("length_constraints:number_paragraphs", kw, "***a***b***") is True
    assert (
        check("detectable_format:number_bullet_lists", {"num_bullets": 1}, "** bold\n* a") is True
    )


def test_postscript_pps_variant():
    kw = {"postscript_marker": "P.P.S"}
    assert check("detectable_content:postscript", kw, "x\nP.P.S later") is True


@pytest.mark.parametrize(
    "exp",
    [
        {"instruction_id_list": ["nope:unknown"], "kwargs": [{}]},
        {"instruction_id_list": ["punctuation:no_comma"], "kwargs": []},
        {"instruction_id_list": [], "kwargs": []},
        {"instruction_id_list": ["keywords:existence"], "kwargs": [{}]},
        {"instruction_id_list": ["keywords:existence"], "kwargs": [{"keywords": ["(unclosed"]}]},
        {
            "instruction_id_list": ["keywords:frequency"],
            "kwargs": [{"keyword": "a", "frequency": 1, "relation": "exactly"}],
        },
        {
            "instruction_id_list": ["length_constraints:number_words"],
            "kwargs": [{"num_words": True, "relation": "at least"}],
        },
        {
            "instruction_id_list": ["keywords:letter_frequency"],
            "kwargs": [{"letter": "ab", "let_frequency": 1, "let_relation": "at least"}],
        },
        {"instruction_id_list": "punctuation:no_comma", "kwargs": [{}]},
        "not a dict",
    ],
)
def test_bad_expected_fails_closed(exp):
    g = grade("ifeval", "anything", exp, {})
    assert g.passed is False and g.score == 0.0 and g.reason.startswith("bad expected")


def test_check_unknown_id_raises():
    from bakeoff.graders import BadExpected

    with pytest.raises(BadExpected):
        check("nope:unknown", {}, "x")


@pytest.mark.parametrize(
    ("iid", "kwargs"),
    [
        ("keywords:existence", {"keywords": [" cat"]}),
        ("keywords:existence", {"keywords": ["   "]}),  # would match any reply with a space
        ("keywords:frequency", {"keyword": "cat ", "frequency": 1, "relation": "at least"}),
        ("detectable_content:postscript", {"postscript_marker": " P.S."}),
        ("startend:end_checker", {"end_phrase": "Thanks. "}),
    ],
)
def test_padded_string_kwargs_are_bad_expected(iid, kwargs):
    """Surrounding whitespace in an instruction argument is a suite-data error, not a pattern."""
    from bakeoff.graders import BadExpected

    with pytest.raises(BadExpected):
        check(iid, kwargs, "a cat. P.S. Thanks.")
