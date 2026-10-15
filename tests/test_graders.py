import json

import pytest

from bakeoff.graders import Grade, grade, validate_schema

OBJ = {
    "type": "object",
    "required": ["a"],
    "properties": {"a": {"type": "integer"}},
    "additionalProperties": False,
}
G = {"must_include": ["degraded"], "must_not_include": ["healthy"], "answerable": True}
U = {"must_include": [], "must_not_include": [], "answerable": False}

CASES = [
    ("exact", "  OK-42 \n", "OK-42", {}, True),
    ("exact", "OK-42.", "OK-42", {}, False),
    ("exact", "OK", "OK ", {}, False),
    ("numeric", "so the total is $1,234.", "1234", {}, True),
    ("numeric", "-3.5 then 7", "7", {}, True),
    ("numeric", "about 72 or 73", "72", {}, False),
    ("numeric", "no digits", "1", {}, False),
    ("numeric", "It is 72.", 72, {}, True),
    ("numeric", "x = -3.5", -3.5, {}, True),
    ("numeric", "$-5 net", "-5", {}, True),
    ("numeric", "-$5 net", "-5", {}, True),
    ("numeric", "1,2", "2", {}, True),
    ("numeric", "1,2", "12", {}, False),
    ("numeric", "1,2345", "2345", {}, True),
    ("numeric", "1,234,567.5", "1234567.5", {}, True),
    ("numeric", "3.14", "3", {"tolerance": 0.2}, True),
    ("numeric", "3.14", "3", {}, False),
    ("numeric", "x", "abc", {}, False),
    ("numeric", "\u22125", "5", {}, False),
    ("numeric", "\u22125", "-5", {}, True),
    ("numeric", "\u20135", "-5", {}, True),
    ("numeric", "+5", "5", {}, True),
    ("numeric", "$\u22125", "-5", {}, True),
    ("numeric", "3\u20135", "5", {}, True),
    ("numeric", "1.5e3", "1500", {}, True),
    ("numeric", "about 2E-2", 0.02, {}, True),
    ("numeric", "1_000", "1000", {}, False),
    ("numeric", "5", True, {}, False),
    ("numeric", "5", "1_000", {}, False),
    ("numeric", "5", None, {}, False),
    ("numeric", "5", "5", {"tolerance": -1}, False),
    ("regex", "Answer: B", r"^Answer: [A-D]$", {}, True),
    ("regex", "answer: b", r"^Answer: [A-D]$", {}, False),
    ("regex", "answer: b", r"^Answer: [A-D]$", {"ignore_case": True}, True),
    ("json_schema", '```json\n{"a": 1}\n```', OBJ, {}, True),
    ("json_schema", '```\n{"a": 1}\n```', OBJ, {}, True),
    ("json_schema", '  ```json\n{"a": 1}\n```  \n', OBJ, {}, True),
    ("json_schema", '{"a": 1}', OBJ, {}, True),
    ("json_schema", '{"a": 1} trailing', {"type": "object"}, {}, False),
    ("json_schema", 'Here: {"a": 1}', {"type": "object"}, {}, False),
    ("json_schema", 'Here:\n```json\n{"a": 1}\n```', OBJ, {}, False),
    ("json_schema", '```json\n{"a": 1}\n``` done', OBJ, {}, False),
    ("json_schema", '{"a": 1.0}', OBJ, {}, False),
    ("json_schema", '{"a": 1, "b": 2}', OBJ, {}, False),
    ("json_schema", '{"a": NaN}', {"type": "object"}, {}, False),
    (
        "json_schema",
        '{"a": "x"}',
        {"type": "object", "properties": {"a": {"enum": ["y"]}}},
        {},
        False,
    ),
    ("needle", "The code is AMBER-4821", "AMBER-4821", {}, True),
    ("needle", "The code is amber-4821", "AMBER-4821", {}, False),
    ("grounding", "Node b is DEGRADED.", G, {}, True),
    ("grounding", "Not sure, maybe healthy.", G, {}, False),
    ("grounding", "degraded but healthy", G, {}, False),
    ("grounding", "NOT IN CONTEXT (degraded)", G, {}, False),
    ("grounding", "NOT IN CONTEXT", U, {}, True),
    ("grounding", "not in context", U, {}, True),
    ("grounding", "It is 42.", U, {}, False),
    ("grounding", "NOT IN CONTEXT", {**U, "must_not_include": ["42"]}, {}, True),
    ("grounding", "NOT IN CONTEXT, maybe 42", {**U, "must_not_include": ["42"]}, {}, False),
    ("grounding", "UNKNOWN", U, {"refusal": "unknown"}, True),
]


@pytest.mark.parametrize("grader,output,expected,options,passed", CASES)
def test_graders(grader, output, expected, options, passed):
    g = grade(grader, output, expected, options)
    assert g.passed is passed
    assert g.score == (1.0 if passed else 0.0)
    assert g.reason


def test_judge_deferred():
    g = grade("judge", "x", None, {})
    assert g == Grade(None, 0.0, "deferred to bakeoff judge")


def test_integer_rejects_bool_and_float():
    assert validate_schema(True, {"type": "integer"})
    assert validate_schema(1.5, {"type": "integer"})
    assert validate_schema(1.0, {"type": "integer"})
    assert validate_schema(True, {"type": "number"})
    assert not validate_schema(2, {"type": "number"})
    assert not validate_schema(2.5, {"type": "number"})


def test_enum_bool_not_int():
    assert validate_schema(True, {"enum": [1]})
    assert not validate_schema(1, {"enum": [1]})


def test_validate_nested_paths():
    schema = {"type": "array", "items": {"type": "object", "required": ["k"]}}
    assert validate_schema([{"k": 1}, {}], schema) == ["$[1]: missing required property 'k'"]


def test_reason_lists_first_three_errors():
    schema = {"type": "object", "required": ["a", "b", "c", "d"]}
    g = grade("json_schema", "{}", schema, {})
    assert g.reason.count("missing required") == 3


def test_unknown_grader():
    with pytest.raises(KeyError, match="nope"):
        grade("nope", "", None, {})


def test_malformed_input_returns_grade_never_raises():
    deep = "[" * 100000
    cases = [
        ("json_schema", deep, {"type": "array"}),
        ("json_schema", "1e999", {"type": "number"}),
        ("json_schema", "[Infinity]", {"type": "array"}),
        ("json_schema", "{}", {"type": ["array", "null"]}),
        ("json_schema", "{}", {"type": "object", "minProperties": 1}),
        ("json_schema", "{}", "object"),
        ("regex", "x", "("),
        ("regex", "x", 5),
        ("needle", "x", ""),
        ("needle", "x", None),
        ("grounding", "x", {"must_include": [], "must_not_include": []}),
        ("grounding", "x", {"answerable": 1}),
        ("grounding", "x", {"answerable": True, "must_include": [""]}),
        ("grounding", "x", {"answerable": True, "must_not_include": [""]}),
        ("grounding", "x", {"answerable": True, "must_include": "x"}),
        ("grounding", "x", None),
    ]
    for name, out, exp in cases:
        g = grade(name, out, exp, {})
        assert isinstance(g, Grade) and g.passed is False, (name, exp)
    g = grade("grounding", "x", {"answerable": True}, {"refusal": ""})
    assert g.passed is False and g.reason.startswith("bad expected")
    assert grade("regex", "x", "(", {}).reason.startswith("bad expected")
    assert grade("json_schema", deep, {"type": "array"}, {}).reason.startswith("not valid JSON")


@pytest.mark.parametrize(
    "schema",
    [
        {"minItems": 1},
        {"pattern": "x"},
        {"const": 1},
        {"anyOf": []},
        {"minimum": 1},
        {"additionalProperties": "string"},
        {"additionalProperties": {"minimum": 1}},
        {"items": [{"type": "string"}]},
        {"type": ["array", "bogus"]},
        {"type": []},
        {"properties": {"a": {"minLength": 1}}},
    ],
)
def test_unsupported_schema_keywords_fail_closed(schema):
    errs = validate_schema({"a": "x"}, schema)
    assert errs and "unsupported" in errs[0]
    assert grade("json_schema", '{"a": "x"}', schema, {}).passed is False


def test_additional_properties_true_ok():
    assert not validate_schema({"a": 1}, {"type": "object", "additionalProperties": True})


VALS = {
    "schema": {
        "type": "object",
        "properties": {"s": {"type": "string"}, "n": {}, "t": {"type": "array"}, "o": {}},
    },
    "values": {"s": "Mixed", "n": 2, "t": ["a", "B"], "o": {"k": "v"}},
}


@pytest.mark.parametrize(
    "out,ok,reason",
    [
        ('{"s": " mixed ", "n": 2.0, "t": ["b", "A"], "o": {"k": "V", "x": 1}}', True, "valid"),
        ('{"s": "positive", "n": 2, "t": ["a", "b"], "o": {"k": "v"}}', False, "'s'"),
        ('{"s": "mixed", "n": true, "t": ["a", "b"], "o": {"k": "v"}}', False, "'n'"),
        ('{"s": "mixed", "n": 2, "t": ["a"], "o": {"k": "v"}}', False, "'t'"),
        ('{"s": "mixed", "n": 2, "t": ["a", "a"], "o": {"k": "v"}}', False, "'t'"),
        ('{"s": "mixed", "n": 2, "t": ["a", "b"], "o": {"k": "w"}}', False, "'o'"),
        ('{"s": "mixed", "t": ["a", "b"], "o": {"k": "v"}}', False, "'n'"),
    ],
)
def test_json_schema_values(out, ok, reason):
    r = grade("json_schema", out, VALS, {})
    assert r.passed is ok and reason in r.reason


def test_json_schema_values_malformed_and_bare_schema_unchanged():
    assert "bad expected" in grade("json_schema", "{}", {"schema": {}, "values": []}, {}).reason
    assert "bad expected" in grade("json_schema", "{}", {"schema": {}, "extra": 1}, {}).reason
    assert grade("json_schema", '{"a": 1}', OBJ, {}).passed


def test_json_values_key_must_be_in_schema():
    exp = {"schema": {"type": "object", "properties": {"a": {}}}, "values": {"b": 1}}
    assert "bad expected" in grade("json_schema", '{"a": 1}', exp, {}).reason


def test_json_values_list_of_dicts_needs_same_keys():
    schema = {"type": "object", "properties": {"t": {}}}
    exp = {"schema": schema, "values": {"t": [{"a": 1}, {"b": 2}]}}
    assert "bad expected" in grade("json_schema", '{"t": []}', exp, {}).reason
    ok = {"schema": schema, "values": {"t": [{"a": 1}, {"a": 2}]}}
    assert grade("json_schema", '{"t": [{"a": 2}, {"a": 1}]}', ok, {}).passed


def test_json_values_reason_is_capped():
    schema = {"type": "object", "properties": {"a": {}}}
    r = grade(
        "json_schema", json.dumps({"a": "x" * 1000}), {"schema": schema, "values": {"a": "y"}}, {}
    )
    assert not r.passed and len(r.reason) <= 200


_OBJ = {"type": "object", "required": ["a"]}
_TWO_FENCES = '```json\n{"a":1}\n```\nand\n```json\n{"a":2}\n```'


@pytest.mark.parametrize(
    "output, strict, extractable",
    [
        ('<json>{"a":1}</json>', False, True),
        ('Sure:\n```json\n{"a":1}\n```\nDone', False, True),
        ('Here: {"a": 1} ok', False, True),
        (_TWO_FENCES, False, False),
        ('{"a":1}', True, True),
        ('{"b":1}', False, False),
        ('<json>{"b":1}</json>', False, False),
        ('Here: {"s": "}{", "a": 1} ok', False, True),
    ],
)
def test_extractable(output, strict, extractable):
    g = grade("json_schema", output, _OBJ, {})
    assert g.passed is strict and g.extractable is extractable


def test_extractable_checks_values():
    exp = {"schema": {**_OBJ, "properties": {"a": {"type": "integer"}}}, "values": {"a": 2}}
    g = grade("json_schema", '<json>{"a":1}</json>', exp, {})
    assert g.passed is False and g.extractable is False
    assert grade("json_schema", '<json>{"a":2}</json>', exp, {}).extractable is True


def test_other_graders_extractable_none():
    assert grade("exact", "x", "x", {}).extractable is None


@pytest.mark.parametrize(
    "output, schema", [("5", {"type": "integer"}), ('"hi"', {"type": "string"})]
)
def test_strict_scalar_pass_is_extractable(output, schema):
    g = grade("json_schema", output, schema, {})
    assert g.passed is True and g.extractable is True


def test_bad_expected_json_schema_extractable_false():
    g = grade("json_schema", "{}", {"type": "bogus"}, {})
    assert g.passed is False and g.extractable is False


def test_type_list_accepts_any_listed_type():
    schema = {"type": ["number", "null"]}
    assert not validate_schema(1.5, schema)
    assert not validate_schema(None, schema)
    errs = validate_schema("x", schema)
    assert errs and "number/null" in errs[0]


def test_additional_properties_schema_validates_map_values():
    schema = {
        "type": "object",
        "properties": {"k": {"type": "string"}},
        "additionalProperties": {"type": "integer"},
    }
    assert not validate_schema({"k": "s", "a": 1, "b": 2}, schema)
    errs = validate_schema({"k": "s", "a": "no"}, schema)
    assert errs == ["$.a: expected integer, got str"]


def test_annotation_keywords_accepted_and_ignored():
    schema = {
        "$schema": "x",
        "$id": "y",
        "title": "t",
        "description": "d",
        "type": "object",
        "properties": {"a": {"description": "d", "type": "integer"}},
    }
    assert not validate_schema({"a": 1}, schema)
    assert validate_schema({"a": "no"}, schema)
    assert "unsupported" in validate_schema({}, {"title": "t", "minItems": 1})[0]
