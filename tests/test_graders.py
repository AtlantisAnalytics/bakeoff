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
