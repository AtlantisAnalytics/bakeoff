"""Grader registry: decides pass or fail for every published quality score."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Grade:
    """Result of grading one output. passed=None means deferred (no verdict yet)."""

    passed: bool | None
    score: float
    reason: str


GRADERS: dict[str, Callable[[str, Any, dict], Grade]] = {}


def register(name: str):
    def deco(fn: Callable[[str, Any, dict], Grade]):
        GRADERS[name] = fn
        return fn

    return deco


def grade(grader: str, output: str, expected: Any, options: dict) -> Grade:
    try:
        fn = GRADERS[grader]
    except KeyError:
        raise KeyError(f"unknown grader: {grader!r}") from None
    return fn(output, expected, options)


def _result(ok: bool, reason: str) -> Grade:
    return Grade(ok, 1.0 if ok else 0.0, reason)


@register("exact")
def _exact(output: str, expected: Any, options: dict) -> Grade:
    ok = output.strip() == expected
    return _result(ok, "match" if ok else "output differs from expected")


# Optional sign, optional "$", digits with valid thousands grouping or plain digits, optional
# decimals. A "." only counts as a decimal point when digits follow it.
_NUM_RE = re.compile(
    r"(?<![\w.])"
    r"(?P<s1>[-+])?\$?(?P<s2>(?<=\$)[-+])?"
    r"(?P<int>\d{1,3}(?:,\d{3})+(?!\d)|\d+)"
    r"(?P<frac>\.\d+)?"
)


def _last_number(text: str) -> float | None:
    last = None
    for m in _NUM_RE.finditer(text):
        last = m
    if last is None:
        return None
    value = float(last["int"].replace(",", "") + (last["frac"] or ""))
    return -value if "-" in (last["s1"], last["s2"]) else value


@register("numeric")
def _numeric(output: str, expected: Any, options: dict) -> Grade:
    try:
        want = float(str(expected).replace(",", "") if isinstance(expected, str) else expected)
    except (TypeError, ValueError):
        return _result(False, f"expected is not numeric: {expected!r}")
    got = _last_number(output)
    if got is None:
        return _result(False, "no number in output")
    tol = options.get("tolerance", 1e-6)
    ok = abs(got - want) <= tol
    return _result(ok, f"got {got:g}, expected {want:g}")


@register("regex")
def _regex(output: str, expected: Any, options: dict) -> Grade:
    flags = re.IGNORECASE if options.get("ignore_case") else 0
    ok = re.search(expected, output, flags) is not None
    return _result(ok, "pattern matched" if ok else "pattern did not match")


_FENCE_RE = re.compile(r"\A\s*```(?:json)?[ \t]*\n?(.*?)\n?[ \t]*```\s*\Z", re.DOTALL)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


_TYPE_CHECKS: dict[str, Callable[[Any], bool]] = {
    "string": lambda v: isinstance(v, str),
    "number": _is_number,
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "null": lambda v: v is None,
}


def _enum_eq(a: Any, b: Any) -> bool:
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    return a == b


def validate_schema(value: Any, schema: dict, path: str = "$") -> list[str]:
    """Validate against a JSON Schema subset. Returns a list of error strings."""
    errors: list[str] = []
    if "type" in schema:
        check = _TYPE_CHECKS.get(schema["type"])
        if check is None:
            return [f"{path}: unsupported type {schema['type']!r}"]
        if not check(value):
            return [f"{path}: expected {schema['type']}, got {type(value).__name__}"]
    if "enum" in schema and not any(_enum_eq(value, e) for e in schema["enum"]):
        errors.append(f"{path}: {value!r} not in enum {schema['enum']!r}")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required property {key!r}")
        for key, sub in props.items():
            if key in value:
                errors.extend(validate_schema(value[key], sub, f"{path}.{key}"))
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in props:
                    errors.append(f"{path}: additional property {key!r} not allowed")
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, item in enumerate(value):
            errors.extend(validate_schema(item, schema["items"], f"{path}[{i}]"))
    return errors


def _reject_constant(name: str):
    raise ValueError(f"non-standard JSON constant {name}")


@register("json_schema")
def _json_schema(output: str, expected: Any, options: dict) -> Grade:
    m = _FENCE_RE.match(output)
    text = m.group(1) if m else output
    try:
        value = json.loads(text, parse_constant=_reject_constant)
    except ValueError as e:
        return _result(False, f"not valid JSON: {e}")
    errors = validate_schema(value, expected)
    if errors:
        return _result(False, "; ".join(errors[:3]))
    return _result(True, "valid")


@register("needle")
def _needle(output: str, expected: Any, options: dict) -> Grade:
    ok = expected in output
    return _result(ok, "needle found" if ok else "needle not found")


@register("grounding")
def _grounding(output: str, expected: Any, options: dict) -> Grade:
    text = output.lower()
    marker = options.get("refusal", "NOT IN CONTEXT").lower()
    refused = marker in text
    forbidden = [s for s in expected.get("must_not_include", []) if s.lower() in text]
    if forbidden:
        return _result(False, f"forbidden present: {forbidden}")
    if expected.get("answerable", True):
        if refused:
            return _result(False, "refused an answerable item")
        missing = [s for s in expected.get("must_include", []) if s.lower() not in text]
        if missing:
            return _result(False, f"missing: {missing}")
        return _result(True, "grounded")
    if not refused:
        return _result(False, "did not refuse an unanswerable item")
    return _result(True, "refused")


@register("judge")
def _judge(output: str, expected: Any, options: dict) -> Grade:
    return Grade(None, 0.0, "deferred to bakeoff judge")
