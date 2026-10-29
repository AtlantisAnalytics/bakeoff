"""Grader registry: decides pass or fail for every published quality score.

A grader never raises on bad output or a malformed `expected`: both become a failing Grade.
Only an unknown grader name raises (KeyError).
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from bakeoff import safejson


@dataclass(frozen=True)
class Grade:
    """Result of grading one output. passed=None means deferred (no verdict yet)."""

    passed: bool | None
    score: float
    reason: str
    extractable: bool | None = None


class BadExpected(Exception):
    """The suite's `expected` (or grader options) is malformed; graded as a fail."""


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
    try:
        return fn(output, expected, options)
    except BadExpected as e:
        return Grade(False, 0.0, f"bad expected: {e}", False if grader == "json_schema" else None)


def _result(ok: bool, reason: str) -> Grade:
    return Grade(ok, 1.0 if ok else 0.0, reason)


@register("exact")
def _exact(output: str, expected: Any, options: dict) -> Grade:
    ok = output.strip() == expected
    return _result(ok, "match" if ok else "output differs from expected")


# Optional sign (ASCII or Unicode minus/en dash), optional "$", digits with valid thousands
# grouping or plain digits, optional decimals and exponent. A "." only counts as a decimal point
# when digits follow it.
_SIGNS = "-+−–"
_NUM_RE = re.compile(
    r"(?<![\w.])"
    rf"(?P<s1>[{_SIGNS}])?\$?(?P<s2>(?<=\$)[{_SIGNS}])?"
    r"(?P<int>\d{1,3}(?:,\d{3})+(?!\d)|\d+)"
    r"(?P<frac>\.\d+)?"
    r"(?P<exp>[eE][-+]?\d+)?"
)
_NEGATIVE = ("-", "−", "–")
_PLAIN_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def _last_number(text: str) -> float | None:
    last = None
    for m in _NUM_RE.finditer(text):
        last = m
    if last is None:
        return None
    value = float(last["int"].replace(",", "") + (last["frac"] or "") + (last["exp"] or ""))
    return -value if last["s1"] in _NEGATIVE or last["s2"] in _NEGATIVE else value


@register("numeric")
def _numeric(output: str, expected: Any, options: dict) -> Grade:
    if isinstance(expected, bool):
        raise BadExpected(f"numeric expected is a bool: {expected!r}")
    if isinstance(expected, str):
        if not _PLAIN_NUMBER_RE.fullmatch(expected.strip()):
            raise BadExpected(f"numeric expected is not a plain number: {expected!r}")
        want = float(expected)
    elif isinstance(expected, (int, float)):
        want = float(expected)
    else:
        raise BadExpected(f"numeric expected must be a str or number: {expected!r}")
    tol = options.get("tolerance", 1e-6)
    if isinstance(tol, bool) or not isinstance(tol, (int, float)) or not tol >= 0:
        raise BadExpected(f"tolerance must be a non-negative number: {tol!r}")
    if not (math.isfinite(want) and math.isfinite(tol)):
        raise BadExpected("expected and tolerance must be finite")
    got = _last_number(output)
    if got is None:
        return _result(False, "no number in output")
    ok = abs(got - want) <= tol
    return _result(ok, f"got {got:g}, expected {want:g}")


@register("regex")
def _regex(output: str, expected: Any, options: dict) -> Grade:
    if not isinstance(expected, str):
        raise BadExpected(f"regex expected must be a str: {expected!r}")
    flags = re.IGNORECASE if options.get("ignore_case") else 0
    try:
        ok = re.search(expected, output, flags) is not None
    except re.error as e:
        raise BadExpected(f"invalid regex: {e}") from None
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
_ANNOTATIONS = {"$schema", "$id", "title", "description"}  # accepted, never enforced
_KEYWORDS = {
    "type",
    "properties",
    "required",
    "enum",
    "items",
    "additionalProperties",
} | _ANNOTATIONS


def _type_names(t: Any) -> list[str] | None:
    """The type names of a `type` keyword (a name or a non-empty list of names), or None."""
    names = t if isinstance(t, list) else [t]
    if not names or not all(isinstance(n, str) and n in _TYPE_CHECKS for n in names):
        return None
    return names


def _enum_eq(a: Any, b: Any) -> bool:
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    return a == b


def schema_problems(schema: Any, path: str = "$") -> list[str]:
    """Fail closed: report anything in the schema outside the supported subset."""
    if not isinstance(schema, dict):
        return [f"{path}: schema must be an object"]
    out = [f"{path}: unsupported keyword {k!r}" for k in schema if k not in _KEYWORDS]
    t = schema.get("type")
    if "type" in schema and _type_names(t) is None:
        out.append(f"{path}: unsupported type {t!r}")
    if "properties" in schema:
        props = schema["properties"]
        if not isinstance(props, dict):
            out.append(f"{path}: unsupported 'properties' (must be an object)")
        else:
            for key, sub in props.items():
                out.extend(schema_problems(sub, f"{path}.{key}"))
    if "required" in schema:
        req = schema["required"]
        if not isinstance(req, list) or not all(isinstance(r, str) for r in req):
            out.append(f"{path}: unsupported 'required' (must be a list of strings)")
    if "enum" in schema and not isinstance(schema["enum"], list):
        out.append(f"{path}: unsupported 'enum' (must be a list)")
    if "items" in schema:
        if isinstance(schema["items"], dict):
            out.extend(schema_problems(schema["items"], f"{path}[]"))
        else:
            out.append(f"{path}: unsupported 'items' (must be an object)")
    if "additionalProperties" in schema:
        ap = schema["additionalProperties"]
        if isinstance(ap, dict):
            out.extend(schema_problems(ap, f"{path}.*"))
        elif not isinstance(ap, bool):
            out.append(f"{path}: unsupported 'additionalProperties' (must be a bool or schema)")
    return out


def _validate(value: Any, schema: dict, path: str) -> list[str]:
    errors: list[str] = []
    if "type" in schema:
        names = _type_names(schema["type"]) or []
        if not any(_TYPE_CHECKS[n](value) for n in names):
            return [f"{path}: expected {'/'.join(names)}, got {type(value).__name__}"]
    if "enum" in schema and not any(_enum_eq(value, e) for e in schema["enum"]):
        errors.append(f"{path}: {value!r} not in enum {schema['enum']!r}")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required property {key!r}")
        for key, sub in props.items():
            if key in value:
                errors.extend(_validate(value[key], sub, f"{path}.{key}"))
        extra = schema.get("additionalProperties")
        for key in value:
            if key in props:
                continue
            if extra is False:
                errors.append(f"{path}: additional property {key!r} not allowed")
            elif isinstance(extra, dict):
                errors.extend(_validate(value[key], extra, f"{path}.{key}"))
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            errors.extend(_validate(item, schema["items"], f"{path}[{i}]"))
    return errors


def validate_schema(value: Any, schema: dict, path: str = "$") -> list[str]:
    """Validate against a JSON Schema subset. Returns a list of error strings.

    Any keyword outside the subset is reported as an error rather than ignored.
    """
    return schema_problems(schema, path) or _validate(value, schema, path)


def _reject_constant(name: str):
    raise ValueError(f"non-standard JSON constant {name}")


def _parse_float(s: str) -> float:
    f = float(s)
    if not math.isfinite(f):
        raise ValueError(f"non-finite number {s}")
    return f


def _norm_eq(a: Any, b: Any) -> bool:
    """Value equality for `values`: strings casefolded and stripped, numbers numeric (bool is
    never a number), dicts compared on the expected keys only, lists as multisets."""
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().casefold() == b.strip().casefold()
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if _is_number(a) and _is_number(b):
        return a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return all(k in a and _norm_eq(a[k], v) for k, v in b.items())
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        rest = list(b)
        for x in a:
            for i, y in enumerate(rest):
                if _norm_eq(x, y):
                    del rest[i]
                    break
            else:
                return False
        return True
    return a is None and b is None


def _check_values_shape(v: Any) -> None:
    """Lists of dicts need identical key sets so greedy multiset matching is unambiguous."""
    if isinstance(v, dict):
        for x in v.values():
            _check_values_shape(x)
    elif isinstance(v, list):
        shapes = {frozenset(x) for x in v if isinstance(x, dict)}
        if len(shapes) > 1:
            raise BadExpected("values list elements are dicts with differing keys")
        for x in v:
            _check_values_shape(x)


def _check_json(text: str, schema: dict, values: dict | None) -> tuple[bool, str]:
    """Parse `text` as strict JSON, then check the schema and `values`. Returns (ok, reason)."""
    try:
        value = safejson.loads(text, parse_constant=_reject_constant, parse_float=_parse_float)
    except ValueError as e:
        return False, f"not valid JSON: {type(e).__name__}: {str(e)[:100]}"
    errors = validate_schema(value, schema)
    if errors:
        return False, "; ".join(errors[:3])
    for key, want in (values or {}).items():
        if not isinstance(value, dict) or key not in value:
            return False, f"value mismatch: {key!r} missing"
        if not _norm_eq(value[key], want):
            return False, f"value mismatch: {key!r} got {value[key]!r}, want {want!r}"[:200]
    return True, "valid"


_ANY_FENCE_RE = re.compile(r"```(?:json)?[ \t]*\n?(.*?)\n?[ \t]*```", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<json>(.*?)</json>", re.DOTALL | re.IGNORECASE)


def _balanced_span(text: str) -> str | None:
    """The span from the first '{' or '[' to its matching close, ignoring brackets in strings."""
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        return None
    start = min(starts)
    depth = 0
    in_str = esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c in "{[":
            depth += 1
        elif c in "}]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _candidates(output: str):
    fences = _ANY_FENCE_RE.findall(output)
    if len(fences) == 1:
        yield fences[0]
    tags = _TAG_RE.findall(output)
    if len(tags) == 1:
        yield tags[0]
    # Several fences or tags make the answer ambiguous, so the bare-span fallback is skipped too.
    span = _balanced_span(output) if len(fences) < 2 and len(tags) < 2 else None
    if span is not None:
        yield span


def _extractable(output: str, schema: dict, values: dict | None) -> bool:
    """Lenient score: does any candidate span in the output parse and pass the same checks?"""
    return any(_check_json(c, schema, values)[0] for c in _candidates(output))


@register("json_schema")
def _json_schema(output: str, expected: Any, options: dict) -> Grade:
    """`expected` is a schema, or {"schema": {...}, "values": {...}} to also check field values."""
    values = None
    if isinstance(expected, dict) and "schema" in expected:
        extra = set(expected) - {"schema", "values"}
        if extra:
            raise BadExpected(f"unknown keys in expected: {sorted(extra)}")
        values = expected.get("values", {})
        if not isinstance(values, dict):
            raise BadExpected("values must be an object")
        expected = expected["schema"]
    problems = schema_problems(expected)
    if problems:
        raise BadExpected("; ".join(problems[:3]))
    if values:
        props = expected.get("properties", {})
        unknown = [k for k in values if k not in props]
        if unknown:
            raise BadExpected(f"values keys not in schema properties: {unknown}")
        _check_values_shape(values)
    m = _FENCE_RE.match(output)
    ok, reason = _check_json(m.group(1) if m else output, expected, values)
    return Grade(ok, 1.0 if ok else 0.0, reason, ok or _extractable(output, expected, values))


@register("needle")
def _needle(output: str, expected: Any, options: dict) -> Grade:
    if not isinstance(expected, str) or not expected:
        raise BadExpected(f"needle expected must be a non-empty str: {expected!r}")
    ok = expected in output
    return _result(ok, "needle found" if ok else "needle not found")


def _phrases(expected: dict, key: str) -> list[str]:
    items = expected.get(key, [])
    if not isinstance(items, list) or not all(isinstance(s, str) and s for s in items):
        raise BadExpected(f"{key} must be a list of non-empty strings")
    return items


_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"), "-")


def _fold(s: str) -> str:
    """Lower case, with Unicode spaces and dashes as their ASCII forms: gpt-oss writes "Mira\u202f
    Svensson" and "2026\u201105\u201126", which must match "Mira Svensson" and "2026-05-26"."""
    return unicodedata.normalize("NFKC", s).translate(_DASHES).lower()


@register("grounding")
def _grounding(output: str, expected: Any, options: dict) -> Grade:
    if not isinstance(expected, dict) or not isinstance(expected.get("answerable"), bool):
        raise BadExpected("grounding expected must be an object with a bool 'answerable'")
    must_include = _phrases(expected, "must_include")
    must_not = _phrases(expected, "must_not_include")
    marker = options.get("refusal", "NOT IN CONTEXT")
    if not isinstance(marker, str) or not marker:
        raise BadExpected("refusal marker must be a non-empty str")
    text = _fold(output)
    refused = _fold(marker) in text
    forbidden = [s for s in must_not if _fold(s) in text]
    if forbidden:
        return _result(False, f"forbidden present: {forbidden}")
    if expected["answerable"]:
        if refused:
            return _result(False, "refused an answerable item")
        missing = [s for s in must_include if _fold(s) not in text]
        if missing:
            return _result(False, f"missing: {missing}")
        return _result(True, "grounded")
    if not refused:
        return _result(False, "did not refuse an unanswerable item")
    return _result(True, "refused")


@register("judge")
def _judge(output: str, expected: Any, options: dict) -> Grade:
    return Grade(None, 0.0, "deferred to bakeoff judge")


from bakeoff import ifeval  # noqa: E402,F401
