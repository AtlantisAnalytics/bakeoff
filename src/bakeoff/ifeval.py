"""IFEval instruction checkers (strict mode), a 16-instruction subset.

Semantics follow the IFEval reference implementation, `instruction_following_eval/instructions.py`
in google-research/google-research (Apache-2.0):
https://github.com/google-research/google-research/tree/master/instruction_following_eval

Deviations from upstream:
- Words are counted with `re.findall(r"\\w+", text)` instead of nltk's tokenizer.
- The language check (langdetect) in `change_case:english_*` is not performed; only
  `isupper()` / `islower()` apply.
- Missing or invalid kwargs are an error (BadExpected) instead of being replaced by random
  values, and unknown instruction ids are an error instead of being skipped.
- Only strict mode is implemented (no loose-mode response variants).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from bakeoff.graders import BadExpected, Grade, register

_RELATIONS = ("less than", "at least")


def _relation(kwargs: dict) -> str:
    rel = kwargs.get("relation")
    if rel not in _RELATIONS:
        raise BadExpected(f"relation must be one of {_RELATIONS}: {rel!r}")
    return rel


def _int(kwargs: dict, key: str, minimum: int = 0) -> int:
    v = kwargs.get(key)
    if isinstance(v, bool) or not isinstance(v, int) or v < minimum:
        raise BadExpected(f"{key} must be an int >= {minimum}: {v!r}")
    return v


def _str(kwargs: dict, key: str) -> str:
    v = kwargs.get(key)
    if not isinstance(v, str) or not v.strip():
        raise BadExpected(f"{key} must be a non-empty str: {v!r}")
    return v


def _str_list(kwargs: dict, key: str) -> list[str]:
    v = kwargs.get(key)
    if not isinstance(v, list) or not v or not all(isinstance(s, str) and s for s in v):
        raise BadExpected(f"{key} must be a non-empty list of non-empty strings: {v!r}")
    return v


def _compare(count: int, relation: str, target: int) -> bool:
    return count < target if relation == "less than" else count >= target


def _findall(pattern: str, text: str, flags: int = 0) -> list:
    try:
        return re.findall(pattern, text, flags)
    except re.error as e:
        raise BadExpected(f"invalid regex {pattern!r}: {e}") from None


def _no_comma(out: str, kw: dict) -> bool:
    return "," not in out


def _lowercase(out: str, kw: dict) -> bool:
    return out.islower()


def _capital(out: str, kw: dict) -> bool:
    return out.isupper()


def _json_format(out: str, kw: dict) -> bool:
    value = (
        out.strip()
        .removeprefix("```json")
        .removeprefix("```Json")
        .removeprefix("```JSON")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    try:
        json.loads(value)
    except (ValueError, RecursionError):
        return False
    return True


def _title(out: str, kw: dict) -> bool:
    return any(t.lstrip("<").rstrip(">").strip() for t in re.findall(r"<<[^\n]+>>", out))


def _number_bullets(out: str, kw: dict) -> bool:
    n = _int(kw, "num_bullets", 1)
    star = re.findall(r"^\s*\*[^\*].*$", out, flags=re.MULTILINE)
    dash = re.findall(r"^\s*-.*$", out, flags=re.MULTILINE)
    return len(star) + len(dash) == n


def _end_checker(out: str, kw: dict) -> bool:
    phrase = _str(kw, "end_phrase").strip().lower()
    return out.strip().strip('"').lower().endswith(phrase)


def _quotation(out: str, kw: dict) -> bool:
    v = out.strip()
    return len(v) > 1 and v[0] == '"' and v[-1] == '"'


def _existence(out: str, kw: dict) -> bool:
    keywords = _str_list(kw, "keywords")
    found = [bool(_findall(k, out, re.IGNORECASE)) for k in keywords]
    return all(found)


def _forbidden(out: str, kw: dict) -> bool:
    words = _str_list(kw, "forbidden_words")
    return not any(_findall(r"\b" + w + r"\b", out, re.IGNORECASE) for w in words)


def _frequency(out: str, kw: dict) -> bool:
    keyword = _str(kw, "keyword")
    freq = _int(kw, "frequency")
    rel = _relation(kw)
    return _compare(len(_findall(keyword, out, re.IGNORECASE)), rel, freq)


def _letter_frequency(out: str, kw: dict) -> bool:
    letter = kw.get("letter")
    if not isinstance(letter, str) or len(letter) != 1 or not letter.isalpha():
        raise BadExpected(f"letter must be a single alphabetic character: {letter!r}")
    freq = _int(kw, "let_frequency")
    rel = _relation({"relation": kw.get("let_relation")})
    return _compare(out.lower().count(letter.lower()), rel, freq)


def _placeholders(out: str, kw: dict) -> bool:
    n = _int(kw, "num_placeholders", 1)
    return len(re.findall(r"\[.*?\]", out)) >= n


def _postscript(out: str, kw: dict) -> bool:
    marker = _str(kw, "postscript_marker")
    value = out.lower()
    if marker == "P.P.S":
        pattern = r"\s*p\.\s?p\.\s?s.*$"
    elif marker == "P.S.":
        pattern = r"\s*p\.\s?s\..*$"
    else:
        pattern = r"\s*" + re.escape(marker.lower()) + r".*$"
    return bool(re.findall(pattern, value, flags=re.MULTILINE))


def _number_words(out: str, kw: dict) -> bool:
    n = _int(kw, "num_words")
    rel = _relation(kw)
    return _compare(len(re.findall(r"\w+", out)), rel, n)


def _number_paragraphs(out: str, kw: dict) -> bool:
    n = _int(kw, "num_paragraphs", 1)
    parts = re.split(r"\s?\*\*\*\s?", out)
    count = len(parts)
    for i, p in enumerate(parts):
        if not p.strip():
            if i == 0 or i == len(parts) - 1:
                count -= 1
            else:
                return False
    return count == n


_CHECKERS: dict[str, Callable[[str, dict], bool]] = {
    "punctuation:no_comma": _no_comma,
    "change_case:english_lowercase": _lowercase,
    "change_case:english_capital": _capital,
    "detectable_format:json_format": _json_format,
    "detectable_format:title": _title,
    "detectable_format:number_bullet_lists": _number_bullets,
    "startend:end_checker": _end_checker,
    "startend:quotation": _quotation,
    "keywords:existence": _existence,
    "keywords:forbidden_words": _forbidden,
    "keywords:frequency": _frequency,
    "keywords:letter_frequency": _letter_frequency,
    "detectable_content:number_placeholders": _placeholders,
    "detectable_content:postscript": _postscript,
    "length_constraints:number_words": _number_words,
    "length_constraints:number_paragraphs": _number_paragraphs,
}

SUPPORTED: frozenset[str] = frozenset(_CHECKERS)


def check(instruction_id: str, kwargs: dict, output: str) -> bool:
    """Strict-mode check of one instruction. Raises BadExpected on a bad id or kwargs."""
    fn = _CHECKERS.get(instruction_id) if isinstance(instruction_id, str) else None
    if fn is None:
        raise BadExpected(f"unsupported instruction id: {instruction_id!r}")
    if not isinstance(kwargs, dict):
        raise BadExpected(f"kwargs for {instruction_id} must be an object: {kwargs!r}")
    return fn(output, kwargs)


@register("ifeval")
def _ifeval(output: str, expected: Any, options: dict) -> Grade:
    if not isinstance(expected, dict):
        raise BadExpected("ifeval expected must be an object")
    ids = expected.get("instruction_id_list")
    kws = expected.get("kwargs")
    if not isinstance(ids, list) or not ids or not isinstance(kws, list):
        raise BadExpected("need a non-empty instruction_id_list and a kwargs list")
    if len(ids) != len(kws):
        raise BadExpected(f"instruction_id_list ({len(ids)}) and kwargs ({len(kws)}) differ")
    results = [(i, check(i, k, output)) for i, k in zip(ids, kws, strict=True)]
    passed = sum(ok for _, ok in results)
    failed = next((i for i, ok in results if not ok), None)
    if failed is None:
        return Grade(True, 1.0, "all instructions followed")
    return Grade(False, passed / len(results), f"failed {failed} ({passed}/{len(results)} passed)")
