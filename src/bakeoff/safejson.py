"""JSON parsing with a fixed nesting limit, for bundles and model output we don't control.

The interpreter's own limit is not a contract: Python 3.12 raises RecursionError on JSON nested
~1000 deep, 3.14 parses 100k deep. A fixed limit keeps the verdict the same on every Python."""

import json
import re
from typing import Any

MAX_DEPTH = 500

_STRING = re.compile(r'"(?:[^"\\]|\\.)*"', re.S)
_BRACKET = re.compile(r"[\[\]{}]")


def nesting_depth(text: str) -> int:
    """Deepest bracket nesting outside strings. An unterminated string can only overcount, and
    such text is not valid JSON anyway."""
    depth = peak = 0
    for m in _BRACKET.finditer(_STRING.sub('""', text)):
        if m.group() in "[{":
            depth += 1
            peak = max(peak, depth)
        else:
            depth -= 1
    return peak


def loads(text: str, **kwargs: Any) -> Any:
    """`json.loads`, but ValueError for nesting deeper than MAX_DEPTH."""
    if nesting_depth(text) > MAX_DEPTH:
        raise ValueError(f"JSON nested deeper than {MAX_DEPTH}")
    return json.loads(text, **kwargs)
