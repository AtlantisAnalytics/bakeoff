import pytest

from bakeoff import safejson
from bakeoff.safejson import MAX_DEPTH, nesting_depth


def test_depth_counts_arrays_and_objects():
    assert nesting_depth("1") == 0
    assert nesting_depth('{"a": [1, {"b": []}]}') == 4


def test_depth_ignores_brackets_in_strings():
    assert nesting_depth('["[[[{{{", "]]]"]') == 1
    assert nesting_depth(r'["a \" [[[ still a string"]') == 1
    assert nesting_depth(r'["ends in a backslash \\", [2]]') == 2


def test_limit_is_inclusive():
    at = "[" * MAX_DEPTH + "]" * MAX_DEPTH
    assert safejson.loads(at) is not None
    with pytest.raises(ValueError, match="nested deeper"):
        safejson.loads("[" + at + "]")


def test_deep_input_rejected_before_parsing():
    deep = "[" * 100000 + "]" * 100000  # 3.14 parses this; the limit must still refuse it
    with pytest.raises(ValueError):
        safejson.loads(deep)


def test_kwargs_pass_through():
    def no_constants(name):
        raise ValueError(name)

    with pytest.raises(ValueError):
        safejson.loads("[NaN]", parse_constant=no_constants)
