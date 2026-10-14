from bakeoff.stats import summarize, wilson


def test_wilson_bounds():
    lo, hi = wilson(8, 10)
    assert round(lo, 3) == 0.490 and round(hi, 3) == 0.943


def test_wilson_empty():
    assert wilson(0, 0) == (0.0, 1.0)


def test_summarize():
    assert summarize([3.0, 1.0, 2.0]) == {"median": 2.0, "min": 1.0, "max": 3.0, "n": 3}
    assert summarize([]) == {"median": None, "min": None, "max": None, "n": 0}
