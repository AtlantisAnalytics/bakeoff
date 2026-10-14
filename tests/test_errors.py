import pytest

from bakeoff.errors import AbortRun, ErrorBudget


def test_budget_aborts_over_threshold():
    b = ErrorBudget("quality")
    for ok in [True] * 7 + [False] * 2:
        b.record(ok)  # 9 requests: below min_requests
    with pytest.raises(AbortRun, match="quality: 3/10") as exc:
        b.record(False)
    assert exc.value.reason == "quality: 3/10 requests failed"


def test_budget_tolerates_threshold():
    b = ErrorBudget("speed")
    for ok in [True] * 8 + [False] * 2:
        b.record(ok)  # exactly 20% at n=10


def test_budget_ignores_errors_below_min_requests():
    b = ErrorBudget("fit")
    for _ in range(9):
        b.record(False)
