import hashlib
import math
import random

import pytest

from bakeoff.resample import SplitMix64, paired_diff_ci, verdict_seed

GOLDEN_LO = 0.05
GOLDEN_HI = 0.16


def test_splitmix64_reference_vector():
    rng = SplitMix64(1234567)
    got = [rng.next() for _ in range(10)]
    assert got == [
        6457827717110365317,
        3203168211198807973,
        9817491932198370423,
        4593380528125082431,
        16408922859458223821,
        7804594928223864054,
        10895525637215051397,
        5078158048327840177,
        8075865375900838704,
        15101793978218222876,
    ]


def test_verdict_seed_derivation():
    msg = b"bakeoff-verdict-v1|2026-10-19|24gb|general|" + b"a" * 64 + b"|" + b"b" * 64
    expected = int.from_bytes(hashlib.sha256(msg).digest()[:8], "big")
    assert verdict_seed("2026-10-19", "24gb", "general", "a" * 64, "b" * 64) == expected


def test_identical_pairs_zero():
    data = [True] * 30 + [False] * 20
    assert paired_diff_ci([(data, list(data))], 1000, 0.95, 7) == (0.0, 0.0, 0.0)


def test_ci_golden():
    cand = [True] * 80 + [False] * 20
    pick = [True] * 70 + [False] * 30
    diff, lo, hi = paired_diff_ci([(cand, pick)], 10000, 0.95, 42)
    assert diff == 0.1
    assert lo < 0.1 < hi
    assert (lo, hi) == (GOLDEN_LO, GOLDEN_HI)


def _analytic(cand, pick):
    n = len(cand)
    b = sum(1 for c, p in zip(cand, pick, strict=True) if c and not p)
    c_ = sum(1 for c, p in zip(cand, pick, strict=True) if p and not c)
    d = (b - c_) / n
    half = 1.96 * math.sqrt((b + c_) - (b - c_) ** 2 / n) / n
    return d - half, d + half


def test_bootstrap_matches_analytic():
    for s in range(5):
        r = random.Random(1000 + s)
        cand = [r.random() < 0.7 for _ in range(200)]
        pick = [(c if r.random() < 0.8 else not c) for c in cand]
        _, lo, hi = paired_diff_ci([(cand, pick)], 4000, 0.95, 100 + s)
        alo, ahi = _analytic(cand, pick)
        assert abs(lo - alo) < 0.015
        assert abs(hi - ahi) < 0.015


def test_golden_close_to_analytic():
    cand = [True] * 80 + [False] * 20
    pick = [True] * 70 + [False] * 30
    alo, ahi = _analytic(cand, pick)
    assert abs(GOLDEN_LO - alo) <= 0.03
    assert abs(GOLDEN_HI - ahi) <= 0.03


def test_multi_suite_mean():
    s1 = ([True] * 6 + [False] * 4, [True] * 4 + [False] * 6)
    s2 = ([True] * 5 + [False] * 5, [True] * 5 + [False] * 5)
    diff, _, _ = paired_diff_ci([s1, s2], 100, 0.95, 1)
    assert math.isclose(diff, 0.1)


def test_multi_suite_golden_unequal_n():
    # Cross-checked against an independent C port (splitmix64 + same loop order).
    s1 = ([(i * 7) % 5 < 3 for i in range(37)], [(i * 3) % 4 < 2 for i in range(37)])
    s2 = ([(i * 5) % 7 < 4 for i in range(101)], [(i * 11) % 6 < 3 for i in range(101)])
    assert paired_diff_ci([s1, s2], 5000, 0.95, 2024) == (
        0.093658014450093668,
        -0.039603960396039604,
        0.23013112122023013,
    )


def test_invalid_inputs():
    ok = ([True], [False])
    with pytest.raises(ValueError):
        paired_diff_ci([], 10, 0.95, 1)
    with pytest.raises(ValueError):
        paired_diff_ci([([], [])], 10, 0.95, 1)
    with pytest.raises(ValueError):
        paired_diff_ci([ok], 0, 0.95, 1)
