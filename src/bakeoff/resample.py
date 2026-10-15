"""Deterministic paired bootstrap, reproducible bit-for-bit in any language.

Reproducible contract:

* Seed: ``verdict_seed`` is the first 8 bytes (big-endian) of SHA-256 over
  ``bakeoff-verdict-v1|week|tier|track|cand_sha|pick_sha`` (UTF-8).
* Generator: splitmix64, one instance seeded once with that seed and shared by every draw.
  All arithmetic is mod 2**64; an index is ``next() % n`` (negligible modulo bias for n < 2**32).
* Data: suites arrive ordered by suite key, items paired and ordered by id. For each suite,
  ``d_i = int(candidate_i) - int(pick_i)``.
* Draw order: resample-major. For each of the ``resamples`` iterations, for each suite in the given
  order, draw ``n`` indices sequentially (``n`` = that suite's item count). The suite value is
  ``sum(d[idx]) / n`` (integer sum, then one float division).
* Statistic: the suite values summed in suite order (float, starting from 0.0) and divided by the
  suite count k. ``diff`` is the same formula on the unresampled data.
* Interval: statistics sorted ascending; with ``ci_milli = round(ci * 1000)`` and B resamples,
  ``lo = s[(1000 - ci_milli) * B // 2000]`` and
  ``hi = s[ceil((1000 + ci_milli) * B / 2000) - 1]`` (integer arithmetic throughout).

Standard library only.
"""

from __future__ import annotations

import hashlib

_MASK = 0xFFFFFFFFFFFFFFFF


class SplitMix64:
    """splitmix64 generator; all arithmetic is mod 2**64."""

    def __init__(self, seed: int) -> None:
        self.state = seed & _MASK

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
        return z ^ (z >> 31)


def verdict_seed(week: str, tier: str, track: str, cand_sha: str, pick_sha: str) -> int:
    msg = f"bakeoff-verdict-v1|{week}|{tier}|{track}|{cand_sha}|{pick_sha}".encode()
    return int.from_bytes(hashlib.sha256(msg).digest()[:8], "big")


def paired_diff_ci(
    suites: list[tuple[list[bool], list[bool]]], resamples: int, ci: float, seed: int
) -> tuple[float, float, float]:
    """Return (diff, lo, hi): mean over suites of the paired pass-rate difference, with a
    percentile bootstrap interval. Suites must arrive ordered by key, items paired and ordered."""
    if resamples < 1:
        raise ValueError("resamples must be at least 1")
    if not suites:
        raise ValueError("suites must not be empty")
    if any(len(cs) == 0 for cs, _ in suites):
        raise ValueError("every suite must contain at least one item")
    diffs = [[int(c) - int(p) for c, p in zip(cs, ps, strict=True)] for cs, ps in suites]
    k = len(diffs)
    observed = sum(sum(d) / len(d) for d in diffs) / k
    rng = SplitMix64(seed)
    nxt = rng.next
    stats = []
    for _ in range(resamples):
        total = 0.0
        for d in diffs:
            n = len(d)
            total += sum([d[nxt() % n] for _ in range(n)]) / n
        stats.append(total / k)
    stats.sort()
    ci_milli = round(ci * 1000)
    lo = stats[(1000 - ci_milli) * resamples // 2000]
    hi = stats[-((-(1000 + ci_milli) * resamples) // 2000) - 1]
    return observed, lo, hi
