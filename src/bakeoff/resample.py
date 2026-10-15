"""Deterministic paired bootstrap, reproducible bit-for-bit in any language.

The generator is splitmix64 (Vigna). Indices are drawn as ``next() % n``; the modulo bias is
negligible for n < 2**32. The seed for a verdict is the first 8 bytes (big-endian) of a SHA-256 over
a fixed, pipe-separated message. Standard library only.
"""

from __future__ import annotations

import hashlib
import math

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
    lo = stats[math.floor((1 - ci) / 2 * resamples)]
    hi = stats[math.ceil((1 + ci) / 2 * resamples) - 1]
    return observed, lo, hi
