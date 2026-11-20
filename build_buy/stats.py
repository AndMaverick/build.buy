"""Sampling and statistics with honest uncertainty."""
from __future__ import annotations

import math
import random
from collections.abc import Sequence
from statistics import NormalDist


def uniforms(n: int, d: int, rng: random.Random, method: str = "lhs") -> list[list[float]]:
    """``d`` columns of ``n`` uniforms. Latin hypercube puts exactly one point in each of ``n`` equal strata of every column, so a few
    thousand trials cover every region of every input. ``method`` is ``lhs`` or ``random``."""
    if n < 1 or d < 0:
        raise ValueError("need n >= 1 and d >= 0")
    if method not in ("lhs", "random"):
        raise ValueError(f"method must be lhs or random, got {method!r}")
    cols = []
    for _ in range(d):
        if method == "random":
            cols.append([min(max(rng.random(), 1e-12), 1 - 1e-12) for _ in range(n)])
        else:
            perm = list(range(n))
            rng.shuffle(perm)
            cols.append([(perm[i] + rng.random()) / n for i in range(n)])
    return cols


def quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear interpolation between order statistics (the usual 'type 7' definition) of an already sorted sample."""
    n = len(sorted_values)
    if n == 0:
        raise ValueError("no values")
    if n == 1:
        return sorted_values[0]
    pos = q * (n - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, n - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def mean_se(values: Sequence[float]) -> tuple[float, float]:
    n = len(values)
    m = sum(values) / n
    if n < 2:
        return m, 0.0
    return m, math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1) / n)


def wilson(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion; stays inside [0, 1] and behaves at 0 and n."""
    if n <= 0:
        raise ValueError("n must be positive")
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = 0.0 if successes == 0 else max(0.0, centre - half)
    hi = 1.0 if successes == n else min(1.0, centre + half)
    return lo, hi


def irr(cashflows: Sequence[float]) -> float:
    """Annualised internal rate of return of monthly flows (month 0 first), or nan when there is no sign change. Bisection on the
    monthly rate; with several sign changes there can be more than one root."""
    if not (any(c < 0 for c in cashflows) and any(c > 0 for c in cashflows)):
        return math.nan

    def npv(rate: float) -> float:
        return sum(c / (1 + rate) ** t for t, c in enumerate(cashflows))

    lo, hi = -0.99, 10.0
    flo, fhi = npv(lo), npv(hi)
    if flo * fhi > 0:
        return math.nan
    for _ in range(200):
        mid = (lo + hi) / 2
        fm = npv(mid)
        if fm == 0 or (hi - lo) < 1e-14:
            break
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return float((1 + (lo + hi) / 2) ** 12 - 1)
