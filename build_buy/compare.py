"""Deterministic comparison: ranking, cost-curve crossover, break-even assumptions and tornado."""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from .engine import Outcome, evaluate
from .expr import ExprError
from .model import Model


def base_values(model: Model) -> dict[str, float]:
    """Every variable at its median: the 'if everything is typical' case."""
    return {n: d.ppf(0.5) for n, d in model.variables.items()}


def compare(model: Model, values: Mapping[str, float] | None = None) -> list[Outcome]:
    """All options, best NPV first (ties broken by name)."""
    v = dict(values) if values is not None else base_values(model)
    return sorted((evaluate(model, n, v) for n in model.options), key=lambda o: (-o.npv, o.name))


def crossover(a: Outcome, b: Outcome, *, discounted: bool = False, rate: float = 0.0) -> float | None:
    """The month (fractional) at which the cumulative cost curves of ``a`` and ``b`` cross, or None if they never do inside the horizon.

    The starting order is the first month in which the curves differ. A crossing is the first later month in which the order is
    reversed; the exact point is found by linear interpolation inside that month, which is exact when costs are flat month to month."""
    ca, cb = a.flows.cost, b.flows.cost
    if discounted:
        ca = [c * (1 + rate) ** (-t / 12) for t, c in enumerate(ca)]
        cb = [c * (1 + rate) ** (-t / 12) for t, c in enumerate(cb)]
    diff, running = [], 0.0
    for x, y in zip(ca, cb, strict=True):
        running += x - y
        diff.append(running)
    start_sign = 0
    prev = 0
    for t, d in enumerate(diff):
        if d == 0:
            continue
        sign = 1 if d > 0 else -1
        if start_sign == 0:
            start_sign, prev = sign, t
        elif sign != start_sign:
            d0 = diff[prev]
            return prev + d0 / (d0 - d) * (t - prev)
        else:
            prev = t
    return None


@dataclass(frozen=True)
class BreakEven:
    variable: str
    value: float | None
    probability_favours_first: float | None
    direction: str
    note: str


def breakeven(model: Model, variable: str, first: str, second: str, values: Mapping[str, float] | None = None) -> BreakEven:
    """The value of ``variable`` at which ``first`` and ``second`` have equal NPV (others fixed), and the probability, under the
    variable's own distribution, that it lands on the side where ``first`` wins."""
    if variable not in model.variables:
        raise ValueError(f"unknown variable {variable!r}")
    base = dict(values) if values is not None else base_values(model)
    dist = model.variables[variable]

    def gap(x: float) -> float:
        v = {**base, variable: x}
        return evaluate(model, first, v).npv - evaluate(model, second, v).npv

    lo0, hi0 = dist.ppf(0.0005), dist.ppf(0.9995)
    span = (hi0 - lo0) or abs(hi0) or 1.0
    for widen in (1.0, 0.25, 0.0):
        lo, hi = lo0 - widen * span, hi0 + widen * span
        if lo0 >= 0:
            lo = max(lo, 0.0)
        try:
            flo, fhi = gap(lo), gap(hi)
        except (ExprError, ValueError, OverflowError):
            continue
        if flo == 0 or fhi == 0 or (flo > 0) != (fhi > 0):
            break
    else:
        return BreakEven(variable, None, None, "none", f"{first} and {second} never swap places for any value of {variable}")
    a, b, fa = lo, hi, flo
    if flo == 0:
        root = lo
    elif fhi == 0:
        root = hi
    else:
        for _ in range(200):
            mid = (a + b) / 2
            fm = gap(mid)
            if fm == 0 or (b - a) < 1e-12 * max(1.0, abs(mid)):
                a = b = mid
                break
            if (fm > 0) == (fa > 0):
                a, fa = mid, fm
            else:
                b = mid
        root = (a + b) / 2
    rising = fhi > flo
    below = dist.cdf(root)
    p_first = (1 - below) if rising else below
    where = f"above {root:.6g}" if rising else f"below {root:.6g}"
    return BreakEven(variable, root, p_first, "first wins when higher" if rising else "first wins when lower",
                     f"{first} beats {second} when {variable} is {where}")


@dataclass(frozen=True)
class Swing:
    variable: str
    low_input: float
    high_input: float
    low_gap: float
    high_gap: float

    @property
    def swing(self) -> float:
        return abs(self.high_gap - self.low_gap)


def tornado(model: Model, first: str, second: str, lo: float = 0.10, hi: float = 0.90) -> list[Swing]:
    """How far the NPV advantage of ``first`` over ``second`` moves when one variable goes from its ``lo`` to its ``hi`` quantile
    with every other variable at its median. Largest swing first."""
    if not 0 < lo < hi < 1:
        raise ValueError("need 0 < lo < hi < 1")
    base = base_values(model)
    out = []
    for name, dist in model.variables.items():
        a, b = dist.ppf(lo), dist.ppf(hi)
        ga = evaluate(model, first, {**base, name: a}).npv - evaluate(model, second, {**base, name: a}).npv
        gb = evaluate(model, first, {**base, name: b}).npv - evaluate(model, second, {**base, name: b}).npv
        out.append(Swing(name, a, b, ga, gb))
    return sorted(out, key=lambda s: (-s.swing, s.variable))


def is_finite(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))
