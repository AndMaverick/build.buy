"""Monthly cash flows for each alternative, and the measures taken from them."""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from .model import Alternative, Model
from .stats import irr


def grow(rate: float, years: float) -> float:
    """``(1 + rate) ** years``, refusing a rate at or below -100 percent (which has no real value for fractional years)."""
    if rate <= -1:
        raise ValueError(f"a yearly rate of {rate:g} is at or below -100 percent")
    return float((1 + rate) ** years)


def whole(x: float) -> int:
    """Round a formula result up to a whole number of months, ignoring floating-point dust (8.000000000000002 is 8)."""
    return math.ceil(round(x, 9))


@dataclass(frozen=True)
class Flows:
    """Monthly flows indexed 0 to horizon. ``lines`` are cost lines (positive is a cost, negative is a credit)."""

    lines: dict[str, list[float]]
    benefit: list[float]
    units: list[float]

    @property
    def cost(self) -> list[float]:
        n = len(self.benefit)
        return [sum(line[t] for line in self.lines.values()) for t in range(n)]

    @property
    def net(self) -> list[float]:
        return [b - c for b, c in zip(self.benefit, self.cost, strict=True)]


@dataclass(frozen=True)
class Outcome:
    name: str
    npv: float
    pv_cost: float
    pv_benefit: float
    tco: float
    total_units: float
    cost_per_unit: float
    payback_month: float
    irr: float
    flows: Flows

    @property
    def cumulative_cost(self) -> list[float]:
        return _cumulative(self.flows.cost)


def _cumulative(xs: list[float]) -> list[float]:
    out, running = [], 0.0
    for x in xs:
        running += x
        out.append(running)
    return out


def volume_at(start: float, growth: float, cap: float | None, t: int) -> float:
    units = start * grow(growth, (t - 1) / 12)
    if cap is not None:
        units = min(units, cap)
    return max(units, 0.0)


def volume(model: Model, v: Mapping[str, float], t: int) -> float:
    """Units served per month at absolute month ``t`` (1 is the first month)."""
    cap = model.volume_cap(v) if model.volume_cap is not None else None
    return volume_at(model.volume_start(v), model.volume_growth_per_year(v), cap, t)


def tiered_cost(units: float, bands: list[tuple[float | None, float]]) -> float:
    """Graduated pricing: each band prices only the units that fall inside it. ``bands`` is (upper bound or None, price per unit)."""
    cost, lower = 0.0, 0.0
    for up_to, price in bands:
        if up_to is not None and up_to < lower:
            raise ValueError("tier bounds must increase")
        top = units if up_to is None else min(units, up_to)
        if top > lower:
            cost += (top - lower) * price
        if up_to is None or units <= up_to:
            break
        lower = up_to
    return cost


def build_flows(model: Model, alt: Alternative, v: Mapping[str, float], start: int = 0) -> Flows:
    """Flows for one alternative whose month 0 falls at absolute month ``start`` (later than 0 when it begins as a fallback)."""
    h = model.horizon_months
    names = ["one_time", "build_team", "run", "usage", "refresh", "terminal"]
    lines = {n: [0.0] * (h + 1) for n in names}
    benefit = [0.0] * (h + 1)
    units = [0.0] * (h + 1)
    live = max(1, whole(alt.live_month(v)))
    quality = alt.quality(v)
    ramp = alt.ramp_months(v)
    spread = max(0, whole(alt.one_time_months(v)))
    one_time = sum(e(v) for e in alt.one_time.values())
    build_monthly = sum(e(v) for e in alt.build_monthly.values())
    run_monthly = sum(e(v) for e in alt.run_monthly.values())
    run_growth = alt.run_growth_per_year(v)
    price_change = alt.price_change_per_year(v)
    minimum = alt.minimum_usage_cost(v)
    bands = [(None if t.up_to is None else t.up_to(v), t.price(v)) for t in alt.tiers]
    unit_cost = alt.unit_cost(v) if alt.unit_cost is not None else 0.0
    benefit_per_unit, benefit_fixed = model.benefit_per_unit(v), model.benefit_fixed_per_month(v)
    vol_start, vol_growth = model.volume_start(v), model.volume_growth_per_year(v)
    vol_cap = model.volume_cap(v) if model.volume_cap is not None else None
    refresh_every = whole(alt.refresh_every_months(v)) if alt.refresh_every_months is not None else 0
    refresh_cost = alt.refresh_cost(v) if alt.refresh_cost is not None else 0.0
    for t in range(start, h + 1):
        k = t - start
        if spread == 0:
            if k == 0:
                lines["one_time"][t] = one_time
        elif 1 <= k <= spread:
            lines["one_time"][t] = one_time / spread
        if 1 <= k < live:
            lines["build_team"][t] = build_monthly
        if k >= live:
            age = k - live
            lines["run"][t] = run_monthly * grow(run_growth, age / 12)
            served = volume_at(vol_start, vol_growth, vol_cap, t)
            units[t] = served
            priced = tiered_cost(served, bands) if bands else served * unit_cost
            lines["usage"][t] = max(priced, minimum) * grow(price_change, age / 12)
            share = 1.0 if ramp <= 0 else min(1.0, (age + 1) / ramp)
            benefit[t] = quality * share * (served * benefit_per_unit + benefit_fixed)
            if refresh_every > 0 and age > 0 and age % refresh_every == 0:
                lines["refresh"][t] = refresh_cost
    lines["terminal"][h] = -alt.terminal(v)
    return Flows(lines, benefit, units)


def failed_flows(model: Model, alt: Alternative, v: Mapping[str, float]) -> Flows:
    """Flows when ``alt`` is abandoned after its ``failure_month``: everything up to then is spent and earned, nothing later, and the
    fallback (if any) begins at that month."""
    h = model.horizon_months
    f = alt.failure_month
    own = build_flows(model, alt, v)
    lines = {n: [x if t <= f and n != "terminal" else 0.0 for t, x in enumerate(series)] for n, series in own.lines.items()}
    benefit = [x if t <= f else 0.0 for t, x in enumerate(own.benefit)]
    units = [x if t <= f else 0.0 for t, x in enumerate(own.units)]
    if alt.fallback is not None:
        fb = build_flows(model, model.options[alt.fallback], v, start=f)
        for n, series in fb.lines.items():
            lines[f"fallback_{n}"] = list(series)
        benefit = [a + b for a, b in zip(benefit, fb.benefit, strict=True)]
        units = [a + b for a, b in zip(units, fb.units, strict=True)]
    assert len(benefit) == h + 1
    return Flows(lines, benefit, units)


def measure(model: Model, name: str, flows: Flows) -> Outcome:
    h = model.horizon_months
    disc = [(1 + model.discount_rate) ** (-t / 12) for t in range(h + 1)]
    cost = flows.cost
    net = flows.net
    pv_cost = sum(c * d for c, d in zip(cost, disc, strict=True))
    pv_benefit = sum(b * d for b, d in zip(flows.benefit, disc, strict=True))
    cum = _cumulative(net)
    trough = min(range(len(cum)), key=lambda i: (cum[i], i))
    payback = 0.0 if cum[trough] >= 0 else next((float(t) for t in range(trough, len(cum)) if cum[t] >= 0), math.inf)
    total_units = sum(flows.units)
    tco = sum(cost)
    return Outcome(name=name, npv=pv_benefit - pv_cost, pv_cost=pv_cost, pv_benefit=pv_benefit, tco=tco, total_units=total_units,
                   cost_per_unit=tco / total_units if total_units > 0 else math.nan, payback_month=payback, irr=irr(net), flows=flows)


def evaluate(model: Model, name: str, v: Mapping[str, float], *, failed: bool = False) -> Outcome:
    """Outcome of one option for given variable values. ``failed`` evaluates the case where it is abandoned and falls back."""
    missing = sorted(set(model.variables) - set(v))
    if missing:
        raise ValueError(f"missing value for variable {missing[0]!r}")
    alt = model.options[name]
    return measure(model, name, failed_flows(model, alt, v) if failed else build_flows(model, alt, v))
