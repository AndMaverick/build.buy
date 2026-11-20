"""The decision model: shared assumptions (volume, benefit per unit, uncertain variables) and one or more alternatives to compare."""
from __future__ import annotations

from dataclasses import dataclass, field

from .distributions import Distribution
from .expr import Expr, compile_expr

ZERO = compile_expr("0")
ONE = compile_expr("1")


@dataclass(frozen=True)
class Tier:
    """One band of graduated usage pricing: units up to ``up_to`` (None for the last band) cost ``price`` each."""

    up_to: Expr | None
    price: Expr


@dataclass(frozen=True)
class Alternative:
    """One way to get the capability: build it, buy it, or a variant of either.

    Month 0 is signing or kickoff, when ``one_time`` costs land unless ``one_time_months`` spreads them over months 1 to N.
    ``build_monthly`` costs run in the months before ``live_month``; from ``live_month`` the option serves volume, earns benefit,
    and pays ``run_monthly`` (growing by ``run_growth_per_year``) plus usage cost (``unit_cost`` or graduated ``tiers``, with a
    ``minimum_usage_cost`` commit, and prices moving by ``price_change_per_year``). ``terminal`` is value (positive) or exit cost
    (negative) at the horizon. ``failure_probability`` matters only in simulation: with that chance the option is abandoned after
    ``failure_month`` and, if there is a ``fallback``, that alternative starts then."""

    name: str
    live_month: Expr = ONE
    quality: Expr = ONE
    ramp_months: Expr = ZERO
    one_time: dict[str, Expr] = field(default_factory=dict)
    one_time_months: Expr = ZERO
    build_monthly: dict[str, Expr] = field(default_factory=dict)
    run_monthly: dict[str, Expr] = field(default_factory=dict)
    run_growth_per_year: Expr = ZERO
    unit_cost: Expr | None = None
    tiers: tuple[Tier, ...] = ()
    price_change_per_year: Expr = ZERO
    minimum_usage_cost: Expr = ZERO
    terminal: Expr = ZERO
    refresh_every_months: Expr | None = None
    refresh_cost: Expr | None = None
    failure_probability: float = 0.0
    failure_month: int = 1
    fallback: str | None = None

    def __post_init__(self) -> None:
        if self.unit_cost is not None and self.tiers:
            raise ValueError(f"{self.name}: give unit_cost or tiers, not both")
        if self.tiers and self.tiers[-1].up_to is not None:
            raise ValueError(f"{self.name}: the last tier must have up_to null (it covers all remaining volume)")
        if any(t.up_to is None for t in self.tiers[:-1]):
            raise ValueError(f"{self.name}: only the last tier may have up_to null")
        if (self.refresh_every_months is None) != (self.refresh_cost is None):
            raise ValueError(f"{self.name}: refresh_every_months and refresh_cost go together")
        if not 0 <= self.failure_probability <= 1:
            raise ValueError(f"{self.name}: failure_probability must be between 0 and 1")
        if self.failure_month < 1:
            raise ValueError(f"{self.name}: failure_month must be at least 1")
        if self.fallback is not None and self.fallback == self.name:
            raise ValueError(f"{self.name}: an option cannot be its own fallback")

    def expressions(self) -> list[tuple[str, Expr]]:
        out: list[tuple[str, Expr]] = [("live_month", self.live_month), ("quality", self.quality), ("ramp_months", self.ramp_months),
                                       ("one_time_months", self.one_time_months), ("run_growth_per_year", self.run_growth_per_year),
                                       ("price_change_per_year", self.price_change_per_year),
                                       ("minimum_usage_cost", self.minimum_usage_cost), ("terminal", self.terminal)]
        for group, items in (("one_time", self.one_time), ("build_monthly", self.build_monthly), ("run_monthly", self.run_monthly)):
            out += [(f"{group}.{k}", e) for k, e in items.items()]
        if self.unit_cost is not None:
            out.append(("unit_cost", self.unit_cost))
        for i, t in enumerate(self.tiers):
            out.append((f"tiers[{i}].price", t.price))
            if t.up_to is not None:
                out.append((f"tiers[{i}].up_to", t.up_to))
        if self.refresh_every_months is not None and self.refresh_cost is not None:
            out += [("refresh_every_months", self.refresh_every_months), ("refresh_cost", self.refresh_cost)]
        return out


@dataclass(frozen=True)
class Model:
    name: str
    variables: dict[str, Distribution]
    volume_start: Expr
    options: dict[str, Alternative]
    horizon_months: int = 60
    discount_rate: float = 0.10
    volume_growth_per_year: Expr = ZERO
    volume_cap: Expr | None = None
    benefit_per_unit: Expr = ZERO
    benefit_fixed_per_month: Expr = ZERO

    def __post_init__(self) -> None:
        if self.horizon_months < 2:
            raise ValueError("horizon_months must be at least 2")
        if self.discount_rate <= -1:
            raise ValueError("discount_rate must be above -100 percent")
        if len(self.options) < 2:
            raise ValueError("a comparison needs at least two options")
        names = set(self.variables)
        shared = [("volume.start", self.volume_start), ("volume.growth_per_year", self.volume_growth_per_year),
                  ("benefit.per_unit", self.benefit_per_unit), ("benefit.fixed_per_month", self.benefit_fixed_per_month)]
        if self.volume_cap is not None:
            shared.append(("volume.cap", self.volume_cap))
        for label, expr in shared:
            missing = sorted(expr.names - names)
            if missing:
                raise ValueError(f"{label} uses unknown variable {missing[0]!r}")
        for key, opt in self.options.items():
            if key != opt.name:
                raise ValueError(f"option key {key!r} does not match its name {opt.name!r}")
            for label, expr in opt.expressions():
                missing = sorted(expr.names - names)
                if missing:
                    raise ValueError(f"option {opt.name!r}, {label} uses unknown variable {missing[0]!r}")
            if opt.fallback is not None:
                if opt.fallback not in self.options:
                    raise ValueError(f"option {opt.name!r}: fallback {opt.fallback!r} is not an option")
                if self.options[opt.fallback].failure_probability > 0 and self.options[opt.fallback].fallback == opt.name:
                    raise ValueError(f"options {opt.name!r} and {opt.fallback!r} fall back on each other")
            if opt.failure_month >= self.horizon_months:
                raise ValueError(f"option {opt.name!r}: failure_month must be inside the horizon")

    @property
    def variable_names(self) -> list[str]:
        """Sorted, so results never depend on the order keys were written in a file."""
        return sorted(self.variables)
