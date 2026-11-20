"""Seeded generators of random expressions and random models, used to test the library against independent oracles."""
from __future__ import annotations

import random

from .distributions import Distribution, Lognormal, Normal, Triangular, Uniform
from .expr import compile_expr
from .model import Alternative, Model, Tier


def random_expression(rng: random.Random, names: list[str], depth: int = 3) -> str:
    """A random formula over ``names`` that is valid both here and in plain Python, so Python's own ``eval`` can act as the oracle."""
    if depth == 0 or rng.random() < 0.25:
        if names and rng.random() < 0.6:
            return rng.choice(names)
        return repr(round(rng.uniform(0.5, 9.5), 2))
    kind = rng.random()
    if kind < 0.55:
        return f"({random_expression(rng, names, depth - 1)} {rng.choice('+-*/')} {random_expression(rng, names, depth - 1)})"
    if kind < 0.65:
        return f"(-{random_expression(rng, names, depth - 1)})"
    if kind < 0.8:
        return f"{rng.choice(['min', 'max'])}({random_expression(rng, names, depth - 1)}, {random_expression(rng, names, depth - 1)})"
    if kind < 0.88:
        return f"abs({random_expression(rng, names, depth - 1)})"
    a, b = random_expression(rng, names, depth - 1), random_expression(rng, names, depth - 1)
    return f"({a} if {random_expression(rng, names, depth - 1)} > {rng.choice(['1', '2', '5'])} else {b})"


def random_distribution(rng: random.Random) -> Distribution:
    kind = rng.choice(["uniform", "triangular", "normal", "lognormal"])
    lo = rng.uniform(1, 50)
    span = rng.uniform(1, 40)
    if kind == "uniform":
        return Uniform(lo, lo + span)
    if kind == "triangular":
        return Triangular(lo, lo + rng.random() * span, lo + span)
    if kind == "normal":
        return Normal(lo + span / 2, span / 4, lower=0.01)
    return Lognormal(rng.uniform(0, 3), rng.uniform(0.1, 0.6))


def random_option(rng: random.Random, name: str, names: list[str], fallback: str | None = None) -> Alternative:
    """A random but valid alternative using a few of the features: spread one-time cost, build team, tiers, ramp, refresh, terminal."""

    def pick() -> str:
        return rng.choice(names)

    kw: dict[str, object] = {
        "live_month": compile_expr(str(rng.randint(1, 8))), "quality": compile_expr(str(rng.choice([1.0, 1.0, 0.9, 1.1]))),
        "ramp_months": compile_expr(str(rng.randint(0, 6))),
        "one_time": {"a": compile_expr(f"{pick()} * {rng.choice([100, 500, 2000])}")},
        "one_time_months": compile_expr(str(rng.randint(0, 5))),
        "build_monthly": {"t": compile_expr(f"{pick()} * {rng.choice([50, 200])}")},
        "run_monthly": {"r": compile_expr(f"{pick()} * {rng.choice([10, 40])}")},
        "run_growth_per_year": compile_expr(str(rng.choice([0.0, 0.05, 0.2]))),
        "price_change_per_year": compile_expr(str(rng.choice([-0.3, -0.1, 0.0, 0.1]))),
        "minimum_usage_cost": compile_expr(str(rng.choice([0, 0, 500]))),
        "terminal": compile_expr(str(rng.choice([0, 1000, -3000]))),
    }
    if rng.random() < 0.5:
        kw["unit_cost"] = compile_expr(f"{pick()} * {rng.choice([0.01, 0.05])}")
    else:
        cut = rng.randint(20, 200)
        kw["tiers"] = (Tier(compile_expr(str(cut)), compile_expr("0.9")), Tier(compile_expr(str(cut * 3)), compile_expr("0.6")),
                       Tier(None, compile_expr("0.3")))
    if rng.random() < 0.4:
        kw["refresh_every_months"] = compile_expr(str(rng.randint(6, 18)))
        kw["refresh_cost"] = compile_expr(f"{pick()} * 300")
    if fallback is not None:
        kw["failure_probability"] = rng.uniform(0.05, 0.4)
        kw["failure_month"] = rng.randint(1, 10)
        kw["fallback"] = fallback
    return Alternative(name=name, **kw)  # type: ignore[arg-type]


def random_model(rng: random.Random, name: str = "random") -> Model:
    """A random but valid model with two or three alternatives."""
    names = [f"x{i}" for i in range(rng.randint(2, 5))]
    variables = {n: random_distribution(rng) for n in names}
    labels = ["a", "b", "c"][: rng.randint(2, 3)]
    options = {lab: random_option(rng, lab, names, fallback=labels[0] if i > 0 and rng.random() < 0.6 else None)
               for i, lab in enumerate(labels)}
    return Model(
        name=name, variables=variables, volume_start=compile_expr(f"{rng.choice(names)} * {rng.choice([5, 20])}"), options=options,
        horizon_months=rng.randint(14, 48), discount_rate=rng.choice([0.0, 0.08, 0.12]),
        volume_growth_per_year=compile_expr(str(rng.choice([0.0, 0.3, 0.8]))),
        volume_cap=compile_expr(str(rng.choice([400, 2000, 100000]))) if rng.random() < 0.5 else None,
        benefit_per_unit=compile_expr(str(rng.choice([1.0, 3.0]))), benefit_fixed_per_month=compile_expr(str(rng.choice([0, 200]))))
