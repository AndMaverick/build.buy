"""Monte Carlo over the uncertain variables and the chance that an option fails, and what the results mean for the decision."""
from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from .engine import evaluate
from .model import Model
from .stats import mean_se, quantile, uniforms, wilson


@dataclass
class Run:
    model: Model
    trials: int
    seed: int
    method: str
    values: dict[str, list[float]]
    failed: dict[str, list[bool]]
    npv: dict[str, list[float]]


def simulate(model: Model, trials: int = 3000, seed: int = 7, method: str = "lhs") -> Run:
    """Same model, trials, seed and method always give the same run, whatever order the variables were written in."""
    if trials < 2:
        raise ValueError("trials must be at least 2")
    rng = random.Random(seed)
    names = model.variable_names
    risky = sorted(n for n, o in model.options.items() if o.failure_probability > 0)
    cols = uniforms(trials, len(names), rng, method)
    draws = uniforms(trials, len(risky), rng, "random")
    values = {n: [model.variables[n].ppf(u) for u in cols[i]] for i, n in enumerate(names)}
    failed: dict[str, list[bool]] = {n: [] for n in risky}
    npv: dict[str, list[float]] = {n: [] for n in sorted(model.options)}
    for i in range(trials):
        v = {n: values[n][i] for n in names}
        gone = {n: draws[j][i] < model.options[n].failure_probability for j, n in enumerate(risky)}
        for n in risky:
            failed[n].append(gone[n])
        for n in npv:
            npv[n].append(evaluate(model, n, v, failed=gone.get(n, False)).npv)
    return Run(model, trials, seed, method, values, failed, npv)


@dataclass(frozen=True)
class OptionSummary:
    name: str
    mean: float
    mean_se: float
    p10: float
    p50: float
    p90: float
    prob_best: float
    expected_regret: float
    p90_regret: float
    prob_loss: float


def summarize(run: Run) -> list[OptionSummary]:
    """Per option: NPV distribution, how often it is the best choice, and its regret (how much NPV is left on the table, in a trial,
    compared with the best option in that trial). Ordered by expected NPV, best first."""
    n = run.trials
    names = sorted(run.npv)
    best = [max(run.npv[k][i] for k in names) for i in range(n)]
    wins = dict.fromkeys(names, 0)
    for i in range(n):
        top = max((run.npv[k][i], k) for k in names)[1]
        wins[top] += 1
    out = []
    for k in names:
        series = sorted(run.npv[k])
        m, se = mean_se(series)
        regrets = sorted(best[i] - run.npv[k][i] for i in range(n))
        out.append(OptionSummary(k, m, se, quantile(series, 0.1), quantile(series, 0.5), quantile(series, 0.9), wins[k] / n,
                                 sum(regrets) / n, quantile(regrets, 0.9), sum(1 for x in series if x < 0) / n))
    return sorted(out, key=lambda s: (-s.mean, s.name))


def prob_beats(run: Run, first: str, second: str) -> tuple[float, tuple[float, float]]:
    """P(first has higher NPV than second) with its Wilson interval."""
    wins = sum(1 for a, b in zip(run.npv[first], run.npv[second], strict=True) if a > b)
    return wins / run.trials, wilson(wins, run.trials)


def evpi(run: Run) -> float:
    """Expected value of perfect information: what you would gain on average by knowing every uncertainty before choosing. It equals
    the smallest expected regret of any single option, and it is the ceiling on what any study could be worth."""
    n = run.trials
    best = sum(max(run.npv[k][i] for k in run.npv) for i in range(n)) / n
    return best - max(sum(v) / n for v in run.npv.values())


def evppi(x: Sequence[float], npv: dict[str, list[float]], bins: int = 10) -> float:
    """Expected value of learning one variable exactly before choosing: cut the trials into equal-count groups by that variable, pick
    the best option on the average of each group, and compare with picking on the overall average. Biased slightly upward with few
    trials per group."""
    n = len(x)
    bins = max(1, min(bins, n))
    order = sorted(range(n), key=lambda i: (x[i], i))
    total = 0.0
    for b in range(bins):
        idx = order[b * n // bins:(b + 1) * n // bins]
        if idx:
            total += max(sum(series[i] for i in idx) / len(idx) for series in npv.values()) * len(idx)
    return total / n - max(sum(series) / n for series in npv.values())


def value_of_information(run: Run, bins: int = 10) -> list[tuple[str, float]]:
    out = [(n, evppi(xs, run.npv, bins)) for n, xs in run.values.items()]
    return sorted(out, key=lambda t: (-t[1], t[0]))


def is_unanimous(run: Run, leader: str) -> bool:
    """True if the leader is never beaten in any trial."""
    return all(run.npv[leader][i] >= max(run.npv[k][i] for k in run.npv) - 1e-12 for i in range(run.trials))


def finite_mean(xs: Sequence[float]) -> float:
    ys = [x for x in xs if not (math.isnan(x) or math.isinf(x))]
    return sum(ys) / len(ys) if ys else math.nan
