"""Run a scenario end to end and render the result as text, markdown or JSON."""
from __future__ import annotations

import math
from typing import Any

from ._version import __version__
from .compare import breakeven, compare, crossover, tornado
from .engine import evaluate
from .scenario import Scenario
from .simulate import evpi, prob_beats, simulate, summarize, value_of_information


def analyze(scenario: Scenario, trials: int = 3000, seed: int = 7, method: str = "lhs") -> dict[str, Any]:
    """The full analysis as plain data (JSON-ready). Deterministic for a given scenario, trials, seed and method."""
    model = scenario.model
    run = simulate(model, trials, seed, method)
    summary = summarize(run)
    leader, runner = summary[0], summary[1]
    typical = {o.name: o for o in compare(model)}
    wins, wins_ci = prob_beats(run, leader.name, runner.name)
    paired = [a - b for a, b in zip(run.npv[leader.name], run.npv[runner.name], strict=True)]
    advantage = sum(paired) / len(paired)
    advantage_se = math.sqrt(sum((x - advantage) ** 2 for x in paired) / (len(paired) - 1) / len(paired))
    swings = tornado(model, leader.name, runner.name)
    options = []
    for s in summary:
        o = typical[s.name]
        item = {"name": s.name, "expected_npv": s.mean, "expected_npv_se": s.mean_se, "npv_p10": s.p10, "npv_p50": s.p50, "npv_p90": s.p90,
                "prob_best": s.prob_best, "expected_regret": s.expected_regret, "p90_regret": s.p90_regret, "prob_loss": s.prob_loss,
                "typical_npv": o.npv, "typical_tco": o.tco, "typical_cost_per_unit": o.cost_per_unit, "typical_payback_month": o.payback_month}
        opt = model.options[s.name]
        if opt.failure_probability > 0:
            item["failure_probability"] = opt.failure_probability
            item["npv_if_it_fails"] = evaluate(model, s.name, {n: d.ppf(0.5) for n, d in model.variables.items()}, failed=True).npv
        options.append(item)
    cross = crossover(typical[leader.name], typical[runner.name])
    actuals: dict[str, Any] = {"min_leader_advantage": advantage, "min_prob_leader_wins": wins, "max_p90_regret": leader.p90_regret,
                               "leader_must_be": leader.name}
    gates = []
    for key, threshold in sorted(scenario.gates.items()):
        actual = actuals[key]
        ok = actual == threshold if key == "leader_must_be" else (actual <= threshold if key.startswith("max_") else actual >= threshold)
        gates.append({"name": key, "threshold": threshold, "actual": actual, "passed": ok})
    return {
        "name": model.name, "version": __version__, "trials": trials, "seed": seed, "method": method, "horizon_months": model.horizon_months,
        "fingerprint": scenario.fingerprint(trials, seed, method, __version__),
        "leader": leader.name, "runner_up": runner.name,
        "typical_case_leader": min(typical.values(), key=lambda o: (-o.npv, o.name)).name,
        "advantage": {"mean": advantage, "se": advantage_se, "prob_leader_beats_runner_up": wins, "prob_ci_low": wins_ci[0], "prob_ci_high": wins_ci[1]},
        "options": options,
        "evpi": evpi(run),
        "evppi": [{"variable": n, "value": v} for n, v in value_of_information(run)[:5]],
        "cost_curves_cross_at_month": cross,
        "breakeven": [{"variable": b.variable, "value": b.value, "probability_leader_wins_side": b.probability_favours_first, "note": b.note}
                      for b in (breakeven(model, w.variable, leader.name, runner.name) for w in swings[:3])],
        "tornado": [{"variable": w.variable, "low_input": w.low_input, "high_input": w.high_input, "gap_at_p10": w.low_gap,
                     "gap_at_p90": w.high_gap, "swing": w.swing} for w in swings[:6]],
        "gates": gates, "passed": all(g["passed"] for g in gates),
    }


def _m(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if math.isinf(x):
        return "never"
    return f"{x:,.0f}"


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _gate_value(name: str) -> Any:
    if name == "min_prob_leader_wins":
        return _pct
    if name == "leader_must_be":
        return str
    return _m


def render_text(a: dict[str, Any]) -> str:
    adv = a["advantage"]
    lines = [f"{a['name']}  (build.buy {a['version']}, {a['trials']} trials, seed {a['seed']}, {a['method']}, {a['horizon_months']} months)", "",
             f"Recommendation: {a['leader']}",
             f"  Expected NPV advantage over {a['runner_up']}: {_m(adv['mean'])}  (standard error {_m(adv['se'])})",
             f"  Chance {a['leader']} beats {a['runner_up']}: {_pct(adv['prob_leader_beats_runner_up'])}  "
             f"(95% interval {_pct(adv['prob_ci_low'])} to {_pct(adv['prob_ci_high'])})"]
    if a["typical_case_leader"] != a["leader"]:
        lines.append(f"  Note: with every input at its median, {a['typical_case_leader']} leads. Uncertainty changes the answer.")
    lines += ["", f"{'option':<14}{'expected NPV':>14}{'p10':>12}{'p90':>12}{'P(best)':>10}{'regret':>10}{'P(loss)':>9}"]
    for o in a["options"]:
        lines.append(f"{o['name']:<14}{_m(o['expected_npv']):>14}{_m(o['npv_p10']):>12}{_m(o['npv_p90']):>12}"
                     f"{_pct(o['prob_best']):>10}{_m(o['expected_regret']):>10}{_pct(o['prob_loss']):>9}")
    lines += ["", "Typical case (every input at its median)"]
    for o in a["options"]:
        pb = "never pays back" if math.isinf(o["typical_payback_month"]) else f"payback month {o['typical_payback_month']:.0f}"
        lines.append(f"  {o['name']:<12}TCO {_m(o['typical_tco'])}, cost per unit {o['typical_cost_per_unit']:.4f}, NPV {_m(o['typical_npv'])}, {pb}")
    risky = [o for o in a["options"] if "failure_probability" in o]
    if risky:
        lines += ["", "If it fails (abandoned, then the fallback starts)"]
        lines += [f"  {o['name']:<12}{_pct(o['failure_probability'])} chance; NPV in that case {_m(o['npv_if_it_fails'])}" for o in risky]
    if a["cost_curves_cross_at_month"] is not None:
        lines += ["", f"Cumulative cost curves of {a['leader']} and {a['runner_up']} cross at month {a['cost_curves_cross_at_month']:.1f}."]
    lines += ["", "What would change the answer (break-even, others at median)"]
    for b in a["breakeven"]:
        lines.append(f"  {b['variable']:<26}" + ("never swaps" if b["value"] is None else
                     f"{b['note']}  (chance it lands where {a['leader']} wins: {_pct(b['probability_leader_wins_side'])})"))
    lines += ["", f"Value of information: knowing everything first is worth at most {_m(a['evpi'])}.",
              "  Learning one input exactly is worth: " + ", ".join(f"{e['variable']} {_m(e['value'])}" for e in a["evppi"][:3])]
    if a["gates"]:
        lines += ["", "Decision gates"]
        for g in a["gates"]:
            fmt = _gate_value(g["name"])
            lines.append(f"  {'PASS' if g['passed'] else 'FAIL'}  {g['name']}: {fmt(g['actual'])} against {fmt(g['threshold'])}")
    lines += ["", f"Fingerprint {a['fingerprint']}", "Same fingerprint on any machine gives the same numbers."]
    return "\n".join(lines)


def render_markdown(a: dict[str, Any]) -> str:
    adv = a["advantage"]
    out = [f"# {a['name']}", "", f"*build.buy {a['version']}, {a['trials']} trials, seed {a['seed']}, {a['method']} sampling, {a['horizon_months']} months*", "",
           f"**Recommendation: {a['leader']}.** Expected NPV advantage over {a['runner_up']} is {_m(adv['mean'])}; "
           f"it is the better choice in {_pct(adv['prob_leader_beats_runner_up'])} of trials "
           f"({_pct(adv['prob_ci_low'])} to {_pct(adv['prob_ci_high'])}).", "",
           "| Option | Expected NPV | p10 | p90 | P(best) | Expected regret |", "|---|---|---|---|---|---|"]
    out += [f"| {o['name']} | {_m(o['expected_npv'])} | {_m(o['npv_p10'])} | {_m(o['npv_p90'])} | {_pct(o['prob_best'])} | {_m(o['expected_regret'])} |"
            for o in a["options"]]
    out += ["", "## What would change the answer", ""]
    out += [f"- {b['note']}" if b["value"] is not None else f"- {b['variable']}: never swaps" for b in a["breakeven"]]
    if a["gates"]:
        out += ["", "## Decision gates", "", "| Gate | Result | Threshold | Status |", "|---|---|---|---|"]
        out += [f"| {g['name']} | {_gate_value(g['name'])(g['actual'])} | {_gate_value(g['name'])(g['threshold'])} | "
                f"{'pass' if g['passed'] else 'FAIL'} |" for g in a["gates"]]
    out += ["", f"Fingerprint `{a['fingerprint']}`"]
    return "\n".join(out)
