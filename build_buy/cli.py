"""Command line: ``python -m build_buy compare scenario.json``. Exit 0 when every gate passes, 1 when a gate fails, 2 on a bad input."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from ._version import __version__
from .compare import base_values, breakeven, compare, crossover
from .report import analyze, render_markdown, render_text
from .scenario import load


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--trials", type=int, default=3000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--method", choices=("lhs", "random"), default="lhs")


def _build() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="build-buy", description="Build-versus-buy decisions for AI capabilities.")
    p.add_argument("--version", action="version", version=f"build.buy {__version__}")
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("compare", help="full analysis under uncertainty, with decision gates")
    c.add_argument("scenario")
    c.add_argument("--format", choices=("text", "markdown", "json"), default="text")
    _common(c)
    t = sub.add_parser("typical", help="deterministic comparison with every input at its median, or at values you set")
    t.add_argument("scenario")
    t.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="override a variable (repeatable)")
    t.add_argument("--format", choices=("text", "json"), default="text")
    b = sub.add_parser("breakeven", help="the value of one variable at which two options tie")
    b.add_argument("scenario")
    b.add_argument("variable")
    b.add_argument("first")
    b.add_argument("second")
    f = sub.add_parser("fingerprint", help="print the reproducibility fingerprint of a scenario and run settings")
    f.add_argument("scenario")
    _common(f)
    return p


def _typical(args: argparse.Namespace) -> int:
    sc = load(args.scenario)
    values = base_values(sc.model)
    for item in args.set:
        name, _, raw = item.partition("=")
        if name not in values:
            raise ValueError(f"unknown variable {name!r}")
        try:
            values[name] = float(raw)
        except ValueError as exc:
            raise ValueError(f"--set {item}: value must be a number") from exc
    ranked = compare(sc.model, values)
    rows: list[dict[str, Any]] = [{"name": o.name, "npv": o.npv, "tco": o.tco, "pv_cost": o.pv_cost, "cost_per_unit": o.cost_per_unit,
                                   "payback_month": o.payback_month} for o in ranked]
    cross = crossover(ranked[0], ranked[1])
    if args.format == "json":
        print(json.dumps({"values": values, "options": rows, "cost_curves_cross_at_month": cross}, indent=2, sort_keys=True))
        return 0
    print("Values: " + ", ".join(f"{k}={v:.6g}" for k, v in sorted(values.items())))
    print(f"{'option':<14}{'NPV':>14}{'TCO':>14}{'cost/unit':>12}{'payback':>10}")
    for r in rows:
        pb = "never" if r["payback_month"] == float("inf") else f"{r['payback_month']:.0f}"
        print(f"{r['name']:<14}{r['npv']:>14,.0f}{r['tco']:>14,.0f}{r['cost_per_unit']:>12.4f}{pb:>10}")
    if cross is not None:
        print(f"Cumulative cost curves of {ranked[0].name} and {ranked[1].name} cross at month {cross:.1f}.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build().parse_args(argv)
    try:
        if args.command == "compare":
            result = analyze(load(args.scenario), args.trials, args.seed, args.method)
            print(json.dumps(result, indent=2, sort_keys=True) if args.format == "json" else
                  render_markdown(result) if args.format == "markdown" else render_text(result))
            return 0 if result["passed"] else 1
        if args.command == "typical":
            return _typical(args)
        if args.command == "breakeven":
            sc = load(args.scenario)
            for name in (args.first, args.second):
                if name not in sc.model.options:
                    raise ValueError(f"unknown option {name!r}")
            be = breakeven(sc.model, args.variable, args.first, args.second)
            print(be.note if be.value is not None else f"{args.first} and {args.second} never swap places for any value of {args.variable}")
            if be.probability_favours_first is not None:
                print(f"Chance {args.variable} lands where {args.first} wins: {100 * be.probability_favours_first:.1f}%")
            return 0
        print(load(args.scenario).fingerprint(args.trials, args.seed, args.method, __version__))
        return 0
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
