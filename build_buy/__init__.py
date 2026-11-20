"""build.buy: build-versus-buy decisions for AI capabilities, with honest uncertainty and reproducible results."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, overload

from ._version import __version__
from .compare import BreakEven, base_values, breakeven, crossover, tornado
from .compare import compare as compare_model
from .distributions import Fixed, Lognormal, Normal, Triangular, Uniform, from_ci90, from_dict
from .engine import Flows, Outcome, build_flows, failed_flows, tiered_cost
from .engine import evaluate as evaluate_model
from .expr import Expr, ExprError, compile_expr
from .model import Alternative, Model, Tier
from .report import analyze, render_markdown, render_text
from .scenario import Scenario, load, parse
from .simple import Option, crossover_year, net_flows, npv, payback
from .simple import compare as compare_yearly
from .simple import evaluate as evaluate_yearly
from .simulate import Run, evpi, evppi, prob_beats, simulate, summarize, value_of_information


@overload
def compare(subject: Sequence[Option], years: int = 5, rate: float = 0.1) -> list[dict[str, Any]]: ...
@overload
def compare(subject: Model, values: Mapping[str, float] | None = None) -> list[Outcome]: ...
def compare(subject: Sequence[Option] | Model, *args: Any, **kwargs: Any) -> list[dict[str, Any]] | list[Outcome]:
    """One name for both generations. A list of ``Option`` runs the original yearly comparison (unchanged); a ``Model`` runs the full one."""
    if isinstance(subject, Model):
        return compare_model(subject, *args, **kwargs)
    return compare_yearly(list(subject), *args, **kwargs)


@overload
def evaluate(subject: Option, years: int, rate: float) -> dict[str, Any]: ...
@overload
def evaluate(subject: Model, name: str, v: Mapping[str, float], *, failed: bool = False) -> Outcome: ...
def evaluate(subject: Option | Model, *args: Any, **kwargs: Any) -> dict[str, Any] | Outcome:
    """One name for both generations: an ``Option`` is evaluated yearly (unchanged), a ``Model`` option monthly."""
    if isinstance(subject, Model):
        return evaluate_model(subject, *args, **kwargs)
    return evaluate_yearly(subject, *args, **kwargs)


__all__ = [
    "Alternative", "BreakEven", "Expr", "ExprError", "Fixed", "Flows", "Lognormal", "Model", "Normal", "Option", "Outcome", "Run",
    "Scenario", "Tier", "Triangular", "Uniform", "__version__", "analyze", "base_values", "breakeven", "build_flows", "compare",
    "compare_yearly", "compile_expr", "crossover", "crossover_year", "evaluate", "evpi", "evppi", "failed_flows", "from_ci90", "from_dict",
    "load", "net_flows", "npv", "parse", "payback", "prob_beats", "render_markdown", "render_text", "simulate", "summarize", "tiered_cost",
    "tornado", "value_of_information",
]
