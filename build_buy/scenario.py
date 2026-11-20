"""Scenario files: strict JSON in, a validated Model and decision gates out. Unknown keys are errors, so a typo cannot silently change an answer."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .distributions import from_dict
from .expr import Expr, ExprError, compile_expr
from .model import Alternative, Model, Tier

GATE_NUMBERS = {"min_leader_advantage", "min_prob_leader_wins", "max_p90_regret"}
GATE_KEYS = GATE_NUMBERS | {"leader_must_be"}
TOP_KEYS = {"name", "description", "horizon_months", "discount_rate", "variables", "volume", "benefit", "options", "gates"}
VOLUME_KEYS = {"start", "growth_per_year", "cap"}
BENEFIT_KEYS = {"per_unit", "fixed_per_month"}
OPTION_KEYS = {"live_month", "quality", "ramp_months", "one_time", "one_time_months", "build_monthly", "run_monthly", "run_growth_per_year",
               "unit_cost", "tiers", "price_change_per_year", "minimum_usage_cost", "terminal", "refresh", "failure"}


@dataclass(frozen=True)
class Scenario:
    model: Model
    gates: dict[str, Any]
    raw: dict[str, Any]

    def fingerprint(self, trials: int, seed: int, method: str, version: str) -> str:
        """SHA-256 of the scenario, the run settings and the library version. Same fingerprint, same numbers, on any machine."""
        canon = json.dumps({"scenario": self.raw, "trials": trials, "seed": seed, "method": method, "version": version},
                           sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canon.encode("ascii")).hexdigest()


def _number(v: Any, where: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{where}: expected a number, got {v!r}")
    return float(v)


def _int(v: Any, where: str) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"{where}: expected a whole number, got {v!r}")
    return v


def _expr(v: Any, where: str) -> Expr:
    try:
        if isinstance(v, str):
            return compile_expr(v)
        return compile_expr(repr(_number(v, where)))
    except ExprError as exc:
        raise ValueError(f"{where}: {exc}") from exc


def _exprs(d: Any, where: str) -> dict[str, Expr]:
    if not isinstance(d, dict):
        raise ValueError(f"{where}: expected an object of name to formula")
    return {k: _expr(v, f"{where}.{k}") for k, v in d.items()}


def _strict(d: Any, allowed: set[str], where: str) -> dict[str, Any]:
    if not isinstance(d, dict):
        raise ValueError(f"{where}: expected an object")
    extra = sorted(set(d) - allowed)
    if extra:
        raise ValueError(f"{where}: unknown key {extra[0]!r} (allowed: {', '.join(sorted(allowed))})")
    return d


def _option(name: str, raw: Any) -> Alternative:
    where = f"options.{name}"
    d = _strict(raw, OPTION_KEYS, where)
    kw: dict[str, Any] = {}
    for key in ("live_month", "quality", "ramp_months", "one_time_months", "run_growth_per_year", "unit_cost", "price_change_per_year",
                "minimum_usage_cost", "terminal"):
        if key in d:
            kw[key] = _expr(d[key], f"{where}.{key}")
    for key in ("one_time", "build_monthly", "run_monthly"):
        if key in d:
            kw[key] = _exprs(d[key], f"{where}.{key}")
    if "tiers" in d:
        if not isinstance(d["tiers"], list) or not d["tiers"]:
            raise ValueError(f"{where}.tiers: expected a non-empty list")
        tiers = []
        for i, t in enumerate(d["tiers"]):
            td = _strict(t, {"up_to", "price"}, f"{where}.tiers[{i}]")
            if "price" not in td or "up_to" not in td:
                raise ValueError(f"{where}.tiers[{i}]: needs up_to (a number, or null for the last tier) and price")
            tiers.append(Tier(None if td["up_to"] is None else _expr(td["up_to"], f"{where}.tiers[{i}].up_to"),
                              _expr(td["price"], f"{where}.tiers[{i}].price")))
        kw["tiers"] = tuple(tiers)
    if "refresh" in d:
        rd = _strict(d["refresh"], {"every_months", "cost"}, f"{where}.refresh")
        if set(rd) != {"every_months", "cost"}:
            raise ValueError(f"{where}.refresh: needs every_months and cost")
        kw["refresh_every_months"] = _expr(rd["every_months"], f"{where}.refresh.every_months")
        kw["refresh_cost"] = _expr(rd["cost"], f"{where}.refresh.cost")
    if "failure" in d:
        fd = _strict(d["failure"], {"probability", "month", "fallback"}, f"{where}.failure")
        if "probability" not in fd:
            raise ValueError(f"{where}.failure: needs probability")
        kw["failure_probability"] = _number(fd["probability"], f"{where}.failure.probability")
        if "month" in fd:
            kw["failure_month"] = _int(fd["month"], f"{where}.failure.month")
        if fd.get("fallback") is not None:
            kw["fallback"] = str(fd["fallback"])
    return Alternative(name=name, **kw)


def parse(raw: Any) -> Scenario:
    d = _strict(raw, TOP_KEYS, "scenario")
    for key in ("name", "variables", "volume", "options"):
        if key not in d:
            raise ValueError(f"scenario: missing required key {key!r}")
    variables_raw = d["variables"]
    if not isinstance(variables_raw, dict) or not variables_raw:
        raise ValueError("variables: expected a non-empty object")
    variables = {k: from_dict(v, f"variables.{k}") for k, v in variables_raw.items()}
    vol = _strict(d["volume"], VOLUME_KEYS, "volume")
    if "start" not in vol:
        raise ValueError("volume: missing required key 'start'")
    ben = _strict(d.get("benefit", {}), BENEFIT_KEYS, "benefit")
    options_raw = d["options"]
    if not isinstance(options_raw, dict):
        raise ValueError("options: expected an object of option name to definition")
    options = {n: _option(n, o) for n, o in options_raw.items()}
    gates_raw = _strict(d.get("gates", {}), GATE_KEYS, "gates")
    gates: dict[str, Any] = {}
    for k, val in gates_raw.items():
        if k == "leader_must_be":
            if val not in options:
                raise ValueError(f"gates.leader_must_be: {val!r} is not an option")
            gates[k] = val
        else:
            gates[k] = _number(val, f"gates.{k}")
    kw: dict[str, Any] = {}
    if "growth_per_year" in vol:
        kw["volume_growth_per_year"] = _expr(vol["growth_per_year"], "volume.growth_per_year")
    if vol.get("cap") is not None:
        kw["volume_cap"] = _expr(vol["cap"], "volume.cap")
    if "per_unit" in ben:
        kw["benefit_per_unit"] = _expr(ben["per_unit"], "benefit.per_unit")
    if "fixed_per_month" in ben:
        kw["benefit_fixed_per_month"] = _expr(ben["fixed_per_month"], "benefit.fixed_per_month")
    model = Model(name=str(d["name"]), variables=variables, volume_start=_expr(vol["start"], "volume.start"), options=options,
                  horizon_months=_int(d.get("horizon_months", 60), "horizon_months"),
                  discount_rate=_number(d.get("discount_rate", 0.10), "discount_rate"), **kw)
    return Scenario(model, gates, json.loads(json.dumps(raw)))


def load(path: str | Path) -> Scenario:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not valid JSON ({exc})") from exc
    try:
        return parse(raw)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc
