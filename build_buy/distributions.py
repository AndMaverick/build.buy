"""Distributions for uncertain assumptions. Each has ``ppf`` (the quantile function that drives sampling), ``cdf`` and ``mean``, so a
sample can be checked against the exact answer."""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Protocol

_N = NormalDist()
Z90 = _N.inv_cdf(0.95)   # a 90 percent interval runs from the 5th to the 95th percentile


class Distribution(Protocol):
    def ppf(self, p: float) -> float: ...
    def cdf(self, x: float) -> float: ...
    def mean(self) -> float: ...


@dataclass(frozen=True)
class Fixed:
    value: float

    def ppf(self, p: float) -> float:
        return self.value

    def cdf(self, x: float) -> float:
        return 1.0 if x >= self.value else 0.0

    def mean(self) -> float:
        return self.value


@dataclass(frozen=True)
class Uniform:
    low: float
    high: float

    def __post_init__(self) -> None:
        if not self.low <= self.high:
            raise ValueError(f"uniform needs low <= high, got {self.low} and {self.high}")

    def ppf(self, p: float) -> float:
        return self.low + p * (self.high - self.low)

    def cdf(self, x: float) -> float:
        if self.high == self.low:
            return 1.0 if x >= self.low else 0.0
        return min(1.0, max(0.0, (x - self.low) / (self.high - self.low)))

    def mean(self) -> float:
        return (self.low + self.high) / 2


@dataclass(frozen=True)
class Triangular:
    """Low, most likely and high: the usual way people state an estimate. Skews naturally when the mode is off-centre."""

    low: float
    mode: float
    high: float

    def __post_init__(self) -> None:
        if not self.low <= self.mode <= self.high:
            raise ValueError(f"triangular needs low <= mode <= high, got {self.low}, {self.mode}, {self.high}")

    def ppf(self, p: float) -> float:
        lo, m, hi = self.low, self.mode, self.high
        if hi == lo:
            return lo
        if p < (m - lo) / (hi - lo):
            return lo + math.sqrt(p * (hi - lo) * (m - lo))
        return hi - math.sqrt((1 - p) * (hi - lo) * (hi - m))

    def cdf(self, x: float) -> float:
        lo, m, hi = self.low, self.mode, self.high
        if x <= lo:
            return 1.0 if hi == lo and x >= lo else 0.0
        if x >= hi:
            return 1.0
        if x <= m:
            return (x - lo) ** 2 / ((hi - lo) * (m - lo))
        return 1 - (hi - x) ** 2 / ((hi - lo) * (hi - m))

    def mean(self) -> float:
        return (self.low + self.mode + self.high) / 3


@dataclass(frozen=True)
class Normal:
    """A normal, optionally cut off at ``lower`` and ``upper`` (renormalised, not clipped)."""

    mu: float
    sigma: float
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        if self.sigma < 0:
            raise ValueError("normal sigma must not be negative")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError("normal lower must not exceed upper")
        if self.sigma > 0 and self._mass() <= 0:
            raise ValueError("the bounds leave no probability mass")

    def _bounds(self) -> tuple[float, float]:
        if self.sigma == 0:
            return 0.0, 1.0
        lo = 0.0 if self.lower is None else _N.cdf((self.lower - self.mu) / self.sigma)
        hi = 1.0 if self.upper is None else _N.cdf((self.upper - self.mu) / self.sigma)
        return lo, hi

    def _mass(self) -> float:
        lo, hi = self._bounds()
        return hi - lo

    def ppf(self, p: float) -> float:
        if self.sigma == 0:
            return self.mu
        lo, hi = self._bounds()
        return self.mu + self.sigma * _N.inv_cdf(lo + min(max(p, 1e-12), 1 - 1e-12) * (hi - lo))

    def cdf(self, x: float) -> float:
        if self.sigma == 0:
            return 1.0 if x >= self.mu else 0.0
        lo, hi = self._bounds()
        return min(1.0, max(0.0, (_N.cdf((x - self.mu) / self.sigma) - lo) / (hi - lo)))

    def mean(self) -> float:
        if self.sigma == 0:
            return self.mu
        lo, hi = self._bounds()
        a = -math.inf if self.lower is None else (self.lower - self.mu) / self.sigma
        b = math.inf if self.upper is None else (self.upper - self.mu) / self.sigma
        return self.mu + self.sigma * (_phi(a) - _phi(b)) / (hi - lo)


def _phi(x: float) -> float:
    return 0.0 if math.isinf(x) else math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


@dataclass(frozen=True)
class Lognormal:
    """Positive and right-skewed: costs and durations overrun more than they underrun. ``mu`` and ``sigma`` are of the log."""

    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if self.sigma < 0:
            raise ValueError("lognormal sigma must not be negative")

    def ppf(self, p: float) -> float:
        if self.sigma == 0:
            return math.exp(self.mu)
        return math.exp(self.mu + self.sigma * _N.inv_cdf(min(max(p, 1e-12), 1 - 1e-12)))

    def cdf(self, x: float) -> float:
        if x <= 0:
            return 0.0
        if self.sigma == 0:
            return 1.0 if x >= math.exp(self.mu) else 0.0
        return _N.cdf((math.log(x) - self.mu) / self.sigma)

    def mean(self) -> float:
        return math.exp(self.mu + self.sigma ** 2 / 2)


def from_ci90(low: float, high: float, *, kind: str = "normal", lower: float | None = None, upper: float | None = None) -> Distribution:
    """A distribution from a 90 percent interval: you are 90 percent sure the value lies between ``low`` and ``high``."""
    if not low < high:
        raise ValueError(f"a 90 percent interval needs low < high, got {low} and {high}")
    if kind == "normal":
        return Normal((low + high) / 2, (high - low) / (2 * Z90), lower, upper)
    if kind == "lognormal":
        if low <= 0:
            raise ValueError("a lognormal interval needs low > 0")
        return Lognormal((math.log(low) + math.log(high)) / 2, (math.log(high) - math.log(low)) / (2 * Z90))
    raise ValueError(f"kind must be normal or lognormal, got {kind!r}")


_SHAPES = {"fixed": {"value"}, "uniform": {"low", "high"}, "triangular": {"low", "mode", "high"}, "normal": {"mu", "sigma", "lower", "upper"},
           "lognormal": {"mu", "sigma"}, "ci90": {"low", "high", "kind", "lower", "upper"}}


def from_dict(spec: Any, where: str = "variable") -> Distribution:
    """A bare number is fixed; ``{"dist": "triangular", "low": ..., "mode": ..., "high": ...}`` and so on. Unknown keys are errors."""
    if isinstance(spec, bool):
        raise ValueError(f"{where}: expected a number or a distribution object")
    if isinstance(spec, (int, float)):
        return Fixed(float(spec))
    if not isinstance(spec, dict) or "dist" not in spec:
        raise ValueError(f"{where}: expected a number or an object with a 'dist' key")
    kind = spec["dist"]
    if kind not in _SHAPES:
        raise ValueError(f"{where}: dist must be one of {', '.join(sorted(_SHAPES))}, got {kind!r}")
    extra = sorted(set(spec) - _SHAPES[kind] - {"dist"})
    if extra:
        raise ValueError(f"{where}: unknown key {extra[0]!r} for {kind} (allowed: {', '.join(sorted(_SHAPES[kind]))})")
    args = {k: v for k, v in spec.items() if k != "dist"}
    try:
        if kind == "fixed":
            return Fixed(float(args["value"]))
        if kind == "ci90":
            return from_ci90(float(args["low"]), float(args["high"]), kind=args.get("kind", "normal"), lower=args.get("lower"), upper=args.get("upper"))
        cls: Any = {"uniform": Uniform, "triangular": Triangular, "normal": Normal, "lognormal": Lognormal}[kind]
        result: Distribution = cls(**args)
        return result
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{where}: {kind} is missing or has a bad value ({exc})") from exc
