"""Reproduces every verification figure quoted in the README. Run: python examples/evaluate.py

Each line compares the library with an answer computed a different way (a closed form, a second implementation, or Python itself), so a
mismatch means a bug, not noise. Everything is seeded, so the numbers are the same on every machine."""
from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path
from statistics import NormalDist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build_buy import (  # noqa: E402
    Alternative,
    Model,
    analyze,
    breakeven,
    compare,
    compile_expr,
    crossover,
    evaluate,
    load,
    simulate,
)
from build_buy.distributions import Normal, Uniform  # noqa: E402
from build_buy.engine import tiered_cost  # noqa: E402
from build_buy.simulate import evpi, summarize, value_of_information  # noqa: E402
from build_buy.synthetic import random_expression, random_model  # noqa: E402

N = NormalDist()
ROOT = Path(__file__).resolve().parent
c = compile_expr


def positive_part(m: float, s: float) -> float:
    z = m / s
    return m * N.cdf(z) + s * N.pdf(z)


def line(label: str, detail: str) -> None:
    print(f"{label:<56}{detail}")


def two_way(a: Alternative, b: Alternative, variables: dict, **kw: object) -> Model:  # type: ignore[type-arg]
    kw.setdefault("volume_start", c("1"))
    return Model(name="t", variables=variables, options={a.name: a, b.name: b}, horizon_months=12, discount_rate=0.0,
                 **kw)  # type: ignore[arg-type]


def main() -> int:
    t0 = time.time()
    rng = random.Random(1)

    names, good, total = ["a", "b", "c"], 0, 0
    for _ in range(5000):
        src = random_expression(rng, names)
        env = {n: round(rng.uniform(-6, 9), 2) for n in names}
        try:
            want = float(eval(src, {"__builtins__": {}}, {"min": min, "max": max, "abs": abs, **env}))  # noqa: S307 (oracle)
        except ZeroDivisionError:
            continue
        total += 1
        good += abs(c(src)(env) - want) <= 1e-9 * max(1.0, abs(want))
    line("formula evaluator vs Python eval", f"{good}/{total} random formulas identical")

    bad = 0
    for _ in range(3000):
        cuts = sorted(rng.sample(range(1, 500), rng.randint(1, 4)))
        prices = [rng.uniform(0.1, 5) for _ in range(len(cuts) + 1)]
        bands = [(float(x), p) for x, p in zip(cuts, prices, strict=False)] + [(None, prices[-1])]
        u = rng.uniform(0, 700)
        lows, his = [0.0, *map(float, cuts)], [*map(float, cuts), math.inf]
        want = sum(p * max(0.0, min(u, hi) - lo) for p, lo, hi in zip(prices, lows, his, strict=True))
        bad += abs(tiered_cost(u, bands) - want) > 1e-8
    line("graduated usage pricing vs independent formula", f"{3000 - bad}/3000 agree to 1e-8")

    worst, count = 0.0, 0
    for i in range(1000):
        m = random_model(rng, f"m{i}")
        v = {n: d.ppf(rng.random()) for n, d in m.variables.items()}
        for name in m.options:
            ok = evaluate(m, name, v).npv
            h = m.horizon_months
            r = m.discount_rate
            net = evaluate(m, name, v).flows.net
            worst = max(worst, abs(ok - sum(x * (1 + r) ** (-t / 12) for t, x in enumerate(net))) / max(1.0, abs(ok)))
            count += 1
            assert len(net) == h + 1
    line("NPV vs discounting the monthly net flow", f"{count} options, largest relative gap {worst:.1e}")

    for upfront, a, b in ((500, 50, 200), (1000, 10, 130), (360, 40, 100)):
        build = Alternative("a", one_time={"o": c(str(upfront))}, run_monthly={"r": c(str(a))})
        buy = Alternative("b", run_monthly={"r": c(str(b))})
        o = {x.name: x for x in compare(two_way(build, buy, {"u": Normal(1, 0)}, benefit_per_unit=c("0")))}
        cross = crossover(o["a"], o["b"])
        line(f"crossover, {upfront} up front, {a} vs {b} a month", f"month {cross:.6f}, closed form {upfront / (b - a):.6f}")

    build = Alternative("build", one_time={"o": c("1200")})
    buy = Alternative("buy", unit_cost=c("2"))
    m = two_way(build, buy, {"v": Uniform(10, 110)}, benefit_per_unit=c("0"), volume_start=c("v"))
    be = breakeven(m, "v", "build", "buy")
    line("break-even volume, build vs per-request price", f"{be.value:.6f}, closed form {1200 / (12 * 2):.6f}; P(build wins) {be.probability_favours_first:.3f}")

    a = Alternative("a", terminal=c("x"))
    mt = two_way(a, Alternative("b"), {"x": Normal(0.5, 2.0)}, benefit_per_unit=c("0"))
    run = simulate(mt, 40000, 3)
    line("EVPI, one normal uncertainty", f"{evpi(run):.4f} against closed form {positive_part(0.5, 2.0) - 0.5:.4f}")
    a2 = Alternative("a", terminal=c("x + y"))
    m2 = two_way(a2, Alternative("b"), {"x": Normal(-1, 2), "y": Normal(0.5, 3)}, benefit_per_unit=c("0"))
    voi = dict(value_of_information(simulate(m2, 80000, 2), bins=40))
    line("EVPPI of the first input", f"{voi['x']:.4f} against closed form {positive_part(-0.5, 2):.4f}")
    line("EVPPI of the second input", f"{voi['y']:.4f} against closed form {positive_part(-0.5, 3):.4f}")

    ident = 0
    for i in range(20):
        mm = random_model(rng, f"s{i}")
        r = simulate(mm, 300, i)
        ident += abs(evpi(r) - summarize(r)[0].expected_regret) <= 1e-7 * max(1.0, abs(summarize(r)[0].mean))
    line("EVPI equals the leader's expected regret", f"{ident}/20 random decisions")

    risky = Alternative("a", one_time={"o": c("1000")}, run_monthly={"r": c("20")}, live_month=c("6"), failure_probability=0.3,
                        failure_month=4, fallback="b")
    safe = Alternative("b", live_month=c("2"), run_monthly={"r": c("30")})
    mf = two_way(risky, safe, {"u": Normal(1, 0)}, benefit_per_unit=c("5"), volume_start=c("20"))
    ok = evaluate(mf, "a", {"u": 1.0}).npv
    failed = evaluate(mf, "a", {"u": 1.0}, failed=True).npv
    runf = simulate(mf, 20000, 1)
    mean = sum(runf.npv["a"]) / 20000
    line("expected NPV with a 30% failure risk", f"{mean:.1f} against {0.7 * ok + 0.3 * failed:.1f} (0.7 x {ok:.1f} + 0.3 x {failed:.1f})")

    sc = load(ROOT / "ai_assistant.json")
    t1 = time.time()
    rep = analyze(sc, 3000, 7)
    line("full analysis, 15 variables, 3 options, 3000 trials", f"{time.time() - t1:.2f}s; leader {rep['leader']}, "
         f"P(beats runner-up) {100 * rep['advantage']['prob_leader_beats_runner_up']:.1f}%")
    line("repeat run is identical", "yes" if analyze(sc, 3000, 7) == rep else "NO")
    print(f"\nTotal {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
