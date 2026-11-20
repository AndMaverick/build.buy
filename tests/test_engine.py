import math
import random
import unittest

from build_buy.distributions import Fixed
from build_buy.engine import build_flows, evaluate, failed_flows, grow, measure, tiered_cost, whole
from build_buy.expr import compile_expr as c
from build_buy.model import Alternative, Model, Tier
from build_buy.synthetic import random_model


def model_with(opt, other=None, **kw):
    options = {"x": opt, "y": other or Alternative("y")}
    base = {"name": "m", "variables": {"u": Fixed(1.0)}, "volume_start": c("50"), "options": options, "horizon_months": 12,
            "discount_rate": 0.0, "benefit_per_unit": c("5")}
    base.update(kw)
    return Model(**base)


def alt(**kw):
    return Alternative(name="x", **kw)


def oracle(model, name, v, failed=False):
    """Independent month-by-month ledger of net cash flow, written from the definitions without sharing code with the engine."""
    h = model.horizon_months

    def one(option, start):
        cost, benefit = [0.0] * (h + 1), [0.0] * (h + 1)
        live = max(1, math.ceil(round(option.live_month(v), 9)))
        spread = max(0, math.ceil(round(option.one_time_months(v), 9)))
        capex = sum(e(v) for e in option.one_time.values())
        team = sum(e(v) for e in option.build_monthly.values())
        run = sum(e(v) for e in option.run_monthly.values())
        for t in range(start, h + 1):
            k = t - start
            if spread == 0 and k == 0:
                cost[t] += capex
            if spread > 0 and 1 <= k <= spread:
                cost[t] += capex / spread
            if 1 <= k < live:
                cost[t] += team
            if k >= live:
                age = k - live
                cost[t] += run * (1 + option.run_growth_per_year(v)) ** (age / 12)
                vol = model.volume_start(v) * (1 + model.volume_growth_per_year(v)) ** ((t - 1) / 12)
                if model.volume_cap is not None:
                    vol = min(vol, model.volume_cap(v))
                if option.tiers:
                    lo, usage = 0.0, 0.0
                    for tier in option.tiers:
                        hi = math.inf if tier.up_to is None else tier.up_to(v)
                        usage += tier.price(v) * max(0.0, min(vol, hi) - lo)
                        lo = hi
                else:
                    usage = vol * (option.unit_cost(v) if option.unit_cost is not None else 0.0)
                cost[t] += max(usage, option.minimum_usage_cost(v)) * (1 + option.price_change_per_year(v)) ** (age / 12)
                ramp = option.ramp_months(v)
                share = 1.0 if ramp <= 0 else min(1.0, (age + 1) / ramp)
                benefit[t] += option.quality(v) * share * (vol * model.benefit_per_unit(v) + model.benefit_fixed_per_month(v))
                if option.refresh_every_months is not None:
                    every = math.ceil(round(option.refresh_every_months(v), 9))
                    if every > 0 and age > 0 and age % every == 0:
                        cost[t] += option.refresh_cost(v)
        return cost, benefit

    option = model.options[name]
    cost, benefit = one(option, 0)
    cost[h] -= option.terminal(v)
    if failed:
        f = option.failure_month
        cost = [x if t <= f else 0.0 for t, x in enumerate(cost)]
        benefit = [x if t <= f else 0.0 for t, x in enumerate(benefit)]
        cost[h] = 0.0
        if option.fallback is not None:
            fb = model.options[option.fallback]
            fcost, fben = one(fb, f)
            fcost[h] -= fb.terminal(v)
            cost = [a + b for a, b in zip(cost, fcost, strict=True)]
            benefit = [a + b for a, b in zip(benefit, fben, strict=True)]
    rate = model.discount_rate
    return sum((b - k) * (1 + rate) ** (-t / 12) for t, (b, k) in enumerate(zip(benefit, cost, strict=True)))


class FlowTests(unittest.TestCase):
    def test_hand_computed_flat_case(self):
        m = model_with(alt(run_monthly={"r": c("100")}, unit_cost=c("2"), one_time={"o": c("300")}))
        o = evaluate(m, "x", {"u": 1.0})
        self.assertAlmostEqual(o.npv, 12 * (250 - 100 - 100) - 300)
        self.assertEqual(o.tco, 300 + 12 * 200)
        self.assertEqual(o.total_units, 12 * 50)
        self.assertAlmostEqual(o.cost_per_unit, (300 + 12 * 200) / 600)
        self.assertEqual(o.payback_month, 6.0)       # -300 + 50 t reaches zero at t = 6

    def test_one_time_cost_spread_over_months(self):
        f = build_flows(model_with(alt(one_time={"o": c("300")}, one_time_months=c("3"))), alt(one_time={"o": c("300")}, one_time_months=c("3")), {"u": 1.0})
        self.assertEqual(f.lines["one_time"][:5], [0.0, 100.0, 100.0, 100.0, 0.0])

    def test_build_team_runs_before_live_month(self):
        a = alt(live_month=c("4"), build_monthly={"t": c("70")}, run_monthly={"r": c("10")})
        f = build_flows(model_with(a), a, {"u": 1.0})
        self.assertEqual(f.lines["build_team"][:6], [0.0, 70.0, 70.0, 70.0, 0.0, 0.0])
        self.assertEqual(f.lines["run"][:6], [0.0, 0.0, 0.0, 0.0, 10.0, 10.0])
        self.assertEqual(f.benefit[3], 0.0)
        self.assertEqual(f.benefit[4], 250.0)

    def test_ramp(self):
        a = alt(ramp_months=c("4"))
        f = build_flows(model_with(a), a, {"u": 1.0})
        self.assertEqual([f.benefit[t] / 250 for t in range(1, 7)], [0.25, 0.5, 0.75, 1.0, 1.0, 1.0])

    def test_quality_multiplies_benefit(self):
        a = alt(quality=c("1.1"))
        self.assertAlmostEqual(build_flows(model_with(a), a, {"u": 1.0}).benefit[1], 275.0)

    def test_growth_after_live_and_price_drift(self):
        a = alt(live_month=c("2"), run_monthly={"r": c("100")}, run_growth_per_year=c("0.5"), unit_cost=c("1"), price_change_per_year=c("-0.2"))
        f = build_flows(model_with(a, horizon_months=24), a, {"u": 1.0})
        self.assertEqual(f.lines["run"][1], 0.0)
        self.assertAlmostEqual(f.lines["run"][2], 100.0)
        self.assertAlmostEqual(f.lines["run"][14], 150.0)             # twelve months after going live: +50 percent
        self.assertAlmostEqual(f.lines["usage"][14], 50 * 0.8)

    def test_volume_growth_and_cap(self):
        a = alt(unit_cost=c("1"))
        m = model_with(a, volume_growth_per_year=c("1.0"), volume_cap=c("80"), horizon_months=24)
        f = build_flows(m, a, {"u": 1.0})
        self.assertAlmostEqual(f.units[1], 50.0)
        self.assertAlmostEqual(f.units[13], 80.0)                      # would be 100 after a year, capped at 80
        self.assertAlmostEqual(f.units[7], 50 * 2 ** 0.5)

    def test_minimum_commit_floors_usage_cost(self):
        a = alt(unit_cost=c("1"), minimum_usage_cost=c("80"))
        self.assertEqual(build_flows(model_with(a), a, {"u": 1.0}).lines["usage"][1], 80.0)

    def test_terminal_value_and_exit_cost(self):
        for terminal, expected in ((500, -500.0), (-300, 300.0)):
            a = alt(terminal=c(str(terminal)))
            self.assertEqual(build_flows(model_with(a), a, {"u": 1.0}).lines["terminal"][12], expected)

    def test_refresh_costs_recur_after_going_live(self):
        a = alt(live_month=c("2"), refresh_every_months=c("4"), refresh_cost=c("1000"))
        f = build_flows(model_with(a, horizon_months=20), a, {"u": 1.0})
        self.assertEqual([t for t, x in enumerate(f.lines["refresh"]) if x], [6, 10, 14, 18])

    def test_failed_flows_stop_and_hand_over_to_the_fallback(self):
        a = Alternative("x", one_time={"o": c("1000")}, build_monthly={"t": c("10")}, live_month=c("20"), failure_probability=0.5,
                        failure_month=4, fallback="y")
        b = Alternative("y", live_month=c("2"), run_monthly={"r": c("30")})
        m = model_with(a, b)
        f = failed_flows(m, a, {"u": 1.0})
        self.assertEqual(f.lines["one_time"][0], 1000.0)
        self.assertEqual(f.lines["build_team"][1:6], [10.0, 10.0, 10.0, 10.0, 0.0])     # spent through month 4, nothing after
        self.assertEqual(f.lines["fallback_run"][5], 0.0)                                # fallback starts at month 4, live at its month 2
        self.assertEqual(f.lines["fallback_run"][6], 30.0)
        self.assertEqual(f.benefit[5], 0.0)
        self.assertEqual(f.benefit[6], 250.0)

    def test_failure_without_fallback_just_stops(self):
        a = Alternative("x", one_time={"o": c("1000")}, failure_probability=0.5, failure_month=3)
        f = failed_flows(model_with(a), a, {"u": 1.0})
        self.assertEqual(sum(f.benefit), 3 * 250.0)
        self.assertEqual(f.lines["terminal"][12], 0.0)

    def test_guards(self):
        a = alt()
        with self.assertRaises(ValueError):
            build_flows(model_with(a, volume_growth_per_year=c("-1.5")), a, {"u": 1.0})
        with self.assertRaises(ValueError):
            grow(-1.0, 0.5)
        with self.assertRaises(ValueError):
            evaluate(model_with(a), "x", {})
        self.assertEqual(whole(8.000000000000002), 8)
        self.assertEqual(whole(8.2), 9)

    def test_model_validation(self):
        with self.assertRaises(ValueError):
            Alternative("x", unit_cost=c("1"), tiers=(Tier(None, c("1")),))
        with self.assertRaises(ValueError):
            Alternative("x", tiers=(Tier(c("10"), c("1")),))
        with self.assertRaises(ValueError):
            Alternative("x", tiers=(Tier(None, c("1")), Tier(None, c("1"))))
        with self.assertRaises(ValueError):
            Alternative("x", failure_probability=1.5)
        with self.assertRaises(ValueError):
            Alternative("x", refresh_every_months=c("3"))
        with self.assertRaises(ValueError):
            Alternative("x", fallback="x")
        with self.assertRaises(ValueError):
            model_with(alt(unit_cost=c("zzz")))
        with self.assertRaises(ValueError):
            model_with(alt(), horizon_months=1)
        with self.assertRaises(ValueError):
            Model("m", {"u": Fixed(1)}, c("1"), {"x": Alternative("x")})
        with self.assertRaises(ValueError):
            model_with(alt(failure_probability=0.5, failure_month=20))
        with self.assertRaises(ValueError):
            model_with(alt(fallback="nope"))


class ClosedFormTests(unittest.TestCase):
    def test_level_annuity(self):
        # a flat monthly cost for the whole horizon, discounted: PV = c (1 - v^N) / i with i the monthly rate
        for rate in (0.0, 0.05, 0.12):
            a = alt(run_monthly={"r": c("100")})
            m = model_with(a, discount_rate=rate, benefit_per_unit=c("0"))
            i = (1 + rate) ** (1 / 12) - 1
            want = 100 * 12 if i == 0 else 100 * (1 - (1 + i) ** -12) / i
            self.assertAlmostEqual(evaluate(m, "x", {"u": 1.0}).pv_cost, want, places=8)

    def test_growing_annuity(self):
        # cost_t = c q^(t - 1) with q = (1 + g)^(1/12): a geometric series discounted by d = (1 + i)^-1
        rate, g, n = 0.10, 0.30, 36
        a = alt(run_monthly={"r": c("100")}, run_growth_per_year=c(str(g)))
        m = model_with(a, discount_rate=rate, benefit_per_unit=c("0"), horizon_months=n)
        q, d = (1 + g) ** (1 / 12), (1 + rate) ** (-1 / 12)
        want = 100 * d * (1 - (q * d) ** n) / (1 - q * d)
        self.assertAlmostEqual(evaluate(m, "x", {"u": 1.0}).pv_cost, want, places=6)

    def test_tiered_cost_by_hand(self):
        bands = [(10.0, 1.0), (20.0, 0.8), (None, 0.5)]
        self.assertAlmostEqual(tiered_cost(25, bands), 10 + 8 + 2.5)
        self.assertEqual(tiered_cost(0, bands), 0.0)
        self.assertEqual(tiered_cost(10, bands), 10.0)
        self.assertEqual(tiered_cost(5, bands), 5.0)

    def test_tiered_cost_matches_an_independent_formula_and_is_continuous_and_monotone(self):
        rng = random.Random(3)
        for _ in range(300):
            cuts = sorted(rng.sample(range(1, 500), rng.randint(1, 4)))
            prices = [rng.uniform(0.1, 5) for _ in range(len(cuts) + 1)]
            bands = [(float(cut), p) for cut, p in zip(cuts, prices, strict=False)] + [(None, prices[-1])]
            lows = [0.0, *map(float, cuts)]
            his = [*map(float, cuts), math.inf]
            prev = 0.0
            for u in sorted(rng.uniform(0, 700) for _ in range(40)):
                want = sum(p * max(0.0, min(u, hi) - lo) for p, lo, hi in zip(prices, lows, his, strict=True))
                got = tiered_cost(u, bands)
                self.assertAlmostEqual(got, want, places=8)
                self.assertGreaterEqual(got, prev - 1e-9)
                prev = got
            eps = 1e-7
            for cut in cuts:
                self.assertAlmostEqual(tiered_cost(cut + eps, bands), tiered_cost(cut - eps, bands), delta=1e-5)

    def test_tier_bounds_must_increase(self):
        with self.assertRaises(ValueError):
            tiered_cost(30, [(20.0, 1.0), (10.0, 0.5), (None, 0.2)])

    def test_irr_makes_the_net_flow_worth_zero(self):
        a = alt(one_time={"o": c("500")}, run_monthly={"r": c("20")}, unit_cost=c("1"))
        o = evaluate(model_with(a), "x", {"u": 1.0})
        self.assertFalse(math.isnan(o.irr))
        monthly = (1 + o.irr) ** (1 / 12) - 1
        self.assertAlmostEqual(sum(x / (1 + monthly) ** t for t, x in enumerate(o.flows.net)), 0.0, delta=1e-6)

    def test_money_scales_linearly(self):
        # multiplying every money amount by k multiplies NPV and TCO by k and leaves the ranking and payback alone
        for k in (0.5, 3.0, 40.0):
            def build(scale):
                a = Alternative("x", one_time={"o": c(str(1000 * scale))}, run_monthly={"r": c(str(30 * scale))}, unit_cost=c(str(0.4 * scale)),
                                live_month=c("3"), build_monthly={"t": c(str(100 * scale))}, ramp_months=c("4"))
                return model_with(a, benefit_per_unit=c(str(5 * scale)), discount_rate=0.1, volume_growth_per_year=c("0.5"))
            base, scaled = evaluate(build(1.0), "x", {"u": 1.0}), evaluate(build(k), "x", {"u": 1.0})
            self.assertAlmostEqual(scaled.npv, k * base.npv, delta=1e-8 * abs(k * base.npv))
            self.assertAlmostEqual(scaled.tco, k * base.tco, delta=1e-8 * k * base.tco)
            self.assertEqual(scaled.payback_month, base.payback_month)

    def test_dearer_inputs_never_raise_npv(self):
        rng = random.Random(7)
        for _ in range(60):
            p = rng.uniform(0.1, 2)
            a = alt(unit_cost=c(str(p)), run_monthly={"r": c(str(rng.uniform(0, 100)))})
            b = alt(unit_cost=c(str(p * rng.uniform(1.0, 3.0))), run_monthly=a.run_monthly)
            self.assertLessEqual(evaluate(model_with(b), "x", {"u": 1.0}).npv, evaluate(model_with(a), "x", {"u": 1.0}).npv + 1e-9)

    def test_later_start_never_raises_npv(self):
        for live in range(1, 9):
            a = alt(live_month=c(str(live)), run_monthly={"r": c("10")}, build_monthly={"t": c("5")})
            nxt = alt(live_month=c(str(live + 1)), run_monthly={"r": c("10")}, build_monthly={"t": c("5")})
            self.assertGreater(evaluate(model_with(a), "x", {"u": 1.0}).npv, evaluate(model_with(nxt), "x", {"u": 1.0}).npv)


class OracleTests(unittest.TestCase):
    def test_matches_independent_ledger_on_random_models(self):
        rng = random.Random(41)
        failed_checked = 0
        for i in range(500):
            m = random_model(rng, f"m{i}")
            v = {n: d.ppf(rng.random()) for n, d in m.variables.items()}
            for name in m.options:
                got = evaluate(m, name, v).npv
                want = oracle(m, name, v)
                self.assertAlmostEqual(got, want, delta=1e-7 * max(1.0, abs(want)), msg=f"model {i} option {name}")
                if m.options[name].failure_probability > 0:
                    gotf = evaluate(m, name, v, failed=True).npv
                    wantf = oracle(m, name, v, failed=True)
                    self.assertAlmostEqual(gotf, wantf, delta=1e-7 * max(1.0, abs(wantf)), msg=f"failed: model {i} option {name}")
                    failed_checked += 1
        self.assertGreater(failed_checked, 100)

    def test_outcome_is_consistent(self):
        rng = random.Random(42)
        for i in range(60):
            m = random_model(rng, f"m{i}")
            v = {n: d.ppf(0.5) for n, d in m.variables.items()}
            o = measure(m, "a", build_flows(m, m.options["a"], v))
            self.assertAlmostEqual(o.npv, o.pv_benefit - o.pv_cost, delta=1e-8 * max(1.0, abs(o.npv)))
            self.assertAlmostEqual(o.tco, sum(o.flows.cost))


if __name__ == "__main__":
    unittest.main()
