import math
import os
import random
import subprocess
import sys
import unittest
from pathlib import Path
from statistics import NormalDist

from build_buy.compare import base_values, breakeven, compare, crossover, tornado
from build_buy.distributions import Fixed, Normal, Uniform
from build_buy.engine import evaluate
from build_buy.expr import compile_expr as c
from build_buy.model import Alternative, Model
from build_buy.simulate import evpi, evppi, prob_beats, simulate, summarize, value_of_information
from build_buy.stats import quantile, uniforms, wilson
from build_buy.synthetic import random_model

N = NormalDist()
ROOT = Path(__file__).resolve().parent.parent


def positive_part(m, s):
    z = m / s
    return m * N.cdf(z) + s * N.pdf(z)


def two_way(a, b, variables, **kw):
    base = {"name": "t", "variables": variables, "volume_start": c("1"), "options": {a.name: a, b.name: b}, "horizon_months": 12,
            "discount_rate": 0.0}
    base.update(kw)
    return Model(**base)


class CrossoverTests(unittest.TestCase):
    def test_flat_costs_give_the_closed_form(self):
        # build: U up front then a a month; buy: b a month. Cumulative costs U + a t and b t cross at t = U / (b - a)
        for upfront, a, b in ((500, 50, 200), (1000, 10, 130), (360, 40, 100), (77, 3, 8)):
            build = Alternative("a", one_time={"o": c(str(upfront))}, run_monthly={"r": c(str(a))})
            buy = Alternative("b", run_monthly={"r": c(str(b))})
            m = two_way(build, buy, {"u": Fixed(1)}, horizon_months=60)
            o = {x.name: x for x in compare(m)}
            self.assertAlmostEqual(crossover(o["a"], o["b"]), upfront / (b - a), places=9)
            self.assertAlmostEqual(crossover(o["b"], o["a"]), upfront / (b - a), places=9)

    def test_no_crossing(self):
        a = Alternative("a", run_monthly={"r": c("10")})
        b = Alternative("b", run_monthly={"r": c("20")})
        o = {x.name: x for x in compare(two_way(a, b, {"u": Fixed(1)}))}
        self.assertIsNone(crossover(o["a"], o["b"]))

    def test_discounting_moves_the_crossing_later(self):
        build = Alternative("a", one_time={"o": c("1000")}, run_monthly={"r": c("10")})
        buy = Alternative("b", run_monthly={"r": c("100")})
        m = two_way(build, buy, {"u": Fixed(1)}, horizon_months=60)
        o = {x.name: x for x in compare(m)}
        plain = crossover(o["a"], o["b"])
        later = crossover(o["a"], o["b"], discounted=True, rate=0.15)
        self.assertGreater(later, plain)


class BreakEvenTests(unittest.TestCase):
    def setUp(self):
        # build: F up front, no usage cost.  buy: p per unit of volume v a month for 12 months.  Equal NPV at v* = F / (12 p)
        self.F, self.p = 1200.0, 2.0
        build = Alternative("build", one_time={"o": c(str(self.F))})
        buy = Alternative("buy", unit_cost=c(str(self.p)))
        self.model = two_way(build, buy, {"v": Uniform(10, 110)}, volume_start=c("v"), benefit_per_unit=c("0"))

    def test_volume_breakeven_matches_the_closed_form(self):
        be = breakeven(self.model, "v", "build", "buy")
        self.assertAlmostEqual(be.value, self.F / (12 * self.p), places=8)
        # build wins when volume is above the break-even; v ~ Uniform(10, 110) so P(v > 50) = 0.6
        self.assertAlmostEqual(be.probability_favours_first, 0.6, places=8)
        self.assertEqual(be.direction, "first wins when higher")

    def test_swapping_the_options_flips_the_side(self):
        be = breakeven(self.model, "v", "buy", "build")
        self.assertAlmostEqual(be.value, 50.0, places=8)
        self.assertAlmostEqual(be.probability_favours_first, 0.4, places=8)

    def test_gap_is_zero_at_the_root_for_random_models(self):
        rng = random.Random(15)
        found = 0
        for i in range(50):
            m = random_model(rng, f"b{i}")
            base = base_values(m)
            for var in m.variables:
                be = breakeven(m, var, "a", "b")
                if be.value is not None:
                    v = {**base, var: be.value}
                    gap = evaluate(m, "a", v).npv - evaluate(m, "b", v).npv
                    self.assertAlmostEqual(gap, 0.0, delta=1e-4 * max(1.0, abs(evaluate(m, "a", v).npv)))
                    self.assertTrue(0 <= be.probability_favours_first <= 1)
                    found += 1
        self.assertGreater(found, 10)

    def test_never_swaps(self):
        a = Alternative("a", run_monthly={"r": c("10")})
        b = Alternative("b", run_monthly={"r": c("50")})
        m = two_way(a, b, {"v": Uniform(1, 2)})
        self.assertIsNone(breakeven(m, "v", "a", "b").value)

    def test_unknown_variable(self):
        with self.assertRaises(ValueError):
            breakeven(self.model, "zzz", "build", "buy")

    def test_search_survives_variables_that_break_the_formulas_when_pushed_out_of_range(self):
        # a growth rate pushed below -100 percent used to produce a complex number; it must be treated as out of range
        a = Alternative("a", one_time={"o": c("1000")})
        b = Alternative("b", unit_cost=c("1"))
        m = two_way(a, b, {"g": Uniform(-0.2, 0.4), "v": Fixed(100)}, volume_start=c("v"), volume_growth_per_year=c("g"))
        breakeven(m, "g", "a", "b")


class TornadoTests(unittest.TestCase):
    def test_swings_are_exact_evaluations_and_sorted(self):
        rng = random.Random(14)
        for i in range(30):
            m = random_model(rng, f"t{i}")
            base = base_values(m)
            sw = tornado(m, "a", "b")
            for s in sw:
                for x, want in ((s.low_input, s.low_gap), (s.high_input, s.high_gap)):
                    v = {**base, s.variable: x}
                    self.assertAlmostEqual(evaluate(m, "a", v).npv - evaluate(m, "b", v).npv, want)
            self.assertEqual([s.swing for s in sw], sorted((s.swing for s in sw), reverse=True))
        with self.assertRaises(ValueError):
            tornado(m, "a", "b", lo=0.9, hi=0.1)


class SimulationTests(unittest.TestCase):
    def terminal_model(self, mu=0.5, sigma=2.0, extra=None):
        # NPV(a) - NPV(b) = x exactly: a is worth x at the horizon, b is worth nothing, discount rate zero
        a = Alternative("a", terminal=c("x"))
        b = Alternative("b")
        variables = {"x": Normal(mu, sigma), **(extra or {})}
        return two_way(a, b, variables, benefit_per_unit=c("0"))

    def test_probability_matches_the_normal_closed_form(self):
        m = self.terminal_model(0.5, 2.0)
        run = simulate(m, 8000, 3)
        p, (lo, hi) = prob_beats(run, "a", "b")
        want = 1 - N.cdf(-0.5 / 2.0)
        self.assertLessEqual(lo, want)
        self.assertGreaterEqual(hi, want)

    def test_evpi_matches_the_closed_form(self):
        m = self.terminal_model(0.5, 2.0)
        run = simulate(m, 20000, 3)
        # choose b (worth 0) unless a is better: gain = E[max(x, 0)] - max(E[x], 0)
        want = positive_part(0.5, 2.0) - 0.5
        self.assertAlmostEqual(evpi(run), want, delta=0.05)

    def test_evpi_equals_the_leaders_expected_regret(self):
        rng = random.Random(8)
        for i in range(6):
            m = random_model(rng, f"s{i}")
            run = simulate(m, 400, 5)
            s = summarize(run)
            self.assertAlmostEqual(evpi(run), s[0].expected_regret, delta=1e-7 * max(1.0, abs(s[0].mean)))
            self.assertAlmostEqual(sum(x.prob_best for x in s), 1.0)
            for x in s:
                self.assertGreaterEqual(x.expected_regret, -1e-9)
                self.assertGreaterEqual(x.p90_regret, -1e-9)
            self.assertEqual([x.mean for x in s], sorted((x.mean for x in s), reverse=True))

    def test_evppi_matches_the_closed_form_for_each_variable(self):
        # NPV(a) - NPV(b) = x + y with x ~ N(-1, 2), y ~ N(0.5, 3). Learning x pins down E[x + y | x] = x + 0.5
        a = Alternative("a", terminal=c("x + y"))
        m = two_way(a, Alternative("b"), {"x": Normal(-1, 2), "y": Normal(0.5, 3)}, benefit_per_unit=c("0"))
        run = simulate(m, 60000, 2)
        voi = dict(value_of_information(run, bins=40))
        self.assertAlmostEqual(voi["x"], positive_part(-0.5, 2), delta=0.04)
        self.assertAlmostEqual(voi["y"], positive_part(-0.5, 3), delta=0.04)
        self.assertLessEqual(max(voi.values()), evpi(run) + 1e-6)

    def test_uninformative_variable_is_worth_about_nothing(self):
        m = self.terminal_model(0.5, 2.0, extra={"noise": Uniform(0, 1)})
        run = simulate(m, 20000, 4)
        self.assertLess(dict(value_of_information(run))["noise"], 0.03)

    def test_failure_frequency_and_expected_value(self):
        # one fixed world: the option's NPV is OK with probability 1 - p and FAILED with probability p
        risky = Alternative("a", one_time={"o": c("1000")}, run_monthly={"r": c("20")}, live_month=c("6"), failure_probability=0.3,
                            failure_month=4, fallback="b")
        safe = Alternative("b", live_month=c("2"), run_monthly={"r": c("30")})
        m = two_way(risky, safe, {"u": Fixed(1.0)}, benefit_per_unit=c("5"), volume_start=c("20"))
        ok = evaluate(m, "a", {"u": 1.0}).npv
        bad = evaluate(m, "a", {"u": 1.0}, failed=True).npv
        run = simulate(m, 6000, 1)
        k = sum(run.failed["a"])
        lo, hi = wilson(k, 6000, 0.999)
        self.assertTrue(lo <= 0.3 <= hi)
        mean = sum(run.npv["a"]) / 6000
        sd = abs(ok - bad) * math.sqrt(0.3 * 0.7 / 6000)
        self.assertLess(abs(mean - (0.7 * ok + 0.3 * bad)), 4 * sd)
        self.assertNotEqual(ok, bad)

    def test_same_arguments_same_run_and_key_order_does_not_matter(self):
        m = random_model(random.Random(31))
        a, b = simulate(m, 300, 5), simulate(m, 300, 5)
        self.assertEqual(a.npv, b.npv)
        self.assertNotEqual(a.npv, simulate(m, 300, 6).npv)
        shuffled = Model(**{**m.__dict__, "variables": dict(reversed(list(m.variables.items())))})
        self.assertEqual(simulate(shuffled, 300, 5).npv, a.npv)

    def test_identical_across_hash_seeds(self):
        code = ("import sys;sys.path.insert(0,sys.argv[1]);import build_buy as b;"
                "s=b.load(sys.argv[1]+'/examples/ai_assistant.json');a=b.analyze(s,400,3);print(a['advantage']['mean'],a['fingerprint'])")
        outs = set()
        for seed in ("0", "1", "12345"):
            env = {**os.environ, "PYTHONHASHSEED": seed}
            outs.add(subprocess.run([sys.executable, "-c", code, str(ROOT)], env=env, capture_output=True, text=True, check=True).stdout)
        self.assertEqual(len(outs), 1)

    def test_latin_hypercube_beats_simple_random(self):
        a = Alternative("a", terminal=c("x + y"))
        m = two_way(a, Alternative("b"), {"x": Uniform(0, 10), "y": Uniform(0, 10)}, benefit_per_unit=c("0"))

        def spread(method):
            means = [sum(simulate(m, 100, s, method).npv["a"]) / 100 for s in range(100)]
            mu = sum(means) / len(means)
            return sum((v - mu) ** 2 for v in means) / len(means)
        self.assertLess(spread("lhs"), spread("random") / 5)

    def test_validation(self):
        with self.assertRaises(ValueError):
            simulate(self.terminal_model(), 1)


class StatsTests(unittest.TestCase):
    def test_lhs_one_point_per_stratum(self):
        n = 400
        for col in uniforms(n, 3, random.Random(1), "lhs"):
            self.assertEqual(sorted(int(u * n) for u in col), list(range(n)))

    def test_quantile_and_wilson(self):
        self.assertAlmostEqual(quantile([1.0, 2.0, 3.0, 4.0], 0.25), 1.75)
        lo, hi = wilson(50, 100)
        self.assertAlmostEqual(lo, 0.4038, places=3)
        self.assertEqual(wilson(0, 10)[0], 0.0)
        self.assertEqual(wilson(10, 10)[1], 1.0)

    def test_wilson_coverage(self):
        rng = random.Random(4)
        for p in (0.1, 0.5, 0.9):
            hits = 0
            for _ in range(2000):
                k = sum(rng.random() < p for _ in range(60))
                lo, hi = wilson(k, 60)
                hits += lo <= p <= hi
            self.assertGreaterEqual(hits / 2000, 0.93)

    def test_evppi_unit(self):
        self.assertEqual(evppi([1.0, 2.0, 3.0], {"a": [5.0, 6.0, 7.0], "b": [1.0, 1.0, 1.0]}, 3), 0.0)


if __name__ == "__main__":
    unittest.main()
