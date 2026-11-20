import math
import random
import unittest

from build_buy.distributions import Fixed, Lognormal, Normal, Triangular, Uniform, from_ci90, from_dict
from build_buy.synthetic import random_distribution


def midpoint(f, n=4000):
    return sum(f((i + 0.5) / n) for i in range(n)) / n


class DistributionTests(unittest.TestCase):
    def test_round_trip_and_mean_for_random_distributions(self):
        rng = random.Random(4)
        for _ in range(80):
            d = random_distribution(rng)
            for p in (0.01, 0.2, 0.5, 0.8, 0.99):
                self.assertAlmostEqual(d.cdf(d.ppf(p)), p, places=7, msg=repr(d))
            self.assertAlmostEqual(midpoint(d.ppf), d.mean(), delta=0.006 * max(1.0, abs(d.mean())), msg=repr(d))

    def test_ppf_is_monotone(self):
        rng = random.Random(5)
        for _ in range(60):
            d = random_distribution(rng)
            xs = [d.ppf(i / 200) for i in range(1, 200)]
            self.assertEqual(xs, sorted(xs))

    def test_triangular_closed_forms(self):
        t = Triangular(1, 2, 6)
        self.assertAlmostEqual(t.mean(), 3.0)
        self.assertAlmostEqual(t.cdf(2), 0.2)

    def test_half_normal(self):
        h = Normal(0, 1, lower=0)
        self.assertAlmostEqual(h.mean(), math.sqrt(2 / math.pi), places=12)
        self.assertGreaterEqual(min(h.ppf(i / 1000) for i in range(1, 1000)), 0.0)

    def test_ci90_puts_five_percent_in_each_tail(self):
        for kind, lo, hi in (("normal", -3.0, 11.0), ("lognormal", 2.0, 40.0)):
            d = from_ci90(lo, hi, kind=kind)
            self.assertAlmostEqual(d.cdf(lo), 0.05, places=9)
            self.assertAlmostEqual(d.cdf(hi), 0.95, places=9)

    def test_fixed(self):
        f = Fixed(7)
        self.assertEqual((f.ppf(0.3), f.mean(), f.cdf(6.9), f.cdf(7)), (7, 7, 0.0, 1.0))

    def test_bad_ranges_rejected(self):
        for build in (lambda: Uniform(2, 1), lambda: Triangular(1, 5, 3), lambda: Normal(0, -1), lambda: Lognormal(0, -1),
                      lambda: from_ci90(5, 5), lambda: Normal(0, 1, lower=9), lambda: from_ci90(-1, 5, kind="lognormal")):
            with self.assertRaises(ValueError):
                build()

    def test_from_dict(self):
        self.assertEqual(from_dict(5), Fixed(5.0))
        for s in ({"dist": "uniform", "low": 1, "high": 2}, {"dist": "triangular", "low": 1, "mode": 2, "high": 3},
                  {"dist": "normal", "mu": 0, "sigma": 1}, {"dist": "lognormal", "mu": 0, "sigma": 1}, {"dist": "ci90", "low": 1, "high": 3}):
            self.assertTrue(hasattr(from_dict(s), "ppf"))
        for bad in ({"dist": "uniform", "low": 1, "high": 2, "colour": 1}, {"dist": "weibull"}, {"low": 1}, True, "5", [1], {"dist": "uniform", "low": 1}):
            with self.assertRaises(ValueError):
                from_dict(bad)


if __name__ == "__main__":
    unittest.main()
