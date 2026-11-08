import unittest

from build_buy import Option, compare, crossover_year, net_flows, npv, payback

BUILD = Option("build", upfront=400, annual=100, annual_benefit=300, delay_years=1)
BUY = Option("buy", upfront=0, annual=150, annual_growth=0.1, annual_benefit=250)


class BuildBuyTests(unittest.TestCase):
    def test_npv_known_value(self):
        self.assertAlmostEqual(npv([-100, 110], 0.1), 0.0)

    def test_flows_include_upfront_and_delay(self):
        flows = net_flows(BUILD, 3)
        self.assertEqual(flows, [-400 - 100, 200, 200])

    def test_recurring_cost_grows(self):
        self.assertEqual([round(c, 2) for c in BUY.costs(3)], [150.0, 165.0, 181.5])

    def test_payback(self):
        self.assertEqual(payback([-500, 200, 200, 200]), 3)
        self.assertIsNone(payback([-500, 100]))

    def test_compare_orders_by_npv(self):
        ranked = compare([BUILD, BUY], years=5, rate=0.1)
        self.assertEqual(ranked[0]["npv"] >= ranked[1]["npv"], True)
        self.assertEqual({r["name"] for r in ranked}, {"build", "buy"})

    def test_crossover(self):
        cheap_run = Option("build", upfront=500, annual=50)
        dear_run = Option("buy", upfront=0, annual=200)
        self.assertEqual(crossover_year(cheap_run, dear_run), 3)
        self.assertEqual(crossover_year(dear_run, cheap_run), 3)
        self.assertIsNone(crossover_year(Option("a", 0, 100), Option("b", 0, 200)))

    def test_horizon_validation(self):
        with self.assertRaises(ValueError):
            compare([BUILD], years=0)


if __name__ == "__main__":
    unittest.main()
