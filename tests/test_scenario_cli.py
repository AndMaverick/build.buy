import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

import build_buy
from build_buy import (
    Model,
    Option,
    analyze,
    compare,
    crossover_year,
    evaluate,
    load,
    net_flows,
    parse,
    payback,
    render_markdown,
    render_text,
)
from build_buy.cli import main

ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / "examples"


def run_cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = main(list(argv))
        except SystemExit as exc:
            code = int(exc.code or 0)
    return code, out.getvalue(), err.getvalue()


def base():
    return json.loads((EX / "ai_assistant.json").read_text())


class ScenarioTests(unittest.TestCase):
    def test_example_loads(self):
        sc = load(EX / "ai_assistant.json")
        self.assertEqual(sorted(sc.model.options), ["api", "build", "platform"])
        self.assertEqual(sc.model.options["build"].fallback, "api")
        self.assertEqual(len(sc.model.options["api"].tiers), 3)

    def test_unknown_keys_rejected_everywhere(self):
        mutations = [
            lambda d: d.update(horizon="48"),
            lambda d: d["volume"].update(growth=1),
            lambda d: d["options"]["api"].update(unit_price=1),
            lambda d: d["options"]["build"]["failure"].update(odds=0.5),
            lambda d: d["options"]["api"]["tiers"][0].update(extra=1),
            lambda d: d["options"]["build"]["refresh"].update(extra=1),
            lambda d: d["gates"].update(min_vibes=1),
            lambda d: d["variables"].update(bad={"dist": "uniform", "low": 1, "high": 2, "typo": 3}),
            lambda d: d.update(benefit={"per_item": 1}),
        ]
        for mutate in mutations:
            d = base()
            mutate(d)
            with self.assertRaises(ValueError):
                parse(d)

    def test_type_and_value_errors(self):
        for key, value in (("horizon_months", "48"), ("horizon_months", 1), ("discount_rate", "ten percent"), ("discount_rate", -2)):
            d = base()
            d[key] = value
            with self.assertRaises(ValueError, msg=f"{key}={value!r}"):
                parse(d)

    def test_formula_problems_name_the_formula(self):
        d = base()
        d["options"]["api"]["run_monthly"]["x"] = "__import__('os')"
        with self.assertRaises(ValueError) as cm:
            parse(d)
        self.assertIn("options.api.run_monthly.x", str(cm.exception))
        d = base()
        d["options"]["api"]["one_time"]["x"] = "ghost * 2"
        with self.assertRaises(ValueError) as cm:
            parse(d)
        self.assertIn("ghost", str(cm.exception))

    def test_structure_errors(self):
        cases = [
            lambda d: d["options"].pop("platform") and d["options"].pop("build"),
            lambda d: d["options"]["build"]["failure"].update(fallback="nope"),
            lambda d: d["options"]["api"].update(unit_cost=1.0),                                    # unit_cost and tiers together
            lambda d: d["options"]["api"]["tiers"].__setitem__(-1, {"up_to": 99, "price": 1}),     # last tier must be open
            lambda d: d["gates"].update(leader_must_be="nope"),
            lambda d: d["options"]["build"]["refresh"].pop("cost"),
            lambda d: d["options"]["build"]["failure"].update(probability=2),
            lambda d: d["options"]["build"]["failure"].update(month=999),
        ]
        for mutate in cases:
            d = base()
            mutate(d)
            with self.assertRaises(ValueError):
                parse(d)

    def test_missing_pieces(self):
        for key in ("name", "variables", "volume", "options"):
            d = base()
            del d[key]
            with self.assertRaises(ValueError):
                parse(d)
        with self.assertRaises(ValueError):
            parse([])

    def test_fingerprint(self):
        d = base()
        a = parse(d).fingerprint(1000, 7, "lhs", "0.2.0")
        self.assertEqual(a, parse(json.loads(json.dumps(d, sort_keys=True))).fingerprint(1000, 7, "lhs", "0.2.0"))
        for args in ((1001, 7, "lhs", "0.2.0"), (1000, 8, "lhs", "0.2.0"), (1000, 7, "random", "0.2.0"), (1000, 7, "lhs", "0.2.1")):
            self.assertNotEqual(a, parse(d).fingerprint(*args))
        d2 = copy.deepcopy(d)
        d2["variables"]["api_price"]["high"] = 5.1
        self.assertNotEqual(a, parse(d2).fingerprint(1000, 7, "lhs", "0.2.0"))
        self.assertEqual(len(a), 64)

    def test_load_reports_the_path(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "bad.json"
            p.write_text("{not json")
            with self.assertRaises(ValueError) as cm:
                load(p)
            self.assertIn("bad.json", str(cm.exception))


class AnalyzeTests(unittest.TestCase):
    def setUp(self):
        self.a = analyze(load(EX / "ai_assistant.json"), 1200, 5)

    def test_deterministic_and_serialisable(self):
        self.assertEqual(self.a, analyze(load(EX / "ai_assistant.json"), 1200, 5))
        json.dumps(self.a)

    def test_internal_consistency(self):
        a = self.a
        self.assertEqual(a["leader"], a["options"][0]["name"])
        self.assertAlmostEqual(sum(o["prob_best"] for o in a["options"]), 1.0)
        means = [o["expected_npv"] for o in a["options"]]
        self.assertEqual(means, sorted(means, reverse=True))
        self.assertAlmostEqual(a["evpi"], min(o["expected_regret"] for o in a["options"]), delta=1e-6 * abs(means[0]))
        adv = a["advantage"]
        self.assertAlmostEqual(adv["mean"], a["options"][0]["expected_npv"] - a["options"][1]["expected_npv"], delta=1e-6 * abs(means[0]))
        self.assertLessEqual(adv["prob_ci_low"], adv["prob_leader_beats_runner_up"])
        self.assertEqual(a["passed"], all(g["passed"] for g in a["gates"]))
        risky = [o for o in a["options"] if "failure_probability" in o]
        self.assertEqual([o["name"] for o in risky], ["build"])
        self.assertLess(risky[0]["npv_if_it_fails"], risky[0]["typical_npv"])

    def test_renderers(self):
        for out in (render_text(self.a), render_markdown(self.a)):
            self.assertIn(self.a["fingerprint"], out)
            self.assertNotIn("\u2014", out)
        self.assertIn("Recommendation", render_text(self.a))
        self.assertTrue(render_markdown(self.a).startswith("# "))

    def test_gate_logic(self):
        d = base()
        d["gates"] = {"min_leader_advantage": 1e12, "max_p90_regret": 1e12, "leader_must_be": "api", "min_prob_leader_wins": 0.0}
        r = analyze(parse(d), 400, 1)
        status = {g["name"]: g["passed"] for g in r["gates"]}
        self.assertFalse(status["min_leader_advantage"])
        self.assertTrue(status["max_p90_regret"])
        self.assertTrue(status["min_prob_leader_wins"])
        self.assertEqual(status["leader_must_be"], r["leader"] == "api")
        self.assertFalse(r["passed"])


class CliTests(unittest.TestCase):
    def test_passing_example_exits_zero(self):
        code, out, _ = run_cli("compare", str(EX / "ai_assistant.json"), "--trials", "800")
        self.assertEqual(code, 0)
        self.assertIn("PASS", out)

    def test_failing_example_exits_one(self):
        code, out, _ = run_cli("compare", str(EX / "close_call.json"), "--trials", "800")
        self.assertEqual(code, 1)
        self.assertIn("FAIL", out)

    def test_formats(self):
        code, out, _ = run_cli("compare", str(EX / "ai_assistant.json"), "--trials", "500", "--format", "json")
        self.assertEqual(json.loads(out)["trials"], 500)
        self.assertTrue(run_cli("compare", str(EX / "ai_assistant.json"), "--trials", "500", "--format", "markdown")[1].startswith("# "))

    def test_bad_input_exits_two(self):
        self.assertEqual(run_cli("compare", "/nonexistent.json")[0], 2)
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "x.json"
            p.write_text('{"name": "x"}')
            code, _, err = run_cli("compare", str(p))
            self.assertEqual(code, 2)
            self.assertIn("error:", err)
        self.assertEqual(run_cli("compare", str(EX / "ai_assistant.json"), "--trials", "1")[0], 2)

    def test_typical_command_and_overrides(self):
        code, out, _ = run_cli("typical", str(EX / "ai_assistant.json"))
        self.assertEqual(code, 0)
        self.assertIn("TCO", out)
        code, out, _ = run_cli("typical", str(EX / "ai_assistant.json"), "--set", "monthly_requests_k=60000", "--format", "json")
        self.assertEqual(json.loads(out)["values"]["monthly_requests_k"], 60000.0)
        self.assertEqual(run_cli("typical", str(EX / "ai_assistant.json"), "--set", "nope=1")[0], 2)
        self.assertEqual(run_cli("typical", str(EX / "ai_assistant.json"), "--set", "api_price=cheap")[0], 2)

    def test_higher_volume_favours_building(self):
        # at the typical case with very large volume the self-hosted build must beat the per-request API
        _, out, _ = run_cli("typical", str(EX / "ai_assistant.json"), "--set", "monthly_requests_k=150000", "--format", "json")
        ranked = [o["name"] for o in json.loads(out)["options"]]
        self.assertLess(ranked.index("build"), ranked.index("api"))

    def test_breakeven_command(self):
        code, out, _ = run_cli("breakeven", str(EX / "ai_assistant.json"), "api_price", "api", "platform")
        self.assertEqual(code, 0)
        self.assertIn("api beats platform", out)
        self.assertEqual(run_cli("breakeven", str(EX / "ai_assistant.json"), "api_price", "api", "ghost")[0], 2)

    def test_fingerprint_command(self):
        code, out, _ = run_cli("fingerprint", str(EX / "ai_assistant.json"), "--trials", "900", "--seed", "4")
        self.assertEqual(out.strip(), load(EX / "ai_assistant.json").fingerprint(900, 4, "lhs", build_buy.__version__))

    def test_version(self):
        code, out, _ = run_cli("--version")
        self.assertEqual(code, 0)
        self.assertIn(build_buy.__version__, out)


class LegacyTests(unittest.TestCase):
    BUILD = Option("build", upfront=400, annual=100, annual_benefit=300, delay_years=1)
    BUY = Option("buy", upfront=0, annual=150, annual_growth=0.1, annual_benefit=250)

    def test_yearly_results_unchanged(self):
        ranked = compare([self.BUILD, self.BUY], years=5, rate=0.10)
        self.assertEqual([r["name"] for r in ranked], ["buy", "build"])
        self.assertEqual((ranked[0]["npv"], ranked[0]["total_cost"], ranked[0]["payback_year"]), (292.47, 915.77, 0))
        self.assertEqual((ranked[1]["npv"], ranked[1]["total_cost"], ranked[1]["payback_year"]), (133.97, 900.0, 3))

    def test_other_legacy_functions(self):
        self.assertEqual(net_flows(self.BUILD, 3), [-500, 200, 200])
        self.assertEqual(payback([-500, 200, 200, 200]), 3)
        self.assertEqual(crossover_year(Option("b", 500, 50), Option("y", 0, 200)), 3)
        self.assertEqual(evaluate(self.BUILD, 5, 0.1)["name"], "build")

    def test_compare_dispatches_on_what_it_is_given(self):
        sc = load(EX / "ai_assistant.json")
        self.assertIsInstance(sc.model, Model)
        self.assertEqual(len(compare(sc.model)), 3)
        self.assertTrue(hasattr(compare(sc.model)[0], "npv"))


if __name__ == "__main__":
    unittest.main()
