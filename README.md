# build.buy

Build versus buy is usually argued on instinct. build.buy puts every option on the same footing, month by month: a one time cost, a build team, run cost that can grow, usage pricing with volume tiers and falling prices, benefits that start late and ramp, the chance a build fails and you buy anyway, and what it is worth or costs to leave. It discounts to net present value, finds the assumption that would flip the answer, and reports how likely each option is to be the right call and what being wrong would cost. Standard library only, Python 3.10 to 3.13.

If you are weighing this decision for real, the commercial question is rarely the arithmetic. It is which assumptions the answer depends on. [roi.lens](https://github.com/AndMaverick/roi.lens) takes the same question to a portfolio and shows how wide the answer really is.

## What it does

A scenario is a JSON file. Variables are the uncertain numbers, options are formulas over them, and gates are the rules a decision must pass.

```
python -m build_buy compare examples/ai_assistant.json
```

```
Customer assistant: managed API, vendor platform or self-hosted build  (build.buy 0.2.0, 3000 trials, seed 7, lhs, 48 months)

Recommendation: platform
  Expected NPV advantage over build: 927,095  (standard error 32,329)
  Chance platform beats build: 76.0%  (95% interval 74.4% to 77.5%)

option          expected NPV         p10         p90   P(best)    regret  P(loss)
platform          14,704,312   3,132,018  31,040,262     49.9%   331,996     0.1%
build             13,777,217   2,405,199  29,926,281     19.7% 1,259,092     0.7%
api               13,664,476   2,858,497  29,454,928     30.4% 1,371,833     0.1%

Typical case (every input at its median)
  platform    TCO 3,653,442, cost per unit 1.5251, NPV 11,447,451, payback month 5
  build       TCO 2,577,006, cost per unit 1.1397, NPV 11,346,918, payback month 14
  api         TCO 4,748,569, cost per unit 1.9661, NPV 10,678,709, payback month 4

If it fails (abandoned, then the fallback starts)
  build       25.0% chance; NPV in that case 9,283,611

Cumulative cost curves of platform and build cross at month 0.8.

What would change the answer (break-even, others at median)
  volume_growth             platform beats build when volume_growth is below 0.594891  (chance it lands where platform wins: 58.4%)
  monthly_requests_k        never swaps
  build_months              platform beats build when build_months is above 7  (chance it lands where platform wins: 71.4%)

Value of information: knowing everything first is worth at most 331,996.
  Learning one input exactly is worth: api_price 107,620, api_price_drift 47,283, monthly_requests_k 9,701

Decision gates
  PASS  max_p90_regret: 1,017,311 against 1,500,000
  PASS  min_prob_leader_wins: 76.0% against 60.0%

Fingerprint 8fd348d1741d65a754c6e18b3ae2d3f5d868d81358f7baae5d42b40cd60856f9
Same fingerprint on any machine gives the same numbers.
```

The CLI exits 0 when every gate passes and 1 when one fails, so a decision rule becomes a CI step or a review checklist. `examples/close_call.json` is the same decision with a gate that says the numbers must support building; they do not, so it exits 1. The ranges in both files are illustrative, not drawn from any client or real deployment. Replace them with yours.

```json
{
  "variables": {
    "monthly_requests_k": {"dist": "ci90", "low": 6000, "high": 60000, "kind": "lognormal"},
    "api_price": {"dist": "ci90", "low": 2.0, "high": 5.0, "kind": "lognormal"},
    "build_months": {"dist": "triangular", "low": 5, "mode": 7, "high": 12}
  },
  "volume": {"start": "monthly_requests_k", "growth_per_year": "volume_growth", "cap": 200000},
  "benefit": {"per_unit": "value_per_k"},
  "options": {
    "api": {
      "live_month": 2,
      "tiers": [{"up_to": 20000, "price": "api_price"}, {"up_to": 80000, "price": "api_price * 0.8"}, {"up_to": null, "price": "api_price * 0.6"}],
      "price_change_per_year": "api_price_drift"
    },
    "build": {
      "live_month": "build_months + 1", "one_time": {"platform_and_data": "build_capex"}, "one_time_months": "build_months",
      "build_monthly": {"team": "team_month"}, "run_monthly": {"ops_and_oncall": "ops_month"}, "unit_cost": "gpu_per_k",
      "refresh": {"every_months": 24, "cost": "refresh_cost"}, "terminal": 120000,
      "failure": {"probability": 0.25, "month": 6, "fallback": "api"}
    }
  },
  "gates": {"min_prob_leader_wins": 0.6, "max_p90_regret": 1500000}
}
```

An abridged excerpt; the full file is `examples/ai_assistant.json`. Other commands: `typical` (every input at its median, or at values you set with `--set name=value`), `breakeven` (the value of one variable at which two options tie) and `fingerprint`.

## The pieces

- **Monthly timeline.** Month 0 is signing or kickoff. One time costs land there or spread over a build period; a build team costs money until the system goes live; run costs, usage, benefit and ramp start then.
- **Usage pricing.** A flat unit cost or graduated tiers (each band prices only the units inside it, so cost is continuous and never jumps at a threshold), a minimum commit, and prices that fall or rise by a yearly rate.
- **Growth.** Volume grows at a yearly rate up to a cap; run costs can escalate (licence uplifts, being locked in).
- **Failure and fallback.** An option can fail with a probability after a given month. Money spent stays spent, then the fallback option starts from that month on its own timeline. This is what makes "build, and buy if it fails" a number rather than a hope.
- **Exit and residual value.** One number at the horizon, positive (what you keep) or negative (what it costs to leave).
- **Uncertainty.** Triangular, uniform, normal (truncated if you like), lognormal and `ci90` ("I am 90 percent sure it is between A and B") inputs, sampled by Latin hypercube. Formulas are parsed with `ast` and restricted to arithmetic, comparisons and a short function list; they never reach `eval`.
- **What the answer depends on.** The value at which any variable makes the answer break even (the volume above which build wins, the price below which the API wins) with the probability that the variable lands on each side, a deterministic tornado, and the month the cumulative cost curves cross.
- **Regret and the value of information.** Per option: how often it is the best choice, its expected regret and 90th percentile regret (what is left on the table compared with the best choice in the same trial). EVPI is the most any study could be worth; EVPPI says what learning one input exactly is worth, so "should we get a quote or run a pilot first" has a number.
- **Reproducible.** Every report ends with a fingerprint: the SHA-256 of the scenario, trial count, seed, sampling method and library version. The same fingerprint gives the same numbers on any machine.

## Verification

`python examples/evaluate.py` reproduces these figures in about 20 seconds. Each compares the library with an answer computed a different way.

| Check | Result |
|---|---|
| Formula evaluator vs Python's own `eval` | 4,989 of 4,989 random formulas identical |
| Graduated usage pricing vs an independent formula | 3,000 of 3,000 agree to 1e-8 |
| NPV vs discounting the monthly net flow | 2,493 random options, largest relative gap 3.2e-14 |
| Crossover of the cost curves against closed form `upfront / (b - a)` | month 3.333333, 8.333333 and 6.000000, all exact |
| Volume at which the options break even, against the closed form | 50.000000 against 50.000000, and P(build wins) 0.600 as the uniform distribution says |
| EVPI of one normal uncertainty | 0.5727 against closed form 0.5727 |
| EVPPI of each input | 0.5796 against 0.5727, and 0.9696 against 0.9634 |
| EVPI equals the leader's expected regret | 20 of 20 random decisions |
| Expected NPV with a 30% failure risk | -460.8 against -461.0, the exact mixture of the two outcomes |
| Full analysis (15 variables, 3 options, 3,000 trials) | about 3 seconds, identical on every repeat |

The test suite has 95 tests. Besides the checks above it includes an independent ledger kept month by month written from the definitions (it agrees with the engine on 500 random models, including the failure paths), the level and growing annuity closed forms, scale invariance (multiply every money amount by k and NPV scales by exactly k), monotonicity (a dearer input never raises NPV, a later start never raises NPV), results that are identical across `PYTHONHASHSEED` values and across the order keys are written in, and every unknown key in a scenario file is an error.

## Honest limits

- Variables are sampled independently. If volume and price move together in your world, that is not captured and the extremes will look slightly less extreme than they are.
- Benefit is `quality x units x value per unit`, the same value per unit for every option. If options deliver genuinely different capabilities, model that through `quality` or a separate benefit variable.
- A failed option is abandoned at one month, and its fallback starts fresh at that month, with no credit for what the failed attempt learned. Partial salvage is not modelled.
- Prices change smoothly by a yearly rate. Step changes at a renewal date, or cliff pricing where crossing a threshold reprices every unit, are not modelled (tiers are graduated).
- Volume is drawn once per trial and then follows its growth path. There is no noise from one month to the next or seasonality.
- EVPPI is estimated by binning and is biased slightly upward with few trials; EVPI is an upper bound on any study.
- The horizon is yours to set and there is one terminal number. A build that pays off in year six looks worse on a four year horizon, by design.
- No tax, and inflation is whatever the discount rate and price drift say it is.
- The quality of the answer is the quality of the ranges. The ranges in the examples are illustrative.

## Compatibility

The 0.1 interface keeps its behaviour (`Option`, `compare`, `evaluate`, `net_flows`, `npv`, `payback`, `crossover_year`), and `compare` and `evaluate` accept either a list of `Option` (yearly, as before) or a `Model` (monthly, the full engine):

```python
from build_buy import Option, compare

build = Option("build", upfront=400, annual=100, annual_benefit=300, delay_years=1)
buy = Option("buy", upfront=0, annual=150, annual_growth=0.1, annual_benefit=250)

compare([build, buy], years=5, rate=0.10)
# [{'name': 'buy',   'npv': 292.47, 'total_cost': 915.77, 'payback_year': 0},
#  {'name': 'build', 'npv': 133.97, 'total_cost': 900.0,  'payback_year': 3}]
```

In this example buy wins on NPV because its benefits start immediately and build's first year earns nothing, even though build costs slightly less in total (900.0 against 915.77). An earlier version of this README printed 800.0 for build's total cost; that was a typo, and the code always returned 900.0.

## Tests

```
python -m unittest discover -s tests -t . -v
```

MIT licensed. Built by [AndMaverick LLC](https://www.andmaverick.com).

---

**Related repositories:** [roi.lens](https://github.com/AndMaverick/roi.lens) puts ranges and risks around any business case. [token.econ](https://github.com/AndMaverick/token.econ) models what usage pricing does to margin as volume grows.

**If this is your problem.** If your AI spend is outrunning your results, that is an architecture problem, and it is solvable. I am happy to look at it with you. [Begin a conversation](https://calendly.com/andmaverick?utm_source=github&utm_medium=repo&utm_campaign=build.buy).
