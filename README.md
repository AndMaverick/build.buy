# build.buy

Build versus buy is usually argued on instinct. build.buy puts both options on the same footing: upfront cost, recurring cost that can grow, benefits that can start late, discounted to net present value, with payback year and the point where the cost curves cross. Standard library only.

If you are weighing this decision for real, the commercial question is rarely the arithmetic. It is which assumptions the answer depends on. I run that conversation with leadership teams through an [architecture review](https://calendly.com/andmaverick/30min?utm_source=github&utm_medium=repo&utm_campaign=build.buy).

## Use

```python
from build_buy import Option, compare

build = Option("build", upfront=400, annual=100, annual_benefit=300, delay_years=1)
buy = Option("buy", upfront=0, annual=150, annual_growth=0.1, annual_benefit=250)

compare([build, buy], years=5, rate=0.10)
# [{'name': 'buy',   'npv': 292.47, 'total_cost': 915.77, 'payback_year': 0},
#  {'name': 'build', 'npv': 133.97, 'total_cost': 800.0,  'payback_year': 3}]
```

In this example buy wins on NPV because benefits start immediately, even though build costs less in total. Change the delay, the growth rate, or the discount rate and the answer can flip, which is the point of modeling it.

## Design

- **Delay is first class.** A build that delivers value a year late is a different investment from one that does not.
- **Run costs can grow.** Usage-priced services rarely stay flat.
- **Crossover year.** crossover_year finds when one option's cumulative cost overtakes the other's, within a horizon you set.
- **Illustrative numbers only.** The figures above are examples, not benchmarks.

## Tests

```
python -m unittest discover -s tests -t . -v
```

MIT licensed. Built by [AndMaverick LLC](https://www.andmaverick.com).

---

**Related reading:** [The Meter Is Running. Nobody Built the Architecture to Stop It.](https://blog.andmaverick.com/the-meter-is-running-nobody-built-the-architecture-to-stop-it?utm_source=github&utm_medium=repo&utm_campaign=build.buy)

**If this is your problem.** If your AI spend is outrunning your results, that is an architecture problem, and it is solvable. I am happy to look at it with you. [Begin a conversation](https://calendly.com/andmaverick?utm_source=github&utm_medium=repo&utm_campaign=build.buy).
