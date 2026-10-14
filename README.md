# onlinerake

[![PyPI version](https://img.shields.io/pypi/v/onlinerake.svg)](https://pypi.org/project/onlinerake/)
[![PyPI Downloads](https://static.pepy.tech/badge/onlinerake)](https://pepy.tech/projects/onlinerake)
[![Documentation](https://img.shields.io/badge/docs-github.io-blue)](https://finite-sample.github.io/onlinerake/)
[![CI](https://github.com/finite-sample/onlinerake/actions/workflows/ci.yml/badge.svg)](https://github.com/finite-sample/onlinerake/actions/workflows/ci.yml)

**Rake a stream without storing it.**

Raking adjusts weights so that weighted shares of groups (women, college
graduates, people over 65) match known population shares. The standard
algorithm, iterative proportional fitting (IPF), needs every row seen so far.
`StreamingRaker` keeps a fixed set of running totals instead: each arriving row
gets its weight in closed form, earlier weights never change, and memory does
not grow with the stream.

## Quick Start

```bash
pip install onlinerake
```

```python
from onlinerake import StreamingRaker, Targets

raker = StreamingRaker(Targets(female=0.5), outcomes=["spend"])
for female, spend in [(1, 10.0), (1, 12.0), (1, 11.0), (0, 20.0)] * 50:
    raker.partial_fit({"female": female}, outcomes={"spend": spend})

print(round(raker.margins["female"], 2))  # 0.52; the raw sample is 75% female
print(round(raker.estimate("spend"), 1))  # 15.3; the unweighted mean is 13.25
```

`partial_fit` returns the row's weight, relative to the mean design weight so
far. Pass `design_weight=` for rows that arrive with base weights, in any units,
and `keep_weights=True` to store the assigned weights (this makes memory grow
with the stream). Results do not depend on the units of design weights or of
continuous features: expansion weights of 5,000 give the same answer as weights
near 1, and income in dollars the same as in thousands.

## Which Tool

| Your situation | Use |
|---|---|
| Rows can be stored and the stream is survey-sized | Refit `BatchIPF` periodically. It is exact at each refit and gives the highest effective sample size. |
| Rows cannot or should not be stored, memory is fixed, or the stream is long | `StreamingRaker`: `p + 5` numbers of state for `p` margins, constant cost per row. |
| You want the batch version of the streaming objective | `fit_penalized_raking` |

Measured cost per arriving row, eight margins (`eval_outputs/streaming_scaling.csv`):

| Rows | `StreamingRaker` | IPF refit every 1,000 rows (amortized) |
|---:|---:|---:|
| 10,000 | 10 µs | 21 µs |
| 100,000 | 13 µs | 150 µs |
| 1,000,000 | 10 µs | 1,314 µs |

The manuscript in [`ms/onlinerake.pdf`](ms/onlinerake.pdf) derives the update and
evaluates it in six simulated scenarios with 50 replications each. With moderate
targets, the error of a weighted outcome mean at the end of the stream is at most
27% above full-data IPF. During the stream, it tracks the margins more closely
than IPF refit every 1,000 rows in five of six scenarios. It does worse with
extreme targets and with 50 margins, and its estimates depend on arrival order.

### How it works

Each row's weight minimizes its raking distance from its design weight plus
`calibration_strength / 2` times the squared running gap between weighted and
target totals. That one-variable problem has a closed-form solution through the
Wright omega function. Larger `calibration_strength` (default 0.3) matches the
targets more closely at the cost of more variable weights.

`fit_penalized_raking(features, targets, calibration_strength)` minimizes the
same objective over all stored rows at once (entropy balancing with a ridge
penalty, which approaches IPF as the strength grows).

## Online Classes That Store Every Row

`OnlineRakingSGD` and `OnlineRakingMWU` store every row and re-optimize all
weights at each arrival, so their cost per row grows with the stream. When rows
can be stored, periodic `BatchIPF` is cheaper and exact at each refit; when they
cannot, use `StreamingRaker`.

```python
from onlinerake import OnlineRakingSGD, Targets

targets = Targets(female=0.51, college=0.32, age_65_plus=0.17)
raker = OnlineRakingSGD(targets, learning_rate=5.0)

survey_stream = [
    {"female": 1, "college": 0, "age_65_plus": 0},
    {"female": 0, "college": 1, "age_65_plus": 1},
    {"female": 1, "college": 1, "age_65_plus": 0},
    {"female": 0, "college": 0, "age_65_plus": 0},
]
for response in survey_stream:
    raker.partial_fit(response)

print(raker.margins, raker.effective_sample_size)
weights = raker.weights
```

### Additive or multiplicative updates

| Update | Algorithm | Default learning rate |
|--------|-----------|-----------------------|
| Additive weight adjustments | `OnlineRakingSGD` | 5.0 |
| Multiplicative weight adjustments | `OnlineRakingMWU` | 1.0 |

Both methods accept records one at a time and initialize each new weight at one.
Unequal starting weights are not supported. Each method takes a fixed number of
steps toward the target means, so always inspect the remaining error. The
multiplicative method need not choose the same weights as a method that fits the
whole dataset at once. `BatchIPF` provides that comparison for binary targets.

Learning rates need evaluation on the intended data, particularly when continuous
features use different units. Better target balance can require more unequal
weights, reducing effective sample size.

### Performance

Run `python examples/streaming_mwu.py` for a reproducible evaluation with 10
seeds and 500 arrivals per stream. With the default MWU settings, mean squared
margin error over arrivals 100–500 falls by 83% in the stationary binary case,
75% after a sampling shift, and 83% with binary and scaled continuous features.
Mean final effective sample sizes are 292, 383, and 155 out of 500, respectively.
These simulations measure calibration on the accumulated sample; they do not
establish improved outcome estimates or a general performance guarantee.

### Continuous Covariates

Target means instead of proportions:

```python
continuous_targets = Targets(
    age=(42.0, "mean"),  # Target mean age = 42
    income=(55000, "mean"),  # Target mean income = $55,000
    female=0.51,  # Binary: 51% female
)
```

### Learning Rate Schedules

To inspect learning-rate summability (not a guarantee of calibration):

```python
from onlinerake import OnlineRakingSGD, Targets, PolynomialDecayLR
from onlinerake.convergence import verify_robbins_monro

schedule = PolynomialDecayLR(initial_lr=10.0, power=0.6)
scheduled_raker = OnlineRakingSGD(targets, learning_rate=schedule)

# Verify Robbins-Monro conditions (analytical for known schedules)
result = verify_robbins_monro(schedule)
print(result.condition_1_satisfied)  # True: Σ η_t = ∞
print(result.condition_2_satisfied)  # True: Σ η_t² < ∞
```

`verify_robbins_monro()` checks the two learning-rate series for known schedule
types. A positive `min_lr` floor fails square summability. These checks do not
establish convergence for a changing, clipped calibration problem.

`.converged` reports whether current squared moment loss is at most `1e-6` after
`convergence_window` observations; later data can reset it. For a different loss
threshold, call `.check_convergence(tolerance=...)`. Batch IPF instead checks the
maximum absolute margin error against its `tolerance`. A stalled or infeasible
fit does not count as converged.

`suggest_mwu_learning_rate` gives an empirical starting point for tuning the MWU
learning rate.

### Diagnostics

```python
from onlinerake import check_target_feasibility, compute_design_effect

# Screen for missing support; this does not prove joint feasibility
feasibility = check_target_feasibility(raker)
print(f"Passes support screen: {feasibility.is_feasible}")

# Measure weighting efficiency
deff = compute_design_effect(raker)
print(f"Design effect: {deff:.2f}")
```

### Batch Comparison

Compare streaming results against traditional IPF:

```python
from onlinerake import BatchIPF

batch_raker = BatchIPF(targets)
batch_raker.fit(survey_stream)

print(f"Online loss: {raker.loss:.6f}")
print(f"Batch loss: {batch_raker.loss:.6f}")
```

## API Reference

### Core Classes

**`StreamingRaker(targets, calibration_strength=0.3, min_weight=0.001, max_weight=100, trim=None, outcomes=())`** - Fixed-memory streaming raker
- `min_weight`, `max_weight` - Bounds on each row's weight relative to its design weight
- `trim` - Cap each weight at this multiple of the mean weight so far (Potter-style trimming; 3 to 6 is common)
- `.partial_fit(obs, design_weight=1.0, outcomes=None)` - Weight one row; returns its weight relative to the mean design weight
- `.margins`, `.loss`, `.effective_sample_size` - Current state
- `.estimate(name)`, `.estimates` - Weighted outcome means
- `.weights` - Assigned weights (requires `keep_weights=True`)

**`fit_penalized_raking(features, targets, calibration_strength=0.3, design_weights=None)`** - Same objective, all rows at once

**`Targets(**features)`** - Define population margins
- Binary features: `female=0.51` (proportion = 1)
- Continuous features: `age=(42.0, "mean")` (target mean)

**`OnlineRakingSGD(targets, learning_rate=5.0)`** - SGD-based streaming raker
- `.partial_fit(obs)` - Process one observation
- `.margins` - Current weighted margins (dict)
- `.loss` - Current squared-error loss
- `.weights` - Copy of the active weight array
- `.effective_sample_size` - ESS accounting for weight variation
- `.converged` - Whether loss is below tolerance

**`OnlineRakingMWU(targets, learning_rate=1.0)`** - Multiplicative weights raker
- Same API as `OnlineRakingSGD`

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 5.0 (SGD), 1.0 (MWU) | Step size for updates |
| `min_weight` | 0.001 | Minimum allowed weight |
| `max_weight` | 100.0 | Maximum allowed weight |
| `n_sgd_steps` | 3 | Gradient steps per observation |
| `convergence_window` | 20 | Minimum observations before reporting convergence |

## Upgrading from 2.x

See [CHANGELOG.md](CHANGELOG.md).

## Installation

```bash
pip install onlinerake
```

Development install:
```bash
git clone https://github.com/finite-sample/onlinerake.git
cd onlinerake
uv sync --all-groups
```

## Testing

```bash
pytest tests/ -v
```

## Examples

See `examples/` for complete worked examples:
- `streaming_mwu.py` - Online multiplicative weighting, drift, and the ESS tradeoff
- `real_survey_example.py` - Basic survey weighting
- `ab_test_calibration.py` - Balancing treatment/control groups
- `ad_targeting_calibration.py` - Real-time ad delivery calibration
- `recommendation_balancing.py` - Content recommendation fairness

Documentation examples in `docs/examples/` run on every docs build:
- `plot_streaming_raker.py` - A biased stream raked in fixed memory, against full-data IPF

## Citation

If you use this package in research, please cite:

```bibtex
@software{onlinerake,
  author = {Sood, Gaurav},
  title = {onlinerake: Streaming Survey Raking},
  url = {https://github.com/finite-sample/onlinerake},
  year = {2026}
}
```

## License

MIT

## Implementation audit

See [the software audit](https://finite-sample.github.io/onlinerake/software-audit.html)
for reproduced defects and their regression checks.
