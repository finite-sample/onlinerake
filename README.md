# onlinerake

[![PyPI version](https://img.shields.io/pypi/v/onlinerake.svg)](https://pypi.org/project/onlinerake/)
[![PyPI Downloads](https://static.pepy.tech/badge/onlinerake)](https://pepy.tech/projects/onlinerake)
[![Documentation](https://img.shields.io/badge/docs-github.io-blue)](https://finite-sample.github.io/onlinerake/)
[![CI](https://github.com/finite-sample/onlinerake/actions/workflows/ci.yml/badge.svg)](https://github.com/finite-sample/onlinerake/actions/workflows/ci.yml)

**Real-time survey weighting for streaming data.**

## The Problem

You're collecting survey responses or observational data one record at a time. Your sample doesn't match population demographics—too many young respondents, too few from certain regions. Traditional weighting methods (raking/IPF) require reprocessing the entire dataset whenever a new response arrives.

**onlinerake** updates weights as observations arrive, reducing squared error
between weighted margins and population targets. Each update revisits all retained
observations: memory is O(n × d), work per arrival is O(n × d × n_sgd_steps),
and total work grows quadratically with stream length for fixed feature count.
Check the remaining margin errors; a finite number of updates need not balance
all targets.

## When to Use This

- **Online surveys** where responses arrive continuously
- **A/B tests** that need demographic balance during collection
- **Passive data collection** (app usage, sensor data) requiring real-time calibration
- **Moderate streams** where updating retained weights after every arrival is affordable

## Quick Start

```bash
pip install onlinerake
```

```python
from onlinerake import OnlineRakingSGD, Targets

# Define population targets (proportion with indicator = 1)
targets = Targets(
    female=0.51,  # 51% female in population
    college=0.32,  # 32% college educated
    age_65_plus=0.17,  # 17% age 65+
)

# Create raker
raker = OnlineRakingSGD(targets, learning_rate=5.0)

survey_stream = [
    {"female": 1, "college": 0, "age_65_plus": 0},
    {"female": 0, "college": 1, "age_65_plus": 1},
    {"female": 1, "college": 1, "age_65_plus": 0},
    {"female": 0, "college": 0, "age_65_plus": 0},
]

# Replace this list with your incoming responses.
for response in survey_stream:
    raker.partial_fit(response)

    # Check current state anytime
    print(f"Weighted margins: {raker.margins}")
    print(f"Effective sample size: {raker.effective_sample_size:.0f}")

# Get final weights
weights = raker.weights
```

## Which Algorithm?

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

## Performance

Run `python examples/streaming_mwu.py` for a reproducible evaluation with 10
seeds and 500 arrivals per stream. With the default MWU settings, mean squared
margin error over arrivals 100–500 falls by 83% in the stationary binary case,
75% after a sampling shift, and 83% with binary and scaled continuous features.
Mean final effective sample sizes are 292, 383, and 155 out of 500, respectively.
These simulations measure calibration on the accumulated sample; they do not
establish improved outcome estimates or a general performance guarantee.

## Features

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

Unsupported numerical regret/loss-bound helpers have been removed. The learning
rate helper is now `suggest_mwu_learning_rate`, explicitly an empirical tuning
heuristic.

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

## Upgrading from 2.0.0

See the [3.0.0 upgrade notes](https://github.com/finite-sample/onlinerake/blob/main/CHANGELOG.md)
for removed bound helpers, the renamed learning-rate heuristic, changed decay
floors, and corrected convergence reporting.

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

Interactive notebooks in `docs/notebooks/`:
- `01_getting_started.ipynb` - Visual introduction
- `02_performance_comparison.ipynb` - Algorithm benchmarking
- `03_advanced_diagnostics.ipynb` - Convergence and diagnostics

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

See [the software audit](https://finite-sample.github.io/onlinerake/software-audit.html) for reproduced defects, regression
checks, and the relationship to the retired `adaptive-eb` prototype. The historical
manuscript in `ms/` describes an update that changes only the newest row and reports
O(Kp) cost; it is not documentation of this implementation and its performance and
convergence claims should not be used for this package.
