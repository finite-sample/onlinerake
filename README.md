# onlinerake

[![PyPI version](https://img.shields.io/pypi/v/onlinerake.svg)](https://pypi.org/project/onlinerake/)
[![PyPI Downloads](https://static.pepy.tech/badge/onlinerake)](https://pepy.tech/projects/onlinerake)
[![Documentation](https://img.shields.io/badge/docs-github.io-blue)](https://finite-sample.github.io/onlinerake/)
[![Tests](https://github.com/finite-sample/onlinerake/actions/workflows/tests.yml/badge.svg)](https://github.com/finite-sample/onlinerake/actions/workflows/tests.yml)

**Real-time survey weighting for streaming data.**

## The Problem

You're collecting survey responses or observational data one record at a time. Your sample doesn't match population demographics—too many young respondents, too few from certain regions. Traditional weighting methods (raking/IPF) require reprocessing the entire dataset whenever a new response arrives.

**onlinerake** updates weights incrementally as each observation streams in, keeping weighted margins aligned with population targets in real time.

## When to Use This

- **Online surveys** where responses arrive continuously
- **A/B tests** that need demographic balance during collection
- **Passive data collection** (app usage, sensor data) requiring real-time calibration
- **Any streaming scenario** where batch reweighting is too slow or impractical

## Quick Start

```bash
pip install onlinerake
```

```python
from onlinerake import OnlineRakingSGD, Targets

# Define population targets (proportion with indicator = 1)
targets = Targets(
    female=0.51,      # 51% female in population
    college=0.32,     # 32% college educated
    age_65_plus=0.17  # 17% age 65+
)

# Create raker
raker = OnlineRakingSGD(targets, learning_rate=5.0)

# Process observations as they arrive
for response in survey_stream:
    raker.partial_fit(response)

    # Check current state anytime
    print(f"Weighted margins: {raker.margins}")
    print(f"Effective sample size: {raker.effective_sample_size:.0f}")

# Get final weights
weights = raker.weights[:raker.n_obs]
```

## Which Algorithm?

| Use Case | Algorithm | Learning Rate |
|----------|-----------|---------------|
| **Most cases** | `OnlineRakingSGD` | 5.0 |
| **Smoother weights, higher ESS** | `OnlineRakingSGD` | 2.0-5.0 |
| **IPF-like multiplicative updates** | `OnlineRakingMWU` | 0.5-1.0 |
| **Starting from unequal base weights** | `OnlineRakingMWU` | 0.5-1.0 |

**Recommendation:** Start with `OnlineRakingSGD(targets, learning_rate=5.0)`. It converges faster, maintains higher effective sample size, and handles most scenarios well.

## Performance

In simulation studies across linear drift, sudden shift, and oscillating bias scenarios:

| Method | Margin Error Reduction | Effective Sample Size |
|--------|----------------------|----------------------|
| SGD | 72-80% | 225-280 (of 300) |
| MWU | 47-52% | 175-276 (of 300) |
| Unweighted | baseline | 300 |

SGD consistently outperforms MWU on margin accuracy while maintaining comparable effective sample sizes.

## Features

### Continuous Covariates (v1.3.0)

Target means instead of proportions:

```python
targets = Targets(
    age=(42.0, "mean"),      # Target mean age = 42
    income=(55000, "mean"),  # Target mean income = $55,000
    female=0.51              # Binary: 51% female
)
```

### Learning Rate Schedules

For theoretical convergence guarantees:

```python
from onlinerake import OnlineRakingSGD, Targets, PolynomialDecayLR
from onlinerake.convergence import verify_robbins_monro

schedule = PolynomialDecayLR(initial_lr=10.0, power=0.6)
raker = OnlineRakingSGD(targets, learning_rate=schedule)

# Verify Robbins-Monro conditions (analytical for known schedules)
result = verify_robbins_monro(schedule)
print(result.condition_1_satisfied)  # True: Σ η_t = ∞
print(result.condition_2_satisfied)  # True: Σ η_t² < ∞
```

The `verify_robbins_monro()` function provides analytical verification for known schedule types with mathematical proofs.

### Diagnostics

```python
from onlinerake import check_target_feasibility, compute_design_effect

# Check if targets are achievable with your data
feasibility = check_target_feasibility(raker)
print(f"Feasible: {feasibility.is_feasible}")

# Measure weighting efficiency
deff = compute_design_effect(raker)
print(f"Design effect: {deff:.2f}")
```

### Batch Comparison

Compare streaming results against traditional IPF:

```python
from onlinerake import BatchIPF

batch_raker = BatchIPF(targets)
batch_raker.fit(all_observations)

print(f"Online loss: {online_raker.loss:.6f}")
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
- `.weights` - Weight array (use `[:raker.n_obs]` to slice)
- `.effective_sample_size` - ESS accounting for weight variation
- `.converged` - Whether loss is below tolerance

**`OnlineRakingMWU(targets, learning_rate=1.0)`** - Multiplicative weights raker
- Same API as `OnlineRakingSGD`

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 5.0 (SGD), 1.0 (MWU) | Step size for updates |
| `min_weight` | 0.1 | Minimum allowed weight |
| `max_weight` | 10.0 | Maximum allowed weight |
| `n_steps` | 3 | Gradient steps per observation |
| `convergence_tol` | 1e-6 | Loss threshold for convergence |

## Installation

```bash
pip install onlinerake
```

Development install:
```bash
git clone https://github.com/finite-sample/onlinerake.git
cd onlinerake
pip install -e ".[docs]"
```

## Testing

```bash
pytest tests/ -v
```

## Examples

See `examples/` for complete worked examples:
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
  version = {1.3.0},
  year = {2026}
}
```

## License

MIT
