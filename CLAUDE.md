# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`onlinerake` is a Python package for streaming survey raking (weight calibration) using two algorithms:
- **SGD raking** (`OnlineRakingSGD`) - stochastic gradient descent with additive updates
- **MWU raking** (`OnlineRakingMWU`) - multiplicative weights update with exponential updates

The package enables real-time weight adjustment for streaming survey data to match known population margins, unlike traditional batch IPF methods.

## Recent Major Updates

### Performance
- **Capacity doubling**: Eliminated O(n²) memory reallocations for weights storage
- **Array optimization**: Moved demographic conversions outside gradient loops
- **Configurable statistics**: Optional weight distribution computations (10-100x speedup for large streams)
- **Cost is quadratic in stream length, by design.** `partial_fit` rewrites all
  `n` accumulated weights (`online_raking_sgd.py:803`) and `_compute_gradient`
  is itself O(n), so per-observation work is Θ(n · n_sgd_steps). "Streaming"
  describes how data arrives, not the cost of taking it. Measured on three
  binary features: 104 µs/obs at n=2,500 rising to 645 µs/obs at n=40,000,
  fitted exponent 1.66 on total time. This module previously claimed
  "performance independent of number of observations" and "3000-6000
  observations per second"; both were false, the second silently depending on
  an unstated n. Plan for tens of thousands per stream, not millions.

### Numerical Stability Improvements
- **MWU exponent clipping**: Dtype-aware bounds prevent overflow (supports extreme learning rates like 1e6)
- **Near-zero loss convergence**: Proper handling when loss approaches machine epsilon
- **Robust weight bounds**: Enhanced clipping prevents numerical instabilities

### Convergence & Streaming Inference Fixes
- **Analytical Robbins-Monro verification**: `verify_robbins_monro()` uses type dispatch for known schedules (ConstantLR, PolynomialDecayLR, InverseTimeDecayLR) with mathematical proofs
- **Streaming inference fix**: `StreamingEstimator.partial_fit()` now correctly captures weights BEFORE update to compute meaningful retroactive impact metrics

## Architecture

Core modules in `onlinerake/`:
- `targets.py` - Defines `Targets` dataclass for population margins (age, gender, education, region)
- `online_raking_sgd.py` - SGD-based streaming raking algorithm 
- `online_raking_mwu.py` - MWU-based algorithm (inherits from SGD)

Interactive documentation in `docs/notebooks/`:
- `01_getting_started.ipynb` - Introduction with visual demonstrations
- `02_performance_comparison.ipynb` - Algorithm benchmarking and performance analysis
- `03_advanced_diagnostics.ipynb` - Convergence monitoring and diagnostic tools

Both raking classes follow scikit-learn's `partial_fit` pattern: call `.partial_fit(obs)` for each observation and inspect `.margins`, `.loss`, and `.effective_sample_size` properties.

## Development Commands

```bash
# Install the package and every dependency group
uv sync --all-groups

# Run the test suite with coverage
uv run pytest --cov --cov-report=term

# Run interactive tutorials
jupyter notebook docs/notebooks/

# Code quality checks (zero tolerance)
uv run ruff check .
uv run ruff format --check .
uv run pyright

# Build documentation (executes the notebooks; warnings are errors)
uv run sphinx-build -W -b html docs _site

# Run comprehensive head-to-head evaluation (quick mode)
uv run python scripts/eval/comprehensive_eval.py --n_seeds 5 --quick

# Run full evaluation (50 seeds, includes large-scale scenario)
uv run python scripts/eval/comprehensive_eval.py --n_seeds 50

# Generate evaluation plots and LaTeX table
uv run python scripts/eval/plot_eval_results.py
```

## Testing

- **Comprehensive test suite**: 390 test cases covering core algorithms, new features, and theoretical foundations
- **Realistic examples**: Gender bias correction, real-time polling, algorithm comparison
- **CI/CD workflows**: py-canon's shared CI, on Python 3.12 and 3.14
- **Coverage**: High test coverage for critical paths, edge cases, and extreme scenarios
- **Performance tests**: Verify linear scaling and optimization effectiveness
- **Numerical stability**: Tests for extreme learning rates (1e6), near-zero loss convergence

## Key Implementation Details

- **Performance**: Capacity doubling for O(log n) weight storage, optimized array conversions
- **Numerical safety**: Dtype-aware exponent clipping, robust convergence detection
- **Algorithms**: SGD uses squared-error loss; MWU uses KL divergence via mirror descent
- **Data structures**: Pre-allocated arrays, configurable weight statistics computation
- **API**: Scikit-learn compatible `partial_fit` pattern with comprehensive diagnostics
- **Dependencies**: Minimal - only numpy and pandas required
- **Compatibility**: Python 3.12+ with modern type hints