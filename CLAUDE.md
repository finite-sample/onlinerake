# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`onlinerake` is a Python package for streaming survey raking (weight calibration) using two algorithms:
- **SGD raking** (`OnlineRakingSGD`) - stochastic gradient descent with additive updates
- **MWU raking** (`OnlineRakingMWU`) - multiplicative weights update with exponential updates

The package enables real-time weight adjustment for streaming survey data to match known population margins, unlike traditional batch IPF methods.

## Recent Major Updates

### Performance Optimizations (Latest)
- **Capacity doubling**: Eliminated O(n²) memory reallocations for weights storage
- **Array optimization**: Moved demographic conversions outside gradient loops  
- **Configurable statistics**: Optional weight distribution computations (10-100x speedup for large streams)
- **Linear scaling**: Performance now scales nearly linearly with data size

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
# Install package in editable mode
pip install -e .

# Install development dependencies
pip install pytest black flake8 sphinx sphinx-rtd-theme myst-parser

# Run comprehensive test suite (105 tests across 3 test files)
pytest tests/ -v --cov=onlinerake --cov-report=term

# Run interactive tutorials
jupyter notebook docs/notebooks/

# Code quality checks (zero tolerance for critical issues)
flake8 onlinerake --count --select=E9,F63,F7,F82 --show-source --statistics
black --check onlinerake

# Build documentation (includes new diagnostics and performance sections)
cd docs && make html

# Run comprehensive head-to-head evaluation (quick mode)
uv run python scripts/eval/comprehensive_eval.py --n_seeds 5 --quick

# Run full evaluation (50 seeds, includes large-scale scenario)
uv run python scripts/eval/comprehensive_eval.py --n_seeds 50

# Generate evaluation plots and LaTeX table
uv run python scripts/eval/plot_eval_results.py
```

## Testing

- **Comprehensive test suite**: 105 test cases covering core algorithms, new features, and theoretical foundations
- **Realistic examples**: Gender bias correction, real-time polling, algorithm comparison
- **CI/CD workflows**: Automated testing on Python 3.10-3.13, code quality checks
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
- **Compatibility**: Python 3.10+ with modern type hints