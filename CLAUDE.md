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

## Architecture

Core modules in `onlinerake/`:
- `targets.py` - Defines `Targets` dataclass for population margins (age, gender, education, region)
- `online_raking_sgd.py` - SGD-based streaming raking algorithm 
- `online_raking_mwu.py` - MWU-based algorithm (inherits from SGD)

Example modules in `examples/`:
- `simulation.py` - Simulation harness for benchmarking algorithms
- `realistic_examples.py` - Real-world usage examples

Both raking classes follow scikit-learn's `partial_fit` pattern: call `.partial_fit(obs)` for each observation and inspect `.margins`, `.loss`, and `.effective_sample_size` properties.

## Development Commands

```bash
# Install package in editable mode
pip install -e .

# Install development dependencies
pip install pytest black flake8 sphinx sphinx-rtd-theme myst-parser

# Run comprehensive test suite (26 tests including performance & numerical stability)
pytest tests/test_onlinerake.py -v --cov=onlinerake --cov-report=term

# Run simulation suite and examples
python3 examples/simulation.py
python3 examples/realistic_examples.py

# Code quality checks (zero tolerance for critical issues)
flake8 onlinerake --count --select=E9,F63,F7,F82 --show-source --statistics
black --check onlinerake

# Build documentation (includes new diagnostics and performance sections)
cd docs && make html
```

## Testing

- **Comprehensive test suite**: 26 test cases covering all core functionality, performance, and numerical stability
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