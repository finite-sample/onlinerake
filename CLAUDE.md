# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`onlinerake` is a Python package for streaming survey raking (weight calibration) using two algorithms:
- **SGD raking** (`OnlineRakingSGD`) - stochastic gradient descent with additive updates
- **MWU raking** (`OnlineRakingMWU`) - multiplicative weights update with exponential updates

The package enables real-time weight adjustment for streaming survey data to match known population margins, unlike traditional batch IPF methods.

## Architecture

Core modules in `onlinerake/`:
- `targets.py` - Defines `Targets` dataclass for population margins (age, gender, education, region)
- `online_raking_sgd.py` - SGD-based streaming raking algorithm 
- `online_raking_mwu.py` - MWU-based algorithm (inherits from SGD)
- `simulation.py` - Simulation harness for benchmarking algorithms
- `__main__.py` - Entry point for command-line simulation execution

Both raking classes follow scikit-learn's `partial_fit` pattern: call `.partial_fit(obs)` for each observation and inspect `.margins`, `.loss`, and `.effective_sample_size` properties.

## Development Commands

```bash
# Install package in editable mode
pip install -e .

# Install development dependencies
pip install pytest black flake8 sphinx sphinx-rtd-theme myst-parser

# Run comprehensive test suite
pytest tests/test_onlinerake.py -v --cov=onlinerake --cov-report=term

# Run simulation suite and examples
python3 -m onlinerake.simulation
python3 examples/realistic_examples.py

# Code quality checks
flake8 onlinerake --count --select=E9,F63,F7,F82 --show-source --statistics
black --check onlinerake

# Build documentation
cd docs && make html
```

## Testing

- **Comprehensive test suite**: 15+ test cases covering all core functionality
- **Realistic examples**: Gender bias correction, real-time polling, algorithm comparison
- **CI/CD workflows**: Automated testing on Python 3.8-3.12, code quality checks
- **Coverage**: High test coverage for critical paths and edge cases

## Key Implementation Details

- Both algorithms maintain weight vectors updated per observation in O(n) time
- SGD uses squared-error loss on margins; MWU uses KL divergence via mirror descent
- `Targets` class uses fixed demographic fields (age, gender, education, region)
- Simulation module generates synthetic streams with bias patterns (linear drift, sudden shift, oscillation)
- Package has minimal dependencies: only numpy and pandas