Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

**Breaking Changes**
- **Minimum Python version increased to 3.10** (from 3.8)
- Modernized type hints using Python 3.10+ syntax (dict, list, | for union types)

**Added**

- **Enhanced Diagnostics & Monitoring**:

  - Gradient norm tracking for convergence analysis
  - Automatic convergence detection with configurable tolerance
  - Oscillation detection for non-converging scenarios
  - Enhanced weight distribution statistics (quartiles, outliers)
  - Verbose mode for debugging with progress indicators
  - Loss moving average calculation
  - New ``diagnostics_demo.py`` example showcasing monitoring features

- **Major Performance Optimizations**:

  - **Capacity doubling for weights storage**: Eliminates O(n²) memory reallocations
  - **Optimized array conversions**: Moved outside gradient computation loops
  - **Configurable weight statistics**: Optional/sampled computation for expensive percentiles
  - **Overall speedup**: 10-100x improvement for large streams (n>1000)
  - Performance scales nearly linearly with data size

- Comprehensive test suite with 21+ test cases
- Realistic examples for common use cases
- Complete documentation with Sphinx
- CI/CD workflows for testing and publishing
- Code formatting and linting checks

**Changed**
- Type hints modernized to use Python 3.10+ built-in types
- Removed ``from __future__ import annotations`` (no longer needed)
- CI/CD now tests Python 3.10, 3.11, 3.12, and 3.13 (dropped 3.8, 3.9)
- Enhanced history tracking with comprehensive diagnostic metrics
- **Internal data structures**: Weights array now uses capacity doubling for O(log n) amortized growth
- **Weight statistics computation**: Now configurable (always, never, or sampled) for performance

**Fixed**

- **Critical Numerical Stability Issues**:

  - MWU algorithm now clips exponential arguments to prevent overflow/underflow
  - Convergence detection properly handles near-zero loss cases
  - Improved robustness with extreme learning rates and gradients
- Import errors for Optional and Any types in simulation module
- Improved docstring formatting and clarity
- Flake8 linting issues with whitespace in slice notation

[0.1.1] - 2024-XX-XX
--------------------

**Added**
- Initial release of onlinerake package
- SGD-based streaming raking algorithm (OnlineRakingSGD)
- MWU-based streaming raking algorithm (OnlineRakingMWU)
- Targets dataclass for population margins
- Simulation module for benchmarking algorithms
- Basic README with usage examples

**Features**
- Real-time weight calibration for streaming survey data
- scikit-learn style partial_fit API
- Support for binary demographic indicators (age, gender, education, region)
- Effective sample size and loss monitoring
- Weight clipping to prevent numerical issues
- Comprehensive margin tracking and reporting

**Dependencies**
- numpy >= 1.21
- pandas >= 1.3
- Python >= 3.10

[0.1.0] - Initial Development
-----------------------------

**Added**
- Core algorithm implementations
- Basic project structure
- Initial documentation