# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - 2026-03-26

### Added
- **Continuous covariate support**: Target means with `(value, "mean")` syntax
- Learning rate schedules: Robbins-Monro, polynomial decay, inverse time (`learning_rate` module)
- Comprehensive diagnostics module: feasibility checks, variance estimation, design effects
- Streaming inference tools: confidence sequences, path-dependent variance estimation
- Sensitivity analysis module for robustness testing
- `BatchIPF` for batch raking comparison
- Convergence analysis with theoretical bounds

### Changed
- Feature storage uses float64 (supports continuous values)
- `BatchIPF` raises informative error for continuous features
- Reorganized scripts directory structure (`scripts/eval/`, `scripts/figures/`)
- `verify_robbins_monro()` now uses analytical verification for known schedule types (ConstantLR, PolynomialDecayLR, InverseTimeDecayLR) with mathematical proofs; falls back to numerical estimation with clear disclaimers for custom schedules

### Fixed
- `StreamingEstimator.partial_fit()` now correctly tracks retroactive weight changes (was comparing current weights to themselves)

## [1.2.0] - 2025-01-XX

### Added
- Interactive Jupyter notebooks with comprehensive examples and visualizations
- Enhanced documentation with step-by-step tutorials

### Changed
- Migrated examples from Python scripts to interactive notebooks
- Streamlined documentation structure focused on usage

### Removed
- Static example scripts in favor of interactive notebooks

## [1.0.0] - 2024-XX-XX

### Added
- Complete rewrite with breaking API changes
- General binary feature support (not limited to demographics)
- Performance improvements with 10x speed boost
- Google-style docstrings and modern type hints
- Comprehensive test suite with 26+ test cases
- Advanced diagnostics and monitoring capabilities

### Changed
- Breaking: Removed hardcoded demographic features
- Breaking: New Targets API for general features
- Enhanced numerical stability and convergence detection
- Improved weight distribution analysis

### Removed
- Hardcoded demographic assumptions
- Legacy API patterns
