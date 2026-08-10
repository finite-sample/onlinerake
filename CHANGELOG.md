# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-08-09

A major version because public API is removed. Two features that put confidence
intervals on a raked margin are gone, and the inference they were attempting now
lives on the quantity that can carry it.

### Removed
- **`diagnostics.compute_confidence_interval()`** and
  **`streaming_inference.compute_confidence_sequence()`**, along with
  `ConfidenceSequence` and `MarginEstimate`.

  R's `survey` reports a standard error of **exactly zero** for a raking
  variable, because after calibration the margin *is* the target the caller
  supplied as known. There is no sampling uncertainty about a number you
  supplied, so neither interval was estimating anything, and a better variance
  estimator would not have changed that.

  Both were also wrong on their own terms before that argument applies. The
  fixed-n interval used `p(1-p)/ESS`, the variance of an *unweighted*
  proportion, measured at 4.43x the margin's actual spread; with an honest width
  its coverage fell from 1.000 to 0.210. The sequence was documented as
  "time-uniform ... valid at all stopping times" and cited a betting
  construction it did not implement — the code was a plug-in Hoeffding-style
  boundary — and measured 0.470 anytime coverage at `OnlineRakingMWU`'s default.
- **`summarize_raking_results()` no longer takes `confidence_level`.** Its
  margin section reports calibration, which has no confidence level.

### Changed
- **Every function taking a replication `method` now defaults to `"auto"`**,
  which picks jackknife when the replicate size `n/G` is below 100 and random
  groups at or above it. Random groups scales the spread among `G` disjoint
  groups by `1/(G(G-1))`, which assumes the margin's variance goes as `1/n`; its
  replicates hold `n/G` observations, which is where that assumption fails.
  Jackknife replicates hold `n(1 - 1/G)` and stay in the safe region. See the
  measurements under Known limitations.

  The rule also bounds its own cost. Both schemes are linear in `n` and
  jackknife refits roughly `9x` the rows, so jackknife gets *more* expensive
  exactly where the rule stops choosing it. Cost peaks at the threshold (0.65 s
  at n = 600) and falls after, against 8.97 s for always-jackknife at n = 4800.

### Added
- **`diagnostics.resolve_replication_method()`**, which turns `"auto"` into the
  scheme that will actually run. A default that changes with the data is harder
  to reason about than either fixed one, so it is inspectable rather than only
  documented. Passing an explicit scheme still bypasses the rule entirely.
- **`diagnostics.margin_calibration()` -> `MarginCalibration`**, reporting what
  is actually true about a raked margin: `gap = estimate - target`, which is
  exact arithmetic rather than an estimate; a replication `std_error`;
  `gap_ratio`, the gap in units of that spread; and `unclosed_fraction`, the
  share of the initial miscalibration the raker did not remove.

  `gap_ratio` carries **no absolute threshold** — at fixed calibration effort
  its numerator is flat in the stream length while its denominator falls, so it
  rises with `n` (3.26, 4.56, 7.02, 16.26 at n = 250 / 1000 / 4000 / 12000 on an
  unchanged raker). `unclosed_fraction` is the scale-free companion and does not
  move with `n`.
- **Inference for the GREG estimate**: `model_assisted_variance()`,
  `model_assisted_std_error()` and `model_assisted_confidence_interval()`.
  `ModelAssistedRaker.model_assisted_estimate` is the one genuine outcome
  estimator in the package and had no variance estimator at all. Signatures
  mirror the margin functions. The interval is conditional on the outcome model,
  which is fixed across replicates by design.
- **Replication variance** underneath both: `method="random_groups"` (default)
  calibrates `n_replicates` disjoint groups and costs less than the original
  fit; `method="jackknife"` is delete-a-group and costs `n_replicates` full
  refits. Each replicate re-runs the whole calibration, which is `survey`'s own
  position — it refuses to calibrate a design after replicate weights exist.

### Fixed
- **`_z_score()` returned the 95% multiplier for any level outside a
  three-entry table.** It fell through to a `scipy` import, and scipy is not a
  dependency of this package, so `except ImportError` returned `Z_SCORES[0.95]`.
  Asking for an 80% interval silently produced a 95% one. Now
  `statistics.NormalDist().inv_cdf`, which is standard library and exact, with a
  range check.
- **A variance of `0.0` below two observations produced a zero-width 95%
  interval** from a single draw. Now `nan`, which is what an unestimable
  variance is. Two tests had asserted the `0.0` and so encoded the defect.
- `estimate_path_dependent_variance()` carried the same `p(1-p)/ESS` and now
  uses the replication variance.
- CI ran four test files never at all — `test_new_features`,
  `test_theoretical_features`, `test_model_assisted`, `test_mwu_ipf_equivalence`
  — because the step named files explicitly. It now runs the directory.

### Known limitations
- The raked margin's variance does not scale as `1/n`: at a sample size of 80
  its spread is 0.72 of what `1/n` predicts from the spread at 800, rising to
  1.00 by 800 (120 replicates per cell). At small sample sizes the margin is
  pinned tighter to the target, because fewer observations satisfy the same
  constraints. The random-groups scaling assumes `1/n` and its replicates are
  `n/G`, so it sits in the region where the assumption fails; jackknife
  replicates are `n(1 - 1/G)` and do not.

  Measured against the margin's own spread over 500-600 streams, with each
  scheme paired on 120 identical further streams, `random_groups` understates by
  an amount governed by `n/G` while jackknife is within noise of the truth
  throughout:

  | n | n/G | random_groups | jackknife | paired jk - rg |
  |---|---|---|---|---|
  | 300 | 30 | 0.838 [0.802, 0.875] | 0.959 [0.920, 0.999] | +14.4% |
  | 600 | 60 | 0.923 [0.871, 0.974] | 0.997 [0.940, 1.055] | +8.1% |
  | 1200 | 120 | 0.958 [0.920, 0.997] | 0.995 [0.954, 1.037] | +3.9% |

  **The size of the gap is specific to the process it was measured on.** On a
  three-margin population under logistic selection the same paired comparison at
  n=300 gives +9.3% rather than +14.4%, with random groups at 0.944 of the
  observed spread and jackknife at 1.032. The direction and the ordering hold in
  both; the magnitude does not transfer, which is why
  `resolve_replication_method` documents the rule as a default rather than a law
  and why `AUTO_REPLICATE_SIZE` is a judgment rather than a derived constant.
- The calibration gap itself is the raker's tracking lag under a fixed-gain
  update and scales as `1 / (learning_rate * n_sgd_steps)`. A decaying
  Robbins-Monro schedule makes it *worse*, not better, because a shrinking step
  cannot track a target that moves as new observations arrive.

## [1.4.0] - 2026-03-30

### Added
- KL divergence tracking for OnlineRakingSGD and OnlineRakingMWU (`track_kl_divergence` parameter)
- `kl_divergence_weights()` - compute KL divergence between weight distributions
- `total_variation_weights()` - compute total variation distance
- `symmetric_kl_divergence()` - symmetric KL measure
- `compare_to_ipf()` - compare streaming raker to batch IPF solution
- `optimal_mwu_learning_rate()` - learning rate guidance for IPF-matching
- `IPFComparison` dataclass for IPF comparison results
- New example: `examples/kl_ipf_comparison.py` demonstrating KL tracking and IPF comparison
- Comprehensive tests for MWU-IPF equivalence

### Changed
- OnlineRakingMWU inherits `track_kl_divergence` from OnlineRakingSGD

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
