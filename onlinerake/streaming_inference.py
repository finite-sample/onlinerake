"""Streaming inference with proper uncertainty quantification.

This module addresses the fundamental question: "What are we estimating at
each time point in a streaming context?"

When weights are updated based on observation t+1, all weighted estimates
for observations 1 through t change retroactively. This module provides:

1. Snapshot estimators: Point-in-time estimates with fixed weights
2. Path-dependent variance: Accounting for sequential weight updates
3. Confidence sequences: Valid confidence intervals over the stream
4. Inference semantics: Clear definition of what each estimate means

Key insight: In streaming raking, there is no single "estimate at time t"
- the estimate depends on all future observations. This module makes these
semantics explicit and provides appropriate statistical tools.

References:
    - Howard, S. R., et al. (2021). Time-uniform, nonparametric, nonasymptotic
      confidence sequences. The Annals of Statistics, 49(2), 1055-1080.
    - Waudby-Smith, I., & Ramdas, A. (2021). Estimating means of bounded
      random variables by betting. arXiv preprint arXiv:2010.09686.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from .online_raking_sgd import OnlineRakingSGD


@dataclass
class StreamingSnapshot:
    """A snapshot of the streaming estimator at a specific time point.

    This captures the state of weighted margins with weights frozen
    at a particular observation count. Unlike the live raker state,
    snapshots don't change when new data arrives.

    Attributes:
        t: Observation number when snapshot was taken.
        margins: Weighted margins at time t.
        weights: Weight vector at time t (copy).
        ess: Effective sample size at time t.
        loss: Loss at time t.
        raw_margins: Unweighted margins at time t.
    """

    t: int
    margins: dict[str, float]
    weights: npt.NDArray[np.float64]
    ess: float
    loss: float
    raw_margins: dict[str, float]


@dataclass
class RetroactiveImpact:
    """Analysis of how new observations retroactively change estimates.

    When observation t+1 arrives and weights update, all previous
    estimates change. This dataclass quantifies that retroactive impact.

    Attributes:
        t_before: Observation count before new data.
        t_after: Observation count after new data.
        margin_changes: Per-feature change in weighted margins.
        ess_change: Change in effective sample size.
        weight_changes: Statistics on weight changes.
    """

    t_before: int
    t_after: int
    margin_changes: dict[str, float]
    ess_change: float
    weight_changes: dict[str, float]


@dataclass
class StreamingEstimator:
    """Stateful estimator that tracks streaming inference semantics.

    This class wraps a raker and provides proper streaming inference
    with snapshot capabilities, confidence sequences, and retroactive
    impact tracking.

    Attributes:
        raker: The underlying OnlineRakingSGD or OnlineRakingMWU.
        snapshots: Historical snapshots (if snapshot_interval > 0).
        snapshot_interval: How often to save snapshots (0 = never).
        track_retroactive: Whether to track retroactive changes.
        retroactive_impacts: History of retroactive changes.
    """

    raker: Any
    snapshots: list[StreamingSnapshot] = field(default_factory=list)
    snapshot_interval: int = 0
    track_retroactive: bool = False
    retroactive_impacts: list[RetroactiveImpact] = field(default_factory=list)
    _last_margins: dict[str, float] | None = None
    _last_ess: float | None = None

    def partial_fit(self, obs: dict[str, Any]) -> None:
        """Process observation with streaming inference tracking.

        Args:
            obs: Observation dictionary.
        """
        # Store state BEFORE update for retroactive tracking
        n_before = self.raker._n_obs
        weights_before: npt.NDArray[np.float64] | None = None

        if self.track_retroactive and n_before > 0:
            self._last_margins = self.raker.margins.copy()
            self._last_ess = self.raker.effective_sample_size
            weights_before = self.raker.weights[:n_before].copy()

        # Process observation
        self.raker.partial_fit(obs)

        # Track retroactive impact
        if self.track_retroactive and self._last_margins is not None:
            current_margins = self.raker.margins
            margin_changes = {
                feat: current_margins[feat] - self._last_margins[feat]
                for feat in current_margins
            }

            # Compare weights of existing observations (before vs after update)
            weights_after = self.raker.weights[:n_before]

            impact = RetroactiveImpact(
                t_before=n_before,
                t_after=self.raker._n_obs,
                margin_changes=margin_changes,
                ess_change=self.raker.effective_sample_size - (self._last_ess or 0),
                weight_changes={
                    "mean_abs_change": (
                        float(np.mean(np.abs(weights_after - weights_before)))
                        if weights_before is not None and len(weights_before) > 0
                        else 0.0
                    ),
                },
            )
            self.retroactive_impacts.append(impact)

        # Take snapshot if needed
        if (
            self.snapshot_interval > 0
            and self.raker._n_obs % self.snapshot_interval == 0
        ):
            self.take_snapshot()

    def take_snapshot(self) -> StreamingSnapshot:
        """Capture current state as immutable snapshot.

        Returns:
            StreamingSnapshot with frozen estimates.
        """
        snapshot = StreamingSnapshot(
            t=self.raker._n_obs,
            margins=self.raker.margins.copy(),
            weights=self.raker.weights.copy(),
            ess=self.raker.effective_sample_size,
            loss=self.raker.loss,
            raw_margins=self.raker.raw_margins.copy(),
        )
        self.snapshots.append(snapshot)
        return snapshot

    def get_snapshot_at(self, t: int) -> StreamingSnapshot | None:
        """Retrieve snapshot closest to time t.

        Args:
            t: Target observation number.

        Returns:
            Closest snapshot, or None if no snapshots exist.
        """
        if not self.snapshots:
            return None

        return min(self.snapshots, key=lambda s: abs(s.t - t))


def estimate_path_dependent_variance(
    raker: OnlineRakingSGD,
    feature: str,
    n_permutations: int = 20,
    seed: int = 0,
) -> dict[str, float]:
    """Estimate variance accounting for path-dependent weight updates.

    Standard variance estimators assume fixed weights. In streaming raking the
    weights depend on the order observations arrived in as well as on which
    observations arrived, so there are two sources to separate:

    * ``sampling_variance`` -- different observations. A replication variance,
      with the calibration re-run inside each replicate. It replaced the
      ``p(1-p)/ESS`` this used to carry, which is the variance of an unweighted
      proportion and measured 4.43x the margin's actual spread.
    * ``path_variance`` -- **the same observations in a different order.** The
      whole sample is refitted ``n_permutations`` times under shuffled arrival
      orders and the spread of the resulting margin is taken.

    The second used to be the variance of the margin over the last ten entries
    of ``history``. Those are ten points along one path at ten different sample
    sizes, serially dependent and still converging, so their spread is not an
    estimate of anything the caller asked for. Measured against the
    permutation spread on a stationary stream it came to 0.08, 0.04 and 0.02 of
    it at n = 200, 400 and 800 -- understating the effect by more than tenfold
    and drifting further out with every observation, because a converging path
    flattens while genuine order-dependence does not.

    **The two components are added on an independence assumption**, which is
    not proved here. Replication holds arrival order fixed and varies the
    sample; permutation holds the sample fixed and varies the order; the sum is
    a reasonable total but the cross-term is not estimated. Read the components
    rather than the total when the distinction matters.

    Cost is ``n_permutations`` full recalibrations on top of the replication,
    so this is much the most expensive diagnostic in the package.

    Args:
        raker: A fitted OnlineRakingSGD or OnlineRakingMWU object.
        feature: Feature name.
        n_permutations: Refits used to measure order dependence. Below two the
            component is unestimable and comes back ``nan``.
        seed: Seed for the permutations, so the answer is reproducible.

    Returns:
        Dictionary with variance components.
    """
    if raker._n_obs == 0:
        return {
            "total_variance": np.nan,
            "sampling_variance": np.nan,
            "path_variance": np.nan,
            "path_contribution_pct": np.nan,
        }

    # Replication variance, not p(1-p)/ESS. The latter is the variance of an
    # UNWEIGHTED proportion and was measured at 4.43x the margin's true spread,
    # because it prices in none of the variance reduction that calibrating onto
    # a fixed target produces. See onlinerake.diagnostics.
    from .diagnostics import estimate_margin_variance

    # Forward the caller's seed: without it this component always used the
    # estimator's own default, so two calls with different seeds shared their
    # sampling-variance term and only the path term moved.
    sampling_variance = estimate_margin_variance(raker, feature, seed=seed)

    # Order dependence, measured by refitting the same observations in shuffled
    # arrival orders. Not the spread of the last few history entries: those sit
    # at different sample sizes on a single path and measure convergence, not
    # order.
    from .diagnostics import _refit_margins

    if raker._n_obs >= 2 and n_permutations >= 2:
        rng = np.random.default_rng(seed)
        permuted = [
            _refit_margins(raker, rng.permutation(raker._n_obs))[feature]
            for _ in range(int(n_permutations))
        ]
        path_variance = float(np.var(permuted, ddof=1))
    else:
        path_variance = float("nan")

    total_variance = sampling_variance + path_variance

    return {
        "total_variance": total_variance,
        "sampling_variance": sampling_variance,
        "path_variance": path_variance,
        # nan, not 0. Below two permutations the component is unestimable, and
        # reporting a 0% share there is a claim that order does not matter --
        # which is precisely what was not measured.
        "path_contribution_pct": (
            100 * path_variance / total_variance
            if np.isfinite(total_variance) and total_variance > 0
            else float("nan")
        ),
    }


def explain_streaming_semantics() -> dict[str, str]:
    """Return documentation of streaming inference semantics.

    Returns:
        Dictionary explaining key concepts.
    """
    return {
        "retroactive_updates": (
            "When observation t+1 arrives and weights are updated, all weighted "
            "estimates for observations 1 through t change. The 'estimate at time t' "
            "is not fixed - it depends on all future observations."
        ),
        "snapshot_vs_live": (
            "A 'snapshot' freezes the weights at time t, giving a fixed estimate. "
            "The 'live' estimate continues to change as new data arrives. Use "
            "snapshots when you need reproducible estimates."
        ),
        "confidence_sequences": (
            "Unlike fixed-sample confidence intervals, confidence sequences remain "
            "valid at any stopping time. They're wider but allow continuous monitoring "
            "without p-hacking concerns."
        ),
        "effective_sample_size": (
            "ESS measures how many unweighted observations would give equivalent "
            "precision. In streaming raking, ESS can fluctuate as weights adjust. "
            "Monitor ESS/n (weight efficiency) for stability."
        ),
        "when_to_stop": (
            "Stop when: (1) loss is below tolerance, (2) ESS/n is acceptable, "
            "(3) confidence sequence is narrow enough. The confidence sequence "
            "approach avoids the 'peeking' problem of repeated testing."
        ),
    }


def analyze_estimate_stability(
    raker: OnlineRakingSGD,
    window: int = 50,
) -> dict[str, Any]:
    """Analyze stability of streaming estimates.

    Helps determine whether the stream has "settled" into stable estimates
    or is still fluctuating significantly.

    Args:
        raker: A fitted OnlineRakingSGD or OnlineRakingMWU object.
        window: Number of recent observations to analyze.

    Returns:
        Dictionary with stability metrics per feature.
    """
    if len(raker.history) < window:
        return {
            "status": "INSUFFICIENT_DATA",
            "observations_needed": window,
            "current_observations": len(raker.history),
        }

    stability_metrics: dict[str, Any] = {"features": {}}

    recent_history = raker.history[-window:]

    for feature in raker._feature_names:
        margins = [state["weighted_margins"][feature] for state in recent_history]

        # Compute stability metrics
        mean = float(np.mean(margins))
        std = float(np.std(margins))
        range_val = float(np.max(margins) - np.min(margins))

        # Coefficient of variation (lower = more stable)
        cv = std / mean if mean > 0 else np.nan

        # Trend (positive = increasing, negative = decreasing)
        if len(margins) >= 10:
            half = len(margins) // 2
            trend = float(np.mean(margins[half:]) - np.mean(margins[:half]))
        else:
            trend = 0.0

        # Stability score (0-1, higher = more stable)
        stability_score = max(0, 1 - cv - abs(trend) * 10) if not np.isnan(cv) else 0.0

        stability_metrics["features"][feature] = {
            "mean": mean,
            "std": std,
            "range": range_val,
            "coefficient_of_variation": cv,
            "trend": trend,
            "stability_score": stability_score,
        }

    # Overall stability
    scores = [m["stability_score"] for m in stability_metrics["features"].values()]
    stability_metrics["overall_stability"] = float(np.mean(scores))
    stability_metrics["status"] = (
        "STABLE" if stability_metrics["overall_stability"] > 0.7 else "UNSTABLE"
    )

    return stability_metrics
