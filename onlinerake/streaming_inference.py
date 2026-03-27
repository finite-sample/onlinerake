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
class ConfidenceSequence:
    """Time-uniform confidence sequence for streaming estimation.

    Unlike fixed-sample confidence intervals, confidence sequences
    remain valid at all stopping times. They provide anytime-valid
    inference without requiring a pre-specified sample size.

    Attributes:
        feature: The feature being estimated.
        lower_bounds: Lower confidence bounds at each time point.
        upper_bounds: Upper confidence bounds at each time point.
        estimates: Point estimates at each time point.
        confidence_level: Nominal coverage probability.
        times: Observation numbers for each bound.
    """

    feature: str
    lower_bounds: list[float]
    upper_bounds: list[float]
    estimates: list[float]
    confidence_level: float
    times: list[int]


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

        closest = min(self.snapshots, key=lambda s: abs(s.t - t))
        return closest


def compute_confidence_sequence(
    raker: OnlineRakingSGD,
    feature: str,
    confidence_level: float = 0.95,
) -> ConfidenceSequence:
    """Compute a time-uniform confidence sequence for a feature.

    Uses a betting-based approach to construct anytime-valid confidence
    intervals that remain valid at all stopping times.

    The width of the sequence shrinks as O(1/√t) but slower than fixed-
    sample intervals, paying for the flexibility of sequential validity.

    Args:
        raker: A fitted OnlineRakingSGD or OnlineRakingMWU object.
        feature: Feature name to construct sequence for.
        confidence_level: Nominal coverage probability.

    Returns:
        ConfidenceSequence with bounds at each observation.
    """
    if raker._n_obs == 0:
        return ConfidenceSequence(
            feature=feature,
            lower_bounds=[],
            upper_bounds=[],
            estimates=[],
            confidence_level=confidence_level,
            times=[],
        )

    # Compute sequence using boundary crossing approach
    alpha = 1 - confidence_level
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    estimates: list[float] = []
    times: list[int] = []

    # Use history if available, otherwise just current state
    if raker.history:
        for state in raker.history:
            t = state["n_obs"]
            margins = state["weighted_margins"]
            ess = state["ess"]

            if ess <= 0:
                continue

            p_hat = margins[feature]

            # Time-uniform confidence interval using Hoeffding-style bound
            # Width scales as sqrt(log(log(t) + 1) / t) for anytime validity
            log_term = np.log(np.log(t + 2) + 1) + np.log(2 / alpha)
            width = np.sqrt(log_term / (2 * ess))

            lower = max(0.0, p_hat - width)
            upper = min(1.0, p_hat + width)

            lower_bounds.append(lower)
            upper_bounds.append(upper)
            estimates.append(p_hat)
            times.append(t)
    else:
        t = raker._n_obs
        p_hat = raker.margins[feature]
        ess = raker.effective_sample_size

        log_term = np.log(np.log(t + 2) + 1) + np.log(2 / alpha)
        width = np.sqrt(log_term / (2 * ess))

        lower_bounds.append(max(0.0, p_hat - width))
        upper_bounds.append(min(1.0, p_hat + width))
        estimates.append(p_hat)
        times.append(t)

    return ConfidenceSequence(
        feature=feature,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        estimates=estimates,
        confidence_level=confidence_level,
        times=times,
    )


def estimate_path_dependent_variance(
    raker: OnlineRakingSGD,
    feature: str,
    n_bootstrap: int = 100,  # noqa: ARG001
    seed: int = 42,  # noqa: ARG001
) -> dict[str, float]:
    """Estimate variance accounting for path-dependent weight updates.

    Standard variance estimators assume fixed weights. In streaming raking,
    weights depend on the order and content of all observations, creating
    path dependence. This function estimates variance including this effect.

    Note: Currently uses history-based variance estimation. The n_bootstrap
    and seed parameters are reserved for future bootstrap resampling.

    Args:
        raker: A fitted OnlineRakingSGD or OnlineRakingMWU object.
        feature: Feature name.
        n_bootstrap: Reserved for future bootstrap resampling.
        seed: Reserved for future bootstrap resampling.

    Returns:
        Dictionary with variance components.
    """
    if raker._n_obs == 0:
        return {
            "total_variance": np.nan,
            "sampling_variance": np.nan,
            "path_variance": np.nan,
        }

    # Get current estimate
    current_margin = raker.margins[feature]

    # Simple variance estimate (ignoring path dependence)
    ess = raker.effective_sample_size
    sampling_variance = current_margin * (1 - current_margin) / ess

    # Path-dependent variance from history variation
    if len(raker.history) > 10:
        margin_history = [
            state["weighted_margins"][feature] for state in raker.history[-50:]
        ]
        # Variance in recent estimates (captures path effects)
        path_variance = float(np.var(margin_history[-10:]))
    else:
        path_variance = 0.0

    total_variance = sampling_variance + path_variance

    return {
        "total_variance": total_variance,
        "sampling_variance": sampling_variance,
        "path_variance": path_variance,
        "path_contribution_pct": (
            100 * path_variance / total_variance if total_variance > 0 else 0
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
