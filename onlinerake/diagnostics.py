"""Diagnostics and variance estimation for online raking.

This module provides tools for:
1. Detecting infeasible target margins
2. Estimating variance of weighted margins
3. Computing confidence intervals
4. Assessing target feasibility

These address the critical gap identified in the original paper:
"What about uncertainty?" - the need for variance estimates and
proper handling of infeasible targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ._utils import requires_observations

if TYPE_CHECKING:
    from .batch_ipf import BatchIPF
    from .online_raking_sgd import OnlineRakingSGD


# Module-level constants for magic numbers
WEIGHT_THRESHOLD_RATIO = 0.95
WEIGHT_LOWER_THRESHOLD_RATIO = 1.05
MIN_FEASIBILITY_SCORE = 0.5
PROGRESS_SCORE_BONUS = 0.5
EXTREME_WEIGHT_RATIO = 1000
MAX_WEIGHT_RATIO_COMPROMISE = 100
Z_SCORES: dict[float, float] = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}


@dataclass
class MarginEstimate:
    """Weighted margin estimate with uncertainty quantification.

    Attributes:
        feature: Name of the feature.
        target: Target population proportion.
        estimate: Weighted sample estimate.
        std_error: Estimated standard error.
        ci_lower: Lower bound of confidence interval.
        ci_upper: Upper bound of confidence interval.
        raw_estimate: Unweighted sample proportion.
        bias_reduction: Percentage reduction in bias from target.
    """

    feature: str
    target: float
    estimate: float
    std_error: float
    ci_lower: float
    ci_upper: float
    raw_estimate: float
    bias_reduction: float


@dataclass
class FeasibilityReport:
    """Report on target feasibility.

    Attributes:
        is_feasible: Whether all targets appear achievable.
        problematic_features: Features with potential feasibility issues.
        feasibility_scores: Per-feature feasibility scores (0-1, higher is better).
        recommendations: Suggested actions if infeasible.
    """

    is_feasible: bool
    problematic_features: list[str]
    feasibility_scores: dict[str, float]
    recommendations: list[str]


def _estimate_margin_variance_impl(
    raker: OnlineRakingSGD,
    feature: str,
) -> float:
    """Implementation of margin variance estimation."""
    ess = raker.effective_sample_size
    if ess <= 0:
        return np.nan

    if raker.targets.is_binary(feature):
        margins = raker.margins
        p_hat = margins[feature]
        variance = p_hat * (1 - p_hat) / ess
    else:
        feature_idx = raker._feature_names.index(feature)
        feature_values = raker._features[: raker._n_obs, feature_idx]
        weights = raker._weights[: raker._n_obs]

        total_w = weights.sum()
        weighted_mean = (weights * feature_values).sum() / total_w
        weighted_var = (weights * (feature_values - weighted_mean) ** 2).sum() / total_w

        variance = weighted_var / ess

    return float(variance)


def estimate_margin_variance(
    raker: OnlineRakingSGD,
    feature: str,
) -> float:
    """Estimate variance of a weighted margin using design-based estimator.

    For binary features, uses a ratio estimator variance formula:
        Var(p̂) ≈ (1/n_eff) * p̂ * (1 - p̂)

    For continuous features, uses the sample variance:
        Var(x̄) ≈ s² / n_eff

    where n_eff is the effective sample size (Kish ESS).

    This is a conservative approximation. The true variance depends on
    the sampling design and weight generation process.

    Args:
        raker: A fitted OnlineRakingSGD or OnlineRakingMWU object.
        feature: Name of the feature to estimate variance for.

    Returns:
        Estimated variance of the weighted margin.
    """
    if raker._n_obs == 0:
        return np.nan

    return _estimate_margin_variance_impl(raker, feature)


def estimate_margin_std_error(
    raker: OnlineRakingSGD,
    feature: str,
) -> float:
    """Estimate standard error of a weighted margin.

    Args:
        raker: A fitted OnlineRakingSGD or OnlineRakingMWU object.
        feature: Name of the feature.

    Returns:
        Estimated standard error.
    """
    var = estimate_margin_variance(raker, feature)
    return float(np.sqrt(var)) if not np.isnan(var) else np.nan


def compute_confidence_interval(
    raker: OnlineRakingSGD,
    feature: str,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Compute confidence interval for a weighted margin.

    Uses normal approximation with estimated standard error.

    Args:
        raker: A fitted OnlineRakingSGD or OnlineRakingMWU object.
        feature: Name of the feature.
        confidence_level: Confidence level (default 0.95 for 95% CI).

    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    margins = raker.margins
    estimate = margins[feature]
    std_error = estimate_margin_std_error(raker, feature)

    if np.isnan(std_error):
        return (np.nan, np.nan)

    # Use closest or interpolate
    if confidence_level in Z_SCORES:
        z = Z_SCORES[confidence_level]
    else:
        # Linear interpolation for other levels (approximate)
        try:
            from scipy import stats  # type: ignore[import-untyped]

            z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        except ImportError:
            # Fall back to 95% CI
            z = 1.960

    # For binary features, clamp to [0, 1]; for continuous, no clamping
    if raker.targets.is_binary(feature):
        lower = max(0.0, estimate - z * std_error)
        upper = min(1.0, estimate + z * std_error)
    else:
        lower = estimate - z * std_error
        upper = estimate + z * std_error

    return (float(lower), float(upper))


def get_margin_estimates(
    raker: OnlineRakingSGD,
    confidence_level: float = 0.95,
) -> list[MarginEstimate]:
    """Get comprehensive margin estimates for all features.

    Args:
        raker: A fitted OnlineRakingSGD or OnlineRakingMWU object.
        confidence_level: Confidence level for intervals.

    Returns:
        List of MarginEstimate objects with full uncertainty quantification.
    """
    estimates = []
    margins = raker.margins
    raw_margins = raker.raw_margins

    for feature in raker._feature_names:
        target = raker.targets[feature]
        estimate = margins[feature]
        raw = raw_margins[feature]
        std_error = estimate_margin_std_error(raker, feature)
        ci_lower, ci_upper = compute_confidence_interval(
            raker, feature, confidence_level
        )

        # Compute bias reduction
        raw_bias = abs(raw - target)
        weighted_bias = abs(estimate - target)
        if raw_bias > 0:
            bias_reduction = 100 * (raw_bias - weighted_bias) / raw_bias
        else:
            bias_reduction = 0.0

        estimates.append(
            MarginEstimate(
                feature=feature,
                target=target,
                estimate=estimate,
                std_error=std_error,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                raw_estimate=raw,
                bias_reduction=bias_reduction,
            )
        )

    return estimates


def check_target_feasibility(
    raker: OnlineRakingSGD,
    tolerance: float = 0.05,
) -> FeasibilityReport:
    """Check whether target margins are feasible given the observed data.

    Feasibility issues arise when:
    1. A target proportion is outside the range achievable by reweighting
    2. Target combinations are mutually inconsistent
    3. Not enough variation in the sample to achieve targets

    This function detects potential feasibility issues and provides
    actionable recommendations.

    Args:
        raker: A fitted OnlineRakingSGD or OnlineRakingMWU object.
        tolerance: Tolerance for feasibility checks.

    Returns:
        FeasibilityReport with diagnosis and recommendations.
    """
    if raker._n_obs == 0:
        return FeasibilityReport(
            is_feasible=False,
            problematic_features=[],
            feasibility_scores={},
            recommendations=["No observations processed yet."],
        )

    raw_margins = raker.raw_margins
    margins = raker.margins
    problematic = []
    scores = {}
    recommendations = []

    for feature in raker._feature_names:
        target = raker.targets[feature]
        raw = raw_margins[feature]
        weighted = margins[feature]
        is_binary = raker.targets.is_binary(feature)

        if is_binary:
            # Binary feature feasibility checks
            # Check if sample has required variation
            if raw == 0:
                # No observations with feature=1, can't increase margin
                if target > tolerance:
                    problematic.append(feature)
                    scores[feature] = 0.0
                    recommendations.append(
                        f"'{feature}': No observations with value=1, "
                        f"cannot achieve target {target:.2%}."
                    )
                    continue

            if raw == 1:
                # All observations have feature=1, can't decrease margin
                if target < 1 - tolerance:
                    problematic.append(feature)
                    scores[feature] = 0.0
                    recommendations.append(
                        f"'{feature}': All observations have value=1, "
                        f"cannot achieve target {target:.2%}."
                    )
                    continue
        else:
            # Continuous feature feasibility checks
            # Check if target is within sample range
            feature_idx = raker._feature_names.index(feature)
            feature_values = raker._features[: raker._n_obs, feature_idx]
            min_val = float(feature_values.min())
            max_val = float(feature_values.max())

            if target < min_val or target > max_val:
                problematic.append(feature)
                scores[feature] = 0.0
                recommendations.append(
                    f"'{feature}': Target mean {target:.2f} is outside sample range "
                    f"[{min_val:.2f}, {max_val:.2f}]. Cannot achieve this target."
                )
                continue

        # Check convergence toward target
        raw_error = abs(raw - target)
        weighted_error = abs(weighted - target)

        # Scale tolerance for continuous features based on target magnitude
        effective_tolerance = (
            tolerance if is_binary else tolerance * max(1.0, abs(target))
        )

        if raw_error > effective_tolerance:
            # We have a bias to correct
            if weighted_error < raw_error:
                # Making progress
                progress = 1 - (weighted_error / raw_error)
                scores[feature] = min(1.0, progress + PROGRESS_SCORE_BONUS)
            else:
                # Not making progress - potential feasibility issue
                scores[feature] = 0.3
                problematic.append(feature)
                if is_binary:
                    recommendations.append(
                        f"'{feature}': Weighted margin ({weighted:.2%}) not converging "
                        f"toward target ({target:.2%}). Consider adjusting learning rate."
                    )
                else:
                    recommendations.append(
                        f"'{feature}': Weighted mean ({weighted:.2f}) not converging "
                        f"toward target ({target:.2f}). Consider adjusting learning rate."
                    )
        else:
            # Already close to target
            scores[feature] = 1.0

        # Check for extreme weights suggesting feasibility strain
        # Use raker's configured bounds rather than hardcoded values
        weight_stats = raker.weight_distribution_stats
        at_max = weight_stats["max"] >= raker.max_weight * WEIGHT_THRESHOLD_RATIO
        at_min = weight_stats["min"] <= raker.min_weight * WEIGHT_LOWER_THRESHOLD_RATIO
        if at_max or at_min:
            if feature not in problematic:
                problematic.append(feature)
                scores[feature] = min(scores.get(feature, 1.0), MIN_FEASIBILITY_SCORE)
            extreme_warning = (
                f"Weights hitting bounds (min={raker.min_weight}, max={raker.max_weight}). "
                "This may indicate target feasibility strain."
            )
            if extreme_warning not in recommendations:
                recommendations.append(extreme_warning)

    # Overall feasibility assessment
    is_feasible = len(problematic) == 0

    if not is_feasible and not recommendations:
        recommendations.append(
            "Consider: (1) Relaxing targets, (2) Increasing sample size, "
            "(3) Using different weight bounds."
        )

    return FeasibilityReport(
        is_feasible=is_feasible,
        problematic_features=problematic,
        feasibility_scores=scores,
        recommendations=recommendations,
    )


@requires_observations(lambda: np.nan)
def compute_design_effect(raker: OnlineRakingSGD) -> float:
    """Compute the design effect (DEFF) due to weighting.

    DEFF = n / ESS, where n is the nominal sample size and ESS is the
    effective sample size. A DEFF of 2 means you need twice the sample
    size to achieve the same precision as an unweighted sample.

    Args:
        raker: A fitted OnlineRakingSGD or OnlineRakingMWU object.

    Returns:
        Design effect. Values near 1 indicate minimal precision loss from weighting.
    """
    ess = raker.effective_sample_size
    if ess <= 0:
        return np.nan

    return float(raker._n_obs / ess)


@requires_observations(lambda: np.nan)
def compute_weight_efficiency(raker: OnlineRakingSGD) -> float:
    """Compute weight efficiency (ESS / n).

    This is the inverse of design effect, ranging from 0 to 1.
    Higher values indicate more efficient use of the sample.

    Args:
        raker: A fitted OnlineRakingSGD or OnlineRakingMWU object.

    Returns:
        Weight efficiency as a proportion.
    """
    ess = raker.effective_sample_size
    return float(ess / raker._n_obs)


def summarize_raking_results(
    raker: OnlineRakingSGD,
    confidence_level: float = 0.95,
) -> dict:
    """Generate a comprehensive summary of raking results.

    Provides all key metrics and diagnostics in a single report.

    Args:
        raker: A fitted OnlineRakingSGD or OnlineRakingMWU object.
        confidence_level: Confidence level for intervals.

    Returns:
        Dictionary with complete summary statistics.
    """
    feasibility = check_target_feasibility(raker)
    margin_estimates = get_margin_estimates(raker, confidence_level)

    return {
        "n_observations": raker._n_obs,
        "effective_sample_size": raker.effective_sample_size,
        "design_effect": compute_design_effect(raker),
        "weight_efficiency": compute_weight_efficiency(raker),
        "final_loss": raker.loss,
        "converged": raker.converged,
        "convergence_step": raker.convergence_step,
        "margin_estimates": [
            {
                "feature": est.feature,
                "target": est.target,
                "estimate": est.estimate,
                "std_error": est.std_error,
                "ci_lower": est.ci_lower,
                "ci_upper": est.ci_upper,
                "raw_estimate": est.raw_estimate,
                "bias_reduction_pct": est.bias_reduction,
            }
            for est in margin_estimates
        ],
        "feasibility": {
            "is_feasible": feasibility.is_feasible,
            "problematic_features": feasibility.problematic_features,
            "scores": feasibility.feasibility_scores,
            "recommendations": feasibility.recommendations,
        },
        "weight_distribution": raker.weight_distribution_stats,
    }


@dataclass
class InfeasibilityAnalysis:
    """Detailed analysis of target infeasibility.

    When targets are infeasible, this class provides:
    1. Diagnosis of why targets are infeasible
    2. Achievable bounds for each feature
    3. Compromise solutions
    4. Recommendations for resolution

    Attributes:
        is_feasible: Whether exact targets are achievable.
        infeasibility_type: Type of infeasibility detected.
        achievable_bounds: For each feature, the achievable range.
        compromise_targets: Adjusted targets that are feasible.
        diagnosis: Detailed explanation of the infeasibility.
    """

    is_feasible: bool
    infeasibility_type: str
    achievable_bounds: dict[str, tuple[float, float]]
    compromise_targets: dict[str, float]
    diagnosis: list[str]


def analyze_infeasibility(
    raker: OnlineRakingSGD,
    max_weight_ratio: float = MAX_WEIGHT_RATIO_COMPROMISE,
) -> InfeasibilityAnalysis:
    """Analyze why targets may be infeasible and suggest compromises.

    This function provides detailed diagnosis when targets cannot be
    exactly achieved by reweighting. It identifies:

    1. **Structural infeasibility**: Some targets require weight=0 or ∞
    2. **Conflicting targets**: Targets are mutually inconsistent
    3. **Numerical infeasibility**: Extreme weights would be required

    Args:
        raker: A fitted OnlineRakingSGD or OnlineRakingMWU object.
        max_weight_ratio: Maximum allowable weight ratio for compromise targets.

    Returns:
        InfeasibilityAnalysis with diagnosis and compromise solutions.

    Examples:
        >>> analysis = analyze_infeasibility(raker)
        >>> if not analysis.is_feasible:
        ...     print("Original targets infeasible")
        ...     print(f"Reason: {analysis.infeasibility_type}")
        ...     print(f"Suggested compromise: {analysis.compromise_targets}")
    """
    if raker._n_obs == 0:
        return InfeasibilityAnalysis(
            is_feasible=False,
            infeasibility_type="no_data",
            achievable_bounds={},
            compromise_targets={},
            diagnosis=["No observations available for analysis."],
        )

    diagnosis: list[str] = []
    achievable_bounds: dict[str, tuple[float, float]] = {}
    compromise_targets: dict[str, float] = {}
    infeasibility_types: list[str] = []

    raw_margins = raker.raw_margins
    n_obs = raker._n_obs

    for feature in raker._feature_names:
        target = raker.targets[feature]
        raw = raw_margins[feature]
        is_binary = raker.targets.is_binary(feature)
        feature_idx = raker._feature_names.index(feature)
        feature_values = raker._features[:n_obs, feature_idx]

        if is_binary:
            # Binary feature analysis
            feature_sum = feature_values.sum()
            n_with_feature = int(feature_sum)
            n_without_feature = n_obs - n_with_feature

            # Compute achievable bounds
            # Lower bound: give all weight to observations without feature
            # Upper bound: give all weight to observations with feature
            if n_without_feature == 0:
                min_achievable = 1.0
            else:
                min_achievable = 0.0

            if n_with_feature == 0:
                max_achievable = 0.0
            else:
                max_achievable = 1.0

            achievable_bounds[feature] = (min_achievable, max_achievable)

            # Check if target is achievable
            if target < min_achievable:
                diagnosis.append(
                    f"'{feature}': Target {target:.1%} below achievable minimum "
                    f"{min_achievable:.1%} (all observations have feature=1)."
                )
                infeasibility_types.append("structural")
                compromise_targets[feature] = min_achievable
            elif target > max_achievable:
                diagnosis.append(
                    f"'{feature}': Target {target:.1%} above achievable maximum "
                    f"{max_achievable:.1%} (no observations have feature=1)."
                )
                infeasibility_types.append("structural")
                compromise_targets[feature] = max_achievable
            else:
                # Target is achievable in principle
                # Check if it requires extreme weights
                weight_ratio_needed = _estimate_required_weight_ratio(
                    raw, target, n_with_feature, n_without_feature
                )

                if weight_ratio_needed > EXTREME_WEIGHT_RATIO:
                    diagnosis.append(
                        f"'{feature}': Achieving target {target:.1%} from raw {raw:.1%} "
                        f"requires ~{weight_ratio_needed:.0f}:1 weight ratio."
                    )
                    infeasibility_types.append("numerical")
                    compromise_targets[feature] = _compute_achievable_with_ratio(
                        raw,
                        target,
                        n_with_feature,
                        n_without_feature,
                        max_ratio=max_weight_ratio,
                    )
                else:
                    compromise_targets[feature] = target
        else:
            # Continuous feature analysis
            min_val = float(feature_values.min())
            max_val = float(feature_values.max())

            achievable_bounds[feature] = (min_val, max_val)

            # Check if target is within sample range
            if target < min_val:
                diagnosis.append(
                    f"'{feature}': Target mean {target:.2f} below sample minimum "
                    f"{min_val:.2f}."
                )
                infeasibility_types.append("structural")
                compromise_targets[feature] = min_val
            elif target > max_val:
                diagnosis.append(
                    f"'{feature}': Target mean {target:.2f} above sample maximum "
                    f"{max_val:.2f}."
                )
                infeasibility_types.append("structural")
                compromise_targets[feature] = max_val
            else:
                # Target is achievable
                compromise_targets[feature] = target

    # Determine overall infeasibility type
    if not diagnosis:
        infeasibility_type = "feasible"
        is_feasible = True
    elif "structural" in infeasibility_types:
        infeasibility_type = "structural"
        is_feasible = False
    elif "numerical" in infeasibility_types:
        infeasibility_type = "numerical"
        is_feasible = False
    else:
        infeasibility_type = "unknown"
        is_feasible = False

    return InfeasibilityAnalysis(
        is_feasible=is_feasible,
        infeasibility_type=infeasibility_type,
        achievable_bounds=achievable_bounds,
        compromise_targets=compromise_targets,
        diagnosis=diagnosis,
    )


def _estimate_required_weight_ratio(
    raw: float,
    target: float,
    n_with: int,
    n_without: int,
) -> float:
    """Estimate the weight ratio required to move from raw to target margin."""
    if n_with == 0 or n_without == 0:
        return float("inf")

    if raw == target:
        return 1.0

    # Simplified model: w_1 for feature=1, w_0 for feature=0
    # Target = (n_with * w_1) / (n_with * w_1 + n_without * w_0)
    # Assuming w_0 = 1 (reference), solve for w_1:
    # target * (n_with * w_1 + n_without) = n_with * w_1
    # target * n_without = n_with * w_1 * (1 - target)
    # w_1 = target * n_without / (n_with * (1 - target))

    if target == 0:
        return 0.0
    if target == 1:
        return float("inf")

    w_1 = target * n_without / (n_with * (1 - target))

    # If w_1 > 1, ratio is w_1:1
    # If w_1 < 1, ratio is 1:1/w_1
    if w_1 > 1:
        return w_1
    else:
        return 1.0 / w_1 if w_1 > 0 else float("inf")


def _compute_achievable_with_ratio(
    raw: float,
    target: float,
    n_with: int,
    n_without: int,
    max_ratio: float,
) -> float:
    """Compute the closest achievable target given a maximum weight ratio."""
    if n_with == 0:
        return 0.0
    if n_without == 0:
        return 1.0

    # Two cases: need to increase or decrease margin
    if target > raw:
        # Need to increase: max weight on feature=1
        w_1 = max_ratio
        achievable = (n_with * w_1) / (n_with * w_1 + n_without)
        return min(target, achievable)
    else:
        # Need to decrease: max weight on feature=0
        w_0 = max_ratio
        achievable = n_with / (n_with + n_without * w_0)
        return max(target, achievable)


def suggest_feasible_targets(
    raker: OnlineRakingSGD,
    max_weight_ratio: float = 100.0,
) -> dict[str, float]:
    """Suggest feasible targets based on observed data and weight constraints.

    Given the observed data distribution and constraints on weight variability,
    this function suggests the closest achievable targets to the original
    targets.

    Args:
        raker: A fitted OnlineRakingSGD or OnlineRakingMWU object.
        max_weight_ratio: Maximum allowable weight ratio (default 100:1).

    Returns:
        Dictionary of suggested feasible targets for each feature.

    Examples:
        >>> # Original targets may be infeasible
        >>> targets = Targets(feature=0.90)  # Very extreme
        >>> raker = OnlineRakingSGD(targets)
        >>> for obs in data:
        ...     raker.partial_fit(obs)
        >>>
        >>> # Get suggested feasible targets
        >>> feasible = suggest_feasible_targets(raker, max_weight_ratio=50)
        >>> print(f"Original: 0.90, Feasible: {feasible['feature']:.2f}")
    """
    analysis = analyze_infeasibility(raker, max_weight_ratio=max_weight_ratio)
    return analysis.compromise_targets


def explain_infeasibility_causes() -> dict[str, str]:
    """Return explanations of different types of target infeasibility.

    Returns:
        Dictionary mapping infeasibility types to explanations.
    """
    return {
        "structural": (
            "STRUCTURAL INFEASIBILITY: The target proportion is mathematically "
            "impossible given the observed data. For example, if no observations "
            "have feature=1, you cannot achieve a target >0 for that feature. "
            "Solution: Adjust targets to be within achievable bounds, or collect "
            "more diverse data."
        ),
        "numerical": (
            "NUMERICAL INFEASIBILITY: The target is theoretically achievable but "
            "would require extreme weights (e.g., 1000:1 ratio). This leads to "
            "very low effective sample size and unstable estimates. "
            "Solution: Relax targets toward observed proportions, or accept "
            "lower weight efficiency."
        ),
        "conflicting": (
            "CONFLICTING TARGETS: Multiple targets are individually achievable but "
            "cannot be satisfied simultaneously. This occurs when features are "
            "correlated and targets imply impossible joint distributions. "
            "Solution: Use iterative methods like IPF that find compromise solutions, "
            "or adjust targets to be mutually consistent."
        ),
        "convergence": (
            "CONVERGENCE ISSUES: The algorithm is not converging toward targets, "
            "possibly due to learning rate issues or oscillation. "
            "Solution: Adjust learning rate, use diminishing schedules, or "
            "increase the number of gradient steps per observation."
        ),
    }


@dataclass
class IPFComparison:
    """Comparison between streaming raker and batch IPF solutions.

    Attributes:
        weight_kl: KL divergence D_KL(w_raker || w_ipf).
        weight_tv: Total variation distance between weight distributions.
        margin_mse: Mean squared error between margins.
        margin_max_diff: Maximum absolute difference in margins.
        ess_ratio: ESS_raker / ESS_ipf ratio.
        raker_loss: Final loss of the streaming raker.
        ipf_loss: Final loss of batch IPF.
    """

    weight_kl: float
    weight_tv: float
    margin_mse: float
    margin_max_diff: float
    ess_ratio: float
    raker_loss: float
    ipf_loss: float


def compare_to_ipf(
    raker: OnlineRakingSGD,
    ipf: BatchIPF | None = None,
) -> IPFComparison:
    """Compare streaming raker solution to batch IPF.

    This function quantifies how close a streaming raker (SGD or MWU) is
    to the batch IPF solution. The key insight is that MWU performs mirror
    descent with KL divergence, and should converge to IPF as the learning
    rate decreases.

    Args:
        raker: Fitted streaming raker (OnlineRakingSGD or OnlineRakingMWU).
        ipf: Pre-fitted BatchIPF object. If None, fits IPF on the raker's
            data (requires accessing internal state).

    Returns:
        IPFComparison with comparison metrics:
        - weight_kl: D_KL(w_raker || w_ipf) - how different the weights are
        - weight_tv: Total variation distance between normalized weights
        - margin_mse: MSE between weighted margins
        - margin_max_diff: Worst-case margin difference
        - ess_ratio: Relative effective sample size

    Raises:
        ValueError: If raker has no observations or if IPF fitting fails.

    Examples:
        >>> from onlinerake import OnlineRakingMWU, BatchIPF, Targets
        >>> targets = Targets(female=0.51, college=0.32)
        >>> mwu = OnlineRakingMWU(targets, learning_rate=0.5)
        >>> for obs in data:
        ...     mwu.partial_fit(obs)
        >>> ipf = BatchIPF(targets).fit(data)
        >>> comparison = compare_to_ipf(mwu, ipf)
        >>> print(f"Weight KL from IPF: {comparison.weight_kl:.6f}")
        >>> print(f"Margin MSE: {comparison.margin_mse:.6f}")

    Note:
        For MWU to closely match IPF:
        1. Use small learning rates (η < 1.0)
        2. Process many observations (n > 100)
        3. Use multiple SGD steps per observation (n_sgd_steps >= 3)

        MWU with η → 0 should produce D_KL → 0 and margin_mse → 0.
    """
    from .batch_ipf import BatchIPF
    from .divergence import kl_divergence_weights, total_variation_weights

    if raker._n_obs == 0:
        raise ValueError("Raker has no observations; cannot compare to IPF")

    # Fit IPF if not provided
    if ipf is None:
        # Reconstruct observations from internal state
        ipf = BatchIPF(raker.targets)
        observations = []
        for i in range(raker._n_obs):
            obs = {}
            for j, name in enumerate(raker._feature_names):
                obs[name] = int(raker._features[i, j])
            observations.append(obs)
        ipf.fit(observations)

    # Ensure same number of observations
    if raker._n_obs != ipf._n_obs:
        raise ValueError(
            f"Raker has {raker._n_obs} observations but IPF has {ipf._n_obs}. "
            "Ensure IPF was fit on the same data."
        )

    # Get weights
    w_raker = raker.weights
    w_ipf = ipf.weights

    # Compute divergence metrics
    weight_kl = kl_divergence_weights(w_raker, w_ipf)
    weight_tv = total_variation_weights(w_raker, w_ipf)

    # Compute margin comparison
    margins_raker = raker.margins
    margins_ipf = ipf.margins

    margin_diffs = []
    for name in raker._feature_names:
        diff = margins_raker[name] - margins_ipf[name]
        margin_diffs.append(diff)

    margin_mse = float(np.mean([d**2 for d in margin_diffs]))
    margin_max_diff = float(np.max([abs(d) for d in margin_diffs]))

    # Compute ESS ratio
    ess_raker = raker.effective_sample_size
    ess_ipf = ipf.effective_sample_size
    ess_ratio = ess_raker / ess_ipf if ess_ipf > 0 else np.nan

    return IPFComparison(
        weight_kl=weight_kl,
        weight_tv=weight_tv,
        margin_mse=margin_mse,
        margin_max_diff=margin_max_diff,
        ess_ratio=ess_ratio,
        raker_loss=raker.loss,
        ipf_loss=ipf.loss,
    )


def optimal_mwu_learning_rate(n_obs: int, n_features: int) -> float:
    """Compute theoretical optimal learning rate for MWU to approximate IPF.

    From mirror descent theory, the optimal learning rate is approximately:
        η* ≈ sqrt(2 * log(n_obs) / T)

    where T is the expected number of iterations. For streaming raking with
    n_sgd_steps per observation, T ≈ n_obs * n_sgd_steps.

    A smaller learning rate means MWU stays closer to IPF at each step,
    but convergence is slower. This function provides a reasonable starting
    point for tuning.

    Args:
        n_obs: Expected number of observations.
        n_features: Number of features being calibrated.

    Returns:
        Recommended learning rate for MWU.

    Examples:
        >>> lr = optimal_mwu_learning_rate(n_obs=1000, n_features=4)
        >>> print(f"Recommended learning rate: {lr:.3f}")

    Note:
        This is a theoretical guideline. In practice:
        - For IPF-matching: use lr < 0.5
        - For faster convergence: use lr 1.0-5.0
        - Monitor loss and adjust as needed
    """
    if n_obs <= 1:
        return 1.0

    # From mirror descent regret bounds:
    # η* = sqrt(2 * D / (T * G^2))
    # where D is diameter of domain (log(n_obs) for simplex)
    # and G is gradient bound (depends on n_features)

    # Simplified heuristic that works well empirically
    log_n = np.log(n_obs)
    eta = np.sqrt(2 * log_n / n_obs)

    # Scale by features (more features = more complex optimization)
    eta *= np.sqrt(n_features)

    # Clamp to reasonable range
    return float(np.clip(eta, 0.01, 5.0))
