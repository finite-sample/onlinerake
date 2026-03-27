"""Sensitivity analysis for online raking hyperparameters.

This module provides tools to systematically evaluate how the raking
algorithms perform under different hyperparameter settings. This addresses
the "garden of forking paths" criticism by making hyperparameter choices
explicit and justified.

Key hyperparameters analyzed:
- Learning rate (η): How aggressively to update weights
- n_sgd_steps (K): Number of gradient updates per observation
- Weight bounds (ε, M): Constraints on weight range
- Convergence tolerance: When to stop iterating
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .online_raking_sgd import OnlineRakingSGD
    from .targets import Targets


@dataclass
class SensitivityResult:
    """Results from a single sensitivity analysis run.

    Attributes:
        params: Hyperparameter values used.
        final_loss: Loss at end of stream.
        final_ess: Effective sample size at end.
        convergence_step: When convergence was detected (None if not converged).
        margin_errors: Per-feature absolute errors from target.
        mean_margin_error: Average absolute error across features.
        weight_efficiency: ESS / n (higher is better).
        oscillation_detected: Whether oscillation was detected.
    """

    params: dict[str, Any]
    final_loss: float
    final_ess: float
    convergence_step: int | None
    margin_errors: dict[str, float]
    mean_margin_error: float
    weight_efficiency: float
    oscillation_detected: bool


@dataclass
class SensitivityReport:
    """Comprehensive sensitivity analysis report.

    Attributes:
        results: List of results for each parameter combination.
        best_params: Parameters that achieved lowest loss.
        param_importance: Estimated importance of each parameter.
        recommendations: Suggested parameter values.
    """

    results: list[SensitivityResult]
    best_params: dict[str, Any]
    param_importance: dict[str, float]
    recommendations: list[str]


def run_sensitivity_analysis(
    observations: list[dict[str, Any]],
    targets: Targets,
    learning_rates: list[float] | None = None,
    n_steps_values: list[int] | None = None,
    min_weights: list[float] | None = None,
    max_weights: list[float] | None = None,
    algorithm: str = "sgd",
    seeds: list[int] | None = None,
) -> SensitivityReport:
    """Run systematic sensitivity analysis over hyperparameter grid.

    This function evaluates the raking algorithms across different
    hyperparameter combinations to identify robust settings and
    understand parameter sensitivity.

    Args:
        observations: List of observations to process.
        targets: Target proportions (Targets object).
        learning_rates: Learning rates to test. Default: [0.5, 1.0, 2.0, 5.0, 10.0]
        n_steps_values: SGD steps per observation to test. Default: [1, 3, 5]
        min_weights: Minimum weight bounds to test. Default: [1e-4, 1e-3, 1e-2]
        max_weights: Maximum weight bounds to test. Default: [10, 100, 1000]
        algorithm: "sgd" or "mwu". Default: "sgd"
        seeds: Random seeds for multiple runs. Default: [42]

    Returns:
        SensitivityReport with all results and recommendations.

    Examples:
        >>> from onlinerake import Targets
        >>> targets = Targets(age=0.4, gender=0.5)
        >>> observations = [{'age': 1, 'gender': 0}, ...]
        >>> report = run_sensitivity_analysis(observations, targets)
        >>> print(report.best_params)
        >>> print(report.recommendations)
    """
    from .online_raking_mwu import OnlineRakingMWU
    from .online_raking_sgd import OnlineRakingSGD

    # Default parameter grids
    if learning_rates is None:
        learning_rates = [0.5, 1.0, 2.0, 5.0, 10.0]
    if n_steps_values is None:
        n_steps_values = [1, 3, 5]
    if min_weights is None:
        min_weights = [1e-4, 1e-3, 1e-2]
    if max_weights is None:
        max_weights = [10.0, 100.0, 1000.0]
    if seeds is None:
        seeds = [42]

    results: list[SensitivityResult] = []

    # Grid search
    for lr in learning_rates:
        for n_steps in n_steps_values:
            for min_w in min_weights:
                for max_w in max_weights:
                    if max_w <= min_w:
                        continue

                    for seed in seeds:
                        # Set seed for reproducibility
                        np.random.seed(seed)

                        # Create raker
                        if algorithm.lower() == "mwu":
                            raker = OnlineRakingMWU(
                                targets,
                                learning_rate=lr,
                                n_steps=n_steps,
                                min_weight=min_w,
                                max_weight=max_w,
                            )
                        else:
                            raker = OnlineRakingSGD(
                                targets,
                                learning_rate=lr,
                                n_sgd_steps=n_steps,
                                min_weight=min_w,
                                max_weight=max_w,
                            )

                        # Process observations
                        for obs in observations:
                            raker.partial_fit(obs)

                        # Collect results
                        margins = raker.margins
                        margin_errors = {
                            name: abs(margins[name] - targets[name])
                            for name in raker._feature_names
                        }

                        result = SensitivityResult(
                            params={
                                "learning_rate": lr,
                                "n_steps": n_steps,
                                "min_weight": min_w,
                                "max_weight": max_w,
                                "seed": seed,
                            },
                            final_loss=raker.loss,
                            final_ess=raker.effective_sample_size,
                            convergence_step=raker.convergence_step,
                            margin_errors=margin_errors,
                            mean_margin_error=float(
                                np.mean(list(margin_errors.values()))
                            ),
                            weight_efficiency=raker.effective_sample_size
                            / raker._n_obs,
                            oscillation_detected=raker.detect_oscillation(),
                        )
                        results.append(result)

    # Find best parameters
    if not results:
        raise ValueError(
            "No valid parameter combinations found. "
            "Check that min_weight < max_weight and parameter ranges are valid."
        )
    best_result = min(results, key=lambda r: r.final_loss)
    best_params = best_result.params

    # Estimate parameter importance using variance analysis
    param_importance = _estimate_param_importance(results)

    # Generate recommendations
    recommendations = _generate_recommendations(results, best_params, param_importance)

    return SensitivityReport(
        results=results,
        best_params=best_params,
        param_importance=param_importance,
        recommendations=recommendations,
    )


def _estimate_param_importance(results: list[SensitivityResult]) -> dict[str, float]:
    """Estimate relative importance of each parameter.

    Uses variance decomposition to estimate how much each parameter
    contributes to variation in the loss.
    """
    if len(results) < 2:
        return {}

    losses = np.array([r.final_loss for r in results])
    total_var = np.var(losses)

    if total_var == 0:
        return {
            "learning_rate": 0.0,
            "n_steps": 0.0,
            "min_weight": 0.0,
            "max_weight": 0.0,
        }

    importance = {}

    # For each parameter, compute variance explained
    for param in ["learning_rate", "n_steps", "min_weight", "max_weight"]:
        # Group results by parameter value
        param_groups: dict[Any, list[float]] = {}
        for r in results:
            val = r.params[param]
            if val not in param_groups:
                param_groups[val] = []
            param_groups[val].append(r.final_loss)

        # Compute between-group variance
        group_means = [np.mean(g) for g in param_groups.values()]
        between_var = np.var(group_means) if len(group_means) > 1 else 0

        importance[param] = float(between_var / total_var) if total_var > 0 else 0

    # Normalize to sum to 1
    total_importance = sum(importance.values())
    if total_importance > 0:
        importance = {k: v / total_importance for k, v in importance.items()}

    return importance


def _generate_recommendations(
    results: list[SensitivityResult],
    best_params: dict[str, Any],
    param_importance: dict[str, float],
) -> list[str]:
    """Generate actionable recommendations based on sensitivity analysis."""
    recommendations = []

    # Recommend best parameters
    recommendations.append(
        f"Best learning rate: {best_params['learning_rate']} "
        f"(importance: {param_importance.get('learning_rate', 0):.1%})"
    )
    recommendations.append(
        f"Best n_steps: {best_params['n_steps']} "
        f"(importance: {param_importance.get('n_steps', 0):.1%})"
    )

    # Check for oscillation at high learning rates
    high_lr_results = [r for r in results if r.params["learning_rate"] >= 5.0]
    if high_lr_results:
        oscillation_rate = sum(
            1 for r in high_lr_results if r.oscillation_detected
        ) / len(high_lr_results)
        if oscillation_rate > 0.3:
            recommendations.append(
                f"Warning: {oscillation_rate:.0%} of high learning rate runs showed oscillation. "
                "Consider using a lower learning rate or diminishing schedule."
            )

    # Check weight efficiency
    efficiencies = [r.weight_efficiency for r in results]
    if np.mean(efficiencies) < 0.5:
        recommendations.append(
            "Low weight efficiency detected. Consider relaxing targets or "
            "increasing sample size."
        )

    # Check for extreme weights indicating feasibility issues
    for r in results:
        if r.params == best_params:
            if r.weight_efficiency < 0.3:
                recommendations.append(
                    "Even best parameters show low weight efficiency. "
                    "Review target feasibility."
                )
            break

    return recommendations


def quick_sensitivity_check(
    raker: OnlineRakingSGD,
    observations: list[dict[str, Any]],
    n_variations: int = 5,
) -> dict[str, Any]:
    """Quick sensitivity check around current parameters.

    Tests small variations around the raker's current settings to
    assess local sensitivity.

    Args:
        raker: A configured (but not fitted) raker to use as baseline.
        observations: Observations to test with.
        n_variations: Number of variations to test per parameter.

    Returns:
        Dictionary with sensitivity metrics.
    """
    base_lr = raker.learning_rate
    base_n_steps = raker.n_sgd_steps

    # Create variations
    lr_range = [base_lr * f for f in [0.5, 0.75, 1.0, 1.25, 1.5]][:n_variations]
    n_steps_range = list(range(max(1, base_n_steps - 2), base_n_steps + 3))[
        :n_variations
    ]

    # Run sensitivity analysis with focused ranges
    report = run_sensitivity_analysis(
        observations,
        raker.targets,
        learning_rates=lr_range,
        n_steps_values=n_steps_range,
        min_weights=[raker.min_weight],
        max_weights=[raker.max_weight],
    )

    return {
        "baseline_lr": base_lr,
        "baseline_n_steps": base_n_steps,
        "best_lr": report.best_params["learning_rate"],
        "best_n_steps": report.best_params["n_steps"],
        "lr_sensitivity": report.param_importance.get("learning_rate", 0),
        "n_steps_sensitivity": report.param_importance.get("n_steps", 0),
        "recommendations": report.recommendations,
    }
