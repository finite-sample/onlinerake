"""Learning-rate series checks and empirical calibration diagnostics.

Schedule summability alone does not establish convergence of a growing,
clipped, path-dependent calibration problem. No numerical loss or regret
certificate is supplied for these online estimators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ._utils import requires_observations, validate_count, validate_positive

if TYPE_CHECKING:
    from .learning_rate import LearningRateSchedule
    from .online_raking_sgd import OnlineRakingSGD


# Module-level constants for magic numbers
LIPSCHITZ_PERTURBATION_SCALE = 0.1
WEIGHT_RATIO_WARN = 100
WEIGHT_RATIO_FAIL = 1000
ESS_WARN_THRESHOLD = 0.5
ESS_FAIL_THRESHOLD = 0.2
LOW_WEIGHT_EFFICIENCY_THRESHOLD = 0.3


@dataclass
class ConvergenceAnalysis:
    """Results of convergence analysis for a raking algorithm.

    Attributes:
        satisfies_robbins_monro: Whether learning rate satisfies RM conditions.
        lipschitz_constant: Estimated Lipschitz constant of the loss gradient.
        convergence_rate: Theoretical convergence rate (if applicable).
        expected_iterations: Rough extrapolation from recent loss, or None.
            Changes in incoming data can invalidate it.
        warnings: List of potential issues with convergence.
    """

    satisfies_robbins_monro: bool
    lipschitz_constant: float
    convergence_rate: str
    expected_iterations: int | None
    warnings: list[str]


@dataclass
class RobbinsMonroVerification:
    """Verification of Robbins-Monro conditions.

    The Robbins-Monro conditions for stochastic approximation convergence are:
        1. Σ η_t = ∞  (sum of step sizes diverges)
        2. Σ η_t² < ∞ (sum of squared step sizes converges)

    Attributes:
        condition_1_satisfied: Whether Σ η_t = ∞ is satisfied.
        condition_2_satisfied: Whether Σ η_t² < ∞ is satisfied.
        sum_lr_estimate: Estimated sum of learning rates over ``n_steps`` steps.
        sum_lr_sq_estimate: Estimated sum of squared learning rates.
        n_steps_evaluated: Number of steps used for evaluation.
        analysis_notes: Detailed notes on the verification.
    """

    condition_1_satisfied: bool
    condition_2_satisfied: bool
    sum_lr_estimate: float
    sum_lr_sq_estimate: float
    n_steps_evaluated: int
    analysis_notes: list[str]


def verify_robbins_monro(
    schedule: LearningRateSchedule | float,
    n_steps: int = 100000,
) -> RobbinsMonroVerification:
    """Verify Robbins-Monro conditions for a learning rate schedule.

    Uses analytical verification for known schedule types (ConstantLR,
    InverseTimeDecayLR, PolynomialDecayLR). Falls back to numerical
    estimation for custom schedules.

    For convergence guarantees in stochastic approximation, the learning
    rate sequence {η_t} must satisfy:
        1. Σ_{t=1}^∞ η_t = ∞
        2. Σ_{t=1}^∞ η_t² < ∞

    Condition 1 ensures we can eventually reach any point in the space.
    Condition 2 ensures the noise averages out and we converge.

    Args:
        schedule: Learning rate schedule or constant learning rate.
        n_steps: Number of steps to evaluate (for computing sum estimates).

    Returns:
        RobbinsMonroVerification with detailed analysis.

    Examples:
        >>> from onlinerake.learning_rate import PolynomialDecayLR
        >>> schedule = PolynomialDecayLR(initial_lr=5.0, power=0.6)
        >>> result = verify_robbins_monro(schedule)
        >>> print(result.condition_1_satisfied)  # Should be True
        True
        >>> print(result.condition_2_satisfied)  # Should be True
        True
    """
    validate_count(n_steps, "n_steps")
    notes: list[str] = []

    # Handle constant float
    if isinstance(schedule, (int, float)):
        return _verify_constant(float(schedule), n_steps, notes)

    # Analytical verification for known types
    params = schedule.get_params()
    if (
        params.get("type") in {"polynomial_decay", "inverse_time_decay"}
        and params.get("min_lr", 0) > 0
    ):
        rates = np.array([schedule(t) for t in range(1, n_steps + 1)])
        return RobbinsMonroVerification(
            condition_1_satisfied=True,
            condition_2_satisfied=False,
            sum_lr_estimate=float(rates.sum()),
            sum_lr_sq_estimate=float((rates**2).sum()),
            n_steps_evaluated=n_steps,
            analysis_notes=[
                "A positive learning-rate floor makes the squared-rate sum diverge."
            ],
        )
    schedule_type = params.get("type", "unknown")

    if schedule_type == "constant":
        return _verify_constant(params["learning_rate"], n_steps, notes)

    if schedule_type == "polynomial_decay":
        return _verify_polynomial(params, n_steps, notes)

    if schedule_type == "inverse_time_decay":
        return _verify_inverse_time(params, n_steps, notes)

    # Unknown schedule: numerical estimation with disclaimer
    return _verify_numerical_fallback(schedule, n_steps, notes)


def _verify_constant(
    lr: float, n_steps: int, notes: list[str]
) -> RobbinsMonroVerification:
    """Constant LR: sum diverges (✓), squared sum diverges (✗)."""
    validate_positive(lr, "learning_rate")
    notes.append(f"Constant learning rate: η = {lr}")
    notes.append("")
    notes.append("Condition 1 (Σ η_t = ∞): SATISFIED")
    notes.append(f"  Sum = η·T = {lr}·T → ∞ as T → ∞")
    notes.append("")
    notes.append("Condition 2 (Σ η_t² < ∞): NOT SATISFIED")
    notes.append(f"  Sum = η²·T = {lr**2}·T → ∞ as T → ∞")
    notes.append("")
    notes.append("CONCLUSION: Constant learning rates do NOT satisfy Robbins-Monro.")
    notes.append("Algorithm may oscillate around optimum without converging.")

    return RobbinsMonroVerification(
        condition_1_satisfied=True,
        condition_2_satisfied=False,
        sum_lr_estimate=lr * n_steps,
        sum_lr_sq_estimate=lr * lr * n_steps,
        n_steps_evaluated=n_steps,
        analysis_notes=notes,
    )


def _verify_polynomial(
    params: dict[str, Any], n_steps: int, notes: list[str]
) -> RobbinsMonroVerification:
    """Polynomial decay η_t = η₀/t^α: satisfies RM iff 0.5 < α ≤ 1."""
    power = params["power"]
    initial_lr = params["initial_lr"]

    notes.append(f"Polynomial decay: η_t = {initial_lr}/t^{power}")
    notes.append("")

    # Condition 1: Σ 1/t^α diverges iff α ≤ 1
    cond1 = power <= 1.0
    if cond1:
        notes.append("Condition 1 (Σ η_t = ∞): SATISFIED")
        notes.append(f"  Σ 1/t^{power} diverges because {power} ≤ 1")
    else:
        notes.append("Condition 1 (Σ η_t = ∞): NOT SATISFIED")
        notes.append(f"  Σ 1/t^{power} converges because {power} > 1")

    notes.append("")

    # Condition 2: Σ 1/t^(2α) converges iff 2α > 1, i.e., α > 0.5
    cond2 = power > 0.5
    if cond2:
        notes.append("Condition 2 (Σ η_t² < ∞): SATISFIED")
        notes.append(f"  Σ 1/t^{2 * power} converges because {2 * power} > 1")
    else:
        notes.append("Condition 2 (Σ η_t² < ∞): NOT SATISFIED")
        notes.append(f"  Σ 1/t^{2 * power} diverges because {2 * power} ≤ 1")

    notes.append("")
    if cond1 and cond2:
        notes.append("CONCLUSION: Robbins-Monro conditions satisfied.")
        notes.append(
            "These series conditions alone do not guarantee raker convergence."
        )
    else:
        notes.append("CONCLUSION: Robbins-Monro conditions NOT satisfied.")
        notes.append(f"Required: 0.5 < power ≤ 1. Current: power = {power}")

    # Compute actual sums for reference
    ts = np.arange(1, n_steps + 1)
    lrs = initial_lr / (ts**power)

    return RobbinsMonroVerification(
        condition_1_satisfied=cond1,
        condition_2_satisfied=cond2,
        sum_lr_estimate=float(np.sum(lrs)),
        sum_lr_sq_estimate=float(np.sum(lrs**2)),
        n_steps_evaluated=n_steps,
        analysis_notes=notes,
    )


def _verify_inverse_time(
    params: dict[str, Any], n_steps: int, notes: list[str]
) -> RobbinsMonroVerification:
    """Inverse time decay η_t = η₀/(1 + decay·t): satisfies RM iff decay > 0."""
    decay = params["decay"]
    initial_lr = params["initial_lr"]

    notes.append(f"Inverse time decay: η_t = {initial_lr}/(1 + {decay}·t)")
    notes.append("")

    # Condition 1: sum always diverges (constant if decay=0, harmonic if decay>0)
    cond1 = True
    # Condition 2: squared sum converges iff decay > 0
    cond2 = decay > 0

    if cond2:
        notes.append("Condition 1 (Σ η_t = ∞): SATISFIED")
        notes.append("  Behaves like harmonic series Σ 1/t → ∞")
    else:
        notes.append("Condition 1 (Σ η_t = ∞): SATISFIED (trivially, constant)")

    notes.append("")

    if cond2:
        notes.append("Condition 2 (Σ η_t² < ∞): SATISFIED")
        notes.append("  Behaves like Σ 1/t² which converges")
    else:
        notes.append("Condition 2 (Σ η_t² < ∞): NOT SATISFIED")
        notes.append("  With decay=0, this is a constant learning rate")

    notes.append("")
    if cond1 and cond2:
        notes.append("CONCLUSION: Robbins-Monro conditions satisfied.")
    else:
        notes.append(
            "CONCLUSION: With decay=0, this is a constant LR (RM not satisfied)."
        )

    ts = np.arange(1, n_steps + 1)
    lrs = initial_lr / (1.0 + decay * ts)

    return RobbinsMonroVerification(
        condition_1_satisfied=cond1,
        condition_2_satisfied=cond2,
        sum_lr_estimate=float(np.sum(lrs)),
        sum_lr_sq_estimate=float(np.sum(lrs**2)),
        n_steps_evaluated=n_steps,
        analysis_notes=notes,
    )


def _verify_numerical_fallback(
    schedule: LearningRateSchedule, n_steps: int, notes: list[str]
) -> RobbinsMonroVerification:
    """Numerical estimation for unknown schedule types."""
    notes.append("Unknown schedule type - using numerical estimation")
    notes.append("NOTE: Results are heuristic, not rigorous proofs")
    notes.append("")

    # Evaluate schedule at many points
    lrs = np.array([schedule(t) for t in range(1, n_steps + 1)])
    sum_lr = np.sum(lrs)
    sum_lr_sq = np.sum(lrs**2)

    # Check condition 1: sum should be large (approaching infinity)
    sum_lr_half = np.sum(lrs[: n_steps // 2])
    growth_ratio = sum_lr / sum_lr_half if sum_lr_half > 0 else 0

    # For divergent series, doubling T should give more than 1.1x the sum
    condition_1 = bool(growth_ratio > 1.1)

    notes.append(f"Evaluated over n_steps = {n_steps} steps")
    notes.append(f"Sum of learning rates: {sum_lr:.4f}")
    notes.append(f"Growth ratio (full/half): {growth_ratio:.4f}")

    if condition_1:
        notes.append("Condition 1 (Σ η_t = ∞): LIKELY SATISFIED (heuristic)")
    else:
        notes.append("Condition 1 (Σ η_t = ∞): LIKELY NOT SATISFIED (heuristic)")
        notes.append("  Sum appears to converge or grow slowly.")

    # Check condition 2: sum of squares should converge (be bounded)
    sum_lr_sq_half = np.sum(lrs[: n_steps // 2] ** 2)
    sq_growth_ratio = sum_lr_sq / sum_lr_sq_half if sum_lr_sq_half > 0 else 0

    # For convergent series, doubling T should give < 1.1x the sum
    condition_2 = bool(sq_growth_ratio < 1.1)

    notes.append(f"Sum of squared learning rates: {sum_lr_sq:.6f}")
    notes.append(f"Squared sum growth ratio: {sq_growth_ratio:.4f}")

    if condition_2:
        notes.append("Condition 2 (Σ η_t² < ∞): LIKELY SATISFIED (heuristic)")
    else:
        notes.append("Condition 2 (Σ η_t² < ∞): LIKELY NOT SATISFIED (heuristic)")
        notes.append("  Sum of squares appears to diverge.")

    # Overall assessment
    notes.append("")
    if condition_1 and condition_2:
        notes.append(
            "CONCLUSION: Robbins-Monro conditions LIKELY satisfied (heuristic)."
        )
        notes.append("For rigorous verification, use a known schedule type.")
    elif condition_1 and not condition_2:
        notes.append("WARNING: Only Condition 1 likely satisfied.")
        notes.append("Algorithm may oscillate around optimum without converging.")
    else:
        notes.append("WARNING: Convergence conditions likely not satisfied.")
        notes.append("Consider using a different learning rate schedule.")

    return RobbinsMonroVerification(
        condition_1_satisfied=condition_1,
        condition_2_satisfied=condition_2,
        sum_lr_estimate=float(sum_lr),
        sum_lr_sq_estimate=float(sum_lr_sq),
        n_steps_evaluated=n_steps,
        analysis_notes=notes,
    )


@requires_observations(lambda: np.nan)
def estimate_lipschitz_constant(
    raker: OnlineRakingSGD,
    n_samples: int = 1000,
    seed: int = 42,
) -> float:
    """Estimate the Lipschitz constant of the loss gradient.

    The Lipschitz constant L bounds how fast the gradient can change:
        ||∇f(w) - ∇f(w')|| ≤ L ||w - w'||

    The sampled ratios are local diagnostics, not a global upper bound on L
    or a certificate that any learning rate is safe.

    For the squared-error margin loss used in raking:
        L(w) = Σ_j (m_j(w) - t_j)²

    where m_j(w) = Σ_i w_i x_{ij} / Σ_i w_i

    Args:
        raker: A fitted OnlineRakingSGD or OnlineRakingMWU object.
        n_samples: Number of weight perturbations to test.
        seed: Random seed for reproducibility.

    Returns:
        Estimated Lipschitz constant L.
    """
    rng = np.random.default_rng(seed)

    # Get current weights and gradient
    w = raker._weights[: raker._n_obs].copy()
    grad_current = raker._compute_gradient()

    # Estimate Lipschitz constant by sampling perturbations
    max_ratio = 0.0

    for _ in range(n_samples):
        # Create small perturbation
        delta = rng.standard_normal(len(w)) * LIPSCHITZ_PERTURBATION_SCALE
        w_perturbed = np.clip(w + delta, raker.min_weight, raker.max_weight)

        # Compute gradient at perturbed point
        try:
            raker._weights[: raker._n_obs] = w_perturbed
            grad_perturbed = raker._compute_gradient()
        finally:
            raker._weights[: raker._n_obs] = w

        # Compute ratio
        grad_diff = np.linalg.norm(grad_perturbed - grad_current)
        w_diff = np.linalg.norm(w_perturbed - w)

        if w_diff > 1e-10:
            ratio = grad_diff / w_diff
            max_ratio = max(max_ratio, float(ratio))

    return float(max_ratio)


def analyze_convergence(
    raker: OnlineRakingSGD,
    loss_tolerance: float = 1e-6,
) -> ConvergenceAnalysis:
    """Perform comprehensive convergence analysis.

    Analyzes the theoretical convergence properties of a raking algorithm
    based on its configuration and current state.

    Args:
        raker: A fitted OnlineRakingSGD or OnlineRakingMWU object.
        loss_tolerance: Loss value at or below which the raker counts as
            converged. This is a squared-error loss, not a margin distance --
            see :func:`~onlinerake.diagnostics.check_target_feasibility` for the
            latter, which lives on a completely different scale.

    Returns:
        ConvergenceAnalysis with detailed theoretical assessment.
    """
    warnings: list[str] = []

    # Check Robbins-Monro conditions
    if hasattr(raker, "_lr_schedule") and raker._lr_schedule is not None:
        rm_result = verify_robbins_monro(raker._lr_schedule)
        satisfies_rm = (
            rm_result.condition_1_satisfied and rm_result.condition_2_satisfied
        )
        if not satisfies_rm:
            warnings.extend(
                [
                    note
                    for note in rm_result.analysis_notes
                    if note.startswith("WARNING")
                ]
            )
    else:
        rm_result = verify_robbins_monro(raker.learning_rate)
        satisfies_rm = False
        warnings.append(
            "Using constant learning rate. Consider a diminishing schedule "
            "when decaying adaptation is appropriate; this does not guarantee balance."
        )

    # Estimate Lipschitz constant
    lipschitz = estimate_lipschitz_constant(raker)

    # Check if learning rate is appropriate
    if (
        not np.isnan(lipschitz)
        and lipschitz > 0
        and raker.learning_rate > 1.0 / lipschitz
    ):
        warnings.append(
            f"Learning rate {raker.learning_rate:.4f} exceeds "
            f"1/L = {1.0 / lipschitz:.4f}. "
            "This may cause oscillation or divergence."
        )

    convergence_rate = "Not established for the growing, clipped calibration problem"
    warnings.append(
        "Robbins-Monro summability and sampled curvature are diagnostics, "
        "not a convergence or entropy-balancing certificate."
    )

    # Estimate iterations to convergence
    if raker._n_obs > 0 and raker.loss > loss_tolerance:
        current_loss = raker.loss
        # Very rough estimate based on observed rate
        if len(raker._loss_history) >= 10:
            recent_losses = raker._loss_history[-10:]
            if recent_losses[0] > recent_losses[-1] > 0:
                rate = (recent_losses[0] / recent_losses[-1]) ** (1 / 10)
                if rate > 1.01:
                    iterations_needed = int(
                        np.log(loss_tolerance / current_loss) / np.log(1 / rate)
                    )
                    expected_iterations = max(0, iterations_needed)
                else:
                    expected_iterations = None
            else:
                expected_iterations = None
        else:
            expected_iterations = None
    else:
        expected_iterations = 0 if raker.loss <= loss_tolerance else None

    # Additional warnings
    if raker._n_obs > 0:
        ess_ratio = raker.effective_sample_size / raker._n_obs
        if ess_ratio < LOW_WEIGHT_EFFICIENCY_THRESHOLD:
            warnings.append(
                f"Low weight efficiency ({ess_ratio:.1%}). "
                "This may indicate target feasibility issues."
            )

        if raker.detect_oscillation():
            warnings.append("Oscillation detected. Consider reducing learning rate.")

    return ConvergenceAnalysis(
        satisfies_robbins_monro=satisfies_rm,
        lipschitz_constant=lipschitz,
        convergence_rate=convergence_rate,
        expected_iterations=expected_iterations,
        warnings=warnings,
    )


def verify_convergence_conditions(
    raker: OnlineRakingSGD,
) -> dict[str, Any]:
    """Report empirical warnings and learning-rate series checks.

    Checks rate summability, sampled local curvature, weight spread, effective
    sample size, and loss variation. PASS means no heuristic warning fired;
    it is not proof of convergence or target feasibility.

    Args:
        raker: A fitted OnlineRakingSGD or OnlineRakingMWU object.

    Returns:
        Dictionary with verification results and recommendations.
    """
    results: dict[str, Any] = {
        "overall_status": "PASS",
        "checks": {},
        "recommendations": [],
    }

    # 1. Check learning rate conditions
    if hasattr(raker, "_lr_schedule") and raker._lr_schedule is not None:
        rm_check = verify_robbins_monro(raker._lr_schedule, n_steps=10000)
        results["checks"]["robbins_monro"] = {
            "status": (
                "PASS"
                if (rm_check.condition_1_satisfied and rm_check.condition_2_satisfied)
                else "FAIL"
            ),
            "condition_1": rm_check.condition_1_satisfied,
            "condition_2": rm_check.condition_2_satisfied,
        }
        if not (rm_check.condition_1_satisfied and rm_check.condition_2_satisfied):
            results["overall_status"] = "WARN"
            results["recommendations"].append(
                "Learning rate schedule does not satisfy Robbins-Monro conditions. "
                "Consider using PolynomialDecayLR with power in (0.5, 1]."
            )
    else:
        results["checks"]["robbins_monro"] = {
            "status": "WARN",
            "note": "Using constant learning rate (RM not satisfied)",
        }
        results["overall_status"] = "WARN"
        results["recommendations"].append(
            "No convergence guarantee follows from the learning-rate choice. "
            "Monitor actual margin errors and weight bounds."
        )

    # 2. Check Lipschitz constant and step size
    if raker._n_obs > 0:
        lipschitz = estimate_lipschitz_constant(raker, n_samples=100)
        if not np.isnan(lipschitz):
            inverse_local_curvature = 1.0 / lipschitz if lipschitz > 0 else np.inf
            current_lr = raker.current_learning_rate

            results["checks"]["lipschitz"] = {
                "status": "PASS" if current_lr <= inverse_local_curvature else "WARN",
                "lipschitz_constant": lipschitz,
                "inverse_local_curvature": inverse_local_curvature,
                "current_lr": current_lr,
            }

            if current_lr > inverse_local_curvature:
                results["recommendations"].append(
                    f"Learning rate {current_lr:.4f} exceeds inverse sampled curvature "
                    f"{inverse_local_curvature:.4f}. This may cause oscillation. "
                    "Consider reducing initial learning rate."
                )
        else:
            results["checks"]["lipschitz"] = {
                "status": "N/A",
                "note": "Could not estimate",
            }

    # 3. Check weight distribution stability
    if raker._n_obs > 0:
        weight_stats = raker.weight_distribution_stats
        weight_ratio = weight_stats["max"] / weight_stats["min"]

        results["checks"]["weight_stability"] = {
            "status": (
                "PASS"
                if weight_ratio < WEIGHT_RATIO_WARN
                else ("WARN" if weight_ratio < WEIGHT_RATIO_FAIL else "FAIL")
            ),
            "weight_ratio": weight_ratio,
            "min_weight": weight_stats["min"],
            "max_weight": weight_stats["max"],
        }

        if weight_ratio > WEIGHT_RATIO_WARN:
            results["recommendations"].append(
                f"Large weight ratio ({weight_ratio:.1f}:1) detected. "
                "This may indicate target feasibility issues or unstable convergence."
            )
            if weight_ratio > WEIGHT_RATIO_FAIL:
                results["overall_status"] = "FAIL"

    # 4. Check effective sample size
    if raker._n_obs > 0:
        ess_ratio = raker.effective_sample_size / raker._n_obs
        results["checks"]["ess_efficiency"] = {
            "status": (
                "PASS"
                if ess_ratio > ESS_WARN_THRESHOLD
                else ("WARN" if ess_ratio > ESS_FAIL_THRESHOLD else "FAIL")
            ),
            "ess": raker.effective_sample_size,
            "n_obs": raker._n_obs,
            "efficiency": ess_ratio,
        }

        if ess_ratio < ESS_WARN_THRESHOLD:
            results["recommendations"].append(
                f"Low weight efficiency ({ess_ratio:.1%}). "
                "Effective sample size is significantly reduced by weighting."
            )
            if ess_ratio < ESS_FAIL_THRESHOLD:
                results["overall_status"] = "FAIL"

    # 5. Check for oscillation
    if raker._n_obs > raker.convergence_window:
        oscillating = raker.detect_oscillation()
        results["checks"]["oscillation"] = {
            "status": "FAIL" if oscillating else "PASS",
            "oscillating": oscillating,
        }

        if oscillating:
            results["overall_status"] = "FAIL"
            results["recommendations"].append(
                "Oscillation detected in loss. Reduce learning rate or use a "
                "diminishing schedule."
            )

    return results
