#!/usr/bin/env python3
"""A/B Test Calibration: Real-time covariate balancing for experiment analysis.

This example demonstrates streaming weight calibration for A/B tests,
where treatment and control groups may have covariate imbalances that
emerge during the experiment.

Use Case:
---------
An A/B test comparing two recommendation algorithms has random assignment,
but due to natural variation or technical issues, the treatment group ends
up with different user characteristics than control. Online raking provides
real-time covariate adjustment to ensure valid causal inference.

This is a streaming problem because:
1. Users enter the experiment continuously
2. Need real-time monitoring of balance metrics
3. Can't wait for experiment end to detect imbalance
4. May want to stop early if treatment effect is clear

Key insight: Unlike post-hoc adjustment, streaming raking allows
continuous monitoring of covariate balance during the experiment.

Example output shows:
- Covariate balance before and after weighting
- Treatment effect estimation with and without adjustment
- Confidence sequences for real-time decision making
"""

import numpy as np

from onlinerake import OnlineRakingSGD, Targets
from onlinerake.diagnostics import compute_design_effect
from onlinerake.streaming_inference import (
    compute_confidence_sequence,
)


def simulate_ab_test(
    n_users: int = 2000,
    true_treatment_effect: float = 0.03,  # 3 percentage point lift
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Simulate an A/B test with covariate imbalance.

    Simulates treatment and control groups where:
    - Random assignment has slight imbalance due to chance
    - There's a covariate (power users) that affects outcome
    - Treatment effect exists but is confounded by imbalance

    Args:
        n_users: Total users (split 50/50).
        true_treatment_effect: True causal effect of treatment.
        seed: Random seed.

    Returns:
        Tuple of (treatment_group, control_group) observation lists.
    """
    np.random.seed(seed)

    n_treatment = n_users // 2
    n_control = n_users - n_treatment

    # Population characteristics
    baseline_conversion = 0.10  # 10% baseline conversion
    power_user_effect = 0.05  # Power users convert 5pp more

    treatment_group = []
    control_group = []

    # Treatment group - accidentally got more power users (55% vs 50%)
    for _ in range(n_treatment):
        is_power_user = 1 if np.random.random() < 0.55 else 0  # IMBALANCE
        is_mobile = 1 if np.random.random() < 0.50 else 0
        is_new_user = 1 if np.random.random() < 0.30 else 0

        # Conversion depends on user type + treatment effect
        conversion_prob = (
            baseline_conversion
            + is_power_user * power_user_effect
            + true_treatment_effect  # Treatment effect
        )
        converted = 1 if np.random.random() < conversion_prob else 0

        treatment_group.append(
            {
                "is_power_user": is_power_user,
                "is_mobile": is_mobile,
                "is_new_user": is_new_user,
                "converted": converted,
                "in_treatment": 1,
            }
        )

    # Control group - 50% power users (balanced)
    for _ in range(n_control):
        is_power_user = 1 if np.random.random() < 0.50 else 0
        is_mobile = 1 if np.random.random() < 0.50 else 0
        is_new_user = 1 if np.random.random() < 0.30 else 0

        # Conversion depends on user type only (no treatment)
        conversion_prob = baseline_conversion + is_power_user * power_user_effect
        converted = 1 if np.random.random() < conversion_prob else 0

        control_group.append(
            {
                "is_power_user": is_power_user,
                "is_mobile": is_mobile,
                "is_new_user": is_new_user,
                "converted": converted,
                "in_treatment": 0,
            }
        )

    return treatment_group, control_group


def compute_treatment_effect(
    treatment: list[dict],
    control: list[dict],
    treatment_weights: np.ndarray | None = None,
    control_weights: np.ndarray | None = None,
) -> dict:
    """Compute treatment effect with optional weighting.

    Args:
        treatment: Treatment group observations.
        control: Control group observations.
        treatment_weights: Weights for treatment group (None = equal).
        control_weights: Weights for control group (None = equal).

    Returns:
        Dictionary with effect estimates.
    """
    if treatment_weights is None:
        treatment_weights = np.ones(len(treatment))
    if control_weights is None:
        control_weights = np.ones(len(control))

    # Normalize weights
    treatment_weights = treatment_weights / treatment_weights.sum()
    control_weights = control_weights / control_weights.sum()

    # Weighted conversion rates
    treatment_rate = sum(
        w * obs["converted"]
        for w, obs in zip(treatment_weights, treatment, strict=False)
    )
    control_rate = sum(
        w * obs["converted"] for w, obs in zip(control_weights, control, strict=False)
    )

    return {
        "treatment_rate": treatment_rate,
        "control_rate": control_rate,
        "absolute_effect": treatment_rate - control_rate,
        "relative_effect": (treatment_rate - control_rate) / control_rate
        if control_rate > 0
        else np.nan,
    }


def run_ab_test_example() -> None:
    """Run the A/B test calibration example."""
    print("=" * 70)
    print("A/B TEST CALIBRATION: Real-time Covariate Balancing")
    print("=" * 70)

    # True treatment effect for comparison
    TRUE_EFFECT = 0.03  # 3 percentage points

    # Simulate experiment
    print("\n🧪 Simulating A/B test with 2,000 users...")
    print(f"   True treatment effect: {TRUE_EFFECT:.1%}")
    treatment, control = simulate_ab_test(
        n_users=2000,
        true_treatment_effect=TRUE_EFFECT,
    )

    print(f"   Treatment group: {len(treatment)} users")
    print(f"   Control group: {len(control)} users")

    # Check raw covariate balance
    print("\n" + "-" * 70)
    print("COVARIATE BALANCE (Before Weighting)")
    print("-" * 70)

    for covariate in ["is_power_user", "is_mobile", "is_new_user"]:
        treatment_prop = np.mean([u[covariate] for u in treatment])
        control_prop = np.mean([u[covariate] for u in control])
        imbalance = treatment_prop - control_prop

        status = "⚠️  IMBALANCED" if abs(imbalance) > 0.03 else "✓ balanced"
        print(
            f"   {covariate}: Treatment={treatment_prop:.1%}, "
            f"Control={control_prop:.1%}, Δ={imbalance:+.1%} {status}"
        )

    # Raw treatment effect (biased due to imbalance)
    raw_effect = compute_treatment_effect(treatment, control)
    print("\n📊 Raw Treatment Effect (potentially biased):")
    print(f"   Treatment conversion: {raw_effect['treatment_rate']:.1%}")
    print(f"   Control conversion: {raw_effect['control_rate']:.1%}")
    print(f"   Absolute effect: {raw_effect['absolute_effect']:.2%}")
    print(f"   True effect: {TRUE_EFFECT:.1%}")
    print(f"   Bias: {raw_effect['absolute_effect'] - TRUE_EFFECT:+.2%}")

    # Define target proportions (control group as reference)
    # We want treatment group covariates to match control group
    control_covariates = {
        "is_power_user": np.mean([u["is_power_user"] for u in control]),
        "is_mobile": np.mean([u["is_mobile"] for u in control]),
        "is_new_user": np.mean([u["is_new_user"] for u in control]),
    }

    print("\n" + "-" * 70)
    print("ONLINE RAKING: Reweight Treatment to Match Control")
    print("-" * 70)

    targets = Targets(**control_covariates)
    print("\n📌 Target proportions (from control group):")
    for feat, val in control_covariates.items():
        print(f"   {feat}: {val:.1%}")

    # Create streaming estimator with snapshot tracking
    raker = OnlineRakingSGD(
        targets,
        learning_rate=5.0,
        track_convergence=True,
    )

    # Process treatment group observations as a stream
    print("\n🔄 Processing treatment group as stream...")
    for i, obs in enumerate(treatment):
        # Only use covariates for raking, not outcome
        raker.partial_fit(
            {
                "is_power_user": obs["is_power_user"],
                "is_mobile": obs["is_mobile"],
                "is_new_user": obs["is_new_user"],
            }
        )

        # Progress updates
        if (i + 1) % 250 == 0:
            print(
                f"   Users: {i + 1:,} | Loss: {raker.loss:.6f} | "
                f"ESS: {raker.effective_sample_size:.1f}"
            )

    # Check weighted covariate balance
    print("\n" + "-" * 70)
    print("COVARIATE BALANCE (After Weighting)")
    print("-" * 70)

    weighted_margins = raker.margins
    for covariate in ["is_power_user", "is_mobile", "is_new_user"]:
        control_prop = control_covariates[covariate]
        weighted_prop = weighted_margins[covariate]
        imbalance = weighted_prop - control_prop

        status = "⚠️  IMBALANCED" if abs(imbalance) > 0.03 else "✓ balanced"
        print(
            f"   {covariate}: Treatment(weighted)={weighted_prop:.1%}, "
            f"Control={control_prop:.1%}, Δ={imbalance:+.1%} {status}"
        )

    # Weighted treatment effect
    weighted_effect = compute_treatment_effect(
        treatment,
        control,
        treatment_weights=raker.weights,
        control_weights=None,  # Control uses equal weights
    )

    print("\n📊 Weighted Treatment Effect (bias-corrected):")
    print(
        f"   Treatment conversion (weighted): {weighted_effect['treatment_rate']:.1%}"
    )
    print(f"   Control conversion: {weighted_effect['control_rate']:.1%}")
    print(f"   Absolute effect: {weighted_effect['absolute_effect']:.2%}")
    print(f"   True effect: {TRUE_EFFECT:.1%}")
    print(f"   Bias: {weighted_effect['absolute_effect'] - TRUE_EFFECT:+.2%}")

    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON: Raw vs Weighted Analysis")
    print("=" * 70)

    raw_bias = abs(raw_effect["absolute_effect"] - TRUE_EFFECT)
    weighted_bias = abs(weighted_effect["absolute_effect"] - TRUE_EFFECT)
    improvement = (raw_bias - weighted_bias) / raw_bias * 100 if raw_bias > 0 else 0

    print(f"\n{'Metric':<30} {'Raw':>12} {'Weighted':>12} {'True':>12}")
    print("-" * 66)
    print(
        f"{'Treatment Effect':<30} {raw_effect['absolute_effect']:>12.2%} {weighted_effect['absolute_effect']:>12.2%} {TRUE_EFFECT:>12.1%}"
    )
    print(
        f"{'Bias from True Effect':<30} {raw_bias:>12.2%} {weighted_bias:>12.2%} {'-':>12}"
    )
    print(f"\n   Bias reduction: {improvement:.0f}%")

    # Diagnostics
    print("\n" + "-" * 70)
    print("DIAGNOSTICS")
    print("-" * 70)

    deff = compute_design_effect(raker)
    print("\n📈 Weighting Statistics:")
    print(f"   Design Effect: {deff:.2f}")
    print(f"   Weight Efficiency: {raker.effective_sample_size / len(treatment):.1%}")
    print(
        f"   Effective Sample Size: {raker.effective_sample_size:.1f} (of {len(treatment)})"
    )

    weight_stats = raker.weight_distribution_stats
    print("\n📊 Weight Distribution:")
    print(f"   Min: {weight_stats['min']:.4f}")
    print(f"   Max: {weight_stats['max']:.4f}")
    print(f"   Median: {weight_stats['median']:.4f}")
    print(f"   Max/Min Ratio: {weight_stats['max'] / weight_stats['min']:.1f}")

    # Confidence sequence for treatment effect monitoring
    print("\n" + "-" * 70)
    print("STREAMING INFERENCE: Real-time Monitoring")
    print("-" * 70)

    conf_seq = compute_confidence_sequence(
        raker, "is_power_user", confidence_level=0.95
    )
    if conf_seq.times:
        print("\n🎯 Confidence Sequence (is_power_user balance):")
        print(
            f"   Final 95% CI: [{conf_seq.lower_bounds[-1]:.3f}, {conf_seq.upper_bounds[-1]:.3f}]"
        )
        target_in_ci = (
            conf_seq.lower_bounds[-1]
            <= targets["is_power_user"]
            <= conf_seq.upper_bounds[-1]
        )
        print(f"   Target in CI: {target_in_ci}")

    print("\n" + "=" * 70)
    print("KEY INSIGHTS FOR A/B TEST CALIBRATION:")
    print("=" * 70)
    print("""
1. IMBALANCE DETECTION: Stream processing reveals covariate imbalance
   in real-time, not just at experiment end.

2. BIAS CORRECTION: Weighting treatment group to match control covariates
   removes confounding and gives unbiased treatment effect estimates.

3. VARIANCE TRADEOFF: Weighting increases variance (design effect > 1).
   Check that effective sample size is still adequate.

4. EARLY STOPPING: Confidence sequences enable valid early stopping
   without inflating Type I error from repeated testing.

5. WHEN TO USE:
   - Unexpected covariate imbalance during experiment
   - Non-random assignment (observational studies)
   - Combining data across time periods with different populations

6. CAUTION:
   - Only balances OBSERVED covariates
   - Assumes no unmeasured confounding
   - Large weights may indicate insufficient overlap
""")


if __name__ == "__main__":
    run_ab_test_example()
