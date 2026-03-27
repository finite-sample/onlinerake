#!/usr/bin/env python3
"""Ad Targeting Calibration: Real-time demographic reweighting for ad delivery.

This example demonstrates streaming weight calibration for ad targeting,
where ad impressions need to be reweighted to match target demographics
in real-time as user data streams in.

Use Case:
---------
An ad campaign targets a specific demographic mix (e.g., 40% age 18-34,
60% female). The actual ad delivery may not match these targets due to
algorithmic bias, user availability, or inventory constraints. Online
raking adjusts impression weights in real-time to ensure reporting and
attribution reflect the target audience, not the delivered audience.

This is a streaming problem because:
1. Ad impressions arrive continuously (millions per hour)
2. Campaign adjustments need real-time feedback
3. Cannot wait for batch processing at end of day
4. Target demographics may shift during campaign

Example output shows:
- How weights evolve as impressions stream in
- Comparison of raw vs weighted demographics
- Convergence behavior and effective sample size
"""

import numpy as np

from onlinerake import (
    BatchIPF,
    OnlineRakingMWU,
    OnlineRakingSGD,
    Targets,
)
from onlinerake.convergence import analyze_convergence, verify_robbins_monro
from onlinerake.diagnostics import summarize_raking_results
from onlinerake.learning_rate import robbins_monro_schedule
from onlinerake.streaming_inference import (
    analyze_estimate_stability,
    compute_confidence_sequence,
)


def simulate_ad_impressions(
    n_impressions: int = 5000,
    seed: int = 42,
) -> list[dict[str, int]]:
    """Simulate a stream of ad impressions with realistic biases.

    Real ad delivery often has systematic biases:
    - Younger users browse more → over-represented
    - Mobile skews certain demographics
    - Time-of-day affects who sees ads

    Args:
        n_impressions: Number of ad impressions to simulate.
        seed: Random seed for reproducibility.

    Returns:
        List of impression records with user attributes.
    """
    np.random.seed(seed)

    impressions = []
    for i in range(n_impressions):
        # Simulate biases in ad delivery
        # Age: actual delivery skews young (60% vs target 40%)
        is_young = 1 if np.random.random() < 0.60 else 0

        # Gender: actual delivery skews male (55% vs target 40%)
        is_male = 1 if np.random.random() < 0.55 else 0

        # Income: high-income users over-represented (40% vs target 25%)
        is_high_income = 1 if np.random.random() < 0.40 else 0

        # Mobile: mobile users more likely during commute hours
        hour_of_day = i % 24
        is_mobile = (
            1
            if np.random.random()
            < (0.7 if 7 <= hour_of_day <= 9 or 17 <= hour_of_day <= 19 else 0.5)
            else 0
        )

        # Engagement signal (correlated with demographics)
        clicked = (
            1
            if (is_young * 0.08 + (1 - is_young) * 0.04 + np.random.random() * 0.02)
            > 0.05
            else 0
        )

        impressions.append(
            {
                "is_young": is_young,  # Age 18-34
                "is_male": is_male,
                "is_high_income": is_high_income,
                "is_mobile": is_mobile,
                "clicked": clicked,
            }
        )

    return impressions


def run_ad_calibration_example() -> None:
    """Run the ad targeting calibration example."""
    print("=" * 70)
    print("AD TARGETING CALIBRATION: Real-time Demographic Reweighting")
    print("=" * 70)

    # Campaign target demographics
    # These represent what the advertiser wants to reach
    targets = Targets(
        is_young=0.40,  # 40% age 18-34 (target)
        is_male=0.40,  # 40% male (target)
        is_high_income=0.25,  # 25% high income (target)
        is_mobile=0.55,  # 55% mobile (target)
    )

    print("\n📊 Campaign Target Demographics:")
    for feature, target in targets.as_dict().items():
        print(f"   {feature}: {target:.0%}")

    # Simulate biased ad impressions
    print("\n🎯 Simulating 5,000 ad impressions with realistic delivery biases...")
    impressions = simulate_ad_impressions(n_impressions=5000)

    # Compute raw (unweighted) delivery demographics
    raw_demographics = {
        feature: sum(imp[feature] for imp in impressions) / len(impressions)
        for feature in ["is_young", "is_male", "is_high_income", "is_mobile"]
    }

    print("\n❌ Raw (Biased) Delivery Demographics:")
    for feature, value in raw_demographics.items():
        target = targets[feature]
        bias = value - target
        print(f"   {feature}: {value:.1%} (target: {target:.0%}, bias: {bias:+.1%})")

    # METHOD 1: Online SGD with fixed learning rate
    print("\n" + "-" * 70)
    print("METHOD 1: Online SGD (Fixed Learning Rate)")
    print("-" * 70)

    sgd_raker = OnlineRakingSGD(
        targets,
        learning_rate=3.0,
        track_convergence=True,
        convergence_window=50,
    )

    # Process impressions
    for i, imp in enumerate(impressions):
        sgd_raker.partial_fit(imp)

        # Progress updates
        if (i + 1) % 1000 == 0:
            print(
                f"   Processed {i + 1:,} impressions | "
                f"Loss: {sgd_raker.loss:.6f} | "
                f"ESS: {sgd_raker.effective_sample_size:.1f} | "
                f"Converged: {sgd_raker.converged}"
            )

    sgd_margins = sgd_raker.margins
    print("\n✅ SGD Weighted Demographics:")
    for feature in ["is_young", "is_male", "is_high_income", "is_mobile"]:
        target = targets[feature]
        weighted = sgd_margins[feature]
        error = abs(weighted - target)
        print(
            f"   {feature}: {weighted:.1%} (target: {target:.0%}, error: {error:.2%})"
        )

    # METHOD 2: Online SGD with Robbins-Monro schedule (theoretical guarantees)
    print("\n" + "-" * 70)
    print("METHOD 2: Online SGD (Robbins-Monro Schedule)")
    print("-" * 70)

    schedule = robbins_monro_schedule(initial_lr=5.0, power=0.6, min_lr=0.1)

    # Verify convergence conditions
    rm_check = verify_robbins_monro(schedule)
    print(
        f"   Robbins-Monro Conditions: Satisfied = {rm_check.condition_1_satisfied and rm_check.condition_2_satisfied}"
    )

    sgd_rm_raker = OnlineRakingSGD(
        targets,
        learning_rate=schedule,
        track_convergence=True,
    )

    for imp in impressions:
        sgd_rm_raker.partial_fit(imp)

    print(f"   Final Learning Rate: {sgd_rm_raker.current_learning_rate:.4f}")
    print(f"   Final Loss: {sgd_rm_raker.loss:.6f}")
    print(f"   ESS: {sgd_rm_raker.effective_sample_size:.1f}")

    # METHOD 3: MWU (Multiplicative Weights Update)
    print("\n" + "-" * 70)
    print("METHOD 3: Online MWU (Mirror Descent)")
    print("-" * 70)

    mwu_raker = OnlineRakingMWU(
        targets,
        learning_rate=1.0,
        track_convergence=True,
    )

    for imp in impressions:
        mwu_raker.partial_fit(imp)

    print(f"   Final Loss: {mwu_raker.loss:.6f}")
    print(f"   ESS: {mwu_raker.effective_sample_size:.1f}")

    mwu_margins = mwu_raker.margins
    print("\n✅ MWU Weighted Demographics:")
    for feature in ["is_young", "is_male", "is_high_income", "is_mobile"]:
        target = targets[feature]
        weighted = mwu_margins[feature]
        error = abs(weighted - target)
        print(
            f"   {feature}: {weighted:.1%} (target: {target:.0%}, error: {error:.2%})"
        )

    # METHOD 4: Batch IPF (baseline for comparison)
    print("\n" + "-" * 70)
    print("METHOD 4: Batch IPF (Gold Standard Baseline)")
    print("-" * 70)

    ipf = BatchIPF(targets)
    ipf.fit(impressions)

    print(f"   Iterations: {ipf.n_iterations}")
    print(f"   Converged: {ipf.converged}")
    print(f"   Final Loss: {ipf.loss:.6f}")
    print(f"   ESS: {ipf.effective_sample_size:.1f}")

    # Compare all methods
    print("\n" + "=" * 70)
    print("COMPARISON: Online vs Batch Methods")
    print("=" * 70)

    print(f"\n{'Method':<25} {'Loss':>10} {'ESS':>10} {'Efficiency':>12}")
    print("-" * 57)
    print(
        f"{'SGD (fixed LR)':<25} {sgd_raker.loss:>10.6f} {sgd_raker.effective_sample_size:>10.1f} {sgd_raker.effective_sample_size / 5000:>12.1%}"
    )
    print(
        f"{'SGD (Robbins-Monro)':<25} {sgd_rm_raker.loss:>10.6f} {sgd_rm_raker.effective_sample_size:>10.1f} {sgd_rm_raker.effective_sample_size / 5000:>12.1%}"
    )
    print(
        f"{'MWU':<25} {mwu_raker.loss:>10.6f} {mwu_raker.effective_sample_size:>10.1f} {mwu_raker.effective_sample_size / 5000:>12.1%}"
    )
    print(
        f"{'Batch IPF':<25} {ipf.loss:>10.6f} {ipf.effective_sample_size:>10.1f} {ipf.effective_sample_size / 5000:>12.1%}"
    )

    # Demonstrate streaming inference features
    print("\n" + "=" * 70)
    print("ADVANCED: Streaming Inference Analysis")
    print("=" * 70)

    # Analyze convergence
    convergence = analyze_convergence(sgd_raker)
    print("\n📈 Convergence Analysis (SGD):")
    print(f"   Satisfies Robbins-Monro: {convergence.satisfies_robbins_monro}")
    print(f"   Lipschitz Constant: {convergence.lipschitz_constant:.4f}")
    print(f"   Convergence Rate: {convergence.convergence_rate}")
    if convergence.warnings:
        print("   Warnings:")
        for w in convergence.warnings:
            print(f"      - {w}")

    # Confidence sequence
    conf_seq = compute_confidence_sequence(sgd_raker, "is_young", confidence_level=0.95)
    if conf_seq.times:
        final_lower = conf_seq.lower_bounds[-1]
        final_upper = conf_seq.upper_bounds[-1]
        print("\n🎯 Confidence Sequence (is_young, 95%):")
        print(f"   Final Interval: [{final_lower:.3f}, {final_upper:.3f}]")
        print(f"   Width: {final_upper - final_lower:.3f}")
        print(
            f"   Target ({targets['is_young']:.2f}) in interval: {final_lower <= targets['is_young'] <= final_upper}"
        )

    # Estimate stability
    stability = analyze_estimate_stability(sgd_raker, window=100)
    if stability["status"] != "INSUFFICIENT_DATA":
        print("\n📊 Estimate Stability (last 100 impressions):")
        print(f"   Overall Stability: {stability['overall_stability']:.2f}")
        print(f"   Status: {stability['status']}")

    # Comprehensive summary
    print("\n" + "-" * 70)
    print("Full Raking Summary (SGD):")
    print("-" * 70)
    summary = summarize_raking_results(sgd_raker)

    print(f"\nObservations: {summary['n_observations']:,}")
    print(f"Effective Sample Size: {summary['effective_sample_size']:.1f}")
    print(f"Design Effect: {summary['design_effect']:.2f}")
    print(f"Weight Efficiency: {summary['weight_efficiency']:.1%}")

    print("\nMargin Estimates:")
    for est in summary["margin_estimates"]:
        print(
            f"   {est['feature']}: {est['estimate']:.3f} ± {est['std_error']:.3f} "
            f"(target: {est['target']:.2f}, bias reduction: {est['bias_reduction_pct']:.1f}%)"
        )

    print("\n" + "=" * 70)
    print("KEY INSIGHTS FOR AD TARGETING:")
    print("=" * 70)
    print("""
1. STREAMING WORKS: Online raking achieves similar loss to batch IPF
   while processing impressions one at a time.

2. CONVERGENCE: With proper learning rate schedules (Robbins-Monro),
   we have theoretical convergence guarantees.

3. WEIGHT EFFICIENCY: Check ESS/n ratio - if too low (<50%), the
   correction is too aggressive and variance increases substantially.

4. REAL-TIME MONITORING: Use confidence sequences for anytime-valid
   inference without p-hacking concerns.

5. ALGORITHM CHOICE:
   - SGD: Faster convergence, may oscillate
   - MWU: More stable weights, connects to IPF
   - Choose based on your variance-bias tradeoff needs
""")


if __name__ == "__main__":
    run_ad_calibration_example()
