#!/usr/bin/env python3
"""MWU vs IPF comparison with KL divergence tracking.

This example demonstrates the key features for comparing streaming MWU
to batch IPF:

1. KL divergence tracking between consecutive weight updates
2. Comparing streaming MWU to batch IPF solution
3. Learning rate tuning for IPF-matching
4. Understanding when MWU approximates IPF well

Background:
    MWU (Multiplicative Weights Update) performs mirror descent with KL
    divergence as the regularizer. As learning rate decreases, MWU converges
    to the same solution as batch IPF, which minimizes D_KL(w || w_0) subject
    to margin constraints.

Usage:
    python examples/kl_ipf_comparison.py
"""

from __future__ import annotations

import numpy as np

from onlinerake import (
    BatchIPF,
    OnlineRakingMWU,
    Targets,
    compare_to_ipf,
    kl_divergence_weights,
    optimal_mwu_learning_rate,
    symmetric_kl_divergence,
    total_variation_weights,
)


def generate_biased_sample(
    n: int,
    targets: Targets,
    bias: float = 0.1,
    seed: int = 42,
) -> list[dict[str, int]]:
    """Generate sample with systematic bias from targets."""
    rng = np.random.default_rng(seed)
    observations = []

    for _ in range(n):
        obs = {}
        for name in targets.feature_names:
            target = targets[name]
            biased_prob = min(1.0, max(0.0, target + bias))
            obs[name] = int(rng.random() < biased_prob)
        observations.append(obs)

    return observations


def main():
    print("=" * 70)
    print("MWU vs IPF Comparison with KL Divergence Tracking")
    print("=" * 70)
    print()

    # Setup targets
    targets = Targets(
        female=0.51,
        college=0.32,
        urban=0.65,
    )

    print("Target Proportions:")
    for feature in targets.feature_names:
        print(f"  {feature}: {targets[feature]:.0%}")
    print()

    # Generate biased sample
    n_obs = 500
    observations = generate_biased_sample(n_obs, targets, bias=0.08)

    raw_props = {
        f: sum(obs[f] for obs in observations) / len(observations)
        for f in targets.feature_names
    }
    print(f"Sample (n={n_obs}) Raw Margins:")
    for feature in targets.feature_names:
        raw = raw_props[feature]
        target = targets[feature]
        print(
            f"  {feature}: {raw:.1%} (target: {target:.0%}, bias: {raw - target:+.1%})"
        )
    print()

    # =========================================================================
    # Section 1: MWU with KL Divergence Tracking
    # =========================================================================
    print("-" * 70)
    print("Section 1: MWU with KL Divergence Tracking")
    print("-" * 70)
    print()

    mwu = OnlineRakingMWU(
        targets,
        learning_rate=0.3,
        n_sgd_steps=5,
        track_kl_divergence=True,
    )

    for obs in observations:
        mwu.partial_fit(obs)

    print("MWU Results:")
    print(f"  Final loss: {mwu.loss:.6f}")
    print(f"  ESS: {mwu.effective_sample_size:.1f}")
    print()

    kl_history = mwu.kl_divergence_history
    print("KL Divergence History (between consecutive updates):")
    print(f"  Total updates tracked: {len(kl_history)}")
    print(f"  First 5 KL values: {[f'{kl:.6f}' for kl in kl_history[:5]]}")
    print(f"  Last 5 KL values: {[f'{kl:.6f}' for kl in kl_history[-5:]]}")
    print(f"  Cumulative KL: {mwu.cumulative_kl_divergence:.4f}")
    print(f"  Average KL per update: {np.mean(kl_history):.6f}")
    print()

    # =========================================================================
    # Section 2: Batch IPF for Comparison
    # =========================================================================
    print("-" * 70)
    print("Section 2: Batch IPF Reference")
    print("-" * 70)
    print()

    ipf = BatchIPF(targets).fit(observations)

    print("IPF Results:")
    print(f"  Final loss: {ipf.loss:.6f}")
    print(f"  ESS: {ipf.effective_sample_size:.1f}")
    print(f"  Iterations: {ipf.n_iterations}")
    print()

    # =========================================================================
    # Section 3: compare_to_ipf() Function
    # =========================================================================
    print("-" * 70)
    print("Section 3: MWU vs IPF Comparison (compare_to_ipf)")
    print("-" * 70)
    print()

    comparison = compare_to_ipf(mwu, ipf)

    print("Comparison Metrics:")
    print(f"  Weight KL divergence (MWU from IPF): {comparison.weight_kl:.6f}")
    print(f"  Weight TV distance: {comparison.weight_tv:.6f}")
    print(f"  Margin MSE: {comparison.margin_mse:.6f}")
    print(f"  Max margin difference: {comparison.margin_max_diff:.4f}")
    print(f"  ESS ratio (MWU/IPF): {comparison.ess_ratio:.3f}")
    print(f"  MWU loss: {comparison.raker_loss:.6f}")
    print(f"  IPF loss: {comparison.ipf_loss:.6f}")
    print()

    print("Margin Comparison:")
    print(f"  {'Feature':<12} {'MWU':>10} {'IPF':>10} {'Diff':>10}")
    print("  " + "-" * 44)
    for feature in targets.feature_names:
        mwu_m = mwu.margins[feature]
        ipf_m = ipf.margins[feature]
        diff = mwu_m - ipf_m
        print(f"  {feature:<12} {mwu_m:>10.4f} {ipf_m:>10.4f} {diff:>+10.4f}")
    print()

    # =========================================================================
    # Section 4: Divergence Functions Demo
    # =========================================================================
    print("-" * 70)
    print("Section 4: Direct Divergence Computation")
    print("-" * 70)
    print()

    mwu_weights = mwu.weights
    ipf_weights = ipf.weights

    kl_mwu_ipf = kl_divergence_weights(mwu_weights, ipf_weights)
    kl_ipf_mwu = kl_divergence_weights(ipf_weights, mwu_weights)
    sym_kl = symmetric_kl_divergence(mwu_weights, ipf_weights)
    tv = total_variation_weights(mwu_weights, ipf_weights)

    print("Weight Distribution Divergences:")
    print(f"  KL(MWU || IPF): {kl_mwu_ipf:.6f}")
    print(f"  KL(IPF || MWU): {kl_ipf_mwu:.6f}")
    print(f"  Symmetric KL:   {sym_kl:.6f}")
    print(f"  Total Variation: {tv:.6f}")
    print()

    print("Interpretation:")
    print("  - KL(MWU || IPF) measures how well MWU approximates IPF")
    print("  - Symmetric KL = (KL_12 + KL_21) / 2 is a symmetric measure")
    print("  - TV distance is bounded [0, 1] and symmetric")
    print()

    # =========================================================================
    # Section 5: Learning Rate Effects
    # =========================================================================
    print("-" * 70)
    print("Section 5: Learning Rate Effects on IPF Matching")
    print("-" * 70)
    print()

    learning_rates = [2.0, 1.0, 0.5, 0.3, 0.1]

    print(
        f"{'LR':>6} {'n_steps':>8} {'KL from IPF':>14} {'Max Diff':>12} {'ESS Ratio':>12}"
    )
    print("-" * 54)

    for lr in learning_rates:
        n_steps = max(3, int(10 / lr))
        mwu_test = OnlineRakingMWU(targets, learning_rate=lr, n_sgd_steps=n_steps)
        for obs in observations:
            mwu_test.partial_fit(obs)

        kl = kl_divergence_weights(mwu_test.weights, ipf_weights)
        max_diff = max(
            abs(mwu_test.margins[f] - ipf.margins[f]) for f in targets.feature_names
        )
        ess_ratio = mwu_test.effective_sample_size / ipf.effective_sample_size

        print(
            f"{lr:>6.2f} {n_steps:>8} {kl:>14.6f} {max_diff:>12.4f} {ess_ratio:>12.3f}"
        )

    print()
    print("Note: Smaller learning rates with more steps generally converge")
    print("closer to the batch IPF solution.")
    print()

    # =========================================================================
    # Section 6: Optimal Learning Rate Guidance
    # =========================================================================
    print("-" * 70)
    print("Section 6: Learning Rate Guidance")
    print("-" * 70)
    print()

    n_features = len(targets.feature_names)
    suggested_lr = optimal_mwu_learning_rate(n_obs=n_obs, n_features=n_features)

    print(f"For n_obs={n_obs}, n_features={n_features}:")
    print(f"  Suggested learning rate: {suggested_lr:.3f}")
    print()

    mwu_optimal = OnlineRakingMWU(targets, learning_rate=suggested_lr, n_sgd_steps=5)
    for obs in observations:
        mwu_optimal.partial_fit(obs)

    opt_comparison = compare_to_ipf(mwu_optimal, ipf)
    print("Results with suggested LR:")
    print(f"  KL from IPF: {opt_comparison.weight_kl:.6f}")
    print(f"  Max margin diff: {opt_comparison.margin_max_diff:.4f}")
    print(f"  ESS ratio: {opt_comparison.ess_ratio:.3f}")
    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("1. MWU with KL tracking monitors weight distribution changes per update")
    print("2. compare_to_ipf() provides comprehensive MWU vs IPF comparison")
    print("3. Smaller learning rates with more iterations yield closer IPF match")
    print("4. optimal_mwu_learning_rate() provides data-driven LR suggestions")
    print("5. KL divergence is asymmetric; symmetric_kl offers a symmetric alternative")
    print()
    print("MWU is theoretically grounded as mirror descent with KL regularization,")
    print("making it naturally suited for approximating the IPF solution in streaming")
    print("settings where batch computation is not feasible.")


if __name__ == "__main__":
    main()
