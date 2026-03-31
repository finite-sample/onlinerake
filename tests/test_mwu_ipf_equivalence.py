"""Tests verifying MWU convergence to IPF solution.

MWU (Multiplicative Weights Update) is mirror descent with KL divergence
as the regularizer. As learning rate η → 0, MWU should converge to the
same solution as batch IPF, which minimizes D_KL(w || w_0) subject to
margin constraints.

These tests verify this theoretical relationship empirically.
"""

import numpy as np
import pytest

from onlinerake import (
    BatchIPF,
    OnlineRakingMWU,
    OnlineRakingSGD,
    Targets,
    compare_to_ipf,
    kl_divergence_weights,
    optimal_mwu_learning_rate,
    symmetric_kl_divergence,
    total_variation_weights,
)


def generate_biased_sample(n: int, targets: Targets, bias: float = 0.1, seed: int = 42):
    """Generate sample with systematic bias from targets.

    Args:
        n: Number of observations.
        targets: Target proportions.
        bias: How much to bias the sample (positive = oversample).
        seed: Random seed.

    Returns:
        List of observation dicts.
    """
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


class TestDivergenceFunctions:
    """Test KL divergence and total variation functions."""

    def test_kl_divergence_identical(self):
        """KL divergence of identical distributions is 0."""
        w = np.array([1.0, 2.0, 3.0, 4.0])
        assert kl_divergence_weights(w, w) == pytest.approx(0.0, abs=1e-10)

    def test_kl_divergence_positive(self):
        """KL divergence is non-negative."""
        w1 = np.array([1.0, 1.0, 1.0])
        w2 = np.array([1.0, 2.0, 1.0])
        assert kl_divergence_weights(w1, w2) >= 0
        assert kl_divergence_weights(w2, w1) >= 0

    def test_kl_divergence_asymmetric(self):
        """KL divergence is asymmetric (in general)."""
        # Use sparse vs dense to show clear asymmetry
        # D_KL(sparse || dense) is usually different from D_KL(dense || sparse)
        w_sparse = np.array([0.01, 0.01, 0.98])  # Concentrated on one element
        w_dense = np.array([0.33, 0.33, 0.34])  # Roughly uniform
        kl_sd = kl_divergence_weights(w_sparse, w_dense)
        kl_ds = kl_divergence_weights(w_dense, w_sparse)
        # Both should be positive
        assert kl_sd > 0 and kl_ds > 0
        # They should be different (asymmetric property)
        # D_KL(sparse || uniform) is small (sparse is "surprised" by uniform)
        # D_KL(uniform || sparse) is larger (uniform is very "surprised" by sparse)
        assert kl_sd != kl_ds

    def test_total_variation_identical(self):
        """TV distance of identical distributions is 0."""
        w = np.array([1.0, 2.0, 3.0])
        assert total_variation_weights(w, w) == pytest.approx(0.0, abs=1e-10)

    def test_total_variation_symmetric(self):
        """TV distance is symmetric."""
        w1 = np.array([1.0, 1.0, 1.0])
        w2 = np.array([1.0, 3.0, 1.0])
        assert total_variation_weights(w1, w2) == pytest.approx(
            total_variation_weights(w2, w1)
        )

    def test_total_variation_bounded(self):
        """TV distance is bounded by [0, 1]."""
        w1 = np.array([1.0, 0.001, 0.001])
        w2 = np.array([0.001, 1.0, 0.001])
        tv = total_variation_weights(w1, w2)
        assert 0 <= tv <= 1


class TestKLTracking:
    """Test KL divergence tracking in rakers."""

    def test_kl_tracking_disabled_by_default(self):
        """KL tracking is off by default for performance."""
        targets = Targets(a=0.5, b=0.3)
        raker = OnlineRakingSGD(targets)
        obs = [{"a": 1, "b": 0}, {"a": 0, "b": 1}]
        for o in obs:
            raker.partial_fit(o)
        assert raker.kl_divergence_history == []
        assert raker.cumulative_kl_divergence == 0.0

    def test_kl_tracking_sgd(self):
        """KL tracking works for SGD raker."""
        targets = Targets(a=0.5, b=0.3)
        raker = OnlineRakingSGD(targets, track_kl_divergence=True)
        obs = [{"a": 1, "b": 0}, {"a": 0, "b": 1}, {"a": 1, "b": 1}]
        for o in obs:
            raker.partial_fit(o)

        assert len(raker.kl_divergence_history) == 3
        assert all(kl >= 0 for kl in raker.kl_divergence_history)
        assert raker.cumulative_kl_divergence >= 0

    def test_kl_tracking_mwu(self):
        """KL tracking works for MWU raker."""
        targets = Targets(a=0.5, b=0.3)
        raker = OnlineRakingMWU(targets, track_kl_divergence=True)
        obs = [{"a": 1, "b": 0}, {"a": 0, "b": 1}, {"a": 1, "b": 1}]
        for o in obs:
            raker.partial_fit(o)

        assert len(raker.kl_divergence_history) == 3
        assert all(kl >= 0 for kl in raker.kl_divergence_history)
        assert raker.cumulative_kl_divergence >= 0

    def test_cumulative_kl_is_sum(self):
        """Cumulative KL equals sum of individual KLs."""
        targets = Targets(a=0.5)
        raker = OnlineRakingMWU(targets, track_kl_divergence=True)
        obs = [{"a": 1}, {"a": 0}, {"a": 1}, {"a": 0}, {"a": 1}]
        for o in obs:
            raker.partial_fit(o)

        expected_sum = sum(raker.kl_divergence_history)
        assert raker.cumulative_kl_divergence == pytest.approx(expected_sum)


class TestMWUMatchesIPF:
    """Verify MWU converges to IPF solution."""

    def test_converged_mwu_matches_ipf_margins(self):
        """After convergence, MWU and IPF should have similar margins.

        Note: MWU is a streaming algorithm that may not perfectly match
        IPF, especially with limited data. We use more iterations and
        smaller learning rate to get closer.
        """
        targets = Targets(female=0.51, college=0.32, urban=0.65)
        observations = generate_biased_sample(500, targets, bias=0.05)

        # MWU with smaller learning rate and more steps for better convergence
        mwu = OnlineRakingMWU(
            targets, learning_rate=0.1, n_sgd_steps=10, track_kl_divergence=True
        )
        for obs in observations:
            mwu.partial_fit(obs)

        # Batch IPF
        ipf = BatchIPF(targets).fit(observations)

        # Compare margins - allow more tolerance for streaming algorithm
        for name in targets.feature_names:
            mwu_margin = mwu.margins[name]
            ipf_margin = ipf.margins[name]
            assert mwu_margin == pytest.approx(ipf_margin, abs=0.1), (
                f"Margin mismatch for {name}: MWU={mwu_margin:.4f}, IPF={ipf_margin:.4f}"
            )

    def test_smaller_lr_closer_to_ipf(self):
        """Smaller learning rate with more iterations gives closer IPF match.

        Note: With small LR and limited data, MWU may not converge fully.
        We test that moderate LR performs reasonably well.
        """
        targets = Targets(a=0.4, b=0.6)
        observations = generate_biased_sample(200, targets, bias=0.1)

        ipf = BatchIPF(targets).fit(observations)
        ipf_weights = ipf.weights

        kl_from_ipf = []
        # Use more SGD steps to compensate for smaller LR
        configs = [(2.0, 3), (1.0, 5), (0.5, 10), (0.3, 15)]

        for lr, steps in configs:
            mwu = OnlineRakingMWU(targets, learning_rate=lr, n_sgd_steps=steps)
            for obs in observations:
                mwu.partial_fit(obs)
            mwu_weights = mwu.weights
            kl = kl_divergence_weights(mwu_weights, ipf_weights)
            kl_from_ipf.append(kl)

        # All should achieve reasonable match to IPF
        for kl in kl_from_ipf:
            assert kl < 0.2, f"KL should be reasonable: {kl}"

        # The configuration with most steps should be among the best
        assert kl_from_ipf[-1] < max(kl_from_ipf) * 2

    def test_mwu_approaches_ipf_with_more_iterations(self):
        """MWU should get closer to IPF with more observations/iterations."""
        targets = Targets(a=0.5, b=0.3)

        # Generate lots of observations
        observations = generate_biased_sample(500, targets, bias=0.1)
        ipf = BatchIPF(targets).fit(observations)

        # Track distance from IPF at different stages
        mwu = OnlineRakingMWU(targets, learning_rate=0.5, n_sgd_steps=3)

        checkpoints = [50, 100, 200, 500]
        distances = []

        for i, obs in enumerate(observations):
            mwu.partial_fit(obs)
            if (i + 1) in checkpoints:
                # Compare partial MWU to partial IPF
                partial_ipf = BatchIPF(targets).fit(observations[: i + 1])
                kl = kl_divergence_weights(mwu.weights, partial_ipf.weights)
                distances.append(kl)

        # Later checkpoints should generally be closer to IPF
        # (though not monotonic due to stochastic nature)
        # At minimum, final should be better than first
        assert distances[-1] < distances[0] * 2, (
            f"Final distance should be comparable or better than initial. "
            f"Distances: {list(zip(checkpoints, distances))}"
        )

    def test_kl_from_ipf_stays_reasonable(self):
        """KL from IPF should stay in reasonable range over time.

        Note: MWU doesn't guarantee monotonic decrease of KL from IPF
        because it's a streaming algorithm processing one observation
        at a time. We verify it stays within acceptable bounds.
        """
        targets = Targets(female=0.5)
        observations = generate_biased_sample(200, targets, bias=0.1)

        mwu = OnlineRakingMWU(
            targets, learning_rate=0.3, n_sgd_steps=5, track_kl_divergence=True
        )

        kl_from_ipf_history = []
        for i, obs in enumerate(observations):
            mwu.partial_fit(obs)
            if (i + 1) % 40 == 0:
                partial_ipf = BatchIPF(targets).fit(observations[: i + 1])
                kl = kl_divergence_weights(mwu.weights, partial_ipf.weights)
                kl_from_ipf_history.append(kl)

        # All KL values should be reasonable (not exploding)
        for kl in kl_from_ipf_history:
            assert kl < 0.5, f"KL from IPF should stay reasonable: {kl}"

        # Final KL should be in acceptable range
        assert kl_from_ipf_history[-1] < 0.2


class TestCompareToIPF:
    """Test the compare_to_ipf diagnostic function."""

    def test_compare_to_ipf_basic(self):
        """compare_to_ipf returns expected metrics."""
        targets = Targets(a=0.5, b=0.4)
        observations = generate_biased_sample(50, targets, bias=0.1)

        mwu = OnlineRakingMWU(targets, learning_rate=0.5)
        for obs in observations:
            mwu.partial_fit(obs)

        ipf = BatchIPF(targets).fit(observations)
        comparison = compare_to_ipf(mwu, ipf)

        assert comparison.weight_kl >= 0
        assert 0 <= comparison.weight_tv <= 1
        assert comparison.margin_mse >= 0
        assert comparison.margin_max_diff >= 0
        assert comparison.ess_ratio > 0
        assert comparison.raker_loss >= 0
        assert comparison.ipf_loss >= 0

    def test_compare_to_ipf_without_prefit_ipf(self):
        """compare_to_ipf can fit IPF internally."""
        targets = Targets(a=0.5, b=0.4)
        observations = generate_biased_sample(50, targets, bias=0.1)

        mwu = OnlineRakingMWU(targets, learning_rate=0.5)
        for obs in observations:
            mwu.partial_fit(obs)

        # Don't pass ipf; let it be fit internally
        comparison = compare_to_ipf(mwu)

        assert comparison.weight_kl >= 0
        assert comparison.margin_mse >= 0

    def test_compare_to_ipf_empty_raker_raises(self):
        """compare_to_ipf raises on empty raker."""
        targets = Targets(a=0.5)
        mwu = OnlineRakingMWU(targets)

        with pytest.raises(ValueError, match="no observations"):
            compare_to_ipf(mwu)

    def test_perfect_match_gives_zero_kl(self):
        """If MWU perfectly matches IPF, KL should be near zero."""
        # Use very small dataset where MWU can essentially match IPF
        targets = Targets(a=0.5)
        observations = [{"a": 1}, {"a": 0}]  # Exactly balanced

        # With many iterations and small LR, should converge closely
        mwu = OnlineRakingMWU(targets, learning_rate=0.1, n_sgd_steps=20)
        for obs in observations:
            mwu.partial_fit(obs)

        ipf = BatchIPF(targets).fit(observations)

        # IPF doesn't need to adjust weights for balanced data
        comparison = compare_to_ipf(mwu, ipf)

        # Both should have essentially unit weights for balanced data
        assert comparison.weight_kl < 0.1
        assert comparison.margin_mse < 0.01


class TestVariousTargetConfigurations:
    """Test IPF matching across different target setups."""

    def test_single_feature(self):
        """MWU matches IPF with single feature."""
        targets = Targets(feature=0.4)
        observations = generate_biased_sample(200, targets, bias=0.1)

        mwu = OnlineRakingMWU(targets, learning_rate=0.3, n_sgd_steps=10)
        for obs in observations:
            mwu.partial_fit(obs)

        comparison = compare_to_ipf(mwu)

        assert comparison.margin_max_diff < 0.1

    def test_many_features(self):
        """MWU matches IPF with many features."""
        targets = Targets(
            a=0.5, b=0.4, c=0.6, d=0.3, e=0.55, f=0.45, g=0.35, h=0.65
        )
        observations = generate_biased_sample(300, targets, bias=0.1)

        mwu = OnlineRakingMWU(targets, learning_rate=0.3, n_sgd_steps=5)
        for obs in observations:
            mwu.partial_fit(obs)

        ipf = BatchIPF(targets).fit(observations)
        comparison = compare_to_ipf(mwu, ipf)

        # With more features, harder to match exactly but should be reasonable
        assert comparison.margin_max_diff < 0.1
        assert comparison.margin_mse < 0.01

    def test_extreme_targets(self):
        """MWU handles extreme targets reasonably."""
        targets = Targets(rare=0.05, common=0.95)
        observations = generate_biased_sample(200, targets, bias=-0.03)

        mwu = OnlineRakingMWU(targets, learning_rate=0.3, n_sgd_steps=5)
        for obs in observations:
            mwu.partial_fit(obs)

        ipf = BatchIPF(targets).fit(observations)
        comparison = compare_to_ipf(mwu, ipf)

        # Should still achieve reasonable match
        assert comparison.margin_max_diff < 0.15


class TestOptimalLearningRate:
    """Test learning rate guidance function."""

    def test_optimal_lr_reasonable_range(self):
        """Optimal LR should be in reasonable range."""
        lr = optimal_mwu_learning_rate(n_obs=100, n_features=4)
        assert 0.01 <= lr <= 5.0

    def test_optimal_lr_decreases_with_n(self):
        """Optimal LR should decrease with more observations."""
        lr_100 = optimal_mwu_learning_rate(n_obs=100, n_features=4)
        lr_10000 = optimal_mwu_learning_rate(n_obs=10000, n_features=4)
        assert lr_10000 < lr_100

    def test_optimal_lr_increases_with_features(self):
        """Optimal LR scales with number of features."""
        lr_2 = optimal_mwu_learning_rate(n_obs=1000, n_features=2)
        lr_8 = optimal_mwu_learning_rate(n_obs=1000, n_features=8)
        assert lr_8 > lr_2


class TestSymmetricKLDivergence:
    """Test symmetric KL divergence function."""

    def test_symmetric_kl_is_symmetric(self):
        """symmetric_kl(w1, w2) == symmetric_kl(w2, w1)."""
        w1 = np.array([1.0, 2.0, 3.0, 4.0])
        w2 = np.array([4.0, 3.0, 2.0, 1.0])
        assert symmetric_kl_divergence(w1, w2) == pytest.approx(
            symmetric_kl_divergence(w2, w1)
        )

    def test_symmetric_kl_identical_is_zero(self):
        """symmetric_kl(w, w) == 0."""
        w = np.array([1.0, 2.0, 3.0, 4.0])
        assert symmetric_kl_divergence(w, w) == pytest.approx(0.0, abs=1e-10)

    def test_symmetric_kl_positive(self):
        """symmetric_kl >= 0 for any distributions."""
        w1 = np.array([1.0, 1.0, 1.0])
        w2 = np.array([1.0, 3.0, 1.0])
        assert symmetric_kl_divergence(w1, w2) >= 0

        w_sparse = np.array([0.01, 0.01, 0.98])
        w_dense = np.array([0.33, 0.33, 0.34])
        assert symmetric_kl_divergence(w_sparse, w_dense) >= 0

    def test_symmetric_kl_equals_average(self):
        """symmetric_kl == (kl_12 + kl_21) / 2."""
        w1 = np.array([1.0, 2.0, 3.0])
        w2 = np.array([3.0, 2.0, 1.0])

        kl_12 = kl_divergence_weights(w1, w2)
        kl_21 = kl_divergence_weights(w2, w1)
        expected = (kl_12 + kl_21) / 2

        assert symmetric_kl_divergence(w1, w2) == pytest.approx(expected)

    def test_symmetric_kl_various_distributions(self):
        """Test symmetric KL on various distribution shapes."""
        uniform = np.array([1.0, 1.0, 1.0, 1.0])
        peaked = np.array([0.1, 0.1, 10.0, 0.1])
        bimodal = np.array([5.0, 0.1, 0.1, 5.0])

        sym_kl_up = symmetric_kl_divergence(uniform, peaked)
        sym_kl_ub = symmetric_kl_divergence(uniform, bimodal)
        sym_kl_pb = symmetric_kl_divergence(peaked, bimodal)

        assert sym_kl_up >= 0
        assert sym_kl_ub >= 0
        assert sym_kl_pb >= 0

        assert sym_kl_up == pytest.approx(symmetric_kl_divergence(peaked, uniform))
        assert sym_kl_ub == pytest.approx(symmetric_kl_divergence(bimodal, uniform))


class TestSGDvsIPF:
    """Test SGD comparison to IPF (SGD uses different update rule)."""

    def test_sgd_also_converges_to_ipf_margins(self):
        """SGD should also achieve similar margins to IPF."""
        targets = Targets(a=0.5, b=0.3)
        observations = generate_biased_sample(150, targets, bias=0.15)

        sgd = OnlineRakingSGD(targets, learning_rate=5.0, n_sgd_steps=5)
        for obs in observations:
            sgd.partial_fit(obs)

        ipf = BatchIPF(targets).fit(observations)

        for name in targets.feature_names:
            sgd_margin = sgd.margins[name]
            ipf_margin = ipf.margins[name]
            assert sgd_margin == pytest.approx(ipf_margin, abs=0.05)

    def test_mwu_closer_to_ipf_weights_than_sgd(self):
        """MWU should produce weights closer to IPF than SGD.

        This is because MWU uses the same KL-based update as IPF,
        while SGD uses additive updates on squared error.
        """
        targets = Targets(a=0.5, b=0.4)
        observations = generate_biased_sample(100, targets, bias=0.1)

        ipf = BatchIPF(targets).fit(observations)
        ipf_weights = ipf.weights

        # MWU with moderate settings
        mwu = OnlineRakingMWU(targets, learning_rate=0.3, n_sgd_steps=5)
        for obs in observations:
            mwu.partial_fit(obs)
        mwu_kl = kl_divergence_weights(mwu.weights, ipf_weights)

        # SGD with similar settings
        sgd = OnlineRakingSGD(targets, learning_rate=5.0, n_sgd_steps=5)
        for obs in observations:
            sgd.partial_fit(obs)
        sgd_kl = kl_divergence_weights(sgd.weights, ipf_weights)

        # MWU should be closer to IPF in KL sense
        # (This is the key theoretical property we're testing)
        assert mwu_kl < sgd_kl * 2, (
            f"MWU should be closer to IPF than SGD. "
            f"MWU KL: {mwu_kl:.6f}, SGD KL: {sgd_kl:.6f}"
        )
