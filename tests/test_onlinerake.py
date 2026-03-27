"""Comprehensive tests for the onlinerake package."""

import numpy as np
import pytest

from onlinerake import OnlineRakingMWU, OnlineRakingSGD, Targets


class TestTargets:
    """Test the Targets dataclass."""

    def test_default_targets(self):
        """Test that targets require explicit features."""
        with pytest.raises(ValueError, match="At least one feature must be specified"):
            Targets()

    def test_custom_targets(self):
        """Test custom target values."""
        targets = Targets(feature_a=0.6, feature_b=0.4, feature_c=0.7)
        assert targets["feature_a"] == 0.6
        assert targets["feature_b"] == 0.4
        assert targets["feature_c"] == 0.7

    def test_as_dict(self):
        """Test conversion to dictionary."""
        targets = Targets(feature_a=0.6, feature_b=0.4, feature_c=0.7)
        target_dict = targets.as_dict()
        expected = {"feature_a": 0.6, "feature_b": 0.4, "feature_c": 0.7}
        assert target_dict == expected

    def test_continuous_feature(self):
        """Test continuous feature with tuple syntax."""
        targets = Targets(age=(35.0, "mean"), income=(65000.0, "mean"))
        assert targets["age"] == 35.0
        assert targets["income"] == 65000.0
        assert targets.is_continuous("age")
        assert targets.is_continuous("income")
        assert targets.feature_type("age") == "continuous"

    def test_mixed_binary_and_continuous(self):
        """Test mixing binary and continuous features."""
        targets = Targets(
            gender=0.5,
            college=0.35,
            age=(42.0, "mean"),
            income=(65000, "mean"),
        )
        assert targets.is_binary("gender")
        assert targets.is_binary("college")
        assert targets.is_continuous("age")
        assert targets.is_continuous("income")
        assert targets["gender"] == 0.5
        assert targets["age"] == 42.0
        assert targets.binary_features == ["college", "gender"]
        assert targets.continuous_features == ["age", "income"]
        assert targets.has_continuous_features

    def test_continuous_validation_bad_type(self):
        """Test that invalid tuple syntax raises error."""
        with pytest.raises(ValueError, match="must be 'mean'"):
            Targets(age=(35.0, "median"))

    def test_continuous_validation_bad_length(self):
        """Test that wrong tuple length raises error."""
        with pytest.raises(ValueError, match="tuple of length"):
            Targets(age=(35.0, "mean", "extra"))

    def test_binary_out_of_range(self):
        """Test that binary targets must be in [0, 1]."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            Targets(gender=1.5)

    def test_repr_mixed(self):
        """Test repr with mixed features."""
        targets = Targets(gender=0.5, age=(35.0, "mean"))
        repr_str = repr(targets)
        assert "gender=0.50" in repr_str
        assert "age=35.00 [mean]" in repr_str


class TestOnlineRakingSGD:
    """Test the SGD-based online raking algorithm."""

    def test_initialization(self):
        """Test proper initialization of SGD raker."""
        targets = Targets(feature_a=0.5, feature_b=0.5, feature_c=0.4, feature_d=0.3)
        raker = OnlineRakingSGD(targets, learning_rate=1.0)

        assert raker.targets == targets
        assert raker.learning_rate == 1.0
        assert raker._n_obs == 0
        assert len(raker.weights) == 0

    def test_single_observation(self):
        """Test processing a single observation."""
        targets = Targets(feature_a=0.5, feature_b=0.5, feature_c=0.4, feature_d=0.3)
        raker = OnlineRakingSGD(targets, learning_rate=1.0)

        obs = {"feature_a": 1, "feature_b": 0, "feature_c": 1, "feature_d": 0}
        raker.partial_fit(obs)

        assert raker._n_obs == 1
        assert len(raker.weights) == 1
        assert raker.weights[0] > 0

    def test_multiple_observations(self):
        """Test processing multiple observations."""
        targets = Targets(feature_a=0.5, feature_b=0.5, feature_c=0.4, feature_d=0.3)
        raker = OnlineRakingSGD(targets, learning_rate=1.0)

        observations = [
            {"feature_a": 1, "feature_b": 0, "feature_c": 1, "feature_d": 0},
            {"feature_a": 0, "feature_b": 1, "feature_c": 0, "feature_d": 1},
            {"feature_a": 1, "feature_b": 1, "feature_c": 1, "feature_d": 1},
        ]

        for obs in observations:
            raker.partial_fit(obs)

        assert raker._n_obs == 3
        assert len(raker.weights) == 3
        assert all(w > 0 for w in raker.weights)

    def test_margins_property(self):
        """Test that margins are computed correctly."""
        targets = Targets(feature_a=0.5, feature_b=0.5, feature_c=0.4, feature_d=0.3)
        raker = OnlineRakingSGD(targets, learning_rate=1.0)

        # Add observations with known demographics
        observations = [
            {"feature_a": 1, "feature_b": 1, "feature_c": 1, "feature_d": 1},  # all 1s
            {"feature_a": 0, "feature_b": 0, "feature_c": 0, "feature_d": 0},  # all 0s
        ]

        for obs in observations:
            raker.partial_fit(obs)

        margins = raker.margins
        assert "feature_a" in margins
        assert "feature_b" in margins
        assert "feature_c" in margins
        assert "feature_d" in margins

        # Test that margins are valid proportions
        for key in ["feature_a", "feature_b", "feature_c", "feature_d"]:
            assert key in margins
            assert 0 <= margins[key] <= 1

    def test_effective_sample_size(self):
        """Test effective sample size calculation."""
        targets = Targets(feature_a=0.5, feature_b=0.5, feature_c=0.4, feature_d=0.3)
        raker = OnlineRakingSGD(targets, learning_rate=1.0)

        obs = {"feature_a": 1, "feature_b": 0, "feature_c": 1, "feature_d": 0}
        raker.partial_fit(obs)

        ess = raker.effective_sample_size
        assert ess > 0
        assert ess <= raker._n_obs

    def test_loss_property(self):
        """Test that loss is computed correctly."""
        targets = Targets(feature_a=0.5, feature_b=0.5, feature_c=0.4, feature_d=0.3)
        raker = OnlineRakingSGD(targets, learning_rate=1.0)

        obs = {"feature_a": 1, "feature_b": 0, "feature_c": 1, "feature_d": 0}
        raker.partial_fit(obs)

        loss = raker.loss
        assert loss >= 0


class TestOnlineRakingMWU:
    """Test the MWU-based online raking algorithm."""

    def test_initialization(self):
        """Test proper initialization of MWU raker."""
        targets = Targets(feature_a=0.5, feature_b=0.5, feature_c=0.4, feature_d=0.3)
        raker = OnlineRakingMWU(targets, learning_rate=1.0)

        assert raker.targets == targets
        assert raker.learning_rate == 1.0
        assert raker._n_obs == 0
        assert len(raker.weights) == 0

    def test_single_observation(self):
        """Test processing a single observation with MWU."""
        targets = Targets(feature_a=0.5, feature_b=0.5, feature_c=0.4, feature_d=0.3)
        raker = OnlineRakingMWU(targets, learning_rate=1.0)

        obs = {"feature_a": 1, "feature_b": 0, "feature_c": 1, "feature_d": 0}
        raker.partial_fit(obs)

        assert raker._n_obs == 1
        assert len(raker.weights) == 1
        assert raker.weights[0] > 0

    def test_weight_clipping(self):
        """Test weight clipping functionality."""
        targets = Targets(feature_a=0.5, feature_b=0.5, feature_c=0.4, feature_d=0.3)
        raker = OnlineRakingMWU(
            targets, learning_rate=10.0, min_weight=0.1, max_weight=10.0
        )

        obs = {"feature_a": 1, "feature_b": 0, "feature_c": 1, "feature_d": 0}
        raker.partial_fit(obs)

        assert raker.weights[0] >= 0.1
        assert raker.weights[0] <= 10.0

    def test_mwu_diagnostics(self):
        """Test that MWU inherits diagnostics features."""
        targets = Targets(feature_a=0.5, feature_b=0.5, feature_c=0.4, feature_d=0.3)
        raker = OnlineRakingMWU(targets, verbose=False, track_convergence=True)

        obs = {"feature_a": 1, "feature_b": 0, "feature_c": 1, "feature_d": 0}
        raker.partial_fit(obs)

        # Test that MWU has all diagnostic features
        assert len(raker.gradient_norm_history) == 1
        assert not np.isnan(raker.loss_moving_average)
        assert isinstance(raker.weight_distribution_stats, dict)
        assert isinstance(raker.converged, bool)
        assert isinstance(raker.detect_oscillation(), bool)


class TestDiagnosticsAndMonitoring:
    """Test enhanced diagnostics and monitoring features."""

    def test_sgd_diagnostics_comprehensive(self):
        """Test comprehensive diagnostics for SGD raker."""
        targets = Targets(feature_a=0.5, feature_b=0.5, feature_c=0.4, feature_d=0.3)
        raker = OnlineRakingSGD(
            targets,
            learning_rate=2.0,
            verbose=False,
            track_convergence=True,
            convergence_window=5,
        )

        # Generate observations that should converge quickly
        observations = [
            {"age": 0, "gender": 1, "education": 0, "region": 1},
            {"age": 1, "gender": 0, "education": 1, "region": 0},
            {"age": 0, "gender": 1, "education": 0, "region": 1},
            {"age": 1, "gender": 0, "education": 1, "region": 0},
            {"age": 0, "gender": 1, "education": 1, "region": 0},
            {"age": 1, "gender": 0, "education": 0, "region": 1},
        ]

        for obs in observations:
            raker.partial_fit(obs)

        # Test gradient norm tracking
        assert len(raker.gradient_norm_history) == len(observations)
        assert all(norm >= 0 for norm in raker.gradient_norm_history)

        # Test loss moving average
        assert not np.isnan(raker.loss_moving_average)
        assert raker.loss_moving_average >= 0

        # Test weight distribution statistics
        weight_stats = raker.weight_distribution_stats
        expected_keys = {
            "min",
            "max",
            "mean",
            "std",
            "median",
            "q25",
            "q75",
            "outliers_count",
        }
        assert set(weight_stats.keys()) == expected_keys
        assert weight_stats["min"] <= weight_stats["max"]
        assert weight_stats["q25"] <= weight_stats["median"] <= weight_stats["q75"]
        assert weight_stats["outliers_count"] >= 0

        # Test convergence detection
        assert isinstance(raker.converged, bool)
        if raker.converged:
            assert isinstance(raker.convergence_step, int)
            assert raker.convergence_step > 0

        # Test oscillation detection
        oscillating = raker.detect_oscillation()
        assert isinstance(oscillating, bool)

        # Test enhanced history
        last_state = raker.history[-1]
        assert "gradient_norm" in last_state
        assert "loss_moving_avg" in last_state
        assert "converged" in last_state
        assert "oscillating" in last_state
        assert "weight_stats" in last_state
        assert isinstance(last_state["weight_stats"], dict)

    def test_convergence_detection_disabled(self):
        """Test that convergence detection can be disabled."""
        targets = Targets(feature_a=0.5, feature_b=0.5, feature_c=0.4, feature_d=0.3)
        raker = OnlineRakingSGD(targets, track_convergence=False)

        obs = {"feature_a": 1, "feature_b": 0, "feature_c": 1, "feature_d": 0}
        raker.partial_fit(obs)

        # Convergence tracking should be disabled
        assert not raker.converged
        assert raker.convergence_step is None

        # But other diagnostics should still work
        assert not np.isnan(raker.loss_moving_average)
        assert len(raker.gradient_norm_history) == 1

    def test_verbose_mode(self):
        """Test verbose output (we just ensure it doesn't crash)."""
        targets = Targets(feature_a=0.5, feature_b=0.5, feature_c=0.4, feature_d=0.3)
        raker = OnlineRakingSGD(targets, verbose=True)

        # Should not crash with verbose=True
        for i in range(105):  # Trigger verbose output at step 100
            obs = {
                "age": i % 2,
                "gender": (i + 1) % 2,
                "education": i % 2,
                "region": (i + 1) % 2,
            }
            raker.partial_fit(obs)

        assert raker._n_obs == 105

    def test_oscillation_detection(self):
        """Test oscillation detection with artificially oscillating loss."""
        targets = Targets(age=0.5, gender=0.5, education=0.5, region=0.5)

        # Use high learning rate to potentially cause oscillation
        raker = OnlineRakingSGD(targets, learning_rate=10.0, convergence_window=10)

        # Generate alternating pattern that might cause oscillation
        for i in range(20):
            if i % 2 == 0:
                obs = {"age": 1, "gender": 1, "education": 1, "region": 1}
            else:
                obs = {"age": 0, "gender": 0, "education": 0, "region": 0}
            raker.partial_fit(obs)

        # After enough observations, we should be able to detect if oscillating
        oscillating = raker.detect_oscillation()
        assert isinstance(oscillating, bool)

    def test_convergence_tolerance(self):
        """Test convergence detection with different tolerance levels."""
        targets = Targets(feature_a=0.5, feature_b=0.5, feature_c=0.4, feature_d=0.3)
        raker = OnlineRakingSGD(targets, learning_rate=1.0, convergence_window=5)

        # Add several similar observations
        for _ in range(10):
            obs = {"feature_a": 1, "feature_b": 0, "feature_c": 1, "feature_d": 0}
            raker.partial_fit(obs)

        # Test with strict tolerance
        converged_strict = raker.check_convergence(tolerance=1e-10)

        # Test with loose tolerance
        converged_loose = raker.check_convergence(tolerance=1e-2)

        # Loose tolerance should be more likely to detect convergence
        assert isinstance(converged_strict, bool)
        assert isinstance(converged_loose, bool)


class TestNumericalStability:
    """Test numerical stability fixes."""

    def test_mwu_extreme_gradients(self):
        """Test MWU handles extreme gradients without overflow."""
        targets = Targets(age=0.5, gender=0.5, education=0.5, region=0.5)

        # Use extreme learning rate that would cause overflow without clipping
        raker = OnlineRakingMWU(targets, learning_rate=100.0)

        # Create extreme observations that would produce large gradients
        extreme_obs = [
            {"age": 1, "gender": 1, "education": 1, "region": 1},  # All 1s
            {"age": 0, "gender": 0, "education": 0, "region": 0},  # All 0s
        ] * 10

        # Should not crash or produce NaN/Inf values
        for obs in extreme_obs:
            raker.partial_fit(obs)

            # Check that weights remain finite
            assert np.all(np.isfinite(raker.weights))
            assert np.all(raker.weights > 0)

            # Check that margins remain finite
            margins = raker.margins
            assert all(np.isfinite(v) for v in margins.values())

    def test_convergence_near_zero_loss(self):
        """Test convergence detection when loss approaches zero."""
        targets = Targets(age=0.5, gender=0.5, education=0.5, region=0.5)
        raker = OnlineRakingSGD(
            targets, learning_rate=1.0, track_convergence=True, convergence_window=5
        )

        # Create observations that exactly match targets (should produce near-zero loss)
        perfect_obs = [
            {"age": 1, "gender": 1, "education": 1, "region": 1},
            {"age": 0, "gender": 0, "education": 0, "region": 0},
        ] * 10

        for obs in perfect_obs:
            raker.partial_fit(obs)

            # After enough observations, should converge due to low loss
            if raker._n_obs >= raker.convergence_window:
                # Force convergence check
                converged = raker.check_convergence(tolerance=1e-6)

                if converged:
                    assert raker.converged
                    assert raker.convergence_step is not None
                    break

        # Should eventually converge with very low loss
        final_loss = raker.loss
        assert final_loss < 0.01  # Very low loss

    def test_convergence_with_zero_tolerance(self):
        """Test convergence detection with zero tolerance (perfect convergence only)."""
        targets = Targets(age=0.5, gender=0.5, education=0.5, region=0.5)
        raker = OnlineRakingSGD(targets, learning_rate=1.0, convergence_window=3)

        # Add observations that create exactly zero loss
        for i in range(10):
            if i % 2 == 0:
                obs = {"age": 1, "gender": 1, "education": 1, "region": 1}
            else:
                obs = {"age": 0, "gender": 0, "education": 0, "region": 0}
            raker.partial_fit(obs)

        # With zero tolerance, should only converge if loss is exactly zero
        converged = raker.check_convergence(tolerance=0.0)

        # Should handle this without error
        assert isinstance(converged, bool)

    def test_mwu_weight_clipping_with_extreme_updates(self):
        """Test that MWU weight clipping works with extreme exponent clipping."""
        targets = Targets(
            age=0.1, gender=0.9, education=0.1, region=0.9
        )  # Extreme targets
        raker = OnlineRakingMWU(
            targets,
            learning_rate=50.0,  # Very high learning rate
            min_weight=0.01,
            max_weight=100.0,
        )

        # Create biased observations
        for _ in range(20):
            obs = {
                "age": 0,
                "gender": 0,
                "education": 0,
                "region": 0,
            }  # Opposite of targets
            raker.partial_fit(obs)

            # Weights should stay within bounds despite extreme updates
            assert np.all(raker.weights >= raker.min_weight)
            assert np.all(raker.weights <= raker.max_weight)
            assert np.all(np.isfinite(raker.weights))


class TestRealisticScenarios:
    """Test with realistic survey scenarios."""

    def test_feature_bias_correction(self):
        """Test correcting feature bias in a stream."""
        # Example: streaming survey with feature bias
        targets = Targets(feature_a=0.5, feature_b=0.51, feature_c=0.4, feature_d=0.3)
        raker = OnlineRakingSGD(targets, learning_rate=2.0)

        # Simulate a biased stream (70% of feature_b=0, 30% feature_b=1)
        np.random.seed(42)
        n_obs = 100
        biased_observations = []

        for _i in range(n_obs):
            # 70% chance of feature_b=0, 30% chance of feature_b=1
            feature_b = 1 if np.random.random() < 0.3 else 0
            obs = {
                "feature_a": np.random.choice([0, 1]),
                "feature_b": feature_b,
                "feature_c": np.random.choice([0, 1]),
                "feature_d": np.random.choice([0, 1]),
            }
            biased_observations.append(obs)
            raker.partial_fit(obs)

        # Check that feature_b margin is closer to target after raking
        final_margins = raker.margins
        raw_feature_b_prop = (
            sum(obs["feature_b"] for obs in biased_observations) / n_obs
        )

        # Raw proportion should be around 0.3 (biased)
        assert 0.25 <= raw_feature_b_prop <= 0.35

        # Weighted margin should be closer to target 0.51
        feature_b_error_raw = abs(raw_feature_b_prop - 0.51)
        feature_b_error_weighted = abs(final_margins["feature_b"] - 0.51)
        assert feature_b_error_weighted < feature_b_error_raw

    def test_education_bias_correction(self):
        """Test correcting education bias."""
        # Target: 40% have higher education
        targets = Targets(feature_a=0.5, feature_b=0.5, feature_c=0.4, feature_d=0.3)
        raker = OnlineRakingSGD(targets, learning_rate=3.0)

        # Simulate over-educated sample (60% have higher education)
        np.random.seed(123)
        n_obs = 150

        for _i in range(n_obs):
            feature_c = 1 if np.random.random() < 0.6 else 0
            obs = {
                "feature_a": np.random.choice([0, 1]),
                "feature_b": np.random.choice([0, 1]),
                "feature_c": feature_c,
                "feature_d": np.random.choice([0, 1]),
            }
            raker.partial_fit(obs)

        final_margins = raker.margins
        # Weighted feature_c margin should be closer to 0.4
        feature_c_error = abs(final_margins["feature_c"] - 0.4)
        assert feature_c_error < 0.15  # Should be reasonably close


def test_sgd_vs_mwu_comparison():
    """Compare SGD and MWU on the same stream."""
    targets = Targets(feature_a=0.6, feature_b=0.5, feature_c=0.3, feature_d=0.4)

    sgd_raker = OnlineRakingSGD(targets, learning_rate=3.0)
    mwu_raker = OnlineRakingMWU(targets, learning_rate=1.0)

    # Generate a biased stream
    np.random.seed(456)
    observations = []

    for _i in range(200):
        # Feature_a bias: 80% of feature_a=0
        feature_a = 1 if np.random.random() < 0.2 else 0
        obs = {
            "feature_a": feature_a,
            "feature_b": np.random.choice([0, 1]),
            "feature_c": np.random.choice([0, 1]),
            "feature_d": np.random.choice([0, 1]),
        }
        observations.append(obs)

        sgd_raker.partial_fit(obs)
        mwu_raker.partial_fit(obs)

    # Both should improve feature_a margin compared to raw data
    raw_feature_a_prop = sum(obs["feature_a"] for obs in observations) / len(
        observations
    )
    sgd_feature_a_margin = sgd_raker.margins["feature_a"]
    mwu_feature_a_margin = mwu_raker.margins["feature_a"]

    raw_feature_a_error = abs(raw_feature_a_prop - targets["feature_a"])
    sgd_feature_a_error = abs(sgd_feature_a_margin - targets["feature_a"])
    mwu_feature_a_error = abs(mwu_feature_a_margin - targets["feature_a"])

    # Both algorithms should reduce the bias
    assert sgd_feature_a_error < raw_feature_a_error
    assert mwu_feature_a_error < raw_feature_a_error

    # Both should maintain reasonable effective sample sizes
    assert sgd_raker.effective_sample_size > 50
    assert mwu_raker.effective_sample_size > 50


def test_mwu_exponent_clipping_no_overflow():
    """Test MWU with extreme learning rates doesn't produce overflow/NaN."""
    from onlinerake import OnlineRakingMWU, Targets

    targets = Targets(age=0.5, gender=0.5, education=0.4, region=0.3)
    # Use extreme learning rate that would cause overflow without proper clipping
    mwu = OnlineRakingMWU(
        targets,
        learning_rate=1e6,  # Extremely high learning rate
        min_weight=1e-3,
        max_weight=1e3,
        n_steps=5,
    )

    # Stream deliberately extreme cases that would produce large gradients
    for _ in range(50):
        obs = {"age": 1, "gender": 0, "education": 1, "region": 0}
        mwu.partial_fit(obs)

        # All weights must remain finite after each update
        assert np.all(np.isfinite(mwu.weights)), "Weights became infinite or NaN"
        assert np.all(mwu.weights > 0), "Weights became zero or negative"

        # All margins must remain finite
        margins = mwu.margins
        assert all(np.isfinite(v) for v in margins.values()), (
            "Margins became infinite or NaN"
        )

        # Loss must remain finite
        assert np.isfinite(mwu.loss), "Loss became infinite or NaN"


class TestContinuousCovariates:
    """Test continuous covariate support."""

    def test_sgd_continuous_only(self):
        """Test SGD with continuous features only."""
        targets = Targets(age=(35.0, "mean"), income=(50000.0, "mean"))
        raker = OnlineRakingSGD(targets, learning_rate=0.5)

        # Process observations
        observations = [
            {"age": 25.0, "income": 30000.0},
            {"age": 30.0, "income": 45000.0},
            {"age": 40.0, "income": 60000.0},
            {"age": 45.0, "income": 70000.0},
        ]

        for obs in observations:
            raker.partial_fit(obs)

        assert raker._n_obs == 4
        assert len(raker.weights) == 4
        assert all(w > 0 for w in raker.weights)
        assert np.all(np.isfinite(raker.weights))

        # Margins should be computed
        margins = raker.margins
        assert "age" in margins
        assert "income" in margins
        assert np.isfinite(margins["age"])
        assert np.isfinite(margins["income"])

    def test_sgd_mixed_features(self):
        """Test SGD with mixed binary and continuous features."""
        targets = Targets(
            gender=0.5,
            college=0.35,
            age=(42.0, "mean"),
            income=(65000.0, "mean"),
        )
        raker = OnlineRakingSGD(targets, learning_rate=1.0)

        # Process observations with mixed types
        observations = [
            {"gender": 1, "college": 0, "age": 35.0, "income": 52000.0},
            {"gender": 0, "college": 1, "age": 28.0, "income": 45000.0},
            {"gender": 1, "college": 1, "age": 55.0, "income": 85000.0},
            {"gender": 0, "college": 0, "age": 42.0, "income": 62000.0},
        ]

        for obs in observations:
            raker.partial_fit(obs)

        assert raker._n_obs == 4
        margins = raker.margins
        assert 0 <= margins["gender"] <= 1
        assert 0 <= margins["college"] <= 1
        assert margins["age"] > 0
        assert margins["income"] > 0

    def test_sgd_continuous_convergence(self):
        """Test that SGD converges toward continuous targets."""
        targets = Targets(age=(40.0, "mean"))
        raker = OnlineRakingSGD(targets, learning_rate=2.0)

        # Create observations with mean different from target
        np.random.seed(42)
        ages = np.random.normal(35, 5, 100)

        for age in ages:
            raker.partial_fit({"age": age})

        # Loss should decrease over time
        initial_losses = [h["loss"] for h in raker.history[:10]]
        final_losses = [h["loss"] for h in raker.history[-10:]]
        assert np.mean(final_losses) < np.mean(initial_losses)

        # Weighted margin should be closer to target than raw
        raw_mean = ages.mean()
        weighted_mean = raker.margins["age"]
        raw_error = abs(raw_mean - 40.0)
        weighted_error = abs(weighted_mean - 40.0)
        assert weighted_error < raw_error

    def test_mwu_continuous(self):
        """Test MWU with continuous features."""
        targets = Targets(age=(35.0, "mean"), height=(170.0, "mean"))
        raker = OnlineRakingMWU(targets, learning_rate=0.5)

        observations = [
            {"age": 25.0, "height": 165.0},
            {"age": 35.0, "height": 175.0},
            {"age": 45.0, "height": 180.0},
            {"age": 30.0, "height": 168.0},
        ]

        for obs in observations:
            raker.partial_fit(obs)

        assert raker._n_obs == 4
        assert np.all(np.isfinite(raker.weights))
        assert np.all(raker.weights > 0)

    def test_large_scale_continuous_feature(self):
        """Test numerical stability with large-scale features like income."""
        targets = Targets(income=(75000.0, "mean"))
        raker = OnlineRakingSGD(targets, learning_rate=0.001)

        np.random.seed(123)
        incomes = np.random.lognormal(mean=10.5, sigma=0.5, size=100)

        for income in incomes:
            raker.partial_fit({"income": income})

        # Should remain numerically stable
        assert np.all(np.isfinite(raker.weights))
        assert np.isfinite(raker.loss)
        assert np.isfinite(raker.margins["income"])

    def test_mixed_feature_bias_correction(self):
        """Test bias correction with mixed features."""
        targets = Targets(
            female=0.5,
            age=(35.0, "mean"),
        )
        raker = OnlineRakingSGD(targets, learning_rate=2.0)

        # Biased sample: more males (female=0), younger (age<35)
        np.random.seed(42)
        n_obs = 100
        for _ in range(n_obs):
            female = 1 if np.random.random() < 0.3 else 0
            age = np.random.normal(28, 5)
            raker.partial_fit({"female": female, "age": age})

        # Both margins should move toward targets
        margins = raker.margins
        assert margins["female"] > 0.3
        assert margins["age"] > 28


class TestBatchIPFContinuousError:
    """Test that BatchIPF properly rejects continuous features."""

    def test_batch_ipf_rejects_continuous(self):
        """Test that BatchIPF raises error for continuous features."""
        from onlinerake import BatchIPF

        targets = Targets(gender=0.5, age=(35.0, "mean"))
        ipf = BatchIPF(targets)

        observations = [
            {"gender": 1, "age": 25.0},
            {"gender": 0, "age": 45.0},
        ]

        with pytest.raises(ValueError, match="BatchIPF only supports binary features"):
            ipf.fit(observations)

    def test_batch_ipf_incremental_rejects_continuous(self):
        """Test that fit_incremental also rejects continuous features."""
        from onlinerake import BatchIPF

        targets = Targets(income=(50000.0, "mean"))
        ipf = BatchIPF(targets)

        with pytest.raises(ValueError, match="BatchIPF only supports binary features"):
            ipf.fit_incremental([{"income": 45000.0}])

    def test_batch_ipf_binary_still_works(self):
        """Test that BatchIPF still works with binary features."""
        from onlinerake import BatchIPF

        targets = Targets(gender=0.5, education=0.4)
        ipf = BatchIPF(targets)

        observations = [
            {"gender": 1, "education": 0},
            {"gender": 0, "education": 1},
            {"gender": 1, "education": 1},
            {"gender": 0, "education": 0},
        ]

        ipf.fit(observations)
        assert ipf._n_obs == 4
        assert len(ipf.weights) == 4


class TestContinuousDiagnostics:
    """Test diagnostics with continuous features."""

    def test_variance_estimation_continuous(self):
        """Test variance estimation for continuous features."""
        from onlinerake.diagnostics import estimate_margin_variance

        targets = Targets(age=(35.0, "mean"))
        raker = OnlineRakingSGD(targets, learning_rate=1.0)

        ages = [25.0, 30.0, 35.0, 40.0, 45.0]
        for age in ages:
            raker.partial_fit({"age": age})

        variance = estimate_margin_variance(raker, "age")
        assert np.isfinite(variance)
        assert variance >= 0

    def test_feasibility_continuous(self):
        """Test feasibility check for continuous features."""
        from onlinerake.diagnostics import check_target_feasibility

        # Infeasible target: outside sample range
        targets = Targets(age=(100.0, "mean"))
        raker = OnlineRakingSGD(targets, learning_rate=1.0)

        ages = [25.0, 30.0, 35.0, 40.0, 45.0]
        for age in ages:
            raker.partial_fit({"age": age})

        report = check_target_feasibility(raker)
        assert not report.is_feasible
        assert "age" in report.problematic_features

    def test_feasibility_continuous_feasible(self):
        """Test feasibility for achievable continuous target."""
        from onlinerake.diagnostics import check_target_feasibility

        # Feasible target: within sample range
        targets = Targets(age=(35.0, "mean"))
        raker = OnlineRakingSGD(targets, learning_rate=1.0)

        ages = [25.0, 30.0, 35.0, 40.0, 45.0]
        for age in ages:
            raker.partial_fit({"age": age})

        report = check_target_feasibility(raker)
        assert report.is_feasible or "age" not in report.problematic_features

    def test_confidence_interval_continuous(self):
        """Test confidence interval for continuous features."""
        from onlinerake.diagnostics import compute_confidence_interval

        targets = Targets(age=(35.0, "mean"))
        raker = OnlineRakingSGD(targets, learning_rate=1.0)

        ages = [25.0, 30.0, 35.0, 40.0, 45.0]
        for age in ages:
            raker.partial_fit({"age": age})

        lower, upper = compute_confidence_interval(raker, "age")
        # For continuous, no clamping to [0,1]
        assert np.isfinite(lower)
        assert np.isfinite(upper)
        assert lower < upper


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
