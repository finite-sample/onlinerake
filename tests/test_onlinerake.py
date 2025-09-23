"""Comprehensive tests for the onlinerake package."""

import pytest
import numpy as np
from onlinerake import OnlineRakingSGD, OnlineRakingMWU, Targets


class TestTargets:
    """Test the Targets dataclass."""

    def test_default_targets(self):
        """Test default target values."""
        targets = Targets()
        assert targets.age == 0.5
        assert targets.gender == 0.5
        assert targets.education == 0.4
        assert targets.region == 0.3

    def test_custom_targets(self):
        """Test custom target values."""
        targets = Targets(age=0.6, gender=0.4, education=0.7, region=0.2)
        assert targets.age == 0.6
        assert targets.gender == 0.4
        assert targets.education == 0.7
        assert targets.region == 0.2

    def test_as_dict(self):
        """Test conversion to dictionary."""
        targets = Targets(age=0.6, gender=0.4, education=0.7, region=0.2)
        target_dict = targets.as_dict()
        expected = {"age": 0.6, "gender": 0.4, "education": 0.7, "region": 0.2}
        assert target_dict == expected


class TestOnlineRakingSGD:
    """Test the SGD-based online raking algorithm."""

    def test_initialization(self):
        """Test proper initialization of SGD raker."""
        targets = Targets(age=0.5, gender=0.5, education=0.4, region=0.3)
        raker = OnlineRakingSGD(targets, learning_rate=1.0)

        assert raker.targets == targets
        assert raker.learning_rate == 1.0
        assert raker._n_obs == 0
        assert len(raker.weights) == 0

    def test_single_observation(self):
        """Test processing a single observation."""
        targets = Targets()
        raker = OnlineRakingSGD(targets, learning_rate=1.0)

        obs = {"age": 1, "gender": 0, "education": 1, "region": 0}
        raker.partial_fit(obs)

        assert raker._n_obs == 1
        assert len(raker.weights) == 1
        assert raker.weights[0] > 0

    def test_multiple_observations(self):
        """Test processing multiple observations."""
        targets = Targets()
        raker = OnlineRakingSGD(targets, learning_rate=1.0)

        observations = [
            {"age": 1, "gender": 0, "education": 1, "region": 0},
            {"age": 0, "gender": 1, "education": 0, "region": 1},
            {"age": 1, "gender": 1, "education": 1, "region": 1},
        ]

        for obs in observations:
            raker.partial_fit(obs)

        assert raker._n_obs == 3
        assert len(raker.weights) == 3
        assert all(w > 0 for w in raker.weights)

    def test_margins_property(self):
        """Test that margins are computed correctly."""
        targets = Targets()
        raker = OnlineRakingSGD(targets, learning_rate=1.0)

        # Add observations with known demographics
        observations = [
            {"age": 1, "gender": 1, "education": 1, "region": 1},  # all 1s
            {"age": 0, "gender": 0, "education": 0, "region": 0},  # all 0s
        ]

        for obs in observations:
            raker.partial_fit(obs)

        margins = raker.margins
        assert "age" in margins
        assert "gender" in margins
        assert "education" in margins
        assert "region" in margins

        # With equal weights, margins should be 0.5 for each category
        for margin in margins.values():
            assert 0 <= margin <= 1

    def test_effective_sample_size(self):
        """Test effective sample size calculation."""
        targets = Targets()
        raker = OnlineRakingSGD(targets, learning_rate=1.0)

        obs = {"age": 1, "gender": 0, "education": 1, "region": 0}
        raker.partial_fit(obs)

        ess = raker.effective_sample_size
        assert ess > 0
        assert ess <= raker._n_obs

    def test_loss_property(self):
        """Test that loss is computed correctly."""
        targets = Targets()
        raker = OnlineRakingSGD(targets, learning_rate=1.0)

        obs = {"age": 1, "gender": 0, "education": 1, "region": 0}
        raker.partial_fit(obs)

        loss = raker.loss
        assert loss >= 0


class TestOnlineRakingMWU:
    """Test the MWU-based online raking algorithm."""

    def test_initialization(self):
        """Test proper initialization of MWU raker."""
        targets = Targets(age=0.5, gender=0.5, education=0.4, region=0.3)
        raker = OnlineRakingMWU(targets, learning_rate=1.0)

        assert raker.targets == targets
        assert raker.learning_rate == 1.0
        assert raker._n_obs == 0
        assert len(raker.weights) == 0

    def test_single_observation(self):
        """Test processing a single observation with MWU."""
        targets = Targets()
        raker = OnlineRakingMWU(targets, learning_rate=1.0)

        obs = {"age": 1, "gender": 0, "education": 1, "region": 0}
        raker.partial_fit(obs)

        assert raker._n_obs == 1
        assert len(raker.weights) == 1
        assert raker.weights[0] > 0

    def test_weight_clipping(self):
        """Test weight clipping functionality."""
        targets = Targets()
        raker = OnlineRakingMWU(
            targets, learning_rate=10.0, min_weight=0.1, max_weight=10.0
        )

        obs = {"age": 1, "gender": 0, "education": 1, "region": 0}
        raker.partial_fit(obs)

        assert raker.weights[0] >= 0.1
        assert raker.weights[0] <= 10.0

    def test_mwu_diagnostics(self):
        """Test that MWU inherits diagnostics features."""
        targets = Targets()
        raker = OnlineRakingMWU(targets, verbose=False, track_convergence=True)

        obs = {"age": 1, "gender": 0, "education": 1, "region": 0}
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
        targets = Targets()
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
        targets = Targets()
        raker = OnlineRakingSGD(targets, track_convergence=False)

        obs = {"age": 1, "gender": 0, "education": 1, "region": 0}
        raker.partial_fit(obs)

        # Convergence tracking should be disabled
        assert not raker.converged
        assert raker.convergence_step is None

        # But other diagnostics should still work
        assert not np.isnan(raker.loss_moving_average)
        assert len(raker.gradient_norm_history) == 1

    def test_verbose_mode(self):
        """Test verbose output (we just ensure it doesn't crash)."""
        targets = Targets()
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
        targets = Targets()
        raker = OnlineRakingSGD(targets, learning_rate=1.0, convergence_window=5)

        # Add several similar observations
        for _ in range(10):
            obs = {"age": 1, "gender": 0, "education": 1, "region": 0}
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

    def test_gender_bias_correction(self):
        """Test correcting gender bias in a stream."""
        # US population is roughly 51% female
        targets = Targets(age=0.5, gender=0.51, education=0.4, region=0.3)
        raker = OnlineRakingSGD(targets, learning_rate=2.0)

        # Simulate a biased stream (70% male respondents)
        np.random.seed(42)
        n_obs = 100
        biased_observations = []

        for i in range(n_obs):
            # 70% chance of male (gender=0), 30% chance of female (gender=1)
            gender = 1 if np.random.random() < 0.3 else 0
            obs = {
                "age": np.random.choice([0, 1]),
                "gender": gender,
                "education": np.random.choice([0, 1]),
                "region": np.random.choice([0, 1]),
            }
            biased_observations.append(obs)
            raker.partial_fit(obs)

        # Check that gender margin is closer to target after raking
        final_margins = raker.margins
        raw_gender_prop = sum(obs["gender"] for obs in biased_observations) / n_obs

        # Raw proportion should be around 0.3 (biased)
        assert 0.25 <= raw_gender_prop <= 0.35

        # Weighted margin should be closer to target 0.51
        gender_error_raw = abs(raw_gender_prop - 0.51)
        gender_error_weighted = abs(final_margins["gender"] - 0.51)
        assert gender_error_weighted < gender_error_raw

    def test_education_bias_correction(self):
        """Test correcting education bias."""
        # Target: 40% have higher education
        targets = Targets(age=0.5, gender=0.5, education=0.4, region=0.3)
        raker = OnlineRakingSGD(targets, learning_rate=3.0)

        # Simulate over-educated sample (60% have higher education)
        np.random.seed(123)
        n_obs = 150

        for i in range(n_obs):
            education = 1 if np.random.random() < 0.6 else 0
            obs = {
                "age": np.random.choice([0, 1]),
                "gender": np.random.choice([0, 1]),
                "education": education,
                "region": np.random.choice([0, 1]),
            }
            raker.partial_fit(obs)

        final_margins = raker.margins
        # Weighted education margin should be closer to 0.4
        education_error = abs(final_margins["education"] - 0.4)
        assert education_error < 0.15  # Should be reasonably close


def test_sgd_vs_mwu_comparison():
    """Compare SGD and MWU on the same stream."""
    targets = Targets(age=0.6, gender=0.5, education=0.3, region=0.4)

    sgd_raker = OnlineRakingSGD(targets, learning_rate=3.0)
    mwu_raker = OnlineRakingMWU(targets, learning_rate=1.0)

    # Generate a biased stream
    np.random.seed(456)
    observations = []

    for i in range(200):
        # Age bias: 80% young people
        age = 1 if np.random.random() < 0.2 else 0
        obs = {
            "age": age,
            "gender": np.random.choice([0, 1]),
            "education": np.random.choice([0, 1]),
            "region": np.random.choice([0, 1]),
        }
        observations.append(obs)

        sgd_raker.partial_fit(obs)
        mwu_raker.partial_fit(obs)

    # Both should improve age margin compared to raw data
    raw_age_prop = sum(obs["age"] for obs in observations) / len(observations)
    sgd_age_margin = sgd_raker.margins["age"]
    mwu_age_margin = mwu_raker.margins["age"]

    raw_age_error = abs(raw_age_prop - targets.age)
    sgd_age_error = abs(sgd_age_margin - targets.age)
    mwu_age_error = abs(mwu_age_margin - targets.age)

    # Both algorithms should reduce the bias
    assert sgd_age_error < raw_age_error
    assert mwu_age_error < raw_age_error

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
        assert all(
            np.isfinite(v) for v in margins.values()
        ), "Margins became infinite or NaN"

        # Loss must remain finite
        assert np.isfinite(mwu.loss), "Loss became infinite or NaN"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
