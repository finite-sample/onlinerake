"""Tests for new features: batch IPF, learning rate schedules, diagnostics, sensitivity."""

import numpy as np
import pytest

from onlinerake import BatchIPF, OnlineRakingMWU, OnlineRakingSGD, Targets
from onlinerake.diagnostics import (
    check_target_feasibility,
    compute_confidence_interval,
    compute_design_effect,
    compute_weight_efficiency,
    estimate_margin_std_error,
    estimate_margin_variance,
    get_margin_estimates,
    summarize_raking_results,
)
from onlinerake.learning_rate import (
    AdaptiveLR,
    ConstantLR,
    InverseTimeDecayLR,
    PolynomialDecayLR,
    robbins_monro_schedule,
)
from onlinerake.sensitivity import quick_sensitivity_check, run_sensitivity_analysis


class TestBatchIPF:
    """Test batch IPF baseline implementation."""

    def test_basic_fit(self):
        """Test basic IPF fitting."""
        targets = Targets(age=0.5, gender=0.5)
        ipf = BatchIPF(targets)

        observations = [
            {"age": 1, "gender": 0},
            {"age": 0, "gender": 1},
            {"age": 1, "gender": 1},
            {"age": 0, "gender": 0},
        ]

        ipf.fit(observations)

        assert ipf._n_obs == 4
        assert len(ipf.weights) == 4
        assert all(w > 0 for w in ipf.weights)

    def test_convergence(self):
        """Test that IPF converges."""
        targets = Targets(age=0.5, gender=0.5)
        ipf = BatchIPF(targets, tolerance=1e-8)

        # Create observations where targets are achievable
        observations = [
            {"age": 1, "gender": 1},
            {"age": 0, "gender": 0},
        ] * 50

        ipf.fit(observations)

        # Should converge quickly with perfect targets
        assert ipf.converged
        assert ipf.loss < 0.01

    def test_margins_match_targets(self):
        """Test that weighted margins approach targets."""
        targets = Targets(age=0.4, gender=0.6)
        ipf = BatchIPF(targets)

        # Create biased observations
        np.random.seed(42)
        observations = []
        for _ in range(200):
            observations.append(
                {
                    "age": int(np.random.random() < 0.6),  # Biased toward 0.6
                    "gender": int(np.random.random() < 0.4),  # Biased toward 0.4
                }
            )

        ipf.fit(observations)
        margins = ipf.margins

        # Margins should be closer to targets than raw proportions
        raw_margins = ipf.raw_margins
        age_improvement = abs(raw_margins["age"] - 0.4) - abs(margins["age"] - 0.4)
        gender_improvement = abs(raw_margins["gender"] - 0.6) - abs(
            margins["gender"] - 0.6
        )

        assert age_improvement > 0
        assert gender_improvement > 0

    def test_online_vs_batch_convergence(self):
        """Test that online raking converges to same solution as batch IPF."""
        targets = Targets(age=0.4, gender=0.5, education=0.3)

        # Generate observations
        np.random.seed(123)
        observations = []
        for _ in range(300):
            observations.append(
                {
                    "age": int(np.random.random() < 0.6),
                    "gender": int(np.random.random() < 0.4),
                    "education": int(np.random.random() < 0.5),
                }
            )

        # Fit batch IPF
        ipf = BatchIPF(targets)
        ipf.fit(observations)

        # Fit online SGD
        sgd = OnlineRakingSGD(targets, learning_rate=5.0)
        for obs in observations:
            sgd.partial_fit(obs)

        # Both should achieve similar final loss
        # Note: They may not be exactly equal due to different algorithms
        assert abs(sgd.loss - ipf.loss) < 0.1 or (sgd.loss < 0.1 and ipf.loss < 0.1)

    def test_incremental_fit(self):
        """Test incremental IPF fitting."""
        targets = Targets(age=0.5, gender=0.5)
        ipf = BatchIPF(targets)

        batch1 = [{"age": 1, "gender": 0}, {"age": 0, "gender": 1}]
        batch2 = [{"age": 1, "gender": 1}, {"age": 0, "gender": 0}]

        ipf.fit(batch1)
        assert ipf._n_obs == 2

        ipf.fit_incremental(batch2)
        assert ipf._n_obs == 4


class TestLearningRateSchedules:
    """Test learning rate schedule implementations."""

    def test_constant_lr(self):
        """Test constant learning rate."""
        lr = ConstantLR(learning_rate=5.0)
        assert lr(1) == 5.0
        assert lr(100) == 5.0
        assert lr(1000) == 5.0

    def test_inverse_time_decay(self):
        """Test inverse time decay learning rate."""
        lr = InverseTimeDecayLR(initial_lr=5.0, decay=0.01, min_lr=0.1)

        assert lr(1) == pytest.approx(5.0 / 1.01, rel=0.01)
        assert lr(100) == pytest.approx(5.0 / 2.0, rel=0.01)
        # Should respect minimum
        assert lr(10000) >= 0.1

    def test_polynomial_decay(self):
        """Test polynomial decay learning rate."""
        lr = PolynomialDecayLR(initial_lr=5.0, power=0.6, min_lr=0.01)

        assert lr(1) == 5.0
        assert lr(10) < lr(1)
        assert lr(100) < lr(10)
        # Should respect minimum
        assert lr(100000) >= 0.01

    def test_robbins_monro_conditions(self):
        """Test that polynomial decay satisfies Robbins-Monro conditions."""
        lr = PolynomialDecayLR(initial_lr=5.0, power=0.6, min_lr=0.0)

        # Sum should diverge (we can't test infinity, but sum over many terms should be large)
        lr_sum = sum(lr(t) for t in range(1, 10001))
        assert lr_sum > 100  # Should be much larger than finite

        # Sum of squares should converge
        lr_sq_sum = sum(lr(t) ** 2 for t in range(1, 10001))
        # With power=0.6, sum(1/t^1.2) converges
        assert lr_sq_sum < 1000  # Should be bounded

    def test_adaptive_lr(self):
        """Test adaptive learning rate."""
        lr = AdaptiveLR(initial_lr=1.0, min_lr=0.1, max_lr=10.0)

        assert lr(1) == 1.0

        # Simulate improving loss
        lr.update(1.0)
        lr.update(0.5)  # Improving
        assert lr(2) > 1.0  # Should increase

        # Simulate worsening loss
        lr.update(1.0)  # Worsening
        current = lr(3)
        lr.update(2.0)  # Worsening again
        assert lr(4) < current  # Should decrease

    def test_sgd_with_lr_schedule(self):
        """Test SGD with learning rate schedule."""
        targets = Targets(age=0.5, gender=0.5)
        schedule = robbins_monro_schedule(initial_lr=5.0, power=0.6)

        raker = OnlineRakingSGD(targets, learning_rate=schedule)

        assert raker.uses_lr_schedule

        # Process some observations
        observations = [
            {"age": 1, "gender": 0},
            {"age": 0, "gender": 1},
        ] * 10

        for obs in observations:
            raker.partial_fit(obs)

        # Learning rate should have decreased
        assert raker.current_learning_rate < 5.0
        assert len(raker.learning_rate_history) == 20

    def test_invalid_polynomial_power(self):
        """Test that invalid polynomial power raises error."""
        with pytest.raises(ValueError, match="power must be in"):
            PolynomialDecayLR(initial_lr=5.0, power=0.4)  # Too low

        with pytest.raises(ValueError, match="power must be in"):
            PolynomialDecayLR(initial_lr=5.0, power=1.1)  # Too high


class TestDiagnostics:
    """Test diagnostic and variance estimation functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.targets = Targets(age=0.4, gender=0.5, education=0.3)
        self.raker = OnlineRakingSGD(self.targets, learning_rate=3.0)

        # Generate observations
        np.random.seed(42)
        for _ in range(100):
            obs = {
                "age": int(np.random.random() < 0.6),
                "gender": int(np.random.random() < 0.4),
                "education": int(np.random.random() < 0.5),
            }
            self.raker.partial_fit(obs)

    def test_variance_estimation(self):
        """Test variance estimation."""
        var = estimate_margin_variance(self.raker, "age")
        assert var > 0
        assert var < 0.1  # Should be reasonably small with 100 obs

    def test_std_error_estimation(self):
        """Test standard error estimation."""
        se = estimate_margin_std_error(self.raker, "age")
        assert se > 0
        assert se < 0.1

    def test_confidence_interval(self):
        """Test confidence interval computation."""
        ci = compute_confidence_interval(self.raker, "age", 0.95)
        lower, upper = ci

        assert lower < upper
        assert 0 <= lower <= 1
        assert 0 <= upper <= 1

        # Estimate should be within CI
        margins = self.raker.margins
        assert lower <= margins["age"] <= upper

    def test_margin_estimates(self):
        """Test comprehensive margin estimates."""
        estimates = get_margin_estimates(self.raker, confidence_level=0.95)

        assert len(estimates) == 3  # Three features
        for est in estimates:
            assert est.feature in ["age", "gender", "education"]
            assert 0 <= est.estimate <= 1
            assert est.std_error > 0
            assert est.ci_lower <= est.estimate <= est.ci_upper

    def test_feasibility_check(self):
        """Test target feasibility checking."""
        report = check_target_feasibility(self.raker)

        assert isinstance(report.is_feasible, bool)
        assert isinstance(report.feasibility_scores, dict)
        assert all(0 <= s <= 1 for s in report.feasibility_scores.values())

    def test_feasibility_with_infeasible_targets(self):
        """Test feasibility check with infeasible targets."""
        # Create targets that are impossible given the data
        targets = Targets(impossible=0.99)
        raker = OnlineRakingSGD(targets)

        # All observations have impossible=0
        for _ in range(50):
            raker.partial_fit({"impossible": 0})

        report = check_target_feasibility(raker)
        assert not report.is_feasible
        assert "impossible" in report.problematic_features

    def test_design_effect(self):
        """Test design effect computation."""
        deff = compute_design_effect(self.raker)
        assert deff >= 1.0  # Design effect should be at least 1

    def test_weight_efficiency(self):
        """Test weight efficiency computation."""
        efficiency = compute_weight_efficiency(self.raker)
        assert 0 < efficiency <= 1

    def test_summarize_results(self):
        """Test comprehensive result summarization."""
        summary = summarize_raking_results(self.raker)

        assert "n_observations" in summary
        assert summary["n_observations"] == 100
        assert "effective_sample_size" in summary
        assert "design_effect" in summary
        assert "margin_estimates" in summary
        assert "feasibility" in summary
        assert "weight_distribution" in summary


class TestSensitivityAnalysis:
    """Test sensitivity analysis functionality."""

    def test_basic_sensitivity_analysis(self):
        """Test basic sensitivity analysis."""
        targets = Targets(age=0.5, gender=0.5)

        observations = []
        np.random.seed(42)
        for _ in range(100):
            observations.append(
                {
                    "age": int(np.random.random() < 0.6),
                    "gender": int(np.random.random() < 0.4),
                }
            )

        report = run_sensitivity_analysis(
            observations,
            targets,
            learning_rates=[1.0, 5.0],
            n_steps_values=[1, 3],
            min_weights=[1e-3],
            max_weights=[100.0],
            seeds=[42],
        )

        assert len(report.results) > 0
        assert "learning_rate" in report.best_params
        assert "n_steps" in report.best_params

    def test_quick_sensitivity_check(self):
        """Test quick sensitivity check."""
        targets = Targets(age=0.5, gender=0.5)
        raker = OnlineRakingSGD(targets, learning_rate=3.0, n_sgd_steps=3)

        observations = []
        np.random.seed(42)
        for _ in range(50):
            observations.append(
                {
                    "age": int(np.random.random() < 0.6),
                    "gender": int(np.random.random() < 0.4),
                }
            )

        result = quick_sensitivity_check(raker, observations, n_variations=3)

        assert "baseline_lr" in result
        assert "best_lr" in result
        assert "lr_sensitivity" in result
        assert "recommendations" in result


class TestMWUWithSchedule:
    """Test MWU algorithm with learning rate schedules."""

    def test_mwu_with_constant_lr(self):
        """Test MWU with constant LR schedule object."""
        targets = Targets(age=0.5, gender=0.5)
        ConstantLR(learning_rate=1.0)

        # MWU doesn't currently support schedules directly,
        # but we should test it works with numeric LR
        raker = OnlineRakingMWU(targets, learning_rate=1.0)

        observations = [
            {"age": 1, "gender": 0},
            {"age": 0, "gender": 1},
        ] * 10

        for obs in observations:
            raker.partial_fit(obs)

        assert raker._n_obs == 20
        assert raker.loss < 0.5


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_raker_diagnostics(self):
        """Test diagnostics on empty raker."""
        targets = Targets(age=0.5)
        raker = OnlineRakingSGD(targets)

        var = estimate_margin_variance(raker, "age")
        assert np.isnan(var)

        se = estimate_margin_std_error(raker, "age")
        assert np.isnan(se)

        ci = compute_confidence_interval(raker, "age")
        assert np.isnan(ci[0]) and np.isnan(ci[1])

    def test_batch_ipf_empty_data(self):
        """Test batch IPF with empty data."""
        targets = Targets(age=0.5)
        ipf = BatchIPF(targets)

        ipf.fit([])
        assert ipf._n_obs == 0
        assert np.isnan(ipf.loss)

    def test_single_observation(self):
        """Test with single observation."""
        targets = Targets(age=0.5)
        raker = OnlineRakingSGD(targets)
        raker.partial_fit({"age": 1})

        var = estimate_margin_variance(raker, "age")
        assert var >= 0  # Should be defined

        report = check_target_feasibility(raker)
        # Single observation may not be feasible
        assert isinstance(report.is_feasible, bool)


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow_with_diagnostics(self):
        """Test full workflow with all new features."""
        # 1. Define targets
        targets = Targets(age=0.4, gender=0.5, education=0.3)

        # 2. Create raker with learning rate schedule
        schedule = robbins_monro_schedule(initial_lr=5.0, power=0.7)
        raker = OnlineRakingSGD(targets, learning_rate=schedule)

        # 3. Generate biased observations
        np.random.seed(42)
        observations = []
        for _ in range(200):
            observations.append(
                {
                    "age": int(np.random.random() < 0.6),
                    "gender": int(np.random.random() < 0.3),
                    "education": int(np.random.random() < 0.5),
                }
            )

        # 4. Process observations
        for obs in observations:
            raker.partial_fit(obs)

        # 5. Get comprehensive summary
        summary = summarize_raking_results(raker)

        # 6. Validate results
        assert summary["n_observations"] == 200
        assert summary["effective_sample_size"] > 50
        assert summary["final_loss"] < 0.5

        # 7. Check that margins improved
        for est in summary["margin_estimates"]:
            # Bias reduction should be positive
            assert est["bias_reduction_pct"] >= 0, f"{est['feature']} didn't improve"

    def test_online_matches_batch_over_time(self):
        """Test that online solution approaches batch solution."""
        targets = Targets(age=0.4, gender=0.5)

        np.random.seed(123)
        observations = []
        for _ in range(500):
            observations.append(
                {
                    "age": int(np.random.random() < 0.6),
                    "gender": int(np.random.random() < 0.4),
                }
            )

        # Fit batch IPF
        ipf = BatchIPF(targets)
        ipf.fit(observations)
        batch_margins = ipf.margins

        # Fit online SGD
        sgd = OnlineRakingSGD(targets, learning_rate=5.0)
        for obs in observations:
            sgd.partial_fit(obs)
        online_margins = sgd.margins

        # Margins should be similar
        for feature in ["age", "gender"]:
            assert abs(batch_margins[feature] - online_margins[feature]) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
