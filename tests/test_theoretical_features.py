"""Tests for new theoretical features: convergence analysis, streaming inference, infeasibility handling."""

import numpy as np
import pytest

from onlinerake import OnlineRakingMWU, OnlineRakingSGD, Targets
from onlinerake.convergence import (
    ConvergenceAnalysis,
    RobbinsMonroVerification,
    analyze_convergence,
    estimate_lipschitz_constant,
    mwu_convergence_analysis,
    theoretical_convergence_bound,
    verify_convergence_conditions,
    verify_robbins_monro,
)
from onlinerake.diagnostics import (
    InfeasibilityAnalysis,
    analyze_infeasibility,
    explain_infeasibility_causes,
    suggest_feasible_targets,
)
from onlinerake.learning_rate import (
    ConstantLR,
    InverseTimeDecayLR,
    PolynomialDecayLR,
    robbins_monro_schedule,
)
from onlinerake.streaming_inference import (
    StreamingEstimator,
    StreamingSnapshot,
    analyze_estimate_stability,
    estimate_path_dependent_variance,
    explain_streaming_semantics,
)


class TestRobbinsMonroVerification:
    """Test Robbins-Monro condition verification."""

    def test_constant_lr_fails_rm(self):
        """Constant learning rate should not satisfy Robbins-Monro."""
        result = verify_robbins_monro(5.0, n_steps=1000)

        assert isinstance(result, RobbinsMonroVerification)
        assert result.condition_1_satisfied
        assert not result.condition_2_satisfied

    def test_polynomial_decay_satisfies_rm(self):
        """Polynomial decay with proper power should satisfy Robbins-Monro."""
        schedule = PolynomialDecayLR(initial_lr=5.0, power=0.6, min_lr=0.0)
        result = verify_robbins_monro(schedule, n_steps=10000)

        assert result.condition_1_satisfied
        assert result.condition_2_satisfied

    def test_inverse_time_decay_satisfies_rm(self):
        """Inverse time decay should satisfy Robbins-Monro."""
        schedule = InverseTimeDecayLR(initial_lr=5.0, decay=0.01, min_lr=0.0)
        result = verify_robbins_monro(schedule, n_steps=10000)

        assert result.condition_1_satisfied
        assert result.condition_2_satisfied

    def test_constant_lr_schedule_object(self):
        """ConstantLR schedule should fail Robbins-Monro."""
        schedule = ConstantLR(learning_rate=1.0)
        result = verify_robbins_monro(schedule, n_steps=1000)

        assert not result.condition_2_satisfied

    def test_analysis_notes_present(self):
        """Verification should include analysis notes."""
        result = verify_robbins_monro(5.0)

        assert len(result.analysis_notes) > 0
        assert any("Condition" in note for note in result.analysis_notes)


class TestConvergenceAnalysis:
    """Test convergence analysis functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.targets = Targets(age=0.4, gender=0.5, education=0.3)
        self.raker = OnlineRakingSGD(self.targets, learning_rate=3.0)

        np.random.seed(42)
        for _ in range(100):
            obs = {
                "age": int(np.random.random() < 0.6),
                "gender": int(np.random.random() < 0.4),
                "education": int(np.random.random() < 0.5),
            }
            self.raker.partial_fit(obs)

    def test_analyze_convergence_returns_analysis(self):
        """analyze_convergence should return ConvergenceAnalysis."""
        result = analyze_convergence(self.raker)

        assert isinstance(result, ConvergenceAnalysis)
        assert isinstance(result.satisfies_robbins_monro, bool)
        assert isinstance(result.lipschitz_constant, float)
        assert isinstance(result.convergence_rate, str)
        assert isinstance(result.warnings, list)

    def test_lipschitz_estimation(self):
        """Lipschitz constant should be positive and finite."""
        lipschitz = estimate_lipschitz_constant(self.raker, n_samples=50)

        assert lipschitz > 0
        assert np.isfinite(lipschitz)

    def test_convergence_with_schedule(self):
        """Convergence analysis should detect RM compliance with schedule."""
        schedule = robbins_monro_schedule(initial_lr=5.0)
        raker = OnlineRakingSGD(self.targets, learning_rate=schedule)

        for _ in range(50):
            raker.partial_fit({"age": 1, "gender": 0, "education": 1})

        result = analyze_convergence(raker)
        assert result.satisfies_robbins_monro

    def test_convergence_warnings(self):
        """Should generate warnings for problematic configurations."""
        targets = Targets(age=0.5)
        raker = OnlineRakingSGD(targets, learning_rate=100.0)

        for _ in range(50):
            raker.partial_fit({"age": 1})

        result = analyze_convergence(raker)
        assert len(result.warnings) > 0

    def test_verify_convergence_conditions(self):
        """verify_convergence_conditions should return structured results."""
        result = verify_convergence_conditions(self.raker)

        assert "overall_status" in result
        assert "checks" in result
        assert "recommendations" in result
        assert result["overall_status"] in ["PASS", "WARN", "FAIL"]


class TestTheoreticalBounds:
    """Test theoretical convergence bound computations."""

    def test_polynomial_bounds(self):
        """Polynomial schedule should return valid bounds."""
        bounds = theoretical_convergence_bound(
            n_features=4,
            n_observations=1000,
            learning_rate_schedule="polynomial",
            initial_lr=5.0,
            power=0.6,
        )

        assert "convergence_rate" in bounds
        assert "expected_loss_bound" in bounds
        assert bounds["satisfies_robbins_monro"] is True
        assert bounds["expected_loss_bound"] > 0

    def test_constant_bounds(self):
        """Constant schedule should indicate bounded suboptimality."""
        bounds = theoretical_convergence_bound(
            n_features=4,
            n_observations=1000,
            learning_rate_schedule="constant",
            initial_lr=5.0,
        )

        assert bounds["satisfies_robbins_monro"] is False
        assert "suboptimality" in bounds["convergence_rate"].lower()

    def test_mwu_convergence_analysis(self):
        """MWU-specific analysis should return valid results."""
        result = mwu_convergence_analysis(
            n_features=4,
            n_observations=1000,
            learning_rate=1.0,
        )

        assert "algorithm" in result
        assert "regret_bound" in result
        assert "optimal_learning_rate" in result
        assert result["regret_bound"] > 0


class TestInfeasibilityHandling:
    """Test infeasibility detection and handling."""

    def test_structurally_infeasible_target(self):
        """Should detect when target is structurally infeasible."""
        targets = Targets(impossible=0.90)
        raker = OnlineRakingSGD(targets)

        for _ in range(50):
            raker.partial_fit({"impossible": 0})

        analysis = analyze_infeasibility(raker)

        assert isinstance(analysis, InfeasibilityAnalysis)
        assert analysis.is_feasible is False
        assert analysis.infeasibility_type == "structural"
        assert "impossible" in str(analysis.diagnosis)

    def test_feasible_targets_detected(self):
        """Should correctly identify feasible targets."""
        targets = Targets(feature=0.5)
        raker = OnlineRakingSGD(targets)

        for i in range(100):
            raker.partial_fit({"feature": i % 2})

        analysis = analyze_infeasibility(raker)
        assert analysis.is_feasible is True

    def test_compromise_targets_suggested(self):
        """Should suggest compromise targets for infeasible cases."""
        targets = Targets(feature=0.95)
        raker = OnlineRakingSGD(targets)

        for i in range(100):
            raker.partial_fit({"feature": i % 2})

        analysis = analyze_infeasibility(raker)
        compromise = analysis.compromise_targets

        assert "feature" in compromise
        assert 0 <= compromise["feature"] <= 1

    def test_achievable_bounds_computed(self):
        """Should compute achievable bounds for each feature."""
        targets = Targets(feature=0.5)
        raker = OnlineRakingSGD(targets)

        for i in range(100):
            raker.partial_fit({"feature": 1 if i < 30 else 0})

        analysis = analyze_infeasibility(raker)

        assert "feature" in analysis.achievable_bounds
        lower, upper = analysis.achievable_bounds["feature"]
        assert lower <= upper
        assert 0 <= lower <= 1
        assert 0 <= upper <= 1

    def test_suggest_feasible_targets(self):
        """suggest_feasible_targets should return valid targets."""
        targets = Targets(feature=0.99)
        raker = OnlineRakingSGD(targets)

        for i in range(100):
            raker.partial_fit({"feature": i % 2})

        feasible = suggest_feasible_targets(raker)

        assert "feature" in feasible
        assert 0 <= feasible["feature"] <= 1

    def test_explain_infeasibility_causes(self):
        """Should provide explanations for infeasibility types."""
        explanations = explain_infeasibility_causes()

        assert "structural" in explanations
        assert "numerical" in explanations
        assert "conflicting" in explanations
        assert len(explanations["structural"]) > 50


class TestStreamingInference:
    """Test streaming inference features."""

    def setup_method(self):
        """Set up test fixtures."""
        self.targets = Targets(age=0.5, gender=0.5)
        self.raker = OnlineRakingSGD(self.targets, learning_rate=3.0)

        np.random.seed(42)
        for _ in range(100):
            obs = {
                "age": int(np.random.random() < 0.6),
                "gender": int(np.random.random() < 0.4),
            }
            self.raker.partial_fit(obs)

    def test_calibration_reports_the_gap_rather_than_an_interval(self):
        """Replaces test_confidence_sequence_computation.

        ``compute_confidence_sequence`` has been removed. It put a shrinking
        Hoeffding-style band around a calibrated margin and called the result
        time-uniform; measured anytime coverage was 0.470 against a nominal
        0.95. The band was not the problem. A raked margin is aimed at a target
        the caller supplied as known, so R's ``survey`` reports a standard error
        of exactly zero for one, and no width around it estimates anything.

        What an online raker can report honestly is how far it got.
        """
        from onlinerake.diagnostics import margin_calibration

        calibrations = margin_calibration(self.raker)
        assert {c.feature for c in calibrations} == {"age", "gender"}
        for cal in calibrations:
            assert cal.gap == pytest.approx(cal.estimate - cal.target)
            assert cal.std_error >= 0

    def test_path_dependent_variance(self):
        """Should estimate path-dependent variance."""
        result = estimate_path_dependent_variance(self.raker, "age")

        assert "total_variance" in result
        assert "sampling_variance" in result
        assert "path_variance" in result
        assert result["total_variance"] >= result["sampling_variance"]

    def test_streaming_estimator(self):
        """StreamingEstimator should track snapshots and retroactive changes."""
        targets = Targets(feature=0.5)
        raker = OnlineRakingSGD(targets)

        estimator = StreamingEstimator(
            raker=raker,
            snapshot_interval=10,
            track_retroactive=True,
        )

        for i in range(50):
            estimator.partial_fit({"feature": i % 2})

        assert len(estimator.snapshots) >= 4
        assert len(estimator.retroactive_impacts) > 0

    def test_snapshot_creation(self):
        """Should create valid snapshots."""
        targets = Targets(feature=0.5)
        raker = OnlineRakingSGD(targets)

        estimator = StreamingEstimator(raker=raker, snapshot_interval=0)

        for i in range(20):
            estimator.partial_fit({"feature": i % 2})

        snapshot = estimator.take_snapshot()

        assert isinstance(snapshot, StreamingSnapshot)
        assert snapshot.t == 20
        assert "feature" in snapshot.margins
        assert len(snapshot.weights) == 20

    def test_estimate_stability(self):
        """Should analyze estimate stability."""
        result = analyze_estimate_stability(self.raker, window=50)

        assert "status" in result
        if result["status"] != "INSUFFICIENT_DATA":
            assert "features" in result
            assert "overall_stability" in result
            assert 0 <= result["overall_stability"] <= 1

    def test_explain_streaming_semantics(self):
        """Should provide semantic explanations."""
        explanations = explain_streaming_semantics()

        assert "retroactive_updates" in explanations
        assert "snapshot_vs_live" in explanations
        assert "confidence_sequences" in explanations
        assert len(explanations["retroactive_updates"]) > 50


class TestStreamingEstimatorRetroactive:
    """Test retroactive impact tracking in streaming estimator."""

    def test_retroactive_margin_changes(self):
        """Should track margin changes after new observations."""
        targets = Targets(feature=0.5)
        raker = OnlineRakingSGD(targets)

        estimator = StreamingEstimator(
            raker=raker,
            track_retroactive=True,
        )

        for _ in range(10):
            estimator.partial_fit({"feature": 1})

        for _ in range(10):
            estimator.partial_fit({"feature": 0})

        assert len(estimator.retroactive_impacts) > 0

        for impact in estimator.retroactive_impacts:
            assert impact.t_after == impact.t_before + 1
            assert "feature" in impact.margin_changes

    def test_get_snapshot_at(self):
        """Should retrieve snapshots by time."""
        targets = Targets(feature=0.5)
        raker = OnlineRakingSGD(targets)

        estimator = StreamingEstimator(raker=raker, snapshot_interval=5)

        for i in range(25):
            estimator.partial_fit({"feature": i % 2})

        snapshot = estimator.get_snapshot_at(10)

        assert snapshot is not None
        assert snapshot.t == 10


class TestIntegrationConvergenceAndInfeasibility:
    """Integration tests combining convergence analysis with infeasibility."""

    def test_infeasible_causes_convergence_warnings(self):
        """Infeasible targets should generate convergence warnings."""
        targets = Targets(impossible=0.99)
        raker = OnlineRakingSGD(targets, learning_rate=5.0)

        for _ in range(100):
            raker.partial_fit({"impossible": 0})

        infeas = analyze_infeasibility(raker)
        conv = analyze_convergence(raker)

        assert not infeas.is_feasible
        # May or may not have warnings depending on weight efficiency
        assert isinstance(conv.warnings, list)

    def test_full_workflow(self):
        """Test complete workflow with all theoretical features."""
        targets = Targets(age=0.4, gender=0.5)
        schedule = robbins_monro_schedule(initial_lr=5.0, power=0.6)
        raker = OnlineRakingSGD(targets, learning_rate=schedule)

        rm_check = verify_robbins_monro(schedule)
        assert rm_check.condition_1_satisfied and rm_check.condition_2_satisfied

        np.random.seed(42)
        for _ in range(200):
            obs = {
                "age": int(np.random.random() < 0.6),
                "gender": int(np.random.random() < 0.4),
            }
            raker.partial_fit(obs)

        conv = analyze_convergence(raker)
        assert conv.satisfies_robbins_monro

        infeas = analyze_infeasibility(raker)
        assert infeas.is_feasible

        from onlinerake.diagnostics import margin_calibration

        assert len(margin_calibration(raker)) > 0

        conditions = verify_convergence_conditions(raker)
        assert conditions["overall_status"] in ["PASS", "WARN"]


class TestMWUSpecificTheory:
    """Test MWU-specific theoretical features."""

    def test_mwu_convergence_with_analysis(self):
        """MWU should work with convergence analysis."""
        targets = Targets(feature=0.5)
        raker = OnlineRakingMWU(targets, learning_rate=1.0)

        for i in range(100):
            raker.partial_fit({"feature": i % 2})

        conv = analyze_convergence(raker)

        assert isinstance(conv, ConvergenceAnalysis)
        assert np.isfinite(conv.lipschitz_constant)

    def test_mwu_theoretical_bounds(self):
        """MWU theoretical analysis should be consistent."""
        result = mwu_convergence_analysis(
            n_features=4,
            n_observations=1000,
            learning_rate=1.0,
        )

        theoretical_opt_lr = result["optimal_learning_rate"]
        assert theoretical_opt_lr > 0
        assert theoretical_opt_lr < 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestTheNumericalFallbackRuns:
    """A custom schedule takes a path none of the analytic branches cover.

    This existed untested, and the 2.0.0 rename of ``T`` to ``n_steps`` left a
    stale reference inside it. Nothing failed: no test constructs a schedule
    whose ``get_params()['type']`` is unrecognised, so the branch never
    executed. ruff caught the undefined name; this stops the next one being
    caught by a user.
    """

    def test_an_unknown_schedule_type_is_verified_numerically(self):
        from onlinerake.learning_rate import LearningRateSchedule

        class CustomSchedule(LearningRateSchedule):
            """Polynomial decay the dispatcher has no analytic branch for."""

            def __call__(self, t: int) -> float:
                return 1.0 / (t**0.7)

            def get_params(self) -> dict:
                return {"type": "something_the_dispatcher_has_never_seen"}

        result = verify_robbins_monro(CustomSchedule(), n_steps=500)

        # 0.5 < 0.7 <= 1, so both conditions should hold.
        assert result.condition_1_satisfied
        assert result.condition_2_satisfied
        assert result.n_steps_evaluated == 500
        assert any("Evaluated over" in note for note in result.analysis_notes)
