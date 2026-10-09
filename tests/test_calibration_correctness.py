import numpy as np
import pytest
from numpy.testing import assert_allclose

from onlinerake import (
    BatchIPF,
    InverseTimeDecayLR,
    OnlineRakingMWU,
    OnlineRakingSGD,
    PolynomialDecayLR,
    Targets,
    verify_robbins_monro,
)


@pytest.mark.parametrize("cls", [OnlineRakingSGD, OnlineRakingMWU])
def test_unreachable_target_does_not_converge(cls):
    raker = cls(Targets(a=0.8), convergence_window=3)
    for _ in range(5):
        raker.partial_fit({"a": 0})
    assert raker.loss == pytest.approx(0.64)
    assert not raker.converged
    assert not raker.history[-1]["converged"]


@pytest.mark.parametrize("cls", [OnlineRakingSGD, OnlineRakingMWU])
def test_convergence_is_current_and_history_agrees(cls):
    raker = cls(Targets(a=0.5), convergence_window=2, learning_rate=1e-6)
    for x in [0, 1]:
        raker.partial_fit({"a": x})
    assert raker.check_convergence()
    assert raker.convergence_step == 2
    assert raker.history[-1]["converged"]
    for _ in range(10):
        raker.partial_fit({"a": 1})
    assert raker.loss > 0.1
    assert not raker.converged
    assert raker.convergence_step is None
    assert not raker.history[-1]["converged"]


def test_ipf_requires_actual_margin_tolerance():
    raker = BatchIPF(Targets(a=0.8)).fit([{"a": 0}] * 5)
    assert not raker.converged
    data = [{"a": 0, "b": 0}] * 7 + [{"a": 1, "b": 0}] * 2 + [{"a": 1, "b": 1}] * 5
    raker = BatchIPF(Targets(a=0.6, b=0.4), tolerance=1e-8, max_iterations=1000).fit(
        data
    )
    assert raker.converged
    assert max(abs(raker.margins[k] - raker.targets[k]) for k in ["a", "b"]) <= 1e-8


@pytest.mark.parametrize("cls", [OnlineRakingSGD, OnlineRakingMWU, BatchIPF])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 2, -1, "yes"])
def test_invalid_binary_observations_are_rejected(cls, value):
    raker = cls(Targets(a=0.5))
    operation = raker.fit if cls is BatchIPF else raker.partial_fit
    observation = [{"a": value}] if cls is BatchIPF else {"a": value}
    with pytest.raises(ValueError, match="a"):
        operation(observation)
    assert len(raker.weights) == 0


@pytest.mark.parametrize("cls", [OnlineRakingSGD, OnlineRakingMWU])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_invalid_continuous_observations_do_not_mutate_state(cls, value):
    raker = cls(Targets(age=(35.0, "mean")))
    raker.partial_fit({"age": 34})
    before = raker.weights
    with pytest.raises(ValueError, match="age"):
        raker.partial_fit({"age": value})
    assert_allclose(raker.weights, before)
    assert len(raker.history) == 1


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_continuous_target_must_be_finite(value):
    with pytest.raises(ValueError, match="finite"):
        Targets(age=(value, "mean"))


@pytest.mark.parametrize("cls", [OnlineRakingSGD, OnlineRakingMWU])
@pytest.mark.parametrize(
    "options",
    [
        {"learning_rate": np.nan},
        {"learning_rate": np.inf},
        {"min_weight": np.nan},
        {"max_weight": np.inf},
        {"n_sgd_steps": 1.5},
        {"convergence_window": 1.5},
        {"max_history": 0},
    ],
)
def test_invalid_online_configuration(cls, options):
    with pytest.raises(ValueError, match=next(iter(options))):
        cls(Targets(a=0.5), **options)


@pytest.mark.parametrize("cls", [PolynomialDecayLR, InverseTimeDecayLR])
def test_positive_schedule_floor_fails_square_summability(cls):
    schedule = cls(initial_lr=0.1, min_lr=0.05)
    result = verify_robbins_monro(schedule, n_steps=100)
    assert result.condition_1_satisfied
    assert not result.condition_2_satisfied
    rates = np.array([schedule(t) for t in range(1, 101)])
    assert result.sum_lr_estimate == pytest.approx(rates.sum())
    assert result.sum_lr_sq_estimate == pytest.approx((rates**2).sum())


def test_gradient_matches_independent_finite_difference():
    raker = OnlineRakingMWU(Targets(a=0.4, z=(0.3, "mean")), learning_rate=0.1)
    for a, z in [(0, -1), (1, 0.2), (0, 1.2), (1, 0.8)]:
        raker.partial_fit({"a": a, "z": z})
    w = raker.weights
    X = raker._features[: len(w)]
    target = np.array([0.4, 0.3])

    def loss(weights):
        return np.sum((weights @ X / weights.sum() - target) ** 2)

    numeric = []
    for i in range(len(w)):
        delta = np.zeros(len(w))
        delta[i] = 1e-6
        numeric.append((loss(w + delta) - loss(w - delta)) / 2e-6)
    assert_allclose(raker._compute_gradient(), numeric, atol=1e-9)


def test_mwu_step_matches_bounded_entropic_update():
    raker = OnlineRakingMWU(Targets(a=0.7), learning_rate=0.8, n_sgd_steps=1)
    for a in [0, 1, 0]:
        raker.partial_fit({"a": a})
    old = np.r_[raker.weights, 1.0]
    x = np.array([0, 1, 0, 1])
    mean = old @ x / old.sum()
    gradient = 2 * (mean - 0.7) * (x - mean) / old.sum()
    expected = np.clip(
        old * np.exp(-0.8 * gradient), raker.min_weight, raker.max_weight
    )
    raker.partial_fit({"a": 1})
    assert_allclose(raker.weights, expected, atol=1e-14)


def test_mwu_extreme_rate_avoids_intermediate_overflow():
    raker = OnlineRakingMWU(Targets(z=(1.5e5, "mean")), learning_rate=1e308)
    with np.errstate(over="raise", invalid="raise"):
        raker.partial_fit({"z": 0})
        raker.partial_fit({"z": 2e5})
        raker.partial_fit({"z": 0})
    assert np.isfinite(raker.weights).all()
    assert np.all(raker.weights >= raker.min_weight)
    assert np.all(raker.weights <= raker.max_weight)


@pytest.mark.parametrize("cls", [OnlineRakingSGD, OnlineRakingMWU])
def test_weight_statistics_describe_current_weights(cls):
    raker = cls(Targets(a=0.8))
    raker.partial_fit({"a": 0})
    assert raker.weight_distribution_stats["mean"] == 1
    for a in [1, 0, 1]:
        raker.partial_fit({"a": a})
    stats = raker.weight_distribution_stats
    assert stats["mean"] == pytest.approx(raker.weights.mean())
    assert stats["min"] == pytest.approx(raker.weights.min())


def test_online_mwu_is_not_batch_entropy_balancing():
    data = [{"a": a} for a in [0, 1, 0, 1]]
    online = OnlineRakingMWU(Targets(a=0.7), n_sgd_steps=20)
    online.partial_fit_batch(data)
    batch = BatchIPF(Targets(a=0.7)).fit(data)
    assert_allclose(batch.weights, [0.6, 1.4, 0.6, 1.4])
    assert not np.isclose(online.weights[0], online.weights[2])


def test_ipf_matches_known_entropy_projection():
    data = [{"a": a, "b": b} for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]]
    fit = BatchIPF(Targets(a=0.7, b=0.3), tolerance=1e-12).fit(data)
    assert fit.converged
    assert_allclose(fit.weights / fit.weights.sum(), [0.21, 0.09, 0.49, 0.21])


def test_incremental_ipf_also_checks_actual_margins():
    fit = BatchIPF(Targets(a=0.8)).fit([{"a": 0}])
    fit.fit_incremental([{"a": 0}])
    assert not fit.converged


def test_incremental_ipf_accepts_empty_input_without_division_by_zero():
    fit = BatchIPF(Targets(a=0.5)).fit_incremental([])
    assert len(fit.weights) == 0
    assert not fit.converged


def test_schedule_series_check_does_not_certify_raker():
    from onlinerake import analyze_convergence, robbins_monro_schedule

    raker = OnlineRakingMWU(Targets(a=0.8), learning_rate=robbins_monro_schedule())
    for _ in range(4):
        raker.partial_fit({"a": 0})
    analysis = analyze_convergence(raker)
    assert analysis.satisfies_robbins_monro
    assert "Not established" in analysis.convergence_rate
    assert not raker.converged


@pytest.mark.parametrize("cls", [OnlineRakingSGD, OnlineRakingMWU])
def test_invalid_scheduled_rate_leaves_observations_unchanged(cls):
    raker = cls(Targets(a=0.5), learning_rate=lambda t: 1.0 if t == 1 else np.nan)
    raker.partial_fit({"a": 0})
    with pytest.raises(ValueError, match="learning_rate"):
        raker.partial_fit({"a": 1})
    assert_allclose(raker.weights, [1.0])
    assert len(raker.history) == 1


def test_curvature_diagnostic_preserves_weights_and_random_state():
    from onlinerake import estimate_lipschitz_constant

    raker = OnlineRakingMWU(Targets(a=0.7))
    raker.partial_fit_batch([{"a": a} for a in [0, 1, 0, 1]])
    before = raker.weights
    state = np.random.get_state()
    estimate_lipschitz_constant(raker, n_samples=5)
    after_state = np.random.get_state()
    assert_allclose(raker.weights, before, rtol=0, atol=0)
    assert state[0] == after_state[0]
    assert_allclose(state[1], after_state[1], rtol=0, atol=0)
    assert state[2:] == after_state[2:]


def test_curvature_diagnostic_restores_weights_on_error(monkeypatch):
    from onlinerake import estimate_lipschitz_constant

    raker = OnlineRakingMWU(Targets(a=0.7))
    raker.partial_fit_batch([{"a": a} for a in [0, 1, 0, 1]])
    before = raker.weights
    original = raker._compute_gradient
    calls = 0

    def fail_on_perturbation():
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("perturbed gradient failed")
        return original()

    monkeypatch.setattr(raker, "_compute_gradient", fail_on_perturbation)
    with pytest.raises(RuntimeError, match="perturbed gradient"):
        estimate_lipschitz_constant(raker)
    assert_allclose(raker.weights, before, rtol=0, atol=0)


def test_weight_bounds_do_not_prove_target_infeasibility():
    from onlinerake import check_target_feasibility

    raker = OnlineRakingSGD(Targets(age=(35.0, "mean")), learning_rate=1.0)
    raker.partial_fit_batch([{"age": age} for age in [25, 30, 35, 40, 45]])
    raker._cached_weight_stats = None
    report = check_target_feasibility(raker)
    assert raker.raw_margins["age"] == 35
    assert report.is_feasible
    assert report.problematic_features == []
    assert any("hitting bounds" in warning for warning in report.recommendations)
