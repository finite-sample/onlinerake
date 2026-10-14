"""Tests for fixed-memory streaming raking."""

import time
from dataclasses import dataclass

import numpy as np
import pytest

from onlinerake import BatchIPF, Targets
from onlinerake.streaming_raker import (
    StreamingRaker,
    arrival_weight,
    fit_penalized_raking,
)


def _first_order_root(delta, residual, design_weight, strength):
    """Root of f'(w) by bisection on log w; shares no code with the closed form."""

    def slope(w):
        return (
            np.log(w / design_weight)
            + strength * float(delta @ residual)
            + strength * float(delta @ delta) * w
        )

    lo, hi = np.log(1e-12 * design_weight), np.log(1e12 * design_weight)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if slope(np.exp(mid)) > 0:
            hi = mid
        else:
            lo = mid
    return float(np.exp(0.5 * (lo + hi)))


def test_arrival_weight_solves_first_order_condition():
    rng = np.random.default_rng(0)
    for _ in range(500):
        p = int(rng.integers(1, 12))
        delta = rng.normal(size=p)
        residual = rng.normal(size=p) * rng.choice([0.0, 1.0, 50.0])
        strength = float(10 ** rng.uniform(-3, 2))
        design_weight = float(rng.uniform(0.2, 5.0))
        weight = arrival_weight(delta, residual, design_weight, strength, 1e-12, 1e12)
        expected = _first_order_root(delta, residual, design_weight, strength)
        assert weight == pytest.approx(expected, rel=1e-9)


def test_arrival_weight_clips_to_bounds():
    delta = np.array([1.0, -1.0])
    pushes_up = arrival_weight(delta, np.array([-1e4, 1e4]), 1.0, 1.0, 0.5, 2.0)
    pushes_down = arrival_weight(delta, np.array([1e4, -1e4]), 1.0, 1.0, 0.5, 2.0)
    assert pushes_up == 2.0
    assert pushes_down == 0.5


def test_running_state_matches_recomputation():
    targets = Targets(a=0.3, b=0.6, c=0.5)
    raker = StreamingRaker(targets, keep_weights=True)
    rng = np.random.default_rng(1)
    rows = (rng.random((500, 3)) < [0.5, 0.4, 0.7]).astype(int)
    for row in rows:
        raker.partial_fit(dict(zip("abc", row.tolist(), strict=True)))

    w = raker.weights
    total = w.sum()
    margins = (w @ rows) / total
    assert raker.n_obs == 500
    assert raker.total_weight == pytest.approx(total, rel=1e-9)
    assert raker.effective_sample_size == pytest.approx(total**2 / (w @ w), rel=1e-9)
    for i, name in enumerate("abc"):
        assert raker.margins[name] == pytest.approx(margins[i], rel=1e-9)
    t = np.array([0.3, 0.6, 0.5])
    assert raker.loss == pytest.approx(float(((margins - t) ** 2).sum()), rel=1e-9)
    np.testing.assert_allclose(raker.residuals, w @ (rows - t), rtol=1e-9)


def test_first_observation_sets_margins_to_that_row():
    raker = StreamingRaker(Targets(a=0.3, b=0.6))
    weight = raker.partial_fit({"a": 1, "b": 0})
    assert raker.margins == pytest.approx({"a": 1.0, "b": 0.0}, rel=1e-12)
    delta = np.array([0.7, -0.6])
    assert weight == pytest.approx(
        _first_order_root(delta, np.zeros(2), 1.0, 0.3), rel=1e-9
    )


def test_row_on_target_keeps_design_weight():
    raker = StreamingRaker(Targets(age=(35.0, "mean")))
    assert raker.partial_fit({"age": 35.0}, design_weight=2.0) == 2.0


def test_infeasible_targets_stay_finite_and_bounded():
    raker = StreamingRaker(Targets(a=0.9, b=0.9), keep_weights=True)
    rng = np.random.default_rng(2)
    for first in rng.random(2000) < 0.5:
        raker.partial_fit({"a": int(first), "b": int(not first)})
    w = raker.weights
    assert np.all((w >= 1e-3) & (w <= 100.0))
    assert np.all(np.isfinite(raker.residuals))
    # a + b = 1 in every row, so no weighting gets below 2 * 0.4**2; the stream
    # should reach that floor, not merely stay above it.
    assert raker.loss == pytest.approx(0.32, abs=1e-4)


def test_bounds_apply_to_weight_relative_to_design_weight():
    # An expansion weight of 5,000 with almost no calibration pull must stay
    # near 5,000: the bounds limit w / d, they do not cap w at 100.
    delta = np.array([0.5])
    weight = arrival_weight(delta, np.zeros(1), 5000.0, 1e-12, 1e-3, 100.0)
    assert weight == pytest.approx(5000.0, rel=1e-6)
    low = arrival_weight(delta, np.array([1e6]), 5000.0, 1.0, 1e-3, 100.0)
    assert low == pytest.approx(5.0)


@pytest.mark.parametrize("strength", [1e-12, 1e12])
def test_extreme_strengths_are_finite(strength):
    raker = StreamingRaker(Targets(a=0.2, b=0.7), calibration_strength=strength)
    rng = np.random.default_rng(3)
    for row in (rng.random((300, 2)) < 0.5).astype(int):
        weight = raker.partial_fit({"a": int(row[0]), "b": int(row[1])})
        assert 1e-3 <= weight <= 100.0
        if strength == 1e-12:
            assert weight == pytest.approx(1.0, rel=1e-6)
    assert np.isfinite(raker.loss)


def test_object_and_missing_features_match_dict_input():
    @dataclass
    class Row:
        a: int
        b: int = 0

    from_objects = StreamingRaker(Targets(a=0.4, b=0.5))
    from_dicts = StreamingRaker(Targets(a=0.4, b=0.5))
    for a in [1, 0, 1, 1, 0]:
        from_objects.partial_fit(Row(a=a))
        from_dicts.partial_fit({"a": a})
    assert from_objects.margins == from_dicts.margins


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"calibration_strength": 0.0}, "calibration_strength must be positive"),
        ({"calibration_strength": -1.0}, "calibration_strength must be positive"),
        (
            {"calibration_strength": float("nan")},
            "calibration_strength must be positive",
        ),
        ({"min_weight": 0.0}, "min_weight must be positive"),
        ({"min_weight": 2.0, "max_weight": 1.0}, "max_weight must exceed"),
    ],
)
def test_invalid_settings_raise(kwargs, message):
    with pytest.raises(ValueError, match=message):
        StreamingRaker(Targets(a=0.5), **kwargs)


def test_empty_raker_reports_nan():
    raker = StreamingRaker(Targets(a=0.5))
    assert np.isnan(raker.margins["a"])
    assert np.isnan(raker.loss)
    assert raker.effective_sample_size == 0.0


def test_outcome_estimates_match_weighted_means():
    raker = StreamingRaker(Targets(a=0.4), outcomes=["y", "z"], keep_weights=True)
    rng = np.random.default_rng(4)
    rows = (rng.random(400) < 0.6).astype(int)
    ys = rng.normal(size=400)
    zs = rng.normal(size=400)
    for a, y, z in zip(rows, ys, zs, strict=True):
        raker.partial_fit({"a": int(a)}, outcomes={"y": float(y), "z": float(z)})
    w = raker.weights
    assert raker.estimate("y") == pytest.approx((w @ ys) / w.sum(), rel=1e-9)
    assert raker.estimates["z"] == pytest.approx((w @ zs) / w.sum(), rel=1e-9)


@pytest.mark.parametrize(
    ("obs", "kwargs", "message"),
    [
        ({"a": 2}, {"outcomes": {"y": 1.0}}, "binary"),
        ({"a": 1}, {"outcomes": {}}, "missing outcomes"),
        ({"a": 1}, {"outcomes": {"y": 1.0, "extra": 2.0}}, "unregistered"),
        ({"a": 1}, {"outcomes": {"y": float("nan")}}, "y must be finite"),
        ({"a": 1}, {"outcomes": {"y": 1.0}, "design_weight": 0.0}, "design_weight"),
    ],
)
def test_rejected_rows_leave_state_untouched(obs, kwargs, message):
    raker = StreamingRaker(Targets(a=0.4), outcomes=["y"])
    raker.partial_fit({"a": 1}, outcomes={"y": 1.0})
    before = (raker.n_obs, raker.total_weight, raker.residuals, raker.estimates)
    with pytest.raises(ValueError, match=message):
        raker.partial_fit(obs, **kwargs)
    after = (raker.n_obs, raker.total_weight, raker.residuals, raker.estimates)
    assert after[:2] == before[:2]
    np.testing.assert_array_equal(after[2], before[2])
    assert after[3] == before[3]


def test_outcomes_without_registration_raise():
    raker = StreamingRaker(Targets(a=0.4))
    with pytest.raises(ValueError, match="unregistered"):
        raker.partial_fit({"a": 1}, outcomes={"y": 1.0})


def test_estimate_errors_and_empty_state():
    raker = StreamingRaker(Targets(a=0.4), outcomes=["y"])
    assert np.isnan(raker.estimate("y"))
    with pytest.raises(ValueError, match="unregistered outcome"):
        raker.estimate("nope")


def test_weights_require_keep_weights():
    raker = StreamingRaker(Targets(a=0.4))
    raker.partial_fit({"a": 1})
    with pytest.raises(RuntimeError, match="keep_weights"):
        _ = raker.weights


def test_state_does_not_grow_with_stream():
    raker = StreamingRaker(Targets(a=0.4, b=0.5), outcomes=["y"])
    rng = np.random.default_rng(5)

    def feed(k):
        for a, b in (rng.random((k, 2)) < 0.5).astype(int):
            raker.partial_fit({"a": int(a), "b": int(b)}, outcomes={"y": 1.0})

    def array_shapes():
        return {
            key: value.shape
            for key, value in vars(raker).items()
            if isinstance(value, np.ndarray)
        }

    feed(10)
    shapes = array_shapes()
    feed(5000)
    assert array_shapes() == shapes
    assert raker._weights is None


def test_time_per_arrival_does_not_grow_with_stream():
    rng = np.random.default_rng(6)
    targets = Targets(**{f"f{i}": 0.5 for i in range(8)})

    def late_cost(n, tail=2000):
        best = float("inf")
        for _ in range(3):
            raker = StreamingRaker(targets)
            rows = (rng.random((n, 8)) < 0.4).astype(int)
            dicts = [
                dict(zip(targets.feature_names, r.tolist(), strict=True)) for r in rows
            ]
            for obs in dicts[:-tail]:
                raker.partial_fit(obs)
            start = time.perf_counter()
            for obs in dicts[-tail:]:
                raker.partial_fit(obs)
            best = min(best, (time.perf_counter() - start) / tail)
        return best

    assert late_cost(40_000) < 2.0 * late_cost(2_500)


def _easy_stream(n, p, seed):
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.3, 0.7, p)
    target = np.clip(raw + rng.uniform(-0.15, 0.15, p), 0.15, 0.85)
    rows = (rng.random((n, p)) < raw).astype(int)
    names = [f"f{i}" for i in range(p)]
    targets = Targets(**{name: float(target[i]) for i, name in enumerate(names)})
    return targets, rows, target


def test_penalized_batch_reproduces_ipf_at_large_strength():
    targets, rows, _ = _easy_stream(300, 3, 0)
    names = targets.feature_names
    ipf = BatchIPF(targets)
    ipf.fit([dict(zip(names, r.tolist(), strict=True)) for r in rows])
    batch = fit_penalized_raking(rows, targets, calibration_strength=1e6)
    expected = ipf.weights / ipf.weights.mean()
    np.testing.assert_allclose(batch / batch.mean(), expected, rtol=1e-6)


def test_penalized_batch_satisfies_tilt_condition():
    targets, rows, target = _easy_stream(500, 4, 1)
    design = np.random.default_rng(2).uniform(0.5, 2.0, 500)
    c = 0.1
    w = fit_penalized_raking(
        rows, targets, calibration_strength=c, design_weights=design
    )
    delta = rows - target
    residual = w @ delta
    np.testing.assert_allclose(w, design * np.exp(-c * (delta @ residual)), rtol=1e-8)


def test_penalized_batch_rejects_bad_shapes():
    targets, rows, _ = _easy_stream(50, 3, 3)
    with pytest.raises(ValueError, match="features must have shape"):
        fit_penalized_raking(rows[:, :2], targets)
    with pytest.raises(ValueError, match="design_weights"):
        fit_penalized_raking(rows, targets, design_weights=np.ones(49))


def test_fixed_step_gradient_update_does_not_calibrate():
    """Why StreamingRaker solves for the arriving weight exactly.

    A fixed step on the gradient (1/W) * sum_j (m_j - t_j)(x_j - m_j) moves the
    arriving weight by O(1/k) after k rows. The moves sum to O(log n) against a
    total weight of n, so margins stay at their unweighted values.
    """
    targets, rows, target = _easy_stream(3000, 8, 4)
    totals = np.zeros(8)
    weight_sum = 0.0
    for x in rows.astype(float):
        w = 1.0
        totals += x
        weight_sum += 1.0
        for _ in range(3):
            m = totals / weight_sum
            grad = float((m - target) @ (x - m)) / weight_sum
            new = min(max(w - 5.0 * grad, 1e-3), 100.0)
            totals += (new - w) * x
            weight_sum += new - w
            w = new
    raw_error = np.abs(rows.mean(axis=0) - target).mean()
    fixed_step_error = np.abs(totals / weight_sum - target).mean()
    assert fixed_step_error == pytest.approx(raw_error, rel=0.02)

    raker = StreamingRaker(targets)
    for r in rows:
        raker.partial_fit(dict(zip(targets.feature_names, r.tolist(), strict=True)))
    streamed_error = np.mean(
        [abs(raker.margins[n] - targets[n]) for n in targets.feature_names]
    )
    assert streamed_error < 0.1 * raw_error
