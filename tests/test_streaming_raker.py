"""Tests for fixed-memory streaming raking."""

from dataclasses import dataclass

import numpy as np
import pytest

from onlinerake import Targets
from onlinerake.streaming_raker import StreamingRaker, arrival_weight


def _first_order_root(delta, residual, design_weight, strength):
    """Root of f'(w) by bisection on log w; shares no code with the closed form."""

    def slope(w):
        return (
            np.log(w / design_weight)
            + strength * float(delta @ residual)
            + strength * float(delta @ delta) * w
        )

    lo, hi = np.log(1e-12), np.log(1e12)
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
        _first_order_root(delta, np.zeros(2), 1.0, 0.1), rel=1e-9
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
    # a + b = 1 in every row, so the best attainable loss is 2 * 0.4**2.
    assert raker.loss >= 0.32 - 1e-9


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
