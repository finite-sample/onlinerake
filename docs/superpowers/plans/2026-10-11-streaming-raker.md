# Streaming Raker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fixed-memory streaming raker and its batch counterpart, measure them against periodic IPF over 50 seeds, rewrite the onlinerake manuscript from those results, and reposition the package documentation.

**Architecture:** One new module, `onlinerake/streaming_raker.py`, holds the closed-form arrival update (`arrival_weight`), the `StreamingRaker` class (O(p + q) state), and `fit_penalized_raking` (batch optimum via a p-dimensional dual). A new evaluation script writes per-seed CSV rows; a separate plotting script turns them into the manuscript's figures and table. The manuscript and docs are then rewritten from those outputs.

**Tech Stack:** Python ≥3.12, numpy, pandas, scipy (`special.wrightomega`, `optimize.minimize`), pytest, ruff, pyright, Sphinx, LaTeX (latexmk, apalike).

**Spec:** `docs/superpowers/specs/2026-10-11-streaming-raker-design.md`

## Global Constraints

- Work on branch `streaming-raker` in `/Users/soodoku/Documents/GitHub/onlinerake`; one PR to `main`.
- Runtime dependencies become `numpy>=1.21`, `pandas>=1.3`, `scipy>=1.11`.
- Zero tolerance: `uv run ruff check .`, `uv run ruff format --check .`, `uv run pyright`, `uv run pytest`, `uv run sphinx-build -W -b html docs _site`.
- pytest runs with `filterwarnings = ["error"]`: any RuntimeWarning (overflow, divide) fails a test. Do not suppress warnings to pass tests.
- Comments explain why, not what. No narrating comments.
- No behavior changes to `OnlineRakingSGD`, `OnlineRakingMWU`, `BatchIPF`; documentation only.
- Default `calibration_strength = 0.1`; `min_weight = 1e-3`; `max_weight = 100.0` (same bounds as `OnlineRakingSGD`).
- Every commit message ends with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- Manuscript numbers come only from `eval_outputs/streaming_results.csv` produced in Task 5; no number is typed from memory or from the prototypes.

## Review Focus

1. A stream whose first row is the only row: margins must equal that row exactly and the weight must solve the first-order condition (no NaN from dividing by zero total weight). Test in Task 1.
2. A row exactly on target (`x == t`, possible with continuous mean targets): weight must equal the design weight, not NaN from `log(0)`. Test in Task 1.
3. Infeasible targets (two mutually exclusive features both targeted at 0.9): weights stay within bounds, state stays finite, loss stays at its feasible floor. Test in Task 1.
4. Extreme `calibration_strength` (1e-12 and 1e12): no overflow warnings; tiny strength returns the design weight. Test in Task 1.
5. A rejected observation (binary value 2, missing outcome) must leave the raker's state untouched, so a caller can catch the error and continue the stream. Test in Task 2.

---

### Task 1: Arrival update and StreamingRaker core

**Files:**
- Create: `onlinerake/streaming_raker.py`
- Create: `tests/test_streaming_raker.py`
- Modify: `pyproject.toml:26-29` (dependencies)

**Interfaces:**
- Produces: `arrival_weight(delta: np.ndarray, residual: np.ndarray, design_weight: float, strength: float, min_weight: float, max_weight: float) -> float`
- Produces: `StreamingRaker(targets: Targets, calibration_strength: float = 0.1, min_weight: float = 1e-3, max_weight: float = 100.0, outcomes: Sequence[str] = (), keep_weights: bool = False)` with `partial_fit(obs, design_weight: float = 1.0, outcomes: Mapping[str, float] | None = None) -> float` and properties `n_obs: int`, `total_weight: float`, `effective_sample_size: float`, `margins: dict[str, float]`, `loss: float`, `residuals: np.ndarray`.

- [ ] **Step 1: Add scipy dependency**

In `pyproject.toml` change the dependencies block to:

```toml
dependencies = [
    "numpy>=1.21",
    "pandas>=1.3",
    "scipy>=1.11",
]
```

Run: `uv sync --all-groups`
Expected: resolves and installs scipy.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_streaming_raker.py`:

```python
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
        weight = arrival_weight(
            delta, residual, design_weight, strength, 1e-12, 1e12
        )
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
    assert raker.effective_sample_size == pytest.approx(
        total**2 / (w @ w), rel=1e-9
    )
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
    "kwargs",
    [
        {"calibration_strength": 0.0},
        {"calibration_strength": -1.0},
        {"calibration_strength": float("nan")},
        {"min_weight": 0.0},
        {"min_weight": 2.0, "max_weight": 1.0},
    ],
)
def test_invalid_settings_raise(kwargs):
    with pytest.raises(ValueError):
        StreamingRaker(Targets(a=0.5), **kwargs)


def test_empty_raker_reports_nan():
    raker = StreamingRaker(Targets(a=0.5))
    assert np.isnan(raker.margins["a"])
    assert np.isnan(raker.loss)
    assert raker.effective_sample_size == 0.0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_streaming_raker.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'onlinerake.streaming_raker'`.

- [ ] **Step 4: Implement the module**

Create `onlinerake/streaming_raker.py`:

```python
"""Streaming raking in fixed memory.

Each arriving row gets the weight that minimizes its raking distance from its
design weight plus a penalty on the running margin residuals, holding earlier
weights fixed. The state is the residual totals, so memory does not grow with
the stream and past rows are never stored.

The batch objective is penalized calibration (Bardsley and Chambers 1984) with
the raking distance of Deville and Sarndal (1992)::

    F(w) = sum_i [w_i log(w_i / d_i) - w_i + d_i]
           + (c / 2) * || sum_i w_i (x_i - t) ||^2

``StreamingRaker`` minimizes F over each arriving weight once.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import log
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.special import wrightomega

from ._utils import feature_value, validate_positive

if TYPE_CHECKING:
    from .targets import Targets


def arrival_weight(
    delta: npt.NDArray[np.float64],
    residual: npt.NDArray[np.float64],
    design_weight: float,
    strength: float,
    min_weight: float,
    max_weight: float,
) -> float:
    """Minimize D(w, d) + (c/2)||residual + w * delta||^2 over [min, max].

    Setting the derivative log(w/d) + c delta.r + c q w to zero, with
    q = ||delta||^2 and u = c q w, gives u + log u = log(c q d) - c delta.r,
    which the Wright omega function solves exactly. The objective is convex,
    so clipping the unconstrained minimizer gives the constrained one.

    Args:
        delta: The arriving row minus the targets.
        residual: Running totals sum_i w_i (x_i - t) before this row.
        design_weight: The row's base weight d.
        strength: Calibration strength c.
        min_weight: Lower weight bound.
        max_weight: Upper weight bound.

    Returns:
        The arriving row's weight.
    """
    q = float(delta @ delta)
    if q == 0.0:
        weight = design_weight
    else:
        y = log(strength * q * design_weight) - strength * float(delta @ residual)
        weight = float(np.real(wrightomega(y))) / (strength * q)
    return min(max(weight, min_weight), max_weight)


class StreamingRaker:
    """Rake a stream without storing it.

    Args:
        targets: Population targets (binary shares or continuous means).
        calibration_strength: Penalty c on the margin residual totals. Larger
            values match the targets more closely at the cost of more
            variable weights. Default 0.1.
        min_weight: Lower weight bound. Default 0.001.
        max_weight: Upper weight bound. Default 100.
        outcomes: Names of outcomes whose weighted means to track.
        keep_weights: Store each assigned weight. Off by default, which keeps
            memory independent of stream length.

    Examples:
        >>> from onlinerake import StreamingRaker, Targets
        >>> raker = StreamingRaker(Targets(female=0.5), outcomes=["spend"])
        >>> for female, spend in [(1, 10.0), (1, 12.0), (1, 11.0), (0, 20.0)] * 50:
        ...     _ = raker.partial_fit({"female": female}, outcomes={"spend": spend})
        >>> round(raker.margins["female"], 2)
        0.56
        >>> round(raker.estimate("spend"), 1)
        15.0
    """

    def __init__(
        self,
        targets: Targets,
        calibration_strength: float = 0.1,
        min_weight: float = 1e-3,
        max_weight: float = 100.0,
        outcomes: Sequence[str] = (),
        keep_weights: bool = False,
    ) -> None:
        validate_positive(calibration_strength, "calibration_strength")
        validate_positive(min_weight, "min_weight")
        validate_positive(max_weight, "max_weight")
        if max_weight <= min_weight:
            raise ValueError("max_weight must exceed min_weight")
        if len(set(outcomes)) != len(outcomes):
            raise ValueError("outcome names must be unique")

        self.targets = targets
        self.calibration_strength = float(calibration_strength)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.outcome_names = tuple(outcomes)
        self._feature_names = targets.feature_names
        self._target = np.array(
            [targets[name] for name in self._feature_names], dtype=np.float64
        )
        self._residual = np.zeros(len(self._feature_names), dtype=np.float64)
        self._outcome_sums = np.zeros(len(self.outcome_names), dtype=np.float64)
        self._total_weight = 0.0
        self._sum_sq_weight = 0.0
        self._n_obs = 0
        self._weights: list[float] | None = [] if keep_weights else None

    @property
    def n_obs(self) -> int:
        """Number of rows processed."""
        return self._n_obs

    @property
    def total_weight(self) -> float:
        """Sum of assigned weights."""
        return self._total_weight

    @property
    def effective_sample_size(self) -> float:
        """(sum w)^2 / sum w^2; 0.0 before any row."""
        if self._sum_sq_weight == 0.0:
            return 0.0
        return self._total_weight**2 / self._sum_sq_weight

    @property
    def residuals(self) -> npt.NDArray[np.float64]:
        """Copy of the running totals sum_i w_i (x_i - t)."""
        return self._residual.copy()

    @property
    def margins(self) -> dict[str, float]:
        """Weighted shares (binary) or means (continuous); NaN before any row."""
        if self._n_obs == 0:
            return dict.fromkeys(self._feature_names, np.nan)
        values = self._target + self._residual / self._total_weight
        return {
            name: float(values[i]) for i, name in enumerate(self._feature_names)
        }

    @property
    def loss(self) -> float:
        """Sum of squared margin errors; NaN before any row."""
        if self._n_obs == 0:
            return np.nan
        errors = self._residual / self._total_weight
        return float(errors @ errors)

    def _features(self, obs: dict[str, Any] | Any) -> npt.NDArray[np.float64]:
        values = np.empty(len(self._feature_names), dtype=np.float64)
        for i, name in enumerate(self._feature_names):
            raw = obs.get(name, 0) if isinstance(obs, dict) else getattr(obs, name, 0)
            values[i] = feature_value(
                raw, name, binary=self.targets.is_binary(name)
            )
        return values

    def partial_fit(
        self,
        obs: dict[str, Any] | Any,
        design_weight: float = 1.0,
        outcomes: Mapping[str, float] | None = None,
    ) -> float:
        """Weight one row and fold it into the running state.

        Args:
            obs: Feature values as a dict or attributes; missing features are 0.
            design_weight: The row's base weight. Default 1.
            outcomes: Values for every registered outcome name.

        Returns:
            The weight assigned to this row.

        Raises:
            ValueError: If a feature, outcome, or design weight is invalid.
                State is unchanged when this is raised.
        """
        validate_positive(design_weight, "design_weight")
        x = self._features(obs)
        y = self._outcome_values(outcomes)
        delta = x - self._target
        weight = arrival_weight(
            delta,
            self._residual,
            float(design_weight),
            self.calibration_strength,
            self.min_weight,
            self.max_weight,
        )
        self._residual += weight * delta
        self._outcome_sums += weight * y
        self._total_weight += weight
        self._sum_sq_weight += weight * weight
        self._n_obs += 1
        if self._weights is not None:
            self._weights.append(weight)
        return weight

    def _outcome_values(
        self, outcomes: Mapping[str, float] | None
    ) -> npt.NDArray[np.float64]:
        given = {} if outcomes is None else dict(outcomes)
        unknown = set(given) - set(self.outcome_names)
        if unknown:
            raise ValueError(f"unregistered outcomes: {sorted(unknown)}")
        missing = [name for name in self.outcome_names if name not in given]
        if missing:
            raise ValueError(f"missing outcomes: {missing}")
        return np.array(
            [
                feature_value(given[name], name, binary=False)
                for name in self.outcome_names
            ],
            dtype=np.float64,
        )
```

Note: `estimate`, `estimates`, and `weights` are added in Task 2; the doctest above is run in Task 2 Step 4.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_streaming_raker.py -v`
Expected: all tests in the file PASS.

- [ ] **Step 6: Lint and type-check**

Run: `uv run ruff check onlinerake/streaming_raker.py tests/test_streaming_raker.py && uv run ruff format --check onlinerake/streaming_raker.py tests/test_streaming_raker.py && uv run pyright onlinerake/streaming_raker.py`
Expected: no findings. If ruff format reports changes, run `uv run ruff format` on the two files and rerun.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock onlinerake/streaming_raker.py tests/test_streaming_raker.py
git commit -m "Add closed-form fixed-memory streaming raker

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Outcome estimates, kept weights, fixed-memory and timing guarantees

**Files:**
- Modify: `onlinerake/streaming_raker.py` (add `weights`, `estimate`, `estimates` to `StreamingRaker`)
- Modify: `tests/test_streaming_raker.py` (append tests)

**Interfaces:**
- Consumes: `StreamingRaker` from Task 1.
- Produces: `StreamingRaker.weights -> np.ndarray` (raises `RuntimeError` when `keep_weights=False`), `StreamingRaker.estimate(name: str) -> float`, `StreamingRaker.estimates -> dict[str, float]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_streaming_raker.py`:

```python
import time


def test_outcome_estimates_match_weighted_means():
    raker = StreamingRaker(
        Targets(a=0.4), outcomes=["y", "z"], keep_weights=True
    )
    rng = np.random.default_rng(4)
    rows = (rng.random(400) < 0.6).astype(int)
    ys = rng.normal(size=400)
    zs = rng.normal(size=400)
    for a, y, z in zip(rows, ys, zs, strict=True):
        raker.partial_fit({"a": int(a)}, outcomes={"y": float(y), "z": float(z)})
    w = raker.weights
    assert raker.estimate("y") == pytest.approx((w @ ys) / w.sum(), rel=1e-9)
    assert raker.estimates["z"] == pytest.approx((w @ zs) / w.sum(), rel=1e-9)


def test_rejected_rows_leave_state_untouched():
    raker = StreamingRaker(Targets(a=0.4), outcomes=["y"])
    raker.partial_fit({"a": 1}, outcomes={"y": 1.0})
    before = (raker.n_obs, raker.total_weight, raker.residuals.copy())
    bad_calls = [
        ({"a": 2}, {"outcomes": {"y": 1.0}}),
        ({"a": 1}, {"outcomes": {}}),
        ({"a": 1}, {"outcomes": {"y": 1.0, "extra": 2.0}}),
        ({"a": 1}, {"outcomes": {"y": float("nan")}}),
        ({"a": 1}, {"outcomes": {"y": 1.0}, "design_weight": 0.0}),
    ]
    for obs, kwargs in bad_calls:
        with pytest.raises(ValueError):
            raker.partial_fit(obs, **kwargs)
    assert (raker.n_obs, raker.total_weight) == before[:2]
    np.testing.assert_array_equal(raker.residuals, before[2])


def test_outcomes_without_registration_raise():
    raker = StreamingRaker(Targets(a=0.4))
    with pytest.raises(ValueError):
        raker.partial_fit({"a": 1}, outcomes={"y": 1.0})


def test_estimate_errors_and_empty_state():
    raker = StreamingRaker(Targets(a=0.4), outcomes=["y"])
    assert np.isnan(raker.estimate("y"))
    with pytest.raises(ValueError):
        raker.estimate("nope")


def test_weights_require_keep_weights():
    raker = StreamingRaker(Targets(a=0.4))
    raker.partial_fit({"a": 1})
    with pytest.raises(RuntimeError):
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
            dicts = [dict(zip(targets.feature_names, r.tolist(), strict=True)) for r in rows]
            for obs in dicts[:-tail]:
                raker.partial_fit(obs)
            start = time.perf_counter()
            for obs in dicts[-tail:]:
                raker.partial_fit(obs)
            best = min(best, (time.perf_counter() - start) / tail)
        return best

    assert late_cost(40_000) < 2.0 * late_cost(2_500)
```

Move the `import time` line to the top-of-file import block when appending (ruff's isort rule requires it).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_streaming_raker.py -v`
Expected: the five new outcome/weights tests FAIL with `AttributeError` (`estimate`, `estimates`, `weights` do not exist); the state-growth, timing and rejected-row tests may already pass.

- [ ] **Step 3: Implement**

Add to `StreamingRaker` in `onlinerake/streaming_raker.py`, after the `loss` property:

```python
    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """Assigned weights in arrival order.

        Raises:
            RuntimeError: If the raker was created with keep_weights=False.
        """
        if self._weights is None:
            raise RuntimeError("weights are not kept; pass keep_weights=True")
        return np.array(self._weights, dtype=np.float64)

    def estimate(self, name: str) -> float:
        """Weighted mean of one registered outcome; NaN before any row.

        Raises:
            ValueError: If the outcome was not registered.
        """
        if name not in self.outcome_names:
            raise ValueError(f"unregistered outcome: {name}")
        if self._n_obs == 0:
            return np.nan
        return float(
            self._outcome_sums[self.outcome_names.index(name)] / self._total_weight
        )

    @property
    def estimates(self) -> dict[str, float]:
        """Weighted means of all registered outcomes."""
        return {name: self.estimate(name) for name in self.outcome_names}
```

- [ ] **Step 4: Run tests and the doctest**

Run: `uv run pytest tests/test_streaming_raker.py -v && uv run pytest --doctest-modules onlinerake/streaming_raker.py -v`
Expected: all PASS. The doctest values (0.56, 15.0) were computed from the prototype of the same update: the female share moves from the raw 0.75 toward the 0.5 target, and the spend estimate rises above the raw mean 13.25 because the under-represented men spend more. If they differ, the implementation differs from the spec; find out why rather than editing the expected output.

- [ ] **Step 5: Lint, type-check, commit**

Run: `uv run ruff check . && uv run ruff format --check . && uv run pyright`
Expected: no findings.

```bash
git add onlinerake/streaming_raker.py tests/test_streaming_raker.py
git commit -m "Track weighted outcome means in the streaming raker

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Batch penalized raking and the superseded-update regression test

**Files:**
- Modify: `onlinerake/streaming_raker.py` (add `fit_penalized_raking`)
- Modify: `tests/test_streaming_raker.py` (append tests)

**Interfaces:**
- Consumes: `StreamingRaker` (Tasks 1–2).
- Produces: `fit_penalized_raking(features: npt.ArrayLike, targets: Targets, calibration_strength: float = 0.1, design_weights: npt.ArrayLike | None = None) -> npt.NDArray[np.float64]`. Columns of `features` follow `targets.feature_names`. Raises `RuntimeError` if the optimizer does not converge.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_streaming_raker.py` (merge imports into the top block; add `BatchIPF` to the `onlinerake` import and `fit_penalized_raking` to the `onlinerake.streaming_raker` import):

```python
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
    w = fit_penalized_raking(rows, targets, calibration_strength=c, design_weights=design)
    delta = rows - target
    residual = w @ delta
    np.testing.assert_allclose(w, design * np.exp(-c * (delta @ residual)), rtol=1e-8)


def test_penalized_batch_rejects_bad_shapes():
    targets, rows, _ = _easy_stream(50, 3, 3)
    with pytest.raises(ValueError):
        fit_penalized_raking(rows[:, :2], targets)
    with pytest.raises(ValueError):
        fit_penalized_raking(rows, targets, design_weights=np.ones(49))


def test_superseded_fixed_step_update_does_not_calibrate():
    """The update in onlinerake.pdf (March 2026) moves weights by O(1/t^2).

    It adjusted only the arriving weight with a fixed step on the gradient
    (1/W) * sum_j (m_j - t_j)(x_j - m_j). After the first few rows the step is
    too small to matter, so margins stay at their unweighted values. This test
    pins that finding; StreamingRaker replaces the update.
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
    streamed_error = np.mean([abs(raker.margins[n] - targets[n]) for n in targets.feature_names])
    assert streamed_error < 0.1 * raw_error
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_streaming_raker.py -v`
Expected: ImportError on `fit_penalized_raking`.

- [ ] **Step 3: Implement**

Add the import `from scipy.optimize import minimize` next to the `wrightomega` import, and append to `onlinerake/streaming_raker.py`:

```python
def fit_penalized_raking(
    features: npt.ArrayLike,
    targets: Targets,
    calibration_strength: float = 0.1,
    design_weights: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    """Minimize the penalized raking objective over all weights at once.

    The minimizer is the exponential tilt w_i = d_i exp(-lambda.delta_i) with
    lambda = c * sum_i w_i delta_i, so it is found from the strictly convex
    p-dimensional dual
    G(lambda) = [sum_i d_i exp(-lambda.delta_i) + ||lambda||^2 / (2c)] / sum_i d_i.
    This is entropy balancing with a ridge on the multipliers; as c grows it
    approaches raking. Weights are not bounded here, unlike StreamingRaker.

    Args:
        features: Array of shape (n, p), columns in targets.feature_names order.
        targets: Population targets.
        calibration_strength: Penalty c on the residual totals.
        design_weights: Base weights of shape (n,). Default all ones.

    Returns:
        Weights of shape (n,).

    Raises:
        ValueError: If shapes or settings are invalid.
        RuntimeError: If the optimizer does not converge.
    """
    validate_positive(calibration_strength, "calibration_strength")
    x = np.asarray(features, dtype=np.float64)
    names = targets.feature_names
    if x.ndim != 2 or x.shape[1] != len(names):
        raise ValueError(f"features must have shape (n, {len(names)})")
    d = (
        np.ones(x.shape[0])
        if design_weights is None
        else np.asarray(design_weights, dtype=np.float64)
    )
    if d.shape != (x.shape[0],) or not np.all(np.isfinite(d)) or np.any(d <= 0):
        raise ValueError("design_weights must be positive with shape (n,)")

    delta = x - np.array([targets[name] for name in names], dtype=np.float64)
    c = float(calibration_strength)
    # Unscaled, the gradient grows with n and SciPy's gtol stops meaning
    # anything; dividing by sum(d) keeps convergence checks on a fixed scale.
    scale = d.sum()
    identity = np.eye(len(names))

    def tilt(lam: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return d * np.exp(-(delta @ lam))

    def objective(lam: npt.NDArray[np.float64]) -> float:
        return float((tilt(lam).sum() + lam @ lam / (2 * c)) / scale)

    def gradient(lam: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return (-(tilt(lam) @ delta) + lam / c) / scale

    def hessian(lam: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        e = tilt(lam)
        return ((delta.T * e) @ delta + identity / c) / scale

    result = minimize(
        objective,
        np.zeros(len(names)),
        jac=gradient,
        hess=hessian,
        method="trust-exact",
        options={"gtol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"penalized raking did not converge: {result.message}")
    return tilt(result.x)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_streaming_raker.py -v`
Expected: all PASS. If `test_penalized_batch_reproduces_ipf_at_large_strength` fails by more than 1e-6, check `BatchIPF`'s own convergence tolerance (`onlinerake/batch_ipf.py`) before loosening anything; the prototype reached 2.2e-8 with a 1e-10 IPF tolerance.

- [ ] **Step 5: Lint, type-check, commit**

Run: `uv run ruff check . && uv run ruff format --check . && uv run pyright`

```bash
git add onlinerake/streaming_raker.py tests/test_streaming_raker.py
git commit -m "Add batch penalized raking and pin the superseded update

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Public API and reference docs

**Files:**
- Modify: `onlinerake/__init__.py` (import block near line 149; `__all__` "Core algorithms")
- Modify: `docs/api_reference.rst:16-19`

**Interfaces:**
- Consumes: `StreamingRaker`, `fit_penalized_raking`.
- Produces: `from onlinerake import StreamingRaker, fit_penalized_raking`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_api_consistency.py`:

```python
def test_streaming_raker_is_exported():
    import onlinerake

    assert "StreamingRaker" in onlinerake.__all__
    assert "fit_penalized_raking" in onlinerake.__all__
    assert onlinerake.StreamingRaker.__module__ == "onlinerake.streaming_raker"
```

Run: `uv run pytest tests/test_api_consistency.py -v`
Expected: the new test FAILS.

- [ ] **Step 2: Export**

In `onlinerake/__init__.py` add, after `from .sensitivity import (...)` block and before `from .streaming_inference import (`:

```python
from .streaming_raker import StreamingRaker, fit_penalized_raking
```

In `__all__`, change the "Core algorithms" group to:

```python
    # Core algorithms
    "Targets",
    "StreamingRaker",
    "fit_penalized_raking",
    "OnlineRakingSGD",
    "OnlineRakingMWU",
    "BatchIPF",
```

In `docs/api_reference.rst` change the autosummary list to:

```rst
   Targets
   StreamingRaker
   fit_penalized_raking
   OnlineRakingSGD
   OnlineRakingMWU
   BatchIPF
```

- [ ] **Step 3: Verify**

Run: `uv run pytest -q && uv run sphinx-build -W -b html docs _site`
Expected: full suite passes; Sphinx builds with no warnings.

- [ ] **Step 4: Commit**

```bash
git add onlinerake/__init__.py docs/api_reference.rst tests/test_api_consistency.py
git commit -m "Export the streaming raker

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Streaming evaluation script and 50-seed run

**Files:**
- Create: `scripts/eval/streaming_eval.py`
- Modify: `.gitignore` (ensure `eval_outputs/` is ignored; the CSV is copied into `ms/` by Task 6)

**Interfaces:**
- Consumes: `StreamingRaker`, `fit_penalized_raking`, `BatchIPF`, `OnlineRakingSGD`, `OnlineRakingMWU`, `generate_scenario_data` from `scripts/eval/comprehensive_eval.py`.
- Produces: `eval_outputs/streaming_results.csv` with columns `scenario, n, p, seed, method, final_err, max_err, anytime_err, ess_frac, bias, sec_per_obs, state_floats`; `eval_outputs/streaming_order.csv` with columns `scenario, seed, method, estimate_sd_across_orders`; `eval_outputs/streaming_scaling.csv` with columns `n, method, sec_per_obs`.

Methods, all on the same rows per seed:
- `raw` (all weights 1)
- `ipf` (BatchIPF on all rows; anytime error refits at each checkpoint)
- `periodic_ipf_100` (n ≤ 5000 only) and `periodic_ipf_1000` (refit on all stored rows every k arrivals; rows since the last refit carry weight 1, the mean of the last fitted weights after normalizing them to mean 1)
- `stream_c{c}` for c in {0.01, 0.03, 0.1, 0.3, 1.0}
- `batch_c{c}` for the same c (`fit_penalized_raking`, final metrics only; anytime = final)
- `sgd_all`, `mwu_all` (package classes, learning rates 5.0 and 1.0; only when n ≤ 5000 because cost is quadratic)

Outcome per seed: `y = 1 + X @ beta + X[:,0] * X[:,1] + N(0,1)`, `beta ~ U(-1,1)^p` from `np.random.default_rng(10_000 + seed)`. Population truth assumes independent features at the target margins: `1 + beta @ t + t[0] * t[1]`. `bias` = weighted mean of y minus truth.

Scenarios (n, p): standard easy (3000, 8); large easy (50000, 8); hard (3000, 8); high-dimensional easy (3000, 50); drift (5000, 8); small easy (500, 4).

`anytime_err`: mean over 20 checkpoints (every 5% of the stream) of mean |margin error| at that point. `state_floats`: floats a method must keep — `p + q + 3` for `stream_*`, `n*(p+1)` for methods that store rows.

- [ ] **Step 1: Write the script**

Create `scripts/eval/streaming_eval.py`:

```python
"""Fixed-memory streaming raking versus methods that store the stream.

Writes one CSV row per (scenario, seed, method) as soon as it is computed, so
an interrupted run keeps everything finished before the interruption.

Usage:
    uv run python scripts/eval/streaming_eval.py --n_seeds 2 --quick
    uv run python scripts/eval/streaming_eval.py --n_seeds 50
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comprehensive_eval import generate_scenario_data  # noqa: E402

from onlinerake import (  # noqa: E402
    BatchIPF,
    OnlineRakingMWU,
    OnlineRakingSGD,
    StreamingRaker,
    fit_penalized_raking,
)

SCENARIOS = {
    "standard": ("easy", 3000, 8),
    "large": ("easy", 50000, 8),
    "hard": ("hard", 3000, 8),
    "high_dim": ("easy", 3000, 50),
    "drift": ("drift", 5000, 8),
    "small": ("easy", 500, 4),
}
STRENGTHS = (0.01, 0.03, 0.1, 0.3, 1.0)
CHECKPOINTS = 20
FIELDS = [
    "scenario", "n", "p", "seed", "method", "final_err", "max_err",
    "anytime_err", "ess_frac", "bias", "sec_per_obs", "state_floats",
]


def outcome(x, seed, t):
    rng = np.random.default_rng(10_000 + seed)
    beta = rng.uniform(-1, 1, x.shape[1])
    y = 1 + x @ beta + x[:, 0] * x[:, 1] + rng.normal(size=len(x))
    return y, float(1 + beta @ t + t[0] * t[1])


def summarize(w, x, t, y, truth):
    m = (w @ x) / w.sum()
    err = np.abs(m - t)
    return {
        "final_err": float(err.mean()),
        "max_err": float(err.max()),
        "ess_frac": float(w.sum() ** 2 / (w @ w) / len(w)),
        "bias": float((w @ y) / w.sum() - truth),
    }


def checkpoints(n):
    return sorted({max(1, round(n * k / CHECKPOINTS)) for k in range(1, CHECKPOINTS + 1)})


def ipf_weights(targets, dicts):
    raker = BatchIPF(targets)
    raker.fit(dicts)
    return raker.weights


def run_stream(targets, x, y, c):
    raker = StreamingRaker(targets, calibration_strength=c, outcomes=["y"], keep_weights=True)
    names = targets.feature_names
    marks = set(checkpoints(len(x)))
    errs = []
    start = time.perf_counter()
    for i, (row, yi) in enumerate(zip(x, y, strict=True), start=1):
        raker.partial_fit(dict(zip(names, row.tolist(), strict=True)), outcomes={"y": float(yi)})
        if i in marks:
            errs.append(np.mean([abs(raker.margins[k] - targets[k]) for k in names]))
    elapsed = time.perf_counter() - start
    return raker.weights, float(np.mean(errs)), elapsed / len(x)


def run_periodic(targets, dicts, x, t, k):
    marks = checkpoints(len(x))
    errs = []
    fitted = np.empty(0)
    start = time.perf_counter()
    for end in range(k, len(x) + k, k):
        end = min(end, len(x))
        fitted = ipf_weights(targets, dicts[:end])
        fitted = fitted / fitted.mean()
    elapsed = time.perf_counter() - start
    for mark in marks:
        last = (mark // k) * k
        w = np.ones(mark)
        if last:
            prior = ipf_weights(targets, dicts[:last])
            w[:last] = prior / prior.mean()
        errs.append(np.abs((w @ x[:mark]) / w.sum() - t).mean())
    return fitted, float(np.mean(errs)), elapsed / len(x)


def run_all_weights(cls, lr, targets, dicts):
    raker = cls(targets, learning_rate=lr, track_convergence=False)
    start = time.perf_counter()
    for obs in dicts:
        raker.partial_fit(obs)
    return raker.weights, (time.perf_counter() - start) / len(dicts)


def evaluate_seed(name, scenario, n, p, seed, writer, handle):
    targets, dicts = generate_scenario_data(scenario, n, p, seed)
    names = targets.feature_names
    x = np.array([[o[k] for k in names] for o in dicts], dtype=float)
    t = np.array([targets[k] for k in names])
    y, truth = outcome(x, seed, t)
    base = {"scenario": name, "n": n, "p": p, "seed": seed}
    stored = n * (p + 1)

    def emit(method, w, anytime, sec, state):
        writer.writerow({**base, "method": method, **summarize(w, x, t, y, truth),
                         "anytime_err": anytime, "sec_per_obs": sec, "state_floats": state})
        handle.flush()

    raw_any = float(np.mean([np.abs(x[:m].mean(0) - t).mean() for m in checkpoints(n)]))
    emit("raw", np.ones(n), raw_any, 0.0, 0)

    start = time.perf_counter()
    w_ipf = ipf_weights(targets, dicts)
    sec = (time.perf_counter() - start) / n
    ipf_errs = []
    for m in checkpoints(n):
        w_m = ipf_weights(targets, dicts[:m])
        ipf_errs.append(np.abs((w_m @ x[:m]) / w_m.sum() - t).mean())
    ipf_any = float(np.mean(ipf_errs))
    emit("ipf", w_ipf, ipf_any, sec, stored)

    # Refitting every 100 rows of a 50,000-row stream means 500 full IPF fits
    # per seed; at that size only the 1,000-row cadence is affordable.
    for k in (100, 1000):
        if k < n and (k == 1000 or n <= 5000):
            w, anytime, sec = run_periodic(targets, dicts, x, t, k)
            emit(f"periodic_ipf_{k}", w, anytime, sec, stored)

    for c in STRENGTHS:
        w, anytime, sec = run_stream(targets, x, y, c)
        emit(f"stream_c{c:g}", w, anytime, sec, p + 1 + 3)
        start = time.perf_counter()
        w = fit_penalized_raking(x, targets, calibration_strength=c)
        sec = (time.perf_counter() - start) / n
        final = summarize(w, x, t, y, truth)["final_err"]
        emit(f"batch_c{c:g}", w, final, sec, stored)

    if n <= 5000:
        for method, cls, lr in (("sgd_all", OnlineRakingSGD, 5.0), ("mwu_all", OnlineRakingMWU, 1.0)):
            w, sec = run_all_weights(cls, lr, targets, dicts)
            emit(method, w, float("nan"), sec, stored)


def order_dependence(out_dir, n_seeds, n_orders=10):
    path = out_dir / "streaming_order.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scenario", "seed", "method", "estimate_sd_across_orders"])
        for seed in range(n_seeds):
            targets, dicts = generate_scenario_data("easy", 3000, 8, seed)
            names = targets.feature_names
            x = np.array([[o[k] for k in names] for o in dicts], dtype=float)
            y, _ = outcome(x, seed, np.array([targets[k] for k in names]))
            rng = np.random.default_rng(seed)
            estimates = []
            for _ in range(n_orders):
                order = rng.permutation(len(x))
                w, _, _ = run_stream(targets, x[order], y[order], 0.1)
                estimates.append(float((w @ y[order]) / w.sum()))
            writer.writerow(["standard", seed, "stream_c0.1", float(np.std(estimates, ddof=1))])
            handle.flush()


def scaling(out_dir, sizes):
    path = out_dir / "streaming_scaling.csv"
    rng = np.random.default_rng(0)
    from onlinerake import Targets

    targets = Targets(**{f"f{i}": 0.5 for i in range(8)})
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["n", "method", "sec_per_obs"])
        for n in sizes:
            x = (rng.random((n, 8)) < rng.uniform(0.3, 0.7, 8)).astype(int)
            raker = StreamingRaker(targets)
            dicts = [dict(zip(targets.feature_names, r.tolist(), strict=True)) for r in x]
            start = time.perf_counter()
            for obs in dicts:
                raker.partial_fit(obs)
            writer.writerow([n, "stream_c0.1", (time.perf_counter() - start) / n])
            start = time.perf_counter()
            ipf_weights(targets, dicts)
            writer.writerow([n, "ipf_rerun_every_1000_amortized", (time.perf_counter() - start) / 1000])
            handle.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_seeds", type=int, default=50)
    parser.add_argument("--quick", action="store_true", help="skip the large scenario and scaling run")
    parser.add_argument("--output_dir", default="eval_outputs")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "streaming_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for name, (scenario, n, p) in SCENARIOS.items():
            if args.quick and name == "large":
                continue
            for seed in range(args.n_seeds):
                evaluate_seed(name, scenario, n, p, seed, writer, handle)
            print(f"done: {name}", flush=True)

    order_dependence(out_dir, min(args.n_seeds, 20))
    if not args.quick:
        scaling(out_dir, (10_000, 100_000, 1_000_000))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Dry run before the long run**

Run: `uv run python scripts/eval/streaming_eval.py --n_seeds 2 --quick --output_dir /tmp/streaming_dry`
Expected: finishes; `/tmp/streaming_dry/streaming_results.csv` has rows for every scenario except `large` and every method listed above.

Then check what the dry run produced against the prototype (5-seed, easy n=3000 p=8: `stream_c0.1` final_err ≈ 0.002, ess_frac ≈ 0.76; `ipf` final_err ≈ 0):

Run: `uv run python -c "import pandas as pd; d=pd.read_csv('/tmp/streaming_dry/streaming_results.csv'); print(d[d.scenario=='standard'].groupby('method')[['final_err','anytime_err','ess_frac','bias','sec_per_obs']].mean().round(4))"`
Expected: numbers in that neighborhood. If `stream_c0.1` final_err exceeds 0.005, stop and investigate before the full run.

- [ ] **Step 3: Lint, commit the script**

Run: `uv run ruff check scripts/eval/streaming_eval.py && uv run ruff format --check scripts/eval/streaming_eval.py`
Make sure `eval_outputs/` is listed in `.gitignore`; add the line if it is missing.

```bash
git add scripts/eval/streaming_eval.py .gitignore
git commit -m "Evaluate streaming raking against stored-data methods

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 4: Full run (expensive, run in background)**

Run: `uv run python scripts/eval/streaming_eval.py --n_seeds 50 > eval_outputs/streaming_run.log 2>&1` with `run_in_background: true`.
Expected: `done: <scenario>` lines for all six scenarios; the CSVs are written incrementally, so a crash keeps finished rows. The `large` scenario's anytime IPF refits dominate runtime.

- [ ] **Step 5: Show the data to the user**

Print, per scenario, the method means of `final_err, anytime_err, ess_frac, |bias|, sec_per_obs, state_floats`, the order-dependence SD next to the across-seed SD of `stream_c0.1`'s estimate, and the scaling table. Ask the user to look at: (a) whether `stream_c0.1` versus `periodic_ipf_1000` on anytime error and ESS is the trade-off the paper should lead with; (b) whether `hard` and `high_dim` make c = 0.1 the wrong default. Wait for their answer before Task 6.

---

### Task 6: Figures and table

**Files:**
- Create: `scripts/eval/plot_streaming_eval.py`
- Create (generated): `ms/fig_frontier.pdf`, `ms/fig_anytime.pdf`, `ms/fig_scaling.pdf`, `ms/table_streaming.tex`

**Interfaces:**
- Consumes: the three CSVs from Task 5.
- Produces: the four files above, referenced by Tasks 7–8.

- [ ] **Step 1: Write the plotting script**

Create `scripts/eval/plot_streaming_eval.py`:

```python
"""Figures and table for the streaming raking manuscript.

Usage:
    uv run python scripts/eval/plot_streaming_eval.py --input_dir eval_outputs --output_dir ms
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ORDER = ["standard", "large", "hard", "high_dim", "drift", "small"]
TABLE_METHODS = [
    "raw", "ipf", "periodic_ipf_1000", "batch_c0.1", "stream_c0.1",
    "stream_c1", "sgd_all", "mwu_all",
]


def frontier(results, out):
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharey=False)
    for ax, scenario in zip(axes.flat, ORDER, strict=True):
        d = results[results.scenario == scenario].groupby("method")[["final_err", "ess_frac"]].mean()
        for family, marker in (("stream_c", "o-"), ("batch_c", "s--")):
            rows = d[d.index.str.startswith(family)]
            rows = rows.assign(c=rows.index.str.removeprefix(family).astype(float)).sort_values("c")
            ax.plot(rows.ess_frac, rows.final_err, marker, label=family.rstrip("_c"))
        for method, marker in (("ipf", "k*"), ("raw", "kx")):
            if method in d.index:
                ax.plot(d.loc[method, "ess_frac"], d.loc[method, "final_err"], marker, label=method)
        ax.set_title(scenario)
        ax.set_xlabel("ESS / n")
        ax.set_ylabel("mean |margin error|")
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_frontier.pdf")


def anytime(results, out):
    d = results[results.scenario.isin(["standard", "drift"])]
    d = d[d.method.isin(["raw", "periodic_ipf_100", "periodic_ipf_1000", "stream_c0.1", "stream_c1"])]
    table = d.groupby(["scenario", "method"]).anytime_err.mean().unstack(0)
    ax = table.plot.bar(figsize=(7, 3.5))
    ax.set_ylabel("anytime mean |margin error|")
    ax.figure.tight_layout()
    ax.figure.savefig(out / "fig_anytime.pdf")


def scaling(scale, out):
    ax = scale.pivot(index="n", columns="method", values="sec_per_obs").mul(1e6).plot(logx=True, logy=True, marker="o", figsize=(5, 3.5))
    ax.set_ylabel("microseconds per arrival")
    ax.figure.tight_layout()
    ax.figure.savefig(out / "fig_scaling.pdf")


def table(results, order, out):
    d = results[results.method.isin(TABLE_METHODS)].copy()
    d["abs_bias"] = d.bias.abs()
    g = d.groupby(["scenario", "method"])[["final_err", "anytime_err", "ess_frac", "abs_bias", "sec_per_obs"]].mean()
    g["sec_per_obs"] *= 1e6
    g = g.rename(columns={"sec_per_obs": "us_per_obs"}).reindex(ORDER, level=0)
    g.round(4).to_latex(out / "table_streaming.tex", na_rep="--", float_format="%.4f", multirow=True)
    sd = order.estimate_sd_across_orders.mean()
    across = results[(results.scenario == "standard") & (results.method == "stream_c0.1")].bias.std()
    print(f"order SD {sd:.4f} vs across-seed SD {across:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", default="eval_outputs")
    parser.add_argument("--output_dir", default="ms")
    args = parser.parse_args()
    src, out = Path(args.input_dir), Path(args.output_dir)
    results = pd.read_csv(src / "streaming_results.csv")
    frontier(results, out)
    anytime(results, out)
    scaling(pd.read_csv(src / "streaming_scaling.csv"), out)
    table(results, pd.read_csv(src / "streaming_order.csv"), out)


if __name__ == "__main__":
    main()
```

Check `matplotlib` is in a dependency group: `grep -n matplotlib pyproject.toml`. If absent, add it to the `dev` group with `uv add --group dev matplotlib`.

- [ ] **Step 2: Run and open the outputs**

Run: `uv run python scripts/eval/plot_streaming_eval.py --input_dir eval_outputs --output_dir ms && open ms/fig_frontier.pdf ms/fig_anytime.pdf ms/fig_scaling.pdf`
Expected: three figures and `ms/table_streaming.tex`. Show them to the user with what to look for: frontier curves should run from `raw` (ESS 1, high error) toward `ipf`; the gap between `stream` and `batch` curves is the cost of not storing rows. Wait for the user's reaction before writing the paper.

- [ ] **Step 3: Lint, commit**

Run: `uv run ruff check scripts/eval/plot_streaming_eval.py && uv run ruff format --check scripts/eval/plot_streaming_eval.py`

```bash
git add scripts/eval/plot_streaming_eval.py ms/fig_frontier.pdf ms/fig_anytime.pdf ms/fig_scaling.pdf ms/table_streaming.tex pyproject.toml uv.lock
git commit -m "Plot the streaming raking evaluation

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Manuscript methods rewrite

**Files:**
- Modify: `ms/onlinerake.tex` (abstract; Sections 1–3; delete Algorithms 1–2)
- Modify: `ms/online_raking_references.bib` (add Bardsley1984)

**Interfaces:**
- Consumes: spec equations; `StreamingRaker`, `fit_penalized_raking` names.
- Produces: section labels `sec:objective`, `sec:update`, `sec:batch`, `sec:superseded`, used by Task 8.

- [ ] **Step 1: Add the reference**

Append to `ms/online_raking_references.bib`:

```bibtex
@article{Bardsley1984,
  author  = {Bardsley, P. and Chambers, R. L.},
  title   = {Multipurpose Estimation from Unbalanced Samples},
  journal = {Applied Statistics},
  volume  = {33},
  number  = {3},
  pages   = {290--299},
  year    = {1984},
  doi     = {10.2307/2347706}
}
```

Verify volume, issue and pages against `https://api.crossref.org/works/10.2307/2347706` before committing; correct them if Crossref differs.

- [ ] **Step 2: Rewrite the front matter and methods**

Replace the title with `Streaming Raking in Fixed Memory`. Rewrite, keeping the existing `\label`s where sections survive:

1. **Abstract** (≤200 words): problem = raking a stream that is not stored; method = per-arrival minimization of raking distance plus a penalty on running residual totals, solved in closed form by the Wright omega function; batch counterpart = ridge-penalized entropy balancing; headline numbers from `ms/table_streaming.tex` (standard scenario: `stream_c0.1` anytime error, ESS/n versus `ipf`; scaling: µs/arrival at n=10^6); one sentence that periodic IPF is preferable when rows can be stored.
2. **Introduction**: replace the cost-based motivation with the storage motivation (privacy, volume, edge collection). State that rerunning IPF is cheap (cite the scaling figure) so the contribution is memory, not speed.
3. **Related work**: keep Deville1992, Deville1993, Kolenikov2015, Hainmueller2012, Zhao2017, Zubizarreta2015 paragraphs; add Bardsley1984 as the source of penalizing residuals instead of imposing constraints. Delete the Platt/isotonic/temperature sentence (different problem). Keep Arora2012/Beck2003 only if Section 3 still uses mirror descent; it does not, so delete them and remove unused bib entries.
4. **Section 3 `sec:objective`**: the batch objective F(w) with D(w,d), c, and residual totals r = Σ wᵢ(xᵢ − t) = W(m − t). Note it is convex, and that as c → ∞ its minimizer tends to the raking solution.
5. **`sec:update`**: the arrival problem f(w), its derivative, the substitution u = c q w, the Wright omega solution w = ω(y)/(cq), the q = 0 case, clipping, O(p) time and O(p + q) memory. Then the interpretation: w = d·exp(−λ·δ) with λ = c·r, a running estimate of the batch multiplier.
6. **`sec:batch`**: the dual G(λ), its gradient and Hessian, and why it is the right comparator (same objective, rows retained).
7. **`sec:superseded`** (short subsection or appendix): the earlier version of this paper proposed a fixed-step update on the arriving weight; its step scales as 1/t², so it does not calibrate (cite the regression test), and its experiments had run a different, all-weights algorithm. One paragraph, factual.
8. Delete the old Algorithm 1/2 subsections, the "scale-invariant" sentence, and every O(Kp) claim.

- [ ] **Step 3: Compile check**

Run: `cd ms && latexmk -pdf -interaction=nonstopmode onlinerake.tex; grep -nE 'undefined|Warning: Citation|Error' onlinerake.log | head`
Expected: builds; no undefined references or citations.

- [ ] **Step 4: Commit**

```bash
git add ms/onlinerake.tex ms/online_raking_references.bib
git commit -m "Rewrite manuscript methods around fixed-memory raking

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Manuscript experiments, discussion, PDF

**Files:**
- Modify: `ms/onlinerake.tex` (Sections 4–6)
- Delete: `ms/fig_algorithm_comparison.pdf`, `ms/fig_efficiency_accuracy.pdf`, `ms/fig_robustness.pdf`, `ms/table_main_results.tex` (superseded outputs)
- Modify: `ms/onlinerake.pdf` (rebuilt)

**Interfaces:**
- Consumes: Task 6 figures/table; Task 7 labels.

- [ ] **Step 1: Rewrite experiments**

Setup paragraph lists the methods, scenarios, outcome model, truth definition and metrics exactly as in Task 5. Results paragraphs, each citing `\input{table_streaming}` or a figure, and each number copied from the table:
- Accuracy versus ESS (frontier figure): where `stream_c*` sits relative to `ipf` and `batch_c*`; the stream-to-batch gap as the price of not storing rows.
- Anytime calibration (anytime figure): `stream` versus `periodic_ipf_100/1000` and `raw`.
- Outcome bias: |bias| for `stream_c0.1` versus `ipf`; note the interaction term is where weights that are not product-form can show bias.
- Order dependence: order SD versus across-seed SD (from the plotting script's printed line).
- Hard and high-dimensional scenarios: what the default c does there.
- Cost: µs/arrival flat in n (scaling figure) versus amortized IPF reruns; state floats.
- All-weights methods (`sgd_all`, `mwu_all`): one paragraph reporting their numbers and that periodic IPF dominates them.

- [ ] **Step 2: Rewrite discussion and conclusion**

"When to use what": rows storable → periodic `BatchIPF`; rows not storable → `StreamingRaker`; choose c by the frontier. Keep "Squared-error versus KL" only if still relevant (it is not: delete). Rewrite "early-vs-late asymmetry" for the frozen-weight setting, using the order-dependence result. Limitations: no convergence proof for the single pass; bounds; drift without forgetting. Conclusion ≤ 150 words.

- [ ] **Step 3: Remove stale outputs and build**

```bash
git rm ms/fig_algorithm_comparison.pdf ms/fig_efficiency_accuracy.pdf ms/fig_robustness.pdf ms/table_main_results.tex
cd ms && latexmk -pdf -interaction=nonstopmode onlinerake.tex && latexmk -c
```

Expected: `ms/onlinerake.pdf` builds; `grep -nE 'undefined|Error' onlinerake.log` empty. Confirm `.gitignore` covers `*.aux *.log *.out *.bbl *.blg *.fls *.fdb_latexmk`; add the missing patterns.

- [ ] **Step 4: Check every number against the table**

Run: `grep -noE '[0-9]+\.[0-9]+' ms/onlinerake.tex | sort -u -t: -k3 | head -80` and check each manuscript number appears in `ms/table_streaming.tex`, the order-dependence line, or `eval_outputs/streaming_scaling.csv`. Fix or remove any that do not.

- [ ] **Step 5: Show the PDF**

Run: `pdftoppm -f 1 -l 1 -r 110 -png -singlefile ms/onlinerake.pdf /tmp/or_p1 && open /tmp/or_p1.png ms/onlinerake.pdf`
Ask the user to read the abstract, Section 3, and the results table. Wait for their review.

- [ ] **Step 6: Commit**

```bash
git add ms/onlinerake.tex ms/onlinerake.pdf .gitignore
git commit -m "Rewrite manuscript results from the 50-seed evaluation

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Reposition package documentation

**Files:**
- Modify: `README.md`, `CHANGELOG.md`, `CLAUDE.md`, `docs/quickstart.rst`, `onlinerake/online_raking_sgd.py:1-15` (module docstring), `onlinerake/online_raking_mwu.py:1-21` (module docstring)

**Interfaces:**
- Consumes: final numbers from Task 8.

- [ ] **Step 1: README**

Replace the opening ("Real-time survey weighting for streaming data.", "The Problem", "When to Use This", "Quick Start") with:
- One line: rake a stream without storing it.
- "Which tool": rows storable → rerun `BatchIPF` periodically (state the measured cost from the scaling table); rows not storable → `StreamingRaker`; `OnlineRakingSGD`/`OnlineRakingMWU` → retained for existing users, dominated by periodic IPF (cite the paper).
- Quick start using `StreamingRaker` with an outcome, copied from the class doctest so it is known to run.
Keep badges and later sections that remain true; delete claims the paper no longer makes.

- [ ] **Step 2: Docstrings of the all-weights classes**

Append to each module docstring of `online_raking_sgd.py` and `online_raking_mwu.py`:

```text
When past rows can be stored, rerunning ``BatchIPF`` periodically is cheaper and
exact at each refit. When they cannot, use ``StreamingRaker``.
```

- [ ] **Step 3: CHANGELOG and CLAUDE.md**

Add a `## [Unreleased]` section (the release skill dates it as 3.1.0): Added `StreamingRaker`, `fit_penalized_raking`; scipy dependency; manuscript rewritten (algorithm in the March 2026 manuscript did not calibrate; experiments had used the all-weights classes); docs recommend periodic `BatchIPF` when data can be stored. Update `CLAUDE.md` "Project Overview" and "Architecture" to list `streaming_raker.py` first and the same recommendation.

Set `version = "3.1.0"` in `pyproject.toml`.

- [ ] **Step 4: Verify and commit**

Run: `uv run pytest -q && uv run sphinx-build -W -b html docs _site && uv run ruff check . && uv run ruff format --check . && uv run pyright`
Expected: all pass.

```bash
git add README.md CHANGELOG.md CLAUDE.md docs/quickstart.rst onlinerake/online_raking_sgd.py onlinerake/online_raking_mwu.py pyproject.toml uv.lock
git commit -m "Recommend periodic IPF or the streaming raker in the docs

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: PR, CI, and the research site

**Files:**
- Modify (other repo): `/Users/soodoku/Documents/GitHub/soodoku.github.io/research/catalog.json`, `research/index.html`, `research/papers/onlinerake.pdf`

- [ ] **Step 1: Push and open the PR**

```bash
git push -u origin streaming-raker
gh pr create --title "Fixed-memory streaming raker; manuscript rewrite" --body-file /tmp/pr_body.md
gh pr checks --watch
```

Write `/tmp/pr_body.md` first with four sections: Findings (the three numbered findings from the spec, with their numbers); Changes (one line per task); Verification (each command from Global Constraints with its result, plus the 50-seed headline numbers from `ms/table_streaming.tex`); Not changed (`OnlineRakingSGD`/`OnlineRakingMWU` behavior). End it with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.

Expected: CI green (coverage floor 89 included). Report the PR URL to the user; merge only on their approval.

- [ ] **Step 2: Update the site after merge**

After the user merges: in `soodoku.github.io`, run the **Sync papers** workflow (`gh workflow run sync-papers.yml`), review its PR's first-page diff for `onlinerake.pdf`, and update the catalog entry `streaming-calibration-with-mwu-and-sgd`: `title` = the new manuscript title; `description` = two sentences from the new abstract. Regenerate and check:

```bash
cd /Users/soodoku/Documents/GitHub/soodoku.github.io
python3 .github/scripts/research_catalog.py && python3 .github/scripts/research_catalog.py --check
python3 -m unittest discover -s .github/scripts -p 'test_*.py'
```

Commit with the attribution line and push after the user confirms.

- [ ] **Step 3: Release (only if the user asks)**

Use the `release` skill for 3.1.0.
