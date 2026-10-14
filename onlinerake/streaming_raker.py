"""Streaming raking in fixed memory.

Each arriving row gets the weight that minimizes its raking distance from its
design weight plus a penalty on the running margin residuals, holding earlier
weights fixed. The state is the residual totals, so memory does not grow with
the stream and past rows are never stored.

The batch objective is penalized calibration (Bardsley and Chambers 1984) with
the raking distance of Deville and Sarndal (1992)::

    F(w) = sum_i [w_i log(w_i / d_i) - w_i + d_i]
           + (c / 2) * || sum_i w_i (x_i - t) ||^2

``StreamingRaker`` minimizes F over each arriving weight once;
``fit_penalized_raking`` minimizes it over all weights with the rows stored.

The penalty is on totals, so c would otherwise carry the units of the design
weights and of continuous features: doubling every design weight would act
like doubling c, and income in dollars instead of thousands would weigh a
million times more. Both functions therefore divide design weights by their
mean and each continuous feature's deviations from its target by their root
mean square, so a given c means the same thing in any units, as with the unit-
free ridge penalty in standard calibration software.
"""

from __future__ import annotations

from math import log
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize
from scipy.special import wrightomega

from ._utils import feature_value, validate_positive

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .targets import Targets


def arrival_weight(
    delta: npt.NDArray[np.float64],
    residual: npt.NDArray[np.float64],
    design_weight: float,
    strength: float,
    min_weight: float,
    max_weight: float,
) -> float:
    """Minimize D(w, d) + (c/2)||residual + w * delta||^2 over w/d in [min, max].

    Setting the derivative log(w/d) + c delta.r + c q w to zero, with
    q = ||delta||^2 and u = c q w, gives u + log u = log(c q d) - c delta.r,
    which the Wright omega function solves exactly. The objective is convex,
    so clipping the unconstrained minimizer gives the constrained one.

    Args:
        delta: The arriving row minus the targets.
        residual: Running totals sum_i w_i (x_i - t) before this row.
        design_weight: The row's base weight d.
        strength: Calibration strength c.
        min_weight: Lower bound on w / d.
        max_weight: Upper bound on w / d.

    Returns:
        The arriving row's weight.
    """
    q = float(delta @ delta)
    if q == 0.0:
        weight = design_weight
    else:
        y = log(strength * q * design_weight) - strength * float(delta @ residual)
        weight = float(np.real(wrightomega(y))) / (strength * q)
    return min(max(weight, min_weight * design_weight), max_weight * design_weight)


class StreamingRaker:
    """Rake a stream without storing it.

    Args:
        targets: Population targets (binary shares or continuous means).
        calibration_strength: Penalty c on the margin residual totals. Larger
            values match the targets more closely at the cost of more
            variable weights. Default 0.3, which held up across the easy,
            extreme-target and many-margin scenarios in ms/onlinerake.pdf.
            Design weights are divided by their running mean and continuous
            features by the running root mean square of their deviations from
            target, so c does not depend on either's units.
        min_weight: Lower bound on a row's weight relative to its design
            weight, w / d. Default 0.001.
        max_weight: Upper bound on w / d. Default 100.
        trim: Cap each weight at this multiple of the mean weight assigned so
            far, trading some bias for lower variance as in Potter's (1990)
            trimming; common choices are 3 to 6. Past weights cannot absorb
            the trimmed amount, so the penalty shifts it onto later rows. The
            cap overrides min_weight. Default None, no trimming.
        outcomes: Names of outcomes whose weighted means to track.
        keep_weights: Store each assigned weight. Off by default, which keeps
            memory independent of stream length.

    Weights are returned relative to the mean design weight so far, so they
    sum to about the number of rows. To express them as population counts,
    multiply each kept weight by N / total_weight at the end.

    Examples:
        >>> from onlinerake import StreamingRaker, Targets
        >>> raker = StreamingRaker(Targets(female=0.5), outcomes=["spend"])
        >>> for female, spend in [(1, 10.0), (1, 12.0), (1, 11.0), (0, 20.0)] * 50:
        ...     _ = raker.partial_fit({"female": female}, outcomes={"spend": spend})
        >>> round(raker.margins["female"], 2)
        0.52
        >>> round(raker.estimate("spend"), 1)
        15.3
    """

    def __init__(
        self,
        targets: Targets,
        calibration_strength: float = 0.3,
        min_weight: float = 1e-3,
        max_weight: float = 100.0,
        trim: float | None = None,
        outcomes: Sequence[str] = (),
        keep_weights: bool = False,
    ) -> None:
        validate_positive(calibration_strength, "calibration_strength")
        validate_positive(min_weight, "min_weight")
        validate_positive(max_weight, "max_weight")
        if max_weight <= min_weight:
            raise ValueError("max_weight must exceed min_weight")
        if trim is not None and not trim > 1.0:
            raise ValueError("trim must exceed 1")
        if len(set(outcomes)) != len(outcomes):
            raise ValueError("outcome names must be unique")

        self.targets = targets
        self.calibration_strength = float(calibration_strength)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.trim = None if trim is None else float(trim)
        self.outcome_names = tuple(outcomes)
        self._feature_names = targets.feature_names
        self._target = np.array(
            [targets[name] for name in self._feature_names], dtype=np.float64
        )
        self._residual = np.zeros(len(self._feature_names), dtype=np.float64)
        self._continuous = np.array(
            [targets.is_continuous(name) for name in self._feature_names]
        )
        self._squared_deviation = np.zeros(
            int(self._continuous.sum()), dtype=np.float64
        )
        self._design_total = 0.0
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
        """Sum of assigned weights, in units of the mean design weight."""
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
        return {name: float(values[i]) for i, name in enumerate(self._feature_names)}

    @property
    def loss(self) -> float:
        """Sum of squared margin errors; NaN before any row."""
        if self._n_obs == 0:
            return np.nan
        errors = self._residual / self._total_weight
        return float(errors @ errors)

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
        """Weighted mean of one registered outcome.

        Args:
            name: An outcome name passed to the constructor.

        Returns:
            The weighted mean, or NaN before any row.

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

    def _features(self, obs: dict[str, Any] | Any) -> npt.NDArray[np.float64]:
        values = np.empty(len(self._feature_names), dtype=np.float64)
        for i, name in enumerate(self._feature_names):
            raw = obs.get(name, 0) if isinstance(obs, dict) else getattr(obs, name, 0)
            values[i] = feature_value(raw, name, binary=self.targets.is_binary(name))
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
            design_weight: The row's base weight, in any units. Default 1.
            outcomes: Values for every registered outcome name.

        Returns:
            The weight assigned to this row, relative to the mean design
            weight so far.

        Raises:
            ValueError: If a feature, outcome, or design weight is invalid.
                State is unchanged when this is raised.
        """
        validate_positive(design_weight, "design_weight")
        x = self._features(obs)
        y = self._outcome_values(outcomes)
        delta = x - self._target
        n = self._n_obs + 1
        self._design_total += float(design_weight)
        self._squared_deviation += delta[self._continuous] ** 2
        spread = np.ones_like(delta)
        rms = np.sqrt(self._squared_deviation / n)
        spread[self._continuous] = np.where(rms > 0.0, rms, 1.0)
        weight = arrival_weight(
            delta / spread,
            self._residual / spread,
            float(design_weight) * n / self._design_total,
            self.calibration_strength,
            self.min_weight,
            self.max_weight,
        )
        if self.trim is not None:
            mean = self._total_weight / self._n_obs if self._n_obs else 1.0
            weight = min(weight, self.trim * mean)
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


def fit_penalized_raking(
    features: npt.ArrayLike,
    targets: Targets,
    calibration_strength: float = 0.3,
    design_weights: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    """Minimize the penalized raking objective over all weights at once.

    The minimizer is the exponential tilt w_i = d_i exp(-lambda.delta_i) with
    lambda = c * sum_i w_i delta_i, so it is found from the strictly convex
    p-dimensional dual
    G(lambda) = [sum_i d_i exp(-lambda.delta_i) + ||lambda||^2 / (2c)] / sum_i d_i.
    This is entropy balancing with a ridge on the multipliers; as c grows it
    approaches raking. Weights are not bounded here, unlike StreamingRaker.
    As in StreamingRaker, the penalty is computed with design weights divided
    by their mean and continuous deviations divided by their root mean square;
    the returned weights are in the units of design_weights.

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
    continuous = np.array([targets.is_continuous(name) for name in names])
    rms = np.sqrt((delta**2).mean(axis=0))
    delta = delta / np.where(continuous & (rms > 0.0), rms, 1.0)
    mean_design = d.mean()
    d = d / mean_design
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
        # Near 1e-9 a Newton step lowers the O(1) objective by about grad^2,
        # below float64 resolution, so SciPy cannot confirm progress and
        # reports failure. 1e-8 is the tightest gradient check that is reachable.
        options={"gtol": 1e-8},
    )
    if not result.success:
        raise RuntimeError(f"penalized raking did not converge: {result.message}")
    return mean_design * tilt(result.x)
