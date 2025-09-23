"""Streaming raking based on stochastic gradient descent.

This module implements a minimalistic yet flexible online raking algorithm
for adjusting survey weights to match known population margins.  It
maintains an internal weight vector that is updated whenever a new
observation arrives.  The weights are adjusted so that the weighted
proportions of each demographic characteristic track the target
population proportions.  The algorithm uses stochastic gradient
descent (SGD) on a squared‑error loss defined on the margins.

Unlike classic batch raking or iterative proportional fitting (IPF),
this implementation works in a streaming fashion: it does **not**
revisit past observations except through their contribution to the
cumulative weight totals.  Each update runs in *O(n)* time for
n observations.  For large data streams you may wish to consider
optimisations such as keeping only aggregate totals or using a single
gradient step per observation.

The class adheres to a simplified scikit‑learn ``partial_fit`` API: each
call to :meth:`partial_fit` consumes a single observation (encoded as a
mapping or any object exposing the relevant demographic attributes) and
updates the internal weights.  After each call, properties such as
``margins``, ``loss`` and ``effective_sample_size`` provide insight
into the current state of the estimator.

Example::

    from onlinerake import OnlineRakingSGD, Targets
    targets = Targets(age=0.5, gender=0.5, education=0.4, region=0.3)
    raker = OnlineRakingSGD(targets, learning_rate=5.0)
    for obs in stream:
        raker.partial_fit(obs)
        print(raker.margins)  # inspect weighted margins after each step

The algorithm is described in the accompanying README and research
notes.  See also :mod:`onlinerake.online_raking_mwu` for an alternative
update strategy based on multiplicative weights.
"""

from typing import Any, MutableSequence

import numpy as np

from .targets import Targets


class OnlineRakingSGD:
    """Online raking via stochastic gradient descent.

    Parameters
    ----------
    targets : :class:`~onlinerake.targets.Targets`
        Target population proportions for each demographic characteristic.
    learning_rate : float, optional
        Step size used in the gradient descent update. Larger values lead
        to more aggressive updates but may cause oscillation or divergence.
    min_weight : float, optional
        Lower bound applied to the weights after each update to prevent
        weights from collapsing to zero.  Must be positive.
    max_weight : float, optional
        Upper bound applied to the weights after each update to prevent
        runaway weights.  Must exceed ``min_weight``.
    n_sgd_steps : int, optional
        Number of gradient steps applied each time a new observation
        arrives.  Values larger than 1 can help reduce oscillations but
        increase computational cost.
    compute_weight_stats : bool or int, optional
        Controls computation of weight distribution statistics for performance.
        If True, compute on every call. If False, use cached values.
        If integer k, compute every k observations. Default is False.

    Notes
    -----
    * For binary demographic indicators the gradient of the margin with
      respect to each weight can be derived analytically.  See the
      documentation for details.
    * The algorithm does not currently support categorical controls with
      more than two levels.  Extending to multi‑level categories would
      require storing one hot encodings and expanding the margin loss
      accordingly.
    """

    def __init__(
        self,
        targets: Targets,
        learning_rate: float = 5.0,
        min_weight: float = 1e-3,
        max_weight: float = 100.0,
        n_sgd_steps: int = 3,
        verbose: bool = False,
        track_convergence: bool = True,
        convergence_window: int = 20,
        compute_weight_stats: bool | int = False,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if min_weight <= 0:
            raise ValueError("min_weight must be strictly positive")
        if max_weight <= min_weight:
            raise ValueError("max_weight must exceed min_weight")
        if n_sgd_steps < 1:
            raise ValueError("n_sgd_steps must be a positive integer")
        if convergence_window < 1:
            raise ValueError("convergence_window must be a positive integer")
        if not isinstance(compute_weight_stats, (bool, int)):
            raise ValueError(
                "compute_weight_stats must be True, False, or a positive integer"
            )
        if (
            isinstance(compute_weight_stats, int)
            and not isinstance(compute_weight_stats, bool)
            and compute_weight_stats < 1
        ):
            raise ValueError(
                "compute_weight_stats must be True, False, or a positive integer"
            )

        self.targets = targets
        self.learning_rate = learning_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.n_sgd_steps = n_sgd_steps
        self.verbose = verbose
        self.track_convergence = track_convergence
        self.convergence_window = convergence_window
        self.compute_weight_stats = compute_weight_stats

        # internal state with capacity doubling for performance
        self._weights_capacity: int = 0
        self._weights: np.ndarray = np.empty(0, dtype=float)
        # store demographic indicators for each observation in separate arrays
        self._age: MutableSequence[int] = []
        self._gender: MutableSequence[int] = []
        self._education: MutableSequence[int] = []
        self._region: MutableSequence[int] = []
        self._n_obs: int = 0

        # cached weight statistics for performance
        self._cached_weight_stats: dict[str, float] | None = None
        self._weight_stats_computed_at: int = 0

        # history: list of metric dicts recorded after each update
        self.history: list[dict[str, Any]] = []

        # convergence tracking
        self._loss_history: list[float] = []
        self._gradient_norms: list[float] = []
        self._converged: bool = False
        self._convergence_step: int | None = None

    # ------------------------------------------------------------------
    # Utility properties
    # ------------------------------------------------------------------
    @property
    def weights(self) -> np.ndarray:
        """Return a copy of the current weight vector."""
        return self._weights[: self._n_obs].copy()

    @property
    def margins(self) -> dict[str, float]:
        """Return current weighted margins as a dictionary."""
        if self._n_obs == 0:
            return {k: np.nan for k in self.targets.as_dict()}
        w = self._weights[: self._n_obs]
        total = w.sum()
        margins = {}
        for name, arr in zip(
            ["age", "gender", "education", "region"],
            [self._age, self._gender, self._education, self._region],
        ):
            margins[name] = float(np.dot(w, arr) / total)
        return margins

    @property
    def raw_margins(self) -> dict[str, float]:
        """Return unweighted (raw) margins as a dictionary."""
        if self._n_obs == 0:
            return {k: np.nan for k in self.targets.as_dict()}
        raw = {}
        for name, arr in zip(
            ["age", "gender", "education", "region"],
            [self._age, self._gender, self._education, self._region],
        ):
            raw[name] = float(np.mean(arr))
        return raw

    @property
    def loss(self) -> float:
        """Return the current squared‑error loss on margins."""
        if self._n_obs == 0:
            return np.nan
        m = self.margins
        loss = 0.0
        for name, target in self.targets.as_dict().items():
            diff = m[name] - target
            loss += diff * diff
        return float(loss)

    @property
    def effective_sample_size(self) -> float:
        """Return the effective sample size (ESS).

        ESS is defined as (sum w_i)^2 / (sum w_i^2).  It reflects
        the number of equally weighted observations that would yield the
        same variance as the current weighted estimator.
        """
        if self._n_obs == 0:
            return 0.0
        w = self._weights
        sum_w = w.sum()
        sum_w2 = (w * w).sum()
        return float((sum_w * sum_w) / sum_w2) if sum_w2 > 0 else 0.0

    @property
    def converged(self) -> bool:
        """Return True if the algorithm has detected convergence."""
        return self._converged

    @property
    def convergence_step(self) -> int | None:
        """Return the step number where convergence was detected, if any."""
        return self._convergence_step

    @property
    def loss_moving_average(self) -> float:
        """Return moving average of loss over convergence window."""
        if len(self._loss_history) == 0:
            return np.nan
        window_size = min(self.convergence_window, len(self._loss_history))
        return float(np.mean(self._loss_history[-window_size:]))

    @property
    def gradient_norm_history(self) -> list[float]:
        """Return history of gradient norms for convergence analysis."""
        return self._gradient_norms.copy()

    @property
    def weight_distribution_stats(self) -> dict[str, float]:
        """Return comprehensive weight distribution statistics."""
        if self._n_obs == 0:
            return {
                k: np.nan
                for k in [
                    "min",
                    "max",
                    "mean",
                    "std",
                    "median",
                    "q25",
                    "q75",
                    "outliers_count",
                ]
            }

        w = self._weights
        q25, median, q75 = np.percentile(w, [25, 50, 75])
        iqr = q75 - q25
        outlier_threshold = 1.5 * iqr
        outliers_count = np.sum(
            (w < (q25 - outlier_threshold)) | (w > (q75 + outlier_threshold))
        )

        return {
            "min": float(w.min()),
            "max": float(w.max()),
            "mean": float(w.mean()),
            "std": float(w.std()),
            "median": float(median),
            "q25": float(q25),
            "q75": float(q75),
            "outliers_count": int(outliers_count),
        }

    def detect_oscillation(self, threshold: float = 0.1) -> bool:
        """Detect if loss is oscillating rather than converging.

        Parameters
        ----------
        threshold : float
            Relative threshold for detecting oscillation vs trend.

        Returns
        -------
        bool
            True if oscillation is detected in recent loss history.
        """
        if len(self._loss_history) < self.convergence_window:
            return False

        recent_losses = self._loss_history[-self.convergence_window :]

        # Calculate variance in recent losses
        loss_variance = np.var(recent_losses)
        mean_loss = np.mean(recent_losses)

        # Check if variance is high relative to mean (indicating oscillation)
        if mean_loss > 0:
            cv = np.sqrt(loss_variance) / mean_loss
            return bool(cv > threshold)
        return False

    def check_convergence(self, tolerance: float = 1e-6) -> bool:
        """Check if algorithm has converged based on loss stability.

        Parameters
        ----------
        tolerance : float
            Convergence tolerance for loss stability.

        Returns
        -------
        bool
            True if convergence is detected.
        """
        if self._converged or len(self._loss_history) < self.convergence_window:
            return self._converged

        recent_losses = self._loss_history[-self.convergence_window :]
        mean_loss = float(np.mean(recent_losses))

        # First check if loss is essentially zero
        if mean_loss <= tolerance:
            if not self._converged:
                self._converged = True
                self._convergence_step = self._n_obs
                if self.verbose:
                    print(
                        f"Convergence detected at observation {self._n_obs} (loss ≈ 0)"
                    )
            return True

        # Then check relative stability for non-zero loss
        loss_std = float(np.std(recent_losses))
        relative_std = loss_std / mean_loss
        if relative_std < tolerance:
            if not self._converged:
                self._converged = True
                self._convergence_step = self._n_obs
                if self.verbose:
                    print(f"Convergence detected at observation {self._n_obs}")
            return True

        return False

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------
    def _compute_gradient(
        self, precomputed_arrays: dict[str, np.ndarray] | None = None
    ) -> np.ndarray:
        """Compute gradient of the margin loss with respect to weights.

        Parameters
        ----------
        precomputed_arrays : dict, optional
            Pre-converted numpy arrays for demographic indicators.
            If None, arrays will be computed from indicator lists.

        Returns
        -------
        np.ndarray
            Gradient vector of shape (n_obs,) containing the gradient for
            each weight.  The gradient expression is derived in the
            accompanying paper/notes and corresponds to the derivative of
            ``(margins - targets)`` squared with respect to each weight.
        """
        n = self._n_obs
        if n == 0:
            return np.empty(0, dtype=float)
        w = self._weights[:n]
        total_w = w.sum()

        # Use precomputed arrays if provided, otherwise compute them
        if precomputed_arrays is not None:
            arrs = precomputed_arrays
        else:
            # Precompute weighted sums for each characteristic
            # Each arr is a list of ints (0/1) of length n
            arrs = {
                "age": np.array(self._age, dtype=float),
                "gender": np.array(self._gender, dtype=float),
                "education": np.array(self._education, dtype=float),
                "region": np.array(self._region, dtype=float),
            }
        targets = self.targets.as_dict()

        gradients = np.zeros(n, dtype=float)
        for name, arr in arrs.items():
            # Weighted sum of this characteristic
            weighted_sum = np.dot(w, arr)
            current_margin = weighted_sum / total_w
            target = targets[name]
            # derivative of margin w.r.t each weight
            # margin = sum_i w_i x_i / sum_i w_i
            # d margin / d w_k = (x_k * total_w - weighted_sum) / total_w^2
            margin_grad = (arr * total_w - weighted_sum) / (total_w * total_w)
            loss_grad = 2.0 * (current_margin - target) * margin_grad
            gradients += loss_grad
        return gradients

    def _record_state(self, gradient_norm: float | None = None) -> None:
        """Record current metrics to history."""
        current_loss = self.loss
        self._loss_history.append(current_loss)

        # Store gradient norm if provided
        if gradient_norm is not None:
            self._gradient_norms.append(gradient_norm)

        state = {
            "n_obs": self._n_obs,
            "loss": current_loss,
            "weighted_margins": self.margins,
            "raw_margins": self.raw_margins,
            "ess": self.effective_sample_size,
            "weight_stats": self.weight_distribution_stats,
            "gradient_norm": gradient_norm if gradient_norm is not None else np.nan,
            "loss_moving_avg": self.loss_moving_average,
            "converged": self.converged,
            "oscillating": (
                self.detect_oscillation() if self.track_convergence else False
            ),
        }
        self.history.append(state)

        # Check convergence if tracking is enabled
        if self.track_convergence and not self._converged:
            self.check_convergence()

    def partial_fit(self, obs: Any) -> None:
        """Consume a single observation and update weights.

        Parameters
        ----------
        obs : mapping or object
            An observation containing demographic indicators.  The
            attributes/keys ``age``, ``gender``, ``education`` and
            ``region`` must be accessible on the object.  The values
            should be 0 or 1.  Anything truthy is interpreted as 1.

        Returns
        -------
        None
            The internal state is updated in place.  The caller can
            inspect the properties ``weights``, ``margins`` and ``loss``
            after the call for diagnostics.
        """

        # Convert to numeric binary indicators
        def _get_indicator(obj: Any, name: str) -> int:
            val = obj[name] if isinstance(obj, dict) else getattr(obj, name)
            return int(bool(val))

        age = _get_indicator(obs, "age")
        gender = _get_indicator(obs, "gender")
        education = _get_indicator(obs, "education")
        region = _get_indicator(obs, "region")

        # Append new observation and weight
        self._age.append(age)
        self._gender.append(gender)
        self._education.append(education)
        self._region.append(region)
        self._n_obs += 1
        # Initialise weight to 1.0 for new obs; enlarge array
        if self._weights.size == 0:
            self._weights = np.array([1.0], dtype=float)
        else:
            self._weights = np.append(self._weights, 1.0)

        # perform n_sgd_steps updates
        final_gradient_norm = 0.0
        for step in range(self.n_sgd_steps):
            grad = self._compute_gradient()

            # Calculate gradient norm for convergence monitoring
            gradient_norm = float(np.linalg.norm(grad))
            if step == self.n_sgd_steps - 1:  # Store only final gradient norm
                final_gradient_norm = gradient_norm

            self._weights -= self.learning_rate * grad
            # clip weights
            np.clip(self._weights, self.min_weight, self.max_weight, out=self._weights)

            # Verbose output for debugging
            if self.verbose and self._n_obs % 100 == 0 and step == 0:
                print(
                    f"Obs {self._n_obs}: loss={self.loss:.6f}, grad_norm={gradient_norm:.6f}, "
                    f"ess={self.effective_sample_size:.1f}"
                )

        # record state with final gradient norm
        self._record_state(gradient_norm=final_gradient_norm)

    # alias for consistency with MWU version
    fit_one = partial_fit
