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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, MutableSequence, Optional

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
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if min_weight <= 0:
            raise ValueError("min_weight must be strictly positive")
        if max_weight <= min_weight:
            raise ValueError("max_weight must exceed min_weight")
        if n_sgd_steps < 1:
            raise ValueError("n_sgd_steps must be a positive integer")

        self.targets = targets
        self.learning_rate = learning_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.n_sgd_steps = n_sgd_steps

        # internal state
        self._weights: np.ndarray = np.empty(0, dtype=float)
        # store demographic indicators for each observation in separate arrays
        self._age: MutableSequence[int] = []
        self._gender: MutableSequence[int] = []
        self._education: MutableSequence[int] = []
        self._region: MutableSequence[int] = []
        self._n_obs: int = 0

        # history: list of metric dicts recorded after each update
        self.history: list[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Utility properties
    # ------------------------------------------------------------------
    @property
    def weights(self) -> np.ndarray:
        """Return a copy of the current weight vector."""
        return self._weights.copy()

    @property
    def margins(self) -> Dict[str, float]:
        """Return current weighted margins as a dictionary."""
        if self._n_obs == 0:
            return {k: np.nan for k in self.targets.as_dict()}
        w = self._weights
        total = w.sum()
        margins = {}
        for name, arr in zip(
            ["age", "gender", "education", "region"],
            [self._age, self._gender, self._education, self._region],
        ):
            margins[name] = float(np.dot(w, arr) / total)
        return margins

    @property
    def raw_margins(self) -> Dict[str, float]:
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

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------
    def _compute_gradient(self) -> np.ndarray:
        """Compute gradient of the margin loss with respect to weights.

        Returns a vector of shape (n_obs,) containing the gradient for
        each weight.  The gradient expression is derived in the
        accompanying paper/notes and corresponds to the derivative of
        ``(margins - targets)`` squared with respect to each weight.
        """
        n = self._n_obs
        if n == 0:
            return np.empty(0, dtype=float)
        w = self._weights
        total_w = w.sum()

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

    def _record_state(self) -> None:
        """Record current metrics to history."""
        state = {
            "n_obs": self._n_obs,
            "loss": self.loss,
            "weighted_margins": self.margins,
            "raw_margins": self.raw_margins,
            "ess": self.effective_sample_size,
            "weight_stats": {
                "min": float(self._weights.min()) if self._n_obs > 0 else np.nan,
                "max": float(self._weights.max()) if self._n_obs > 0 else np.nan,
                "mean": float(self._weights.mean()) if self._n_obs > 0 else np.nan,
                "std": float(self._weights.std()) if self._n_obs > 0 else np.nan,
            },
        }
        self.history.append(state)

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
        for _ in range(self.n_sgd_steps):
            grad = self._compute_gradient()
            self._weights -= self.learning_rate * grad
            # clip weights
            np.clip(self._weights, self.min_weight, self.max_weight, out=self._weights)

        # record state
        self._record_state()

    # alias for consistency with MWU version
    fit_one = partial_fit