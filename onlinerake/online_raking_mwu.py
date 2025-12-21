"""Streaming raking using multiplicative weights updates (MWU).

This module implements a streaming survey calibration algorithm based on
the multiplicative weights update (MWU) rule.  Instead of subtracting a
scaled gradient like stochastic gradient descent, the algorithm
multiplies each weight by an exponential of its gradient.  This
corresponds to a mirror descent step on the Kullback–Leibler
divergence, which is the natural Bregman divergence for non‑negative
weights.

The resulting updates maintain positivity by construction and tend to
produce weight distributions more similar to classical raking (IPF)
weights.  However, overly aggressive learning rates can lead to weight
explosions or collapses.  Use the optional weight clipping to keep
weights within reasonable bounds.

The class is a drop‑in replacement for
:class:`~onlinerake.online_raking_sgd.OnlineRakingSGD`; it shares
nearly the same API and internal metrics.  See the base class for
attribute definitions and usage examples.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy import finfo, log

from .online_raking_sgd import OnlineRakingSGD


class OnlineRakingMWU(OnlineRakingSGD):
    """Online raking via multiplicative weights updates.

    Parameters
    ----------
    targets : :class:`~onlinerake.targets.Targets`
        Target population proportions for each feature.
    learning_rate : float, optional
        Step size used in the exponent of the multiplicative update.  A
        typical default is ``learning_rate=1.0``.  The algorithm automatically
        clips extreme exponents based on the weights dtype to prevent numerical
        overflow/underflow, making it robust even with very large learning rates.
    min_weight : float, optional
        Lower bound applied to the weights after each update.  This
        prevents weights from collapsing to zero.  Must be positive.
    max_weight : float, optional
        Upper bound applied to the weights after each update.  This
        prevents runaway weights.  Must exceed ``min_weight``.
    n_steps : int, optional
        Number of multiplicative updates applied each time a new
        observation arrives.
    compute_weight_stats : bool or int, optional
        Controls computation of weight distribution statistics for performance.
        If True, compute on every call. If False, never compute.
        If integer k, compute every k observations. Default is False.
    """

    def __init__(
        self,
        targets,
        learning_rate: float = 1.0,
        min_weight: float = 1e-3,
        max_weight: float = 100.0,
        n_steps: int = 3,
        verbose: bool = False,
        track_convergence: bool = True,
        convergence_window: int = 20,
        compute_weight_stats: bool | int = False,
    ) -> None:
        super().__init__(
            targets=targets,
            learning_rate=learning_rate,
            min_weight=min_weight,
            max_weight=max_weight,
            n_sgd_steps=n_steps,
            verbose=verbose,
            track_convergence=track_convergence,
            convergence_window=convergence_window,
            compute_weight_stats=compute_weight_stats,
        )

    def partial_fit(self, obs: dict[str, Any] | Any) -> None:
        """Consume a single observation and update weights multiplicatively.

        Parameters
        ----------
        obs : dict or object
            An observation containing feature indicators. For dict input,
            keys should match feature names in targets. For object input,
            features are accessed as attributes. Values should be binary
            (0/1 or False/True).

        Returns
        -------
        None
            The internal state is updated in place.
        """
        # Ensure we have capacity
        self._expand_capacity()

        # Extract feature values in the correct order
        feature_values = np.zeros(self._n_features, dtype=np.int8)
        for i, name in enumerate(self._feature_names):
            if isinstance(obs, dict):
                val = obs.get(name, 0)
            else:
                val = getattr(obs, name, 0)
            feature_values[i] = int(bool(val))

        # Store the observation
        self._features[self._n_obs] = feature_values
        self._weights[self._n_obs] = 1.0
        self._n_obs += 1

        # MWU steps (entropic mirror descent) with safe exponent clipping
        # Compute log(max float) from dtype to avoid transient inf/0 before clipping
        # Example: float64 -> ~709.78; float32 -> ~88.72
        max_log = float(log(finfo(self._weights.dtype).max))
        # Give ourselves a safety margin to stay away from the asymptote
        max_log *= 0.95  # Slightly more conservative than original 0.99

        final_gradient_norm = 0.0
        for step in range(self.n_sgd_steps):
            grad = self._compute_gradient()

            # Calculate gradient norm for convergence monitoring
            gradient_norm = float(np.linalg.norm(grad))
            if step == self.n_sgd_steps - 1:  # Store only final gradient norm
                final_gradient_norm = gradient_norm

            # Clip the exponent argument BEFORE exp to keep everything finite
            expo = -self.learning_rate * grad
            np.clip(expo, -max_log, max_log, out=expo)
            update = np.exp(expo, dtype=self._weights.dtype)

            # Multiplicative update + in-range clipping
            # Update only the active portion of the weights array
            self._weights[: self._n_obs] *= update
            np.clip(
                self._weights[: self._n_obs],
                self.min_weight,
                self.max_weight,
                out=self._weights[: self._n_obs],
            )

            # Verbose output for debugging
            if self.verbose and self._n_obs % 100 == 0 and step == 0:
                logging.info(
                    f"MWU Obs {self._n_obs}: loss={self.loss:.6f}, grad_norm={gradient_norm:.6f}, "
                    f"ess={self.effective_sample_size:.1f}"
                )

        # record state with final gradient norm
        self._record_state(gradient_norm=final_gradient_norm)

    # alias for consistency with base class
    fit_one = partial_fit
