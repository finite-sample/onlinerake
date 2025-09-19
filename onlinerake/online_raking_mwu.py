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

from typing import Any

import numpy as np

from .online_raking_sgd import OnlineRakingSGD


class OnlineRakingMWU(OnlineRakingSGD):
    """Online raking via multiplicative weights updates.

    Parameters
    ----------
    targets : :class:`~onlinerake.targets.Targets`
        Target population proportions for each demographic characteristic.
    learning_rate : float, optional
        Step size used in the exponent of the multiplicative update.  A
        typical default is ``learning_rate=1.0``.  Larger values may
        accelerate convergence but risk weight explosion.
    min_weight : float, optional
        Lower bound applied to the weights after each update.  This
        prevents weights from collapsing to zero.  Must be positive.
    max_weight : float, optional
        Upper bound applied to the weights after each update.  This
        prevents runaway weights.  Must exceed ``min_weight``.
    n_steps : int, optional
        Number of multiplicative updates applied each time a new
        observation arrives.
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
        )

    def partial_fit(self, obs: Any) -> None:
        """Consume a single observation and update weights multiplicatively."""

        # Add new observation and initialise weight as in base class
        def _get_indicator(obj: Any, name: str) -> int:
            val = obj[name] if isinstance(obj, dict) else getattr(obj, name)
            return int(bool(val))

        age = _get_indicator(obs, "age")
        gender = _get_indicator(obs, "gender")
        education = _get_indicator(obs, "education")
        region = _get_indicator(obs, "region")

        self._age.append(age)
        self._gender.append(gender)
        self._education.append(education)
        self._region.append(region)
        self._n_obs += 1
        if self._weights.size == 0:
            self._weights = np.array([1.0], dtype=float)
        else:
            self._weights = np.append(self._weights, 1.0)

        # multiplicative update steps
        final_gradient_norm = 0.0
        for step in range(self.n_sgd_steps):
            grad = self._compute_gradient()
            
            # Calculate gradient norm for convergence monitoring
            gradient_norm = float(np.linalg.norm(grad))
            if step == self.n_sgd_steps - 1:  # Store only final gradient norm
                final_gradient_norm = gradient_norm
            
            # exponentiate negative gradient times LR
            # clip exponent to avoid overflow
            update = np.exp(-self.learning_rate * grad)
            self._weights *= update
            np.clip(self._weights, self.min_weight, self.max_weight, out=self._weights)
            
            # Verbose output for debugging
            if self.verbose and self._n_obs % 100 == 0 and step == 0:
                print(f"MWU Obs {self._n_obs}: loss={self.loss:.6f}, grad_norm={gradient_norm:.6f}, "
                      f"ess={self.effective_sample_size:.1f}")

        # record state with final gradient norm
        self._record_state(gradient_norm=final_gradient_norm)

    # alias for consistency with base class
    fit_one = partial_fit
