"""Streaming raking based on stochastic gradient descent.

This module implements a minimalistic yet flexible online raking algorithm
for adjusting survey weights to match known population margins. It maintains
an internal weight vector that is updated whenever a new observation arrives.
The weights are adjusted so that the weighted proportions of each feature
track the target population proportions using stochastic gradient descent (SGD)
on a squared-error loss defined on the margins.

Unlike classic batch raking or iterative proportional fitting (IPF), this
implementation works in a streaming fashion: it does not revisit past
observations except through their contribution to the cumulative weight totals.
Each update runs in O(k) time for k features (independent of n observations).

The class follows the scikit-learn ``partial_fit`` API pattern.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from .targets import Targets

if TYPE_CHECKING:
    from .learning_rate import LearningRateSchedule


class OnlineRakingSGD:
    """Online raking via stochastic gradient descent.

    A streaming weight calibration algorithm that adjusts observation weights
    to match target population margins using stochastic gradient descent (SGD).
    The algorithm minimizes squared-error loss between weighted margins and
    target proportions.

    Args:
        targets: Target population proportions for each feature.
        learning_rate: Step size for gradient descent updates. Can be:
            - float: Fixed learning rate (default: 5.0)
            - LearningRateSchedule: Dynamic schedule (e.g., robbins_monro_schedule())
            Larger values lead to faster convergence but may cause oscillation.
        min_weight: Lower bound for weights to prevent collapse. Must be positive.
            Default: 0.001.
        max_weight: Upper bound for weights to prevent explosion. Must exceed
            min_weight. Default: 100.0.
        n_sgd_steps: Number of gradient steps per observation. More steps can
            reduce oscillations but increase computation. Default: 3.
        verbose: If True, log progress information. Default: False.
        track_convergence: If True, monitor convergence metrics. Default: True.
        convergence_window: Number of observations for convergence detection.
            Default: 20.
        compute_weight_stats: Control weight statistics computation.
            If True: compute every observation.
            If False: never compute (best performance).
            If int k: compute every k observations. Default: False.
        max_history: Maximum historical states to retain. None for unlimited
            (may cause memory issues). Default: 1000.

    Attributes:
        targets: The target proportions.
        history: List of historical states after each update.

    Examples:
        >>> # General features
        >>> targets = Targets(owns_car=0.4, is_subscriber=0.2)
        >>> raker = OnlineRakingSGD(targets, learning_rate=5.0)
        >>> raker.partial_fit({'owns_car': 1, 'is_subscriber': 0})
        >>> print(f"Loss: {raker.loss:.4f}")

        >>> # Process multiple observations
        >>> for obs in stream:
        ...     raker.partial_fit(obs)
        ...     if raker.converged:
        ...         break

    Raises:
        ValueError: If any parameter is invalid (negative learning rate, invalid
            weight bounds, non-positive convergence window, invalid compute_weight_stats).

    Note:
        The algorithm supports arbitrary binary features, not limited to
        demographics. Feature names must match those defined in targets.
    """

    def __init__(
        self,
        targets: Targets,
        learning_rate: float | LearningRateSchedule = 5.0,
        min_weight: float = 1e-3,
        max_weight: float = 100.0,
        n_sgd_steps: int = 3,
        verbose: bool = False,
        track_convergence: bool = True,
        convergence_window: int = 20,
        compute_weight_stats: bool | int = False,
        max_history: int | None = 1000,
    ) -> None:
        # Handle learning rate - can be float or schedule
        if isinstance(learning_rate, (int, float)):
            if learning_rate <= 0:
                raise ValueError("learning_rate must be positive")
            self._lr_schedule = None
            self._base_learning_rate = float(learning_rate)
        else:
            # Assume it's a LearningRateSchedule
            self._lr_schedule = learning_rate
            self._base_learning_rate = learning_rate(1)

        if min_weight <= 0:
            raise ValueError("min_weight must be strictly positive")
        if max_weight <= min_weight:
            raise ValueError("max_weight must exceed min_weight")
        if n_sgd_steps < 1:
            raise ValueError("n_sgd_steps must be a positive integer")
        if convergence_window < 1:
            raise ValueError("convergence_window must be a positive integer")
        if not isinstance(compute_weight_stats, bool) and not isinstance(
            compute_weight_stats, int
        ):
            raise ValueError(
                "compute_weight_stats must be True, False, or a positive integer"
            )
        if isinstance(compute_weight_stats, int) and not isinstance(
            compute_weight_stats, bool
        ):
            if compute_weight_stats < 1:
                raise ValueError(
                    "compute_weight_stats must be True, False, or a positive integer"
                )

        self.targets = targets
        self.learning_rate = self._base_learning_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.n_sgd_steps = n_sgd_steps
        self.verbose = verbose
        self.track_convergence = track_convergence
        self.convergence_window = convergence_window
        self.compute_weight_stats = compute_weight_stats
        self.max_history = max_history

        # Extract feature information from targets
        self._feature_names = targets.feature_names
        self._n_features = targets.n_features

        # internal state with capacity doubling for performance
        self._weights_capacity: int = 0
        self._weights: np.ndarray = np.empty(0, dtype=np.float64)

        # Store features as a single 2D array for efficiency
        # Use float64 to support both binary (0/1) and continuous features
        self._features_capacity: int = 0
        self._features: np.ndarray = np.empty((0, self._n_features), dtype=np.float64)
        self._n_obs: int = 0

        # Precomputed target array for efficiency
        self._target_array = np.array(
            [targets[name] for name in self._feature_names], dtype=np.float64
        )

        # cached weight statistics for performance
        self._cached_weight_stats: dict[str, float] | None = None
        self._weight_stats_computed_at: int = 0

        # history: list of metric dicts recorded after each update
        self.history: list[dict[str, Any]] = []

        # convergence tracking
        self._loss_history: list[float] = []
        self._gradient_norms: list[float] = []
        self._learning_rate_history: list[float] = []
        self._converged: bool = False
        self._convergence_step: int | None = None

    # ------------------------------------------------------------------
    # Utility properties
    # ------------------------------------------------------------------
    @property
    def current_learning_rate(self) -> float:
        """Get current learning rate (may vary if using a schedule)."""
        if self._lr_schedule is not None and self._n_obs > 0:
            return self._lr_schedule(self._n_obs)
        return self._base_learning_rate

    @property
    def learning_rate_history(self) -> list[float]:
        """Get history of learning rates used at each observation."""
        return self._learning_rate_history.copy()

    @property
    def uses_lr_schedule(self) -> bool:
        """Return True if using a dynamic learning rate schedule."""
        return self._lr_schedule is not None

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """Get copy of current weight vector.

        Returns:
            Array of shape (n_obs,) containing current weights.

        Examples:
            >>> raker = OnlineRakingSGD(targets)
            >>> raker.partial_fit({'feature_a': 1, 'feature_b': 0})
            >>> weights = raker.weights
            >>> print(weights.shape)
            (1,)
        """
        return self._weights[: self._n_obs].copy()

    @property
    def margins(self) -> dict[str, float]:
        """Get current weighted margins.

        Computes the weighted proportion of observations where each feature
        equals 1, using the current weight vector.

        Returns:
            Dictionary mapping feature names to weighted proportions.
            Returns NaN for all features if no observations processed.

        Examples:
            >>> targets = Targets(a=0.5, b=0.3)
            >>> raker = OnlineRakingSGD(targets)
            >>> raker.partial_fit({'a': 1, 'b': 0})
            >>> margins = raker.margins
            >>> print(margins['a'] > margins['b'])  # a=1, b=0 in observation
            True
        """
        if self._n_obs == 0:
            return self._create_nan_margins()

        w = self._weights[: self._n_obs]
        total_w = w.sum()

        # Efficient vectorized computation
        weighted_sums = w @ self._features[: self._n_obs]
        margins = {
            name: float(weighted_sums[i] / total_w)
            for i, name in enumerate(self._feature_names)
        }
        return margins

    @property
    def raw_margins(self) -> dict[str, float]:
        """Get unweighted (raw) margins.

        Computes the simple proportion of observations where each feature
        equals 1, without using weights.

        Returns:
            Dictionary mapping feature names to unweighted proportions.
            Returns NaN for all features if no observations processed.

        Note:
            Useful for comparing weighted vs unweighted margins to assess
            the impact of the raking process.
        """
        if self._n_obs == 0:
            return self._create_nan_margins()

        # Mean of each feature column
        feature_means = self._features[: self._n_obs].mean(axis=0)
        raw = {
            name: float(feature_means[i]) for i, name in enumerate(self._feature_names)
        }
        return raw

    @property
    def loss(self) -> float:
        """Get current squared-error loss.

        Computes sum of squared differences between current weighted margins
        and target proportions.

        Returns:
            Squared-error loss. Returns NaN if no observations processed.
            Lower values indicate better calibration to targets.

        Examples:
            >>> # Perfect calibration would have loss near 0
            >>> raker = OnlineRakingSGD(targets)
            >>> # Process many observations...
            >>> if raker.loss < 0.001:
            ...     print("Well calibrated")
        """
        if self._n_obs == 0:
            return np.nan

        margins = self.margins
        loss = 0.0
        for name in self._feature_names:
            diff = margins[name] - self.targets[name]
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
        w = self._weights[: self._n_obs]
        sum_w = w.sum()
        sum_w2 = (w * w).sum()
        return float((sum_w * sum_w) / sum_w2) if sum_w2 > 0 else 0.0

    @property
    def converged(self) -> bool:
        """Return True if the algorithm has detected convergence."""
        return self._converged

    @property
    def convergence_step(self) -> int | None:
        """Get step number where convergence was detected.

        Returns:
            Observation number where convergence detected, or None if
            not yet converged.
        """
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
        """Get history of gradient norms.

        Returns:
            List of gradient norms from each SGD step. Useful for
            analyzing convergence behavior.
        """
        return self._gradient_norms.copy()

    @property
    def weight_distribution_stats(self) -> dict[str, float]:
        """Return comprehensive weight distribution statistics."""
        if self._n_obs == 0:
            return dict.fromkeys(
                ["min", "max", "mean", "std", "median", "q25", "q75", "outliers_count"],
                np.nan,
            )

        # Check if we should use cached values
        if isinstance(self.compute_weight_stats, bool):
            if not self.compute_weight_stats and self._cached_weight_stats is not None:
                return self._cached_weight_stats
        elif isinstance(self.compute_weight_stats, int):
            if (
                self._n_obs - self._weight_stats_computed_at
            ) < self.compute_weight_stats:
                if self._cached_weight_stats is not None:
                    return self._cached_weight_stats

        w = self._weights[: self._n_obs]
        q25, median, q75 = np.percentile(w, [25, 50, 75])
        iqr = q75 - q25
        outlier_threshold = 1.5 * iqr
        outliers_count = np.sum(
            (w < (q25 - outlier_threshold)) | (w > (q75 + outlier_threshold))
        )

        stats = {
            "min": float(w.min()),
            "max": float(w.max()),
            "mean": float(w.mean()),
            "std": float(w.std()),
            "median": float(median),
            "q25": float(q25),
            "q75": float(q75),
            "outliers_count": int(outliers_count),
        }

        # Cache the results
        self._cached_weight_stats = stats
        self._weight_stats_computed_at = self._n_obs

        return stats

    def detect_oscillation(self, threshold: float = 0.1) -> bool:
        """Detect if loss is oscillating rather than converging.

        Args:
            threshold: Relative threshold for detecting oscillation vs trend.
                Higher values are less sensitive to oscillation. Default: 0.1.

        Returns:
            True if oscillation detected in recent loss history, False otherwise.

        Note:
            Oscillation suggests the learning rate may be too high.
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

        Args:
            tolerance: Convergence tolerance. Smaller values require more
                stable loss. Default: 1e-6.

        Returns:
            True if convergence detected, False otherwise.

        Note:
            Convergence is detected when loss is near zero or when relative
            standard deviation of recent losses is below tolerance.
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
                    logging.info(
                        f"Convergence detected at observation {self._n_obs} (loss ≈ 0)"
                    )
            return True

        # Then check relative stability for non-zero loss
        loss_std = float(np.std(recent_losses))
        # Avoid division by very small mean_loss
        if mean_loss > tolerance:
            relative_std = loss_std / mean_loss
            if relative_std < tolerance:
                if not self._converged:
                    self._converged = True
                    self._convergence_step = self._n_obs
                    if self.verbose:
                        logging.info(
                            f"Convergence detected at observation {self._n_obs}"
                        )
                return True

        return False

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------
    def _get_current_learning_rate(self) -> float:
        """Get the learning rate for the current observation.

        Returns:
            Current learning rate, either fixed or from schedule.
        """
        if self._lr_schedule is not None:
            lr = self._lr_schedule(self._n_obs)
            self.learning_rate = lr  # Update for inspection
            return lr
        return self._base_learning_rate

    def _compute_gradient(self) -> npt.NDArray[np.float64]:
        """Compute gradient of the margin loss with respect to weights.

        Returns:
            Gradient vector of shape (n_obs,) where each element is the
            partial derivative of the loss with respect to that weight.

        Note:
            Internal method. The gradient formula is derived analytically
            from the squared-error loss on weighted margins.
        """
        n = self._n_obs
        if n == 0:
            return np.empty(0, dtype=np.float64)

        w = self._weights[:n]
        total_w = w.sum()

        # Get active features
        features = self._features[:n]  # Shape: (n_obs, n_features)

        # Compute current margins efficiently
        weighted_sums = w @ features  # Shape: (n_features,)
        current_margins = weighted_sums / total_w

        # Compute gradient for each feature
        gradients = np.zeros(n, dtype=np.float64)

        for i, feature_name in enumerate(self._feature_names):
            target = self.targets[feature_name]
            current_margin = current_margins[i]

            # Gradient of margin w.r.t. weights
            # margin = sum_j(w_j * x_j) / sum_j(w_j)
            # d(margin)/d(w_k) = (x_k * total_w - weighted_sum) / total_w^2
            feature_col = features[:, i]
            margin_grad = (feature_col * total_w - weighted_sums[i]) / (
                total_w * total_w
            )

            # Gradient of squared error loss
            loss_grad = 2.0 * (current_margin - target) * margin_grad
            gradients += loss_grad

        return gradients

    def _expand_capacity(self) -> None:
        """Expand internal arrays when capacity is reached.

        Uses capacity doubling strategy to amortize allocation cost to O(1)
        per observation. This avoids O(n²) complexity from repeated resizing.

        Note:
            Internal method. Called automatically by partial_fit.
        """
        # Expand weights array
        if self._n_obs >= self._weights_capacity:
            new_capacity = max(8, self._weights_capacity * 2, self._n_obs + 1)
            new_weights = np.ones(new_capacity, dtype=np.float64)
            if self._weights_capacity > 0:
                new_weights[: self._weights_capacity] = self._weights[
                    : self._weights_capacity
                ]
            self._weights = new_weights
            self._weights_capacity = new_capacity

        # Expand features array
        if self._n_obs >= self._features_capacity:
            new_capacity = max(8, self._features_capacity * 2, self._n_obs + 1)
            new_features = np.zeros((new_capacity, self._n_features), dtype=np.float64)
            if self._features_capacity > 0:
                new_features[: self._features_capacity] = self._features[
                    : self._features_capacity
                ]
            self._features = new_features
            self._features_capacity = new_capacity

    def _record_state(self, gradient_norm: float | None = None) -> None:
        """Record current metrics to history.

        Args:
            gradient_norm: Norm of the gradient at current step.
                If provided, will be included in historical record.

        Note:
            Internal method. Automatically manages history size based on
            max_history parameter.
        """
        current_loss = self.loss
        self._loss_history.append(current_loss)
        self._learning_rate_history.append(self.learning_rate)

        # Store gradient norm if provided
        if gradient_norm is not None:
            self._gradient_norms.append(gradient_norm)

        state = {
            "n_obs": self._n_obs,
            "loss": current_loss,
            "learning_rate": self.learning_rate,
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

        # Limit history size if specified
        if self.max_history is not None and len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

        # Check convergence if tracking is enabled
        if self.track_convergence and not self._converged:
            self.check_convergence()

    def _create_nan_margins(self) -> dict[str, float]:
        """Create margins dict with NaN for all features.

        Returns:
            Dictionary mapping all feature names to NaN.
        """
        return dict.fromkeys(self._feature_names, np.nan)

    def _extract_feature_values(self, obs: dict[str, Any] | Any) -> np.ndarray:
        """Extract feature values from observation in correct order.

        Args:
            obs: Observation containing feature values. Can be dict or object.

        Returns:
            Array of feature values in the same order as self._feature_names.
        """
        feature_values = np.zeros(self._n_features, dtype=np.float64)
        for i, name in enumerate(self._feature_names):
            if isinstance(obs, dict):
                val = obs.get(name, 0)
            else:
                val = getattr(obs, name, 0)

            if self.targets.is_binary(name):
                feature_values[i] = 1.0 if val else 0.0
            else:
                feature_values[i] = float(val)

        return feature_values

    def _log_update(self, prefix: str, gradient_norm: float) -> None:
        """Log update if verbose mode and every 100 observations.

        Args:
            prefix: Prefix for the log message (e.g., "Obs" or "MWU Obs").
            gradient_norm: Current gradient norm to log.
        """
        if self.verbose and self._n_obs % 100 == 0:
            logging.info(
                f"{prefix} {self._n_obs}: loss={self.loss:.6f}, "
                f"grad_norm={gradient_norm:.6f}, ess={self.effective_sample_size:.1f}"
            )

    def partial_fit(self, obs: dict[str, Any] | Any) -> None:
        """Process single observation and update weights.

        Args:
            obs: Observation containing feature values. Can be:
                - dict: Keys should match feature names in targets
                - object: Features accessed as attributes
                For binary features: values should be 0/1 or False/True.
                For continuous features: values should be numeric (float/int).
                Missing features default to 0.

        Returns:
            None. Updates internal state in place.

        Examples:
            >>> # Binary features only
            >>> targets = Targets(owns_car=0.4, is_subscriber=0.2)
            >>> raker = OnlineRakingSGD(targets)
            >>> raker.partial_fit({'owns_car': 1, 'is_subscriber': 0})

            >>> # Mixed binary and continuous features
            >>> targets = Targets(gender=0.5, age=(35.0, "mean"))
            >>> raker = OnlineRakingSGD(targets)
            >>> raker.partial_fit({'gender': 1, 'age': 42.5})

            >>> # Object input (e.g., dataclass or namedtuple)
            >>> from dataclasses import dataclass
            >>> @dataclass
            ... class Obs:
            ...     owns_car: int
            ...     is_subscriber: int
            >>> raker.partial_fit(Obs(owns_car=1, is_subscriber=0))

        Note:
            After calling, inspect `weights`, `margins`, and `loss` properties
            for current state.
        """
        # Ensure we have capacity
        self._expand_capacity()

        # Extract feature values using helper
        feature_values = self._extract_feature_values(obs)

        # Store the observation
        self._features[self._n_obs] = feature_values
        self._weights[self._n_obs] = 1.0
        self._n_obs += 1

        # Get current learning rate (may be from schedule)
        current_lr = self._get_current_learning_rate()

        # perform n_sgd_steps updates
        final_gradient_norm = 0.0
        for step in range(self.n_sgd_steps):
            grad = self._compute_gradient()

            # Calculate gradient norm for convergence monitoring
            gradient_norm = float(np.linalg.norm(grad))
            if step == self.n_sgd_steps - 1:  # Store only final gradient norm
                final_gradient_norm = gradient_norm

            # Update only active weights
            self._weights[: self._n_obs] -= current_lr * grad
            # clip weights
            np.clip(
                self._weights[: self._n_obs],
                self.min_weight,
                self.max_weight,
                out=self._weights[: self._n_obs],
            )

            # Verbose output for debugging
            if step == 0:
                self._log_update("Obs", gradient_norm)

        # record state with final gradient norm
        self._record_state(gradient_norm=final_gradient_norm)

    def partial_fit_batch(self, observations: list[dict[str, Any] | Any]) -> None:
        """Process multiple observations in batch.

        Args:
            observations: List of observations, each in same format as
                for partial_fit method.

        Returns:
            None. Updates internal state for all observations.

        Examples:
            >>> observations = [
            ...     {'feature_a': 1, 'feature_b': 0},
            ...     {'feature_a': 0, 'feature_b': 1},
            ...     {'feature_a': 1, 'feature_b': 1},
            ... ]
            >>> raker.partial_fit_batch(observations)

        Note:
            Currently processes observations sequentially. Future versions
            may implement true batch processing for better performance.
        """
        for obs in observations:
            self.partial_fit(obs)

    # Backward compatibility aliases
    fit_one = partial_fit
