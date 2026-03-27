"""Batch Iterative Proportional Fitting (IPF) for survey raking.

This module implements classical batch IPF (also known as raking) as a baseline
for comparison with the online raking algorithms. Unlike streaming methods,
batch IPF requires all data to be available upfront and iterates over the
entire dataset multiple times until convergence.

The implementation supports arbitrary binary features to match the online
raking algorithms' interface.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from .targets import Targets


class BatchIPF:
    """Classical Iterative Proportional Fitting for survey raking.

    IPF (also called raking ratio estimation) iteratively adjusts weights
    to match marginal population totals. This is the gold standard batch
    method that online raking algorithms should converge to.

    Args:
        targets: Target population proportions for each feature.
        max_iterations: Maximum number of full passes over features.
        tolerance: Convergence tolerance for margin differences.
        min_weight: Lower bound for weights to prevent collapse.
        max_weight: Upper bound for weights to prevent explosion.

    Examples:
        >>> targets = Targets(age=0.4, gender=0.5, education=0.3)
        >>> ipf = BatchIPF(targets)
        >>> data = [{'age': 1, 'gender': 0, 'education': 1}, ...]
        >>> ipf.fit(data)
        >>> print(ipf.weights)
        >>> print(ipf.margins)
    """

    def __init__(
        self,
        targets: Targets,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        min_weight: float = 1e-6,
        max_weight: float = 1e6,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if min_weight <= 0:
            raise ValueError("min_weight must be strictly positive")
        if max_weight <= min_weight:
            raise ValueError("max_weight must exceed min_weight")

        self.targets = targets
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.min_weight = min_weight
        self.max_weight = max_weight

        self._feature_names = targets.feature_names
        self._n_features = targets.n_features
        self._target_array = np.array(
            [targets[name] for name in self._feature_names], dtype=np.float64
        )

        self._weights: np.ndarray = np.empty(0, dtype=np.float64)
        # Use float64 for consistency with online raking algorithms
        self._features: np.ndarray = np.empty((0, self._n_features), dtype=np.float64)
        self._n_obs: int = 0
        self._converged: bool = False
        self._n_iterations: int = 0
        self._loss_history: list[float] = []

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """Get copy of current weight vector."""
        return self._weights.copy()

    @property
    def margins(self) -> dict[str, float]:
        """Get current weighted margins."""
        if self._n_obs == 0:
            return dict.fromkeys(self._feature_names, np.nan)

        total_w = self._weights.sum()
        weighted_sums = self._weights @ self._features
        return {
            name: float(weighted_sums[i] / total_w)
            for i, name in enumerate(self._feature_names)
        }

    @property
    def raw_margins(self) -> dict[str, float]:
        """Get unweighted (raw) margins."""
        if self._n_obs == 0:
            return dict.fromkeys(self._feature_names, np.nan)

        feature_means = self._features.mean(axis=0)
        return {
            name: float(feature_means[i]) for i, name in enumerate(self._feature_names)
        }

    @property
    def loss(self) -> float:
        """Get current squared-error loss."""
        if self._n_obs == 0:
            return np.nan

        margins = self.margins
        loss = sum(
            (margins[name] - self.targets[name]) ** 2 for name in self._feature_names
        )
        return float(loss)

    @property
    def effective_sample_size(self) -> float:
        """Return the effective sample size (ESS)."""
        if self._n_obs == 0:
            return 0.0
        sum_w = self._weights.sum()
        sum_w2 = (self._weights**2).sum()
        return float((sum_w**2) / sum_w2) if sum_w2 > 0 else 0.0

    @property
    def converged(self) -> bool:
        """Return True if IPF has converged."""
        return self._converged

    @property
    def n_iterations(self) -> int:
        """Return number of iterations performed."""
        return self._n_iterations

    @property
    def loss_history(self) -> list[float]:
        """Return loss at each iteration."""
        return self._loss_history.copy()

    def _extract_features(self, observations: list[dict[str, Any] | Any]) -> np.ndarray:
        """Extract feature matrix from observations."""
        n = len(observations)
        features = np.zeros((n, self._n_features), dtype=np.float64)

        for i, obs in enumerate(observations):
            for j, name in enumerate(self._feature_names):
                if isinstance(obs, dict):
                    val = obs.get(name, 0)
                else:
                    val = getattr(obs, name, 0)

                # Handle binary vs continuous features
                if self.targets.is_binary(name):
                    features[i, j] = 1.0 if val else 0.0
                else:
                    features[i, j] = float(val)

        return features

    def fit(self, observations: list[dict[str, Any] | Any]) -> BatchIPF:
        """Fit IPF weights to the provided observations.

        Args:
            observations: List of observations, each containing feature indicators.

        Returns:
            self: The fitted IPF object.

        Raises:
            ValueError: If targets contain continuous features. BatchIPF only
                supports binary features. Use OnlineRakingSGD or OnlineRakingMWU
                for continuous features.
        """
        # Check for continuous features
        if self.targets.has_continuous_features:
            continuous = self.targets.continuous_features
            raise ValueError(
                f"BatchIPF only supports binary features. "
                f"Continuous features found: {continuous}. "
                f"Use OnlineRakingSGD or OnlineRakingMWU for continuous features."
            )

        self._features = self._extract_features(observations)
        self._n_obs = len(observations)
        self._weights = np.ones(self._n_obs, dtype=np.float64)
        self._loss_history = []
        self._converged = False
        self._n_iterations = 0

        for iteration in range(self.max_iterations):
            self._n_iterations = iteration + 1

            # Track loss before this iteration
            current_loss = self.loss
            self._loss_history.append(current_loss)

            # IPF: adjust weights for each feature dimension
            for j in range(self._n_features):
                feature_col = self._features[:, j]
                target = self._target_array[j]

                # Compute current weighted margin for this feature
                total_w = self._weights.sum()
                weighted_sum = (self._weights * feature_col).sum()
                current_margin = weighted_sum / total_w

                # Skip if margin is at extreme (to avoid division issues)
                if current_margin < 1e-10 or current_margin > 1 - 1e-10:
                    continue

                # Compute adjustment factor
                # For observations where feature=1: multiply by target/current_margin
                # For observations where feature=0: multiply by (1-target)/(1-current_margin)
                adjustment_1 = target / current_margin if current_margin > 0 else 1.0
                adjustment_0 = (
                    (1 - target) / (1 - current_margin) if current_margin < 1 else 1.0
                )

                # Apply adjustments
                self._weights = np.where(
                    feature_col == 1,
                    self._weights * adjustment_1,
                    self._weights * adjustment_0,
                )

                # Clip weights
                np.clip(
                    self._weights, self.min_weight, self.max_weight, out=self._weights
                )

            # Check convergence
            new_loss = self.loss
            if abs(new_loss - current_loss) < self.tolerance:
                self._converged = True
                self._loss_history.append(new_loss)
                break

        return self

    def fit_incremental(
        self,
        new_observations: list[dict[str, Any] | Any],
    ) -> BatchIPF:
        """Incrementally add observations and re-run IPF.

        This simulates what periodic batch IPF would look like in a
        streaming setting: accumulate data, then re-run full IPF.

        Args:
            new_observations: New observations to add.

        Returns:
            self: The refitted IPF object.

        Raises:
            ValueError: If targets contain continuous features. BatchIPF only
                supports binary features. Use OnlineRakingSGD or OnlineRakingMWU
                for continuous features.
        """
        # Check for continuous features
        if self.targets.has_continuous_features:
            continuous = self.targets.continuous_features
            raise ValueError(
                f"BatchIPF only supports binary features. "
                f"Continuous features found: {continuous}. "
                f"Use OnlineRakingSGD or OnlineRakingMWU for continuous features."
            )

        # Extract new features
        new_features = self._extract_features(new_observations)

        # Concatenate with existing data
        if self._n_obs > 0:
            self._features = np.vstack([self._features, new_features])
        else:
            self._features = new_features

        self._n_obs = len(self._features)

        # Re-initialize weights and run IPF
        self._weights = np.ones(self._n_obs, dtype=np.float64)
        self._loss_history = []
        self._converged = False
        self._n_iterations = 0

        # Re-run the same fitting procedure
        for iteration in range(self.max_iterations):
            self._n_iterations = iteration + 1
            current_loss = self.loss
            self._loss_history.append(current_loss)

            for j in range(self._n_features):
                feature_col = self._features[:, j]
                target = self._target_array[j]

                total_w = self._weights.sum()
                weighted_sum = (self._weights * feature_col).sum()
                current_margin = weighted_sum / total_w

                if current_margin < 1e-10 or current_margin > 1 - 1e-10:
                    continue

                adjustment_1 = target / current_margin if current_margin > 0 else 1.0
                adjustment_0 = (
                    (1 - target) / (1 - current_margin) if current_margin < 1 else 1.0
                )

                self._weights = np.where(
                    feature_col == 1,
                    self._weights * adjustment_1,
                    self._weights * adjustment_0,
                )
                np.clip(
                    self._weights, self.min_weight, self.max_weight, out=self._weights
                )

            new_loss = self.loss
            if abs(new_loss - current_loss) < self.tolerance:
                self._converged = True
                self._loss_history.append(new_loss)
                break

        return self
