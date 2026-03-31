"""KL divergence and related distance measures for weight distributions.

This module provides functions for computing divergence between weight
distributions, useful for monitoring MWU convergence and comparing
streaming rakers to batch IPF solutions.

The key insight is that MWU performs mirror descent with KL divergence
as the regularizer. As learning rate decreases, MWU should converge
to the IPF solution (which minimizes KL divergence from uniform weights
subject to margin constraints).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def kl_divergence_weights(
    w_new: npt.NDArray[np.float64],
    w_old: npt.NDArray[np.float64],
    epsilon: float = 1e-10,
) -> float:
    """Compute KL divergence between normalized weight distributions.

    Computes D_KL(p_new || p_old) where p = w / sum(w) are the normalized
    weight distributions.

    This measures how much information is lost when using p_old to
    approximate p_new. In the context of MWU, this tracks how much
    the weight distribution changed between updates.

    Args:
        w_new: New weight vector (will be normalized).
        w_old: Old/reference weight vector (will be normalized).
        epsilon: Small constant for numerical stability. Weights below
            this threshold are clipped to prevent log(0).

    Returns:
        KL divergence D_KL(p_new || p_old). Always non-negative.
        Returns 0 when distributions are identical.

    Examples:
        >>> w1 = np.array([1.0, 1.0, 1.0])
        >>> w2 = np.array([1.0, 2.0, 1.0])
        >>> kl = kl_divergence_weights(w2, w1)
        >>> print(f"KL divergence: {kl:.4f}")

    Note:
        KL divergence is asymmetric: D_KL(p || q) != D_KL(q || p).
        The first argument is the "true" distribution, the second is
        the approximation.
    """
    if len(w_new) == 0 or len(w_old) == 0:
        return 0.0

    if len(w_new) != len(w_old):
        raise ValueError(
            f"Weight vectors must have same length: {len(w_new)} vs {len(w_old)}"
        )

    # Normalize to probability distributions
    sum_new = w_new.sum()
    sum_old = w_old.sum()

    if sum_new <= 0 or sum_old <= 0:
        return np.nan

    p_new = w_new / sum_new
    p_old = w_old / sum_old

    # Clip for numerical stability
    p_new_safe = np.clip(p_new, epsilon, None)
    p_old_safe = np.clip(p_old, epsilon, None)

    # KL divergence: sum(p_new * log(p_new / p_old))
    kl = float(np.sum(p_new_safe * np.log(p_new_safe / p_old_safe)))

    # KL should be non-negative; small negative values are numerical artifacts
    return max(0.0, kl)


def total_variation_weights(
    w1: npt.NDArray[np.float64],
    w2: npt.NDArray[np.float64],
) -> float:
    """Compute total variation distance between normalized weight distributions.

    Total variation distance is defined as:
        TV(p, q) = 0.5 * sum(|p_i - q_i|)

    This is a symmetric distance bounded between 0 and 1.

    Args:
        w1: First weight vector (will be normalized).
        w2: Second weight vector (will be normalized).

    Returns:
        Total variation distance, in range [0, 1].
        Returns 0 when distributions are identical.

    Examples:
        >>> w1 = np.array([1.0, 1.0, 1.0])
        >>> w2 = np.array([1.0, 2.0, 1.0])
        >>> tv = total_variation_weights(w1, w2)
        >>> print(f"Total variation: {tv:.4f}")
    """
    if len(w1) == 0 or len(w2) == 0:
        return 0.0

    if len(w1) != len(w2):
        raise ValueError(
            f"Weight vectors must have same length: {len(w1)} vs {len(w2)}"
        )

    sum1 = w1.sum()
    sum2 = w2.sum()

    if sum1 <= 0 or sum2 <= 0:
        return np.nan

    p1 = w1 / sum1
    p2 = w2 / sum2

    return float(0.5 * np.sum(np.abs(p1 - p2)))


def symmetric_kl_divergence(
    w1: npt.NDArray[np.float64],
    w2: npt.NDArray[np.float64],
    epsilon: float = 1e-10,
) -> float:
    """Compute symmetric KL divergence (Jensen-Shannon-like).

    Returns (D_KL(p1 || p2) + D_KL(p2 || p1)) / 2.

    This is a symmetric measure unlike standard KL divergence.

    Args:
        w1: First weight vector.
        w2: Second weight vector.
        epsilon: Small constant for numerical stability.

    Returns:
        Symmetric KL divergence.
    """
    kl_12 = kl_divergence_weights(w1, w2, epsilon)
    kl_21 = kl_divergence_weights(w2, w1, epsilon)

    if np.isnan(kl_12) or np.isnan(kl_21):
        return np.nan

    return (kl_12 + kl_21) / 2
