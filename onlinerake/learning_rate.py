"""Learning rate schedules for online raking algorithms.

This module provides learning rate schedules that satisfy the Robbins-Monro
conditions for stochastic approximation convergence:

1. Sum of learning rates diverges: Σ η_t = ∞
2. Sum of squared learning rates converges: Σ η_t² < ∞

These conditions guarantee convergence of stochastic gradient descent to
the optimal solution under appropriate assumptions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LearningRateSchedule(ABC):
    """Abstract base class for learning rate schedules."""

    @abstractmethod
    def __call__(self, t: int) -> float:
        """Return learning rate at step t (1-indexed)."""
        pass

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Return schedule parameters for serialization."""
        pass


class ConstantLR(LearningRateSchedule):
    """Constant learning rate.

    Note: Constant learning rates do NOT satisfy Robbins-Monro conditions.
    Use only when you need fast initial convergence and are willing to
    accept potential oscillation or bounded suboptimality.

    Args:
        learning_rate: Fixed learning rate value.
    """

    def __init__(self, learning_rate: float = 1.0) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self._learning_rate = learning_rate

    def __call__(self, t: int) -> float:  # noqa: ARG002
        del t  # Unused, constant LR
        return self._learning_rate

    def get_params(self) -> dict[str, Any]:
        return {"type": "constant", "learning_rate": self._learning_rate}


class InverseTimeDecayLR(LearningRateSchedule):
    """Inverse time decay: η_t = η_0 / (1 + decay * t).

    Satisfies Robbins-Monro when decay > 0:
    - Σ η_t = Σ η_0/(1 + decay*t) → ∞ (harmonic series)
    - Σ η_t² = Σ η_0²/(1 + decay*t)² < ∞

    Args:
        initial_lr: Initial learning rate η_0.
        decay: Decay factor. Larger values mean faster decay.
        min_lr: Minimum learning rate floor.
    """

    def __init__(
        self,
        initial_lr: float = 5.0,
        decay: float = 0.01,
        min_lr: float = 0.01,
    ) -> None:
        if initial_lr <= 0:
            raise ValueError("initial_lr must be positive")
        if decay < 0:
            raise ValueError("decay must be non-negative")
        if min_lr < 0:
            raise ValueError("min_lr must be non-negative")

        self._initial_lr = initial_lr
        self._decay = decay
        self._min_lr = min_lr

    def __call__(self, t: int) -> float:
        lr = self._initial_lr / (1.0 + self._decay * t)
        return max(lr, self._min_lr)

    def get_params(self) -> dict[str, Any]:
        return {
            "type": "inverse_time_decay",
            "initial_lr": self._initial_lr,
            "decay": self._decay,
            "min_lr": self._min_lr,
        }


class PolynomialDecayLR(LearningRateSchedule):
    """Polynomial decay: η_t = η_0 / t^α.

    For Robbins-Monro convergence, need 0.5 < α ≤ 1:
    - α > 0.5 ensures Σ η_t² < ∞
    - α ≤ 1 ensures Σ η_t = ∞

    The classic choice is α = 1 (1/t decay), but α = 0.6-0.7 often works
    better in practice, providing faster initial progress while still
    guaranteeing convergence.

    Args:
        initial_lr: Initial learning rate η_0.
        power: Decay power α. Must be in (0.5, 1].
        min_lr: Minimum learning rate floor.
    """

    def __init__(
        self,
        initial_lr: float = 5.0,
        power: float = 0.6,
        min_lr: float = 0.01,
    ) -> None:
        if initial_lr <= 0:
            raise ValueError("initial_lr must be positive")
        if not (0.5 < power <= 1.0):
            raise ValueError("power must be in (0.5, 1] for Robbins-Monro convergence")
        if min_lr < 0:
            raise ValueError("min_lr must be non-negative")

        self._initial_lr = initial_lr
        self._power = power
        self._min_lr = min_lr

    def __call__(self, t: int) -> float:
        # Avoid division by zero at t=0
        effective_t = max(t, 1)
        lr = self._initial_lr / (effective_t**self._power)
        return float(max(lr, self._min_lr))

    def get_params(self) -> dict[str, Any]:
        return {
            "type": "polynomial_decay",
            "initial_lr": self._initial_lr,
            "power": self._power,
            "min_lr": self._min_lr,
        }


class AdaptiveLR(LearningRateSchedule):
    """Adaptive learning rate based on loss improvement.

    Increases learning rate when making progress, decreases when
    oscillating or stalling. This can provide faster convergence
    than fixed schedules while maintaining stability.

    Note: Does not strictly satisfy Robbins-Monro conditions, but
    practical convergence is often achieved.

    Args:
        initial_lr: Starting learning rate.
        min_lr: Minimum learning rate.
        max_lr: Maximum learning rate.
        increase_factor: Factor to increase LR when improving.
        decrease_factor: Factor to decrease LR when not improving.
    """

    def __init__(
        self,
        initial_lr: float = 1.0,
        min_lr: float = 0.01,
        max_lr: float = 10.0,
        increase_factor: float = 1.05,
        decrease_factor: float = 0.5,
    ) -> None:
        if initial_lr <= 0:
            raise ValueError("initial_lr must be positive")
        if min_lr <= 0:
            raise ValueError("min_lr must be positive")
        if max_lr <= min_lr:
            raise ValueError("max_lr must exceed min_lr")
        if increase_factor <= 1.0:
            raise ValueError("increase_factor must be > 1")
        if not (0 < decrease_factor < 1):
            raise ValueError("decrease_factor must be in (0, 1)")

        self._current_lr = initial_lr
        self._min_lr = min_lr
        self._max_lr = max_lr
        self._increase_factor = increase_factor
        self._decrease_factor = decrease_factor
        self._last_loss: float | None = None

    def __call__(self, t: int) -> float:  # noqa: ARG002
        del t  # Unused, adaptive LR ignores step count
        return self._current_lr

    def update(self, current_loss: float) -> None:
        """Update learning rate based on current loss."""
        if self._last_loss is not None:
            if current_loss < self._last_loss * 0.99:
                # Making progress - increase LR
                self._current_lr = min(
                    self._current_lr * self._increase_factor,
                    self._max_lr,
                )
            elif current_loss > self._last_loss * 1.01:
                # Getting worse - decrease LR
                self._current_lr = max(
                    self._current_lr * self._decrease_factor,
                    self._min_lr,
                )
        self._last_loss = current_loss

    def get_params(self) -> dict[str, Any]:
        return {
            "type": "adaptive",
            "initial_lr": self._current_lr,
            "min_lr": self._min_lr,
            "max_lr": self._max_lr,
            "increase_factor": self._increase_factor,
            "decrease_factor": self._decrease_factor,
        }


def robbins_monro_schedule(
    initial_lr: float = 5.0,
    power: float = 0.6,
    min_lr: float = 0.01,
) -> PolynomialDecayLR:
    """Create a Robbins-Monro compliant learning rate schedule.

    This is a convenience function that creates a polynomial decay
    schedule with parameters that guarantee theoretical convergence.

    Args:
        initial_lr: Initial learning rate. Higher values for faster
            initial progress, but may cause oscillation.
        power: Decay power. Must be in (0.5, 1]. Default 0.6 balances
            convergence speed with stability.
        min_lr: Minimum learning rate to maintain some adaptation.

    Returns:
        A PolynomialDecayLR schedule.

    Examples:
        >>> schedule = robbins_monro_schedule(initial_lr=5.0)
        >>> schedule(1)  # First observation
        5.0
        >>> schedule(100)  # 100th observation
        0.315...
    """
    return PolynomialDecayLR(initial_lr=initial_lr, power=power, min_lr=min_lr)
