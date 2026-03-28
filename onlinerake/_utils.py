"""Internal utility functions for onlinerake package.

This module provides common helper functions used across the package to
reduce code duplication and ensure consistent behavior.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .online_raking_sgd import OnlineRakingSGD


def requires_observations[T](default_factory: Callable[[], T]) -> Callable:
    """Decorator that checks if raker has observations before executing.

    Many diagnostic and analysis functions require at least one observation
    to produce meaningful results. This decorator provides a consistent way
    to handle the zero-observation case.

    Args:
        default_factory: A callable that returns the default value when
            no observations are available.

    Returns:
        Decorator function.

    Examples:
        >>> @requires_observations(lambda: np.nan)
        ... def compute_metric(raker):
        ...     return raker.loss

        >>> @requires_observations(lambda: FeasibilityReport(...))
        ... def check_feasibility(raker, tolerance=0.05):
        ...     ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(raker: OnlineRakingSGD, *args: Any, **kwargs: Any) -> T:
            if raker._n_obs == 0:
                return default_factory()
            return func(raker, *args, **kwargs)

        return wrapper

    return decorator


def validate_positive(value: float, name: str) -> None:
    """Validate that a value is strictly positive.

    Args:
        value: The value to validate.
        name: The name of the parameter (for error messages).

    Raises:
        ValueError: If value is not positive.
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def validate_non_negative(value: float, name: str) -> None:
    """Validate that a value is non-negative.

    Args:
        value: The value to validate.
        name: The name of the parameter (for error messages).

    Raises:
        ValueError: If value is negative.
    """
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
