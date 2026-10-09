"""Internal utility functions for onlinerake package.

This module provides common helper functions used across the package to
reduce code duplication and ensure consistent behavior.
"""

from __future__ import annotations

from functools import wraps
from math import isfinite
from numbers import Integral
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

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
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive")


def validate_non_negative(value: float, name: str) -> None:
    """Validate that a value is non-negative.

    Args:
        value: The value to validate.
        name: The name of the parameter (for error messages).

    Raises:
        ValueError: If value is negative.
    """
    if not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be non-negative")


def validate_count(value: int, name: str) -> None:
    """Require a positive integer count.

    Args:
        value: Count to validate.
        name: Parameter name for errors.

    Raises:
        ValueError: If the count is not a positive integer.
    """
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def feature_value(value: Any, name: str, *, binary: bool) -> float:
    """Validate one observation value before it enters estimator state.

    Args:
        value: Observed feature value.
        name: Feature name for errors.
        binary: Whether only zero and one are allowed.

    Returns:
        Finite numeric feature value.

    Raises:
        ValueError: If the value is nonfinite, nonnumeric, or invalid binary data.
    """
    try:
        numeric = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite numeric data") from exc
    if not isfinite(numeric) or (binary and numeric not in (0.0, 1.0)):
        raise ValueError(
            f"{name} must be finite" + (" and binary (0/1)" if binary else "")
        )
    return numeric
