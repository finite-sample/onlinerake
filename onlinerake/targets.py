"""Target population proportions for online raking algorithms.

This module defines the :class:`Targets` class which captures target
population margins for features under study. It provides a flexible,
typed container for passing target proportions into the online raking
algorithms. Each value represents the proportion of the population
where that binary feature equals 1.

The class supports arbitrary binary features. Features can represent any
binary characteristic you need to calibrate: product preferences, behaviors,
medical conditions, demographics, or any other binary indicators.
"""

from __future__ import annotations

from typing import Any


class Targets:
    """Target population proportions for binary features.

    A flexible container for specifying target proportions for any set of
    binary features. Each feature should have a proportion between 0 and 1
    representing the fraction of the population where that feature is True/1.

    Args:
        **kwargs: Named feature proportions. Each key is a feature name and
            each value is the target proportion (between 0 and 1) for that
            feature being 1/True.

    Attributes:
        _targets (dict[str, float]): Internal storage of target proportions.
        _feature_names (list[str]): Sorted list of feature names for consistent ordering.

    Examples:
        >>> # Product preferences
        >>> targets = Targets(owns_car=0.4, is_subscriber=0.2, likes_coffee=0.7)
        >>> print(targets.feature_names)
        ['is_subscriber', 'likes_coffee', 'owns_car']

        >>> # Medical indicators
        >>> targets = Targets(has_diabetes=0.08, exercises=0.35, smoker=0.15)

        >>> # Access target values
        >>> print(targets['owns_car'])
        0.4

        >>> # Check if feature exists
        >>> print('owns_car' in targets)
        True

    Raises:
        ValueError: If any target proportion is not between 0 and 1.

    Note:
        Feature names are stored in sorted order for consistent behavior
        across different Python versions and hash randomization settings.
    """

    def __init__(self, **kwargs: float) -> None:
        """Initialize targets with named proportions.

        Args:
            **kwargs: Feature name to target proportion mappings.
                Each value must be between 0 and 1 inclusive.
                At least one feature must be specified.

        Raises:
            ValueError: If any proportion is outside [0, 1] or if no features provided.
        """
        if not kwargs:
            raise ValueError(
                "At least one feature must be specified. "
                "Example: Targets(feature1=0.3, feature2=0.7)"
            )

        # Validate proportions are between 0 and 1
        for name, value in kwargs.items():
            if not 0 <= value <= 1:
                raise ValueError(
                    f"Target proportion for '{name}' must be between 0 and 1, got {value}"
                )

        self._targets: dict[str, float] = kwargs
        # Store feature names in consistent order
        self._feature_names: list[str] = sorted(kwargs.keys())

    @property
    def feature_names(self) -> list[str]:
        """Get ordered list of feature names.

        Returns:
            list[str]: Sorted list of feature names.

        Examples:
            >>> targets = Targets(b=0.5, a=0.3, c=0.7)
            >>> targets.feature_names
            ['a', 'b', 'c']
        """
        return self._feature_names.copy()

    @property
    def n_features(self) -> int:
        """Get number of features.

        Returns:
            int: Number of features defined in these targets.

        Examples:
            >>> targets = Targets(a=0.5, b=0.3, c=0.7)
            >>> targets.n_features
            3
        """
        return len(self._feature_names)

    def as_dict(self) -> dict[str, float]:
        """Convert targets to a dictionary.

        Returns:
            dict[str, float]: Dictionary mapping feature names to target proportions.

        Examples:
            >>> targets = Targets(owns_car=0.4, is_subscriber=0.2)
            >>> targets.as_dict()
            {'owns_car': 0.4, 'is_subscriber': 0.2}
        """
        return self._targets.copy()

    def __getitem__(self, key: str) -> float:
        """Get target proportion for a specific feature.

        Args:
            key: Feature name to look up.

        Returns:
            float: Target proportion for the specified feature.

        Raises:
            KeyError: If feature name is not defined in targets.

        Examples:
            >>> targets = Targets(owns_car=0.4, is_subscriber=0.2)
            >>> targets['owns_car']
            0.4
        """
        return self._targets[key]

    def __contains__(self, key: str) -> bool:
        """Check if a feature is defined in targets.

        Args:
            key: Feature name to check.

        Returns:
            bool: True if feature is defined, False otherwise.

        Examples:
            >>> targets = Targets(owns_car=0.4, is_subscriber=0.2)
            >>> 'owns_car' in targets
            True
            >>> 'unknown_feature' in targets
            False
        """
        return key in self._targets

    def __repr__(self) -> str:
        """Return string representation of targets.

        Returns:
            str: String representation showing all target proportions.

        Examples:
            >>> targets = Targets(a=0.5, b=0.3)
            >>> repr(targets)
            "Targets(a=0.50, b=0.30)"
        """
        items = [f"{k}={v:.2f}" for k, v in self._targets.items()]
        return f"Targets({', '.join(items)})"

    def __eq__(self, other: Any) -> bool:
        """Check equality with another Targets object.

        Args:
            other: Object to compare with.

        Returns:
            bool: True if other is a Targets object with same proportions.

        Examples:
            >>> t1 = Targets(a=0.5, b=0.3)
            >>> t2 = Targets(a=0.5, b=0.3)
            >>> t1 == t2
            True
        """
        if not isinstance(other, Targets):
            return False
        return self._targets == other._targets
