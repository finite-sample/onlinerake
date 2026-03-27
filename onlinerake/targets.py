"""Target population margins for online raking algorithms.

This module defines the :class:`Targets` class which captures target
population margins for features under study. It provides a flexible,
typed container for passing target proportions or means into the online
raking algorithms.

The class supports:
- **Binary features**: Target proportion in [0, 1] representing the fraction
  of the population where that feature equals 1.
- **Continuous features**: Target mean (any real number) specified using
  tuple syntax ``(value, "mean")``.

Features can represent any characteristic you need to calibrate: product
preferences, behaviors, medical conditions, demographics, continuous
measurements like age or income, or any other indicators.
"""

from __future__ import annotations

from typing import Any


class Targets:
    """Target population margins for binary and continuous features.

    A flexible container for specifying target proportions (for binary features)
    or target means (for continuous features).

    Args:
        **kwargs: Named feature targets. Each key is a feature name and each
            value specifies the target:
            - For binary features: a float in [0, 1] representing the target
              proportion of the population where that feature is 1/True.
            - For continuous features: a tuple ``(value, "mean")`` where value
              is the target mean (any real number).

    Attributes:
        _targets (dict[str, float]): Internal storage of target values.
        _feature_types (dict[str, str]): Maps feature names to "binary" or "continuous".
        _feature_names (list[str]): Sorted list of feature names for consistent ordering.

    Examples:
        >>> # Binary features only (backward compatible)
        >>> targets = Targets(owns_car=0.4, is_subscriber=0.2, likes_coffee=0.7)
        >>> print(targets.feature_names)
        ['is_subscriber', 'likes_coffee', 'owns_car']

        >>> # Mixed binary and continuous features
        >>> targets = Targets(
        ...     gender=0.5,                 # binary: 50% female
        ...     college=0.35,               # binary: 35% college educated
        ...     age=(42.0, "mean"),         # continuous: mean age 42
        ...     income=(65000, "mean"),     # continuous: mean income $65k
        ... )
        >>> print(targets.is_binary("gender"))
        True
        >>> print(targets.is_continuous("age"))
        True
        >>> print(targets["age"])
        42.0

        >>> # Access target values
        >>> print(targets['owns_car'])
        0.4

        >>> # Check if feature exists
        >>> print('owns_car' in targets)
        True

    Raises:
        ValueError: If any binary target proportion is not between 0 and 1,
            or if the tuple syntax is malformed.

    Note:
        Feature names are stored in sorted order for consistent behavior
        across different Python versions and hash randomization settings.
    """

    def __init__(self, **kwargs: float | tuple[float, str]) -> None:
        if not kwargs:
            raise ValueError(
                "At least one feature must be specified. "
                "Example: Targets(feature1=0.3, feature2=0.7) or "
                "Targets(age=(35.0, 'mean'))"
            )

        self._targets: dict[str, float] = {}
        self._feature_types: dict[str, str] = {}

        for name, value in kwargs.items():
            if isinstance(value, tuple):
                # Continuous feature: (value, "mean")
                if len(value) != 2:
                    raise ValueError(
                        f"Continuous target for '{name}' must be (value, 'mean'), "
                        f"got tuple of length {len(value)}"
                    )
                target_value, type_spec = value
                if not isinstance(target_value, (int, float)):
                    raise ValueError(
                        f"Target value for '{name}' must be numeric, "
                        f"got {type(target_value).__name__}"
                    )
                if type_spec != "mean":
                    raise ValueError(
                        f"Continuous target type for '{name}' must be 'mean', "
                        f"got '{type_spec}'"
                    )
                self._targets[name] = float(target_value)
                self._feature_types[name] = "continuous"
            else:
                # Binary feature: proportion in [0, 1]
                if not isinstance(value, (int, float)):
                    raise ValueError(
                        f"Target for '{name}' must be numeric or (value, 'mean') tuple, "
                        f"got {type(value).__name__}"
                    )
                if not 0 <= value <= 1:
                    raise ValueError(
                        f"Binary target proportion for '{name}' must be between 0 and 1, "
                        f"got {value}. For continuous features, use (value, 'mean') syntax."
                    )
                self._targets[name] = float(value)
                self._feature_types[name] = "binary"

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

    @property
    def binary_features(self) -> list[str]:
        """Get list of binary feature names.

        Returns:
            list[str]: Sorted list of binary feature names.

        Examples:
            >>> targets = Targets(gender=0.5, age=(35.0, "mean"))
            >>> targets.binary_features
            ['gender']
        """
        return sorted(
            name for name, ftype in self._feature_types.items() if ftype == "binary"
        )

    @property
    def continuous_features(self) -> list[str]:
        """Get list of continuous feature names.

        Returns:
            list[str]: Sorted list of continuous feature names.

        Examples:
            >>> targets = Targets(gender=0.5, age=(35.0, "mean"))
            >>> targets.continuous_features
            ['age']
        """
        return sorted(
            name for name, ftype in self._feature_types.items() if ftype == "continuous"
        )

    @property
    def has_continuous_features(self) -> bool:
        """Check if any continuous features are defined.

        Returns:
            bool: True if at least one continuous feature is defined.
        """
        return "continuous" in self._feature_types.values()

    def is_binary(self, feature: str) -> bool:
        """Check if a feature is binary.

        Args:
            feature: Feature name to check.

        Returns:
            bool: True if feature is binary, False otherwise.

        Raises:
            KeyError: If feature name is not defined in targets.

        Examples:
            >>> targets = Targets(gender=0.5, age=(35.0, "mean"))
            >>> targets.is_binary("gender")
            True
            >>> targets.is_binary("age")
            False
        """
        if feature not in self._feature_types:
            raise KeyError(f"Feature '{feature}' not defined in targets")
        return self._feature_types[feature] == "binary"

    def is_continuous(self, feature: str) -> bool:
        """Check if a feature is continuous.

        Args:
            feature: Feature name to check.

        Returns:
            bool: True if feature is continuous, False otherwise.

        Raises:
            KeyError: If feature name is not defined in targets.

        Examples:
            >>> targets = Targets(gender=0.5, age=(35.0, "mean"))
            >>> targets.is_continuous("age")
            True
            >>> targets.is_continuous("gender")
            False
        """
        if feature not in self._feature_types:
            raise KeyError(f"Feature '{feature}' not defined in targets")
        return self._feature_types[feature] == "continuous"

    def feature_type(self, feature: str) -> str:
        """Get the type of a feature.

        Args:
            feature: Feature name to look up.

        Returns:
            str: Either "binary" or "continuous".

        Raises:
            KeyError: If feature name is not defined in targets.

        Examples:
            >>> targets = Targets(gender=0.5, age=(35.0, "mean"))
            >>> targets.feature_type("gender")
            'binary'
            >>> targets.feature_type("age")
            'continuous'
        """
        if feature not in self._feature_types:
            raise KeyError(f"Feature '{feature}' not defined in targets")
        return self._feature_types[feature]

    def as_dict(self) -> dict[str, float]:
        """Convert targets to a dictionary of values.

        Returns:
            dict[str, float]: Dictionary mapping feature names to target values
                (proportions for binary, means for continuous).

        Examples:
            >>> targets = Targets(owns_car=0.4, is_subscriber=0.2)
            >>> targets.as_dict()
            {'owns_car': 0.4, 'is_subscriber': 0.2}
        """
        return self._targets.copy()

    def __getitem__(self, key: str) -> float:
        """Get target value for a specific feature.

        Args:
            key: Feature name to look up.

        Returns:
            float: Target value (proportion for binary, mean for continuous).

        Raises:
            KeyError: If feature name is not defined in targets.

        Examples:
            >>> targets = Targets(owns_car=0.4, age=(35.0, "mean"))
            >>> targets['owns_car']
            0.4
            >>> targets['age']
            35.0
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
            str: String representation showing all target values and types.

        Examples:
            >>> targets = Targets(a=0.5, b=0.3)
            >>> repr(targets)
            "Targets(a=0.50, b=0.30)"
            >>> targets = Targets(gender=0.5, age=(35.0, "mean"))
            >>> repr(targets)
            "Targets(age=35.00 [mean], gender=0.50)"
        """
        items = []
        for k in self._feature_names:
            v = self._targets[k]
            if self._feature_types[k] == "continuous":
                items.append(f"{k}={v:.2f} [mean]")
            else:
                items.append(f"{k}={v:.2f}")
        return f"Targets({', '.join(items)})"

    def __eq__(self, other: Any) -> bool:
        """Check equality with another Targets object.

        Args:
            other: Object to compare with.

        Returns:
            bool: True if other is a Targets object with same targets and types.

        Examples:
            >>> t1 = Targets(a=0.5, b=0.3)
            >>> t2 = Targets(a=0.5, b=0.3)
            >>> t1 == t2
            True
        """
        if not isinstance(other, Targets):
            return False
        return (
            self._targets == other._targets
            and self._feature_types == other._feature_types
        )
