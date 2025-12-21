"""Streaming survey weight calibration via stochastic gradient descent and multiplicative weights update.

This package provides two high-performance streaming weight calibration algorithms
for adjusting observation weights to match known population margins in real time:

- **SGD raking** (:class:`OnlineRakingSGD`): Uses stochastic gradient descent with
  additive weight updates
- **MWU raking** (:class:`OnlineRakingMWU`): Uses multiplicative weights update with
  exponential weight updates

Both algorithms follow the scikit-learn ``partial_fit`` pattern for streaming data.
Each raker accepts observations with binary feature indicators and updates its
internal weight vector to minimize squared-error loss between weighted margins
and target proportions.

The algorithms support arbitrary binary features - not limited to demographics.
Features can represent product preferences, behaviors, medical conditions,
or any binary characteristics you need to calibrate.

Examples:
    >>> from onlinerake import OnlineRakingSGD, OnlineRakingMWU, Targets

    >>> # Product preference calibration
    >>> targets = Targets(owns_car=0.4, is_subscriber=0.2, likes_coffee=0.7)
    >>> sgd_raker = OnlineRakingSGD(targets, learning_rate=5.0)
    >>>
    >>> # Process observations one at a time
    >>> for obs in stream:
    ...     sgd_raker.partial_fit(obs)
    ...     if sgd_raker.converged:
    ...         break
    >>>
    >>> # Inspect current state
    >>> print(f"Loss: {sgd_raker.loss:.6f}")
    >>> print(f"Margins: {sgd_raker.margins}")
    >>> print(f"ESS: {sgd_raker.effective_sample_size:.1f}")

    >>> # Medical survey calibration with MWU
    >>> medical_targets = Targets(has_diabetes=0.08, exercises=0.35, smoker=0.15)
    >>> mwu_raker = OnlineRakingMWU(medical_targets, learning_rate=1.0)
    >>> mwu_raker.partial_fit({'has_diabetes': 0, 'exercises': 1, 'smoker': 0})

Performance:
    - **High throughput**: 3000-6000 observations per second
    - **Memory efficient**: O(n) memory with capacity doubling
    - **Scalable**: Performance independent of number of observations
    - **Flexible**: Works with any number of binary features

Note:
    This is version 1.0.0 with breaking changes. The old demographic-specific
    interface has been removed in favor of a general feature interface.
    Users must explicitly specify their features and target proportions.
"""

from .online_raking_mwu import OnlineRakingMWU
from .online_raking_sgd import OnlineRakingSGD
from .targets import Targets

__all__ = [
    "Targets",
    "OnlineRakingSGD",
    "OnlineRakingMWU",
]

__version__ = "1.0.0"
