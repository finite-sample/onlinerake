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
    >>> from onlinerake import OnlineRakingSGD, Targets
    >>> targets = Targets(owns_car=0.4, is_subscriber=0.2, likes_coffee=0.7)
    >>> raker = OnlineRakingSGD(targets, learning_rate=5.0)

    Feed it observations one at a time, in whatever order they arrive:

    >>> stream = [
    ...     {
    ...         "owns_car": i % 2,
    ...         "is_subscriber": (i % 5) == 0,
    ...         "likes_coffee": (i % 3) != 0,
    ...     }
    ...     for i in range(60)
    ... ]
    >>> for obs in stream:
    ...     raker.partial_fit(obs)

    Every quantity is readable at any point in the stream, not only at the end:

    >>> {name: round(value, 2) for name, value in raker.margins.items()}
    {'is_subscriber': 0.19, 'likes_coffee': 0.7, 'owns_car': 0.42}
    >>> raker.loss < 1e-3
    True
    >>> round(raker.effective_sample_size, 1)
    45.1

Performance:
    "Streaming" here describes how data *arrives*, not the cost of taking it.
    Each observation re-solves the calibration over everything seen so far --
    ``partial_fit`` rewrites all ``n`` accumulated weights and the gradient is
    itself ``O(n)`` -- so per-observation cost is ``Theta(n * n_sgd_steps)``
    and a full pass is quadratic in the stream length.

    Measured on one machine, three binary features, default settings:

    ======  ===========  ============
    n       time/obs     obs/sec
    ======  ===========  ============
    2,500       104 us         9,645
    10,000      222 us         4,508
    40,000      645 us         1,551
    ======  ===========  ============

    Absolute rates are hardware-specific; the trend is not. Fitted exponent on
    total time over that range is 1.66. Plan for tens of thousands of
    observations per stream, not millions, and prefer
    :class:`BatchIPF` when the whole sample is already in hand.

    Memory is ``O(n)`` with capacity doubling, so it is the compute that binds.
"""

from importlib.metadata import version

from .batch_ipf import BatchIPF
from .convergence import (
    ConvergenceAnalysis,
    RobbinsMonroVerification,
    analyze_convergence,
    estimate_lipschitz_constant,
    mwu_convergence_analysis,
    theoretical_convergence_bound,
    verify_convergence_conditions,
    verify_robbins_monro,
)
from .diagnostics import (
    FeasibilityReport,
    InfeasibilityAnalysis,
    IPFComparison,
    MarginCalibration,
    analyze_infeasibility,
    check_target_feasibility,
    compare_to_ipf,
    compute_design_effect,
    compute_weight_efficiency,
    estimate_margin_std_error,
    estimate_margin_variance,
    explain_infeasibility_causes,
    margin_calibration,
    optimal_mwu_learning_rate,
    resolve_replication_method,
    suggest_feasible_targets,
    summarize_raking_results,
)
from .divergence import (
    kl_divergence_weights,
    symmetric_kl_divergence,
    total_variation_weights,
)
from .learning_rate import (
    AdaptiveLR,
    ConstantLR,
    InverseTimeDecayLR,
    LearningRateSchedule,
    PolynomialDecayLR,
    robbins_monro_schedule,
)
from .model_assisted import (
    ModelAssistedRaker,
    ModelAssistedTargets,
    PoststratificationCell,
    PoststratificationCells,
    StreamingMRP,
    model_assisted_confidence_interval,
    model_assisted_std_error,
    model_assisted_variance,
)
from .models import (
    ExternalModelWrapper,
    LinearOutcomeModel,
    LogisticOutcomeModel,
    OutcomeModel,
)
from .online_raking_mwu import OnlineRakingMWU
from .online_raking_sgd import OnlineRakingSGD
from .sensitivity import (
    SensitivityReport,
    SensitivityResult,
    quick_sensitivity_check,
    run_sensitivity_analysis,
)
from .streaming_inference import (
    RetroactiveImpact,
    StreamingEstimator,
    StreamingSnapshot,
    analyze_estimate_stability,
    estimate_path_dependent_variance,
    explain_streaming_semantics,
)
from .targets import Targets

__all__ = [
    # Core algorithms
    "Targets",
    "OnlineRakingSGD",
    "OnlineRakingMWU",
    "BatchIPF",
    # Learning rate schedules
    "LearningRateSchedule",
    "ConstantLR",
    "InverseTimeDecayLR",
    "PolynomialDecayLR",
    "AdaptiveLR",
    "robbins_monro_schedule",
    # Convergence analysis
    "ConvergenceAnalysis",
    "RobbinsMonroVerification",
    "verify_robbins_monro",
    "estimate_lipschitz_constant",
    "analyze_convergence",
    "theoretical_convergence_bound",
    "mwu_convergence_analysis",
    "verify_convergence_conditions",
    # Diagnostics and variance estimation
    "MarginCalibration",
    "FeasibilityReport",
    "InfeasibilityAnalysis",
    "IPFComparison",
    "estimate_margin_variance",
    "estimate_margin_std_error",
    "margin_calibration",
    "resolve_replication_method",
    "check_target_feasibility",
    "analyze_infeasibility",
    "suggest_feasible_targets",
    "explain_infeasibility_causes",
    "compute_design_effect",
    "compute_weight_efficiency",
    "summarize_raking_results",
    "compare_to_ipf",
    "optimal_mwu_learning_rate",
    # Divergence metrics
    "kl_divergence_weights",
    "total_variation_weights",
    "symmetric_kl_divergence",
    # Streaming inference
    "StreamingSnapshot",
    "RetroactiveImpact",
    "StreamingEstimator",
    "estimate_path_dependent_variance",
    "explain_streaming_semantics",
    "analyze_estimate_stability",
    # Sensitivity analysis
    "SensitivityResult",
    "SensitivityReport",
    "run_sensitivity_analysis",
    "quick_sensitivity_check",
    # Model-assisted calibration (GREG/MRP)
    "OutcomeModel",
    "LinearOutcomeModel",
    "LogisticOutcomeModel",
    "ExternalModelWrapper",
    "ModelAssistedTargets",
    "ModelAssistedRaker",
    "model_assisted_variance",
    "model_assisted_std_error",
    "model_assisted_confidence_interval",
    "PoststratificationCell",
    "PoststratificationCells",
    "StreamingMRP",
]

__version__ = version("onlinerake")
