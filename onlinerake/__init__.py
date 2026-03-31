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
    MarginEstimate,
    analyze_infeasibility,
    check_target_feasibility,
    compare_to_ipf,
    compute_confidence_interval,
    compute_design_effect,
    compute_weight_efficiency,
    estimate_margin_std_error,
    estimate_margin_variance,
    explain_infeasibility_causes,
    get_margin_estimates,
    optimal_mwu_learning_rate,
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
    ConfidenceSequence,
    RetroactiveImpact,
    StreamingEstimator,
    StreamingSnapshot,
    analyze_estimate_stability,
    compute_confidence_sequence,
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
    "MarginEstimate",
    "FeasibilityReport",
    "InfeasibilityAnalysis",
    "IPFComparison",
    "estimate_margin_variance",
    "estimate_margin_std_error",
    "compute_confidence_interval",
    "get_margin_estimates",
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
    "ConfidenceSequence",
    "RetroactiveImpact",
    "StreamingEstimator",
    "compute_confidence_sequence",
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
    "PoststratificationCell",
    "PoststratificationCells",
    "StreamingMRP",
]

__version__ = "1.4.0"
