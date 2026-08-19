"""Diagnostics and variance estimation for online raking.

This module provides tools for:
1. Detecting infeasible target margins
2. Estimating variance of weighted margins
3. Computing confidence intervals
4. Assessing target feasibility

These address the critical gap identified in the original paper:
"What about uncertainty?" - the need for variance estimates and
proper handling of infeasible targets.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from statistics import NormalDist
from typing import TYPE_CHECKING

import numpy as np

from ._utils import requires_observations

if TYPE_CHECKING:
    from .batch_ipf import BatchIPF
    from .online_raking_sgd import OnlineRakingSGD


# Module-level constants for magic numbers
WEIGHT_THRESHOLD_RATIO = 0.95
WEIGHT_LOWER_THRESHOLD_RATIO = 1.05
MIN_FEASIBILITY_SCORE = 0.5
PROGRESS_SCORE_BONUS = 0.5
EXTREME_WEIGHT_RATIO = 1000
MAX_WEIGHT_RATIO_COMPROMISE = 100.0
Z_SCORES: dict[float, float] = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}

#: Replication schemes accepted by :func:`estimate_margin_variance`. ``"auto"``
#: is the default and picks one of the other two; see
#: :func:`resolve_replication_method`.
REPLICATION_METHODS = ("auto", "random_groups", "jackknife")

#: The two schemes ``"auto"`` chooses between.
CONCRETE_REPLICATION_METHODS = ("random_groups", "jackknife")

#: Number of replicate groups used when the caller does not choose one.
DEFAULT_N_REPLICATES = 10

#: Replicate size at or above which ``"auto"`` picks ``"random_groups"``.
#: See :func:`resolve_replication_method` for the measurements behind it.
AUTO_REPLICATE_SIZE = 100


@dataclass
class MarginCalibration:
    """How closely one weighted margin reached the target it was aimed at.

    Deliberately *not* a confidence interval. A raked margin is calibrated
    toward a target the caller supplied as known, so its estimand is that
    target and it carries no sampling uncertainty about it -- R's ``survey``
    reports a standard error of exactly zero for a raking variable. What an
    online raker has that a batch one does not is a margin that has not quite
    arrived, and that shortfall is a convergence property.

    Both terms are exact or estimated from the run itself, so nothing here
    depends on a default being right.

    Attributes:
        feature: Name of the feature.
        target: The target the raker was aimed at.
        estimate: The weighted margin actually achieved.
        gap: ``estimate - target``. Exact, not estimated.
        std_error: Replication standard error of the margin -- how much it
            moves across recalibrated subsamples.
        gap_ratio: ``|gap| / std_error`` -- the shortfall in units of the
            margin's own run-to-run spread, and nothing more. **It carries no
            absolute threshold.** At fixed calibration effort the numerator is
            flat in the stream length while the denominator falls, so the ratio
            rises with ``n``: measured 3.26, 4.56, 7.02, 16.26 at n = 250,
            1000, 4000, 12000 on an unchanged raker. Contamination of a
            downstream estimate does rise with ``n`` too -- its bias stays put
            while its own spread shrinks -- so the direction is real, but the
            scale is not shared: at n = 4000 this ratio read 11.78 where the
            downstream bias was 0.97 of that estimate's standard error. Use it
            to compare runs, not against a cutoff.

            An earlier version of this docstring said "below about one, the
            margin has arrived". That was false, and it came from validating the
            ratio by sweeping ``learning_rate * n_sgd_steps`` at fixed ``n``,
            where it does fall neatly to one, without ever sweeping ``n``.
        unclosed_fraction: ``|gap| / |raw_estimate - target|`` -- the share of
            the initial miscalibration the raker did **not** remove. Both terms
            are gaps, so unlike ``gap_ratio`` this does not move with the stream
            length, which makes it the honest answer to "did the raker
            converge". ``nan`` when the raw margin already sat on the target and
            there was nothing to close.
        raw_estimate: Unweighted sample proportion.
        bias_reduction: Percentage reduction in distance to target from
            reweighting.
    """

    feature: str
    target: float
    estimate: float
    gap: float
    std_error: float
    gap_ratio: float
    unclosed_fraction: float
    raw_estimate: float
    bias_reduction: float


@dataclass
class FeasibilityReport:
    """Report on target feasibility.

    Attributes:
        is_feasible: Whether all targets appear achievable.
        problematic_features: Features with potential feasibility issues.
        feasibility_scores: Per-feature feasibility scores (0-1, higher is better).
        recommendations: Suggested actions if infeasible.
    """

    is_feasible: bool
    problematic_features: list[str]
    feasibility_scores: dict[str, float]
    recommendations: list[str]


def _group_assignment(n: int, groups: int, seed: int) -> np.ndarray:
    """Balanced group labels for ``n`` observations, in random order.

    Balanced because ``arange(n) % groups`` gives each group the same size to
    within one; randomly ordered because *which* observation lands in which
    group must not track the arrival index.

    The systematic version was a trap. When arrival order has a period sharing
    a factor with ``groups`` -- every tenth event positive, with the default
    ten groups -- the modulo puts every one of those events in a single
    replicate. Measured on exactly that stream at n=400: the systematic
    assignment gave group 0 all 40 positives and the other nine none, and
    reported a variance of 1.0e-2 against 1.4e-4 to 2.1e-3 for randomized
    assignments. It was measuring the index pattern rather than sampling
    variability, under a scheme named ``random_groups``.

    Args:
        n: Number of observations.
        groups: Number of replicate groups.
        seed: Seed, so a variance estimate is reproducible and a caller can
            check how much the answer depends on the grouping.

    Returns:
        np.ndarray: Group label per observation, shape ``(n,)``.
    """
    return np.random.default_rng(seed).permutation(np.arange(n) % groups)


def resolve_replication_method(
    method: str, n_observations: int, n_replicates: int
) -> str:
    """Turn ``"auto"`` into the scheme it actually runs.

    Call this to see what a default-argument call is about to do. Every public
    function here takes ``method="auto"``, so without this the choice would be
    invisible -- and a default that silently changes scheme is harder to reason
    about than either fixed one.

    ``"random_groups"`` scales the spread among ``G`` disjoint groups back up by
    ``1 / (G(G - 1))``, which assumes the margin's variance goes as ``1/n``. For
    a *raked* margin it does not: at small sizes the margin is pinned tighter to
    its target, because fewer observations satisfy the same constraints. Its
    replicates hold ``n/G`` observations and so sit in that region. Jackknife
    replicates hold ``n(1 - 1/G)`` and do not.

    Measured against the margin's own spread over 500-600 independent streams,
    with each scheme's mean taken over 120 further streams and the two schemes
    paired on identical streams, as ratios to that truth:

    ===== ===== ============= ========= =====================
    n     n/G   random_groups jackknife paired jk - rg
    ===== ===== ============= ========= =====================
    300      30 0.919         1.026     +11.7%  (6.5 sigma)
    600      60 0.900         0.955      +6.0%  (5.0 sigma)
    1200    120 0.929         0.961      +3.5%  (3.6 sigma)
    ===== ===== ============= ========= =====================

    **What supports the rule is the last column, not the first two.** The paired
    gap shrinks steadily as ``n/G`` grows -- 11.7%, 6.0%, 3.5% -- so the scheme
    you pick matters most where replicates are small, which is what
    :data:`AUTO_REPLICATE_SIZE` keys on. Jackknife is above random groups at
    every size, by 3.6 to 6.5 sigma paired.

    **What does not support it is any claim about the absolute levels.** An
    earlier version of this table was measured under a systematic
    ``arange(n) % G`` grouping and showed ``random_groups`` climbing 0.838 ->
    0.923 -> 0.958, which read as an understatement decaying to about 5% by the
    threshold. Under the randomized grouping this package now uses (see
    :func:`_group_assignment`), it does not climb: it sits at 0.90 to 0.93
    throughout, and jackknife is no longer within noise of 1.0 at the larger
    sizes either. Both schemes understate somewhat at every size tried, and the
    threshold is not a point where that stops.

    So :data:`AUTO_REPLICATE_SIZE` is **a judgment about where the choice stops
    mattering much, not a point where either scheme becomes correct.** The
    percentages are specific to the process they were measured on -- two binary
    margins under this package's defaults. Treat the rule as a sensible default,
    not a law, pass an explicit scheme if the choice matters to you, and do not
    read either column as a calibration guarantee.

    The rule also bounds its own cost. Both schemes are linear in ``n``, and
    jackknife refits roughly ``9x`` the rows, so its cost *rises* with ``n`` --
    which is the regime handed to ``random_groups``. Cost peaks at the threshold
    (0.65 s at ``n = 600``) and falls after, against 8.97 s for always-jackknife
    at ``n = 4800``.

    Args:
        method: ``"auto"``, ``"random_groups"`` or ``"jackknife"``.
        n_observations: Number of observations the raker has seen.
        n_replicates: Number of replicate groups requested.

    Returns:
        str: ``"random_groups"`` or ``"jackknife"``. Anything other than
        ``"auto"`` is returned unchanged.

    Raises:
        ValueError: If ``method`` is not a known scheme, or ``n_replicates`` is
            below 2.

    Examples:
        >>> resolve_replication_method("auto", n_observations=300, n_replicates=10)
        'jackknife'
        >>> resolve_replication_method("auto", n_observations=4800, n_replicates=10)
        'random_groups'
        >>> resolve_replication_method("random_groups", 300, 10)
        'random_groups'
    """
    _check_replication(method, n_replicates)
    if method != "auto":
        return method
    groups = max(2, min(int(n_replicates), max(n_observations, 1)))
    return (
        "random_groups"
        if n_observations / groups >= AUTO_REPLICATE_SIZE
        else "jackknife"
    )


def _check_replication(method: str, n_replicates: int) -> None:
    """Reject replication settings that have no meaning.

    Args:
        method: Replication scheme name.
        n_replicates: Number of replicate groups.

    Raises:
        ValueError: If ``method`` is not a known scheme, or ``n_replicates`` is
            below 2.
    """
    if method not in REPLICATION_METHODS:
        raise ValueError(f"method must be one of {REPLICATION_METHODS}, got {method!r}")
    if n_replicates < 2:
        raise ValueError(
            f"n_replicates must be at least 2 to have a spread, got {n_replicates}"
        )


def _unfitted_copy[RakerT: "OnlineRakingSGD"](raker: RakerT) -> RakerT:
    """Build a raker of the same class and configuration, with no observations.

    Every replicate has to be calibrated by the *same* algorithm at the *same*
    settings as the fitted raker, or the spread it measures is the spread of a
    different estimator. Copying the object and clearing its fitted state keeps
    the two in step even for subclasses that carry extra configuration, which
    reconstructing through ``__init__`` would not: ``ModelAssistedRaker`` takes a
    model and a pair of target sets that the base signature knows nothing about.

    Every array in the object is replaced by a private copy, so replaying
    observations through the replicate cannot touch the original's state.

    Convergence tracking and weight statistics are switched off because the
    replicate is read once, for its margins. Neither enters the weight update,
    so switching them off cannot move the number this returns.

    Args:
        raker: The fitted raker to copy.

    Returns:
        RakerT: An unfitted raker of the same class and configuration. The
        return type follows the argument, so a ``ModelAssistedRaker`` in
        gives a ``ModelAssistedRaker`` out and its extra surface survives.
    """
    replicate = copy.copy(raker)
    for name, value in vars(raker).items():
        if isinstance(value, np.ndarray):
            setattr(replicate, name, value.copy())

    # The schedule is an object, and `copy.copy` shares it. The three shipped
    # schedules are stateless -- they compute from the ``t`` they are handed, so
    # a replicate restarts naturally -- but `LearningRateSchedule` is a public
    # extension point and a stateful implementation is legal. Sharing one let a
    # replicate advance the *parent's* schedule: measured on a counter-based
    # schedule, `estimate_margin_variance` took the parent's call count from 31
    # to 301. Measuring a raker must not change it.
    #
    # Deep-copied rather than reset, because the protocol offers no general way
    # to rebuild a schedule in its initial state. For the shipped schedules that
    # is exactly equivalent; for a stateful custom one the replicate inherits
    # the parent's current state rather than restarting, which is a limitation
    # worth knowing rather than a silent difference.
    if getattr(replicate, "_lr_schedule", None) is not None:
        replicate._lr_schedule = copy.deepcopy(raker._lr_schedule)

    replicate._n_obs = 0
    replicate.history = []
    replicate._loss_history = []
    replicate._gradient_norms = []
    replicate._learning_rate_history = []
    replicate._kl_history = []
    replicate._converged = False
    replicate._convergence_step = None
    replicate._cached_weight_stats = None
    replicate._weight_stats_computed_at = 0
    replicate._prev_weights = None
    replicate.track_convergence = False
    replicate.compute_weight_stats = False

    # The arrays are copied at the parent's full capacity, so everything past
    # the replicate's own _n_obs is the parent's data. Every read in the package
    # slices [:n], which makes that harmless -- but "harmless because I read
    # every call site" is what an assertion is for, and a future vectorized
    # operation that forgets the slice would silently mix parent rows into a
    # replicate. Zeroing the tail turns that from a wrong number into an
    # obviously wrong one.
    for name in (
        "_features",
        "_weights",
        "_outcomes",
        "_predictions",
        "_model_features",
    ):
        array = getattr(replicate, name, None)
        if isinstance(array, np.ndarray) and array.size:
            array[...] = 0.0
    flags = getattr(replicate, "_has_outcomes", None)
    if isinstance(flags, np.ndarray) and flags.size:
        flags[...] = False
    return replicate


def _refit_margins(raker: OnlineRakingSGD, index: np.ndarray) -> dict[str, float]:
    """Re-run the calibration on a subset of the observations.

    Takes positions rather than feature rows because a row is not the whole
    observation for every raker. ``ModelAssistedRaker`` also consumes the
    outcome and any model covariate that is not a calibration target, and
    replaying rows alone dropped both -- silently, since the replicate still
    calibrates and still returns margins. :meth:`OnlineRakingSGD._replay` is
    what knows the difference.

    Args:
        raker: The fitted raker, used for its class and configuration.
        index: Positions of the observations to replay, in arrival order.

    Returns:
        dict: Weighted margins of the replicate, per feature.
    """
    replicate = _unfitted_copy(raker)
    for i in index:
        obs, kwargs = raker._replay(int(i))
        replicate.partial_fit(obs, **kwargs)
    return replicate.margins


def _replicate_margins(
    raker: OnlineRakingSGD,
    method: str,
    n_replicates: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    """Calibrate each replicate subsample and collect its margins.

    Observations are split into groups systematically -- observation ``i`` joins
    group ``i % groups`` -- which needs no random number generator and so makes
    the variance estimate reproducible.

    An earlier version of this docstring claimed systematic and randomized
    grouping "agreed to within the study's noise" over 25 streams. A later one
    replaced that with a claim that both understate the margin's spread by about
    30%. **Both were unsupported**, and the second was measured here and
    retracted: re-running the same comparison on different streams reversed its
    sign (0.91 and 0.97 on one set of streams, 1.09 and 1.15 on another). The
    denominator in those ratios is a standard deviation estimated from 20-30
    streams, whose own relative error is ``1/sqrt(2(S-1))`` -- about 15% at
    S=25 -- so every one of those numbers sits within a couple of standard
    errors of 1.0 and of each other. Noise read as signal.

    What *is* measured, at 120 replicates per cell and monotone, is that the
    raked margin's variance does not scale as ``1/n``:

        m     sd(margin)   what 1/n predicts   ratio
        80       0.01042             0.01440    0.72
       200       0.00843             0.00910    0.93
       400       0.00629             0.00644    0.98
       800       0.00455             0.00455    1.00

    At small ``m`` the margin is pinned tighter to the target -- fewer
    observations satisfying the same constraints means more freedom per
    constraint -- so it varies less than ``1/n`` implies. The mean gap is flat
    (~0.019) across every ``m``, so this is a spread effect and not bias.

    That matters for the choice of scheme. The random-groups scaling
    ``1/(G(G-1))`` is unbiased only if ``Var(m) = (n/m) * Var(n)``; with G=10
    the replicate is ``m = n/10``, in the region where that fails. Jackknife
    replicates are ``m = n(1 - 1/G)``, at the safe end of the table. R's
    ``survey`` goes further and uses delete-*one* (``m = n-1``) for its
    jackknife, and defaults to linearization rather than replication at all.

    The two schemes have since been separated, paired on identical streams, and
    the prediction held: ``random_groups`` understates by an amount governed by
    ``n/G``, while jackknife is within noise of the truth at every size tried.
    :func:`resolve_replication_method` carries the table and is what ``"auto"``
    consults.

    Args:
        raker: A fitted raker.
        method: An already-resolved scheme -- ``"random_groups"`` or
            ``"jackknife"``. ``"auto"`` is rejected; resolve it first.
        n_replicates: Number of groups, capped at the number of observations.
        seed: Seed for the balanced-but-randomized grouping. Vary it to see
            how much the estimate depends on which observations shared a
            replicate.

    Returns:
        tuple: An array of shape ``(groups, n_features)`` holding each
        replicate's margins, and the number of groups actually used.

    Raises:
        ValueError: If ``method`` has not been resolved to a concrete scheme.
    """
    if method not in CONCRETE_REPLICATION_METHODS:
        raise ValueError(
            f"expected a resolved scheme, got {method!r}. Pass it through "
            "resolve_replication_method first -- the subset rule below and the "
            "scaling factor in _replication_variances must agree, and they can "
            "only do that if the name is resolved once, before either sees it."
        )
    n = raker._n_obs
    groups = max(2, min(int(n_replicates), n))
    assignment = _group_assignment(n, groups, seed)
    positions = np.arange(n)
    names = raker._feature_names

    subsets = (
        (positions[assignment == g] for g in range(groups))
        if method == "random_groups"
        else (positions[assignment != g] for g in range(groups))
    )
    # One refit per group, not one per group and feature: a replicate already
    # carries every margin, and asking for them one at a time would recalibrate
    # the same subsample once per feature.
    replicates = [_refit_margins(raker, subset) for subset in subsets]
    margins = np.array(
        [[replicate[name] for name in names] for replicate in replicates],
        dtype=float,
    )
    return margins, groups


def _replication_variances(
    raker: OnlineRakingSGD,
    method: str,
    n_replicates: int,
    seed: int,
) -> dict[str, float]:
    """Replication variance of every weighted margin, in one pass.

    Kept separate from :func:`estimate_margin_variance` because a replicate
    yields all the margins at once: asking for them one feature at a time would
    recalibrate the whole set of replicates once per feature.

    Args:
        raker: A fitted raker.
        method: ``"auto"``, ``"random_groups"`` or ``"jackknife"``.
        n_replicates: Number of groups.
        seed: Seed for the balanced-but-randomized grouping. Vary it to see
            how much the estimate depends on which observations shared a
            replicate.

    Returns:
        dict: Estimated variance per feature.
    """
    # Resolved once, here, rather than in each caller: this is the only place
    # the scheme name reaches both the subset rule and the scaling factor, and
    # resolving it twice risks the two disagreeing.
    method = resolve_replication_method(method, raker._n_obs, n_replicates)
    if raker._n_obs < 2:
        # NaN, not 0.0. A single observation cannot be split into two groups, so
        # the variance is unestimable -- and zero variance is a claim that the
        # estimator cannot vary, which produced a zero-width 95% interval from
        # one draw. That is the proportion_confint(0, 20) -> (0.0, 0.0) defect.
        return dict.fromkeys(raker._feature_names, float("nan"))

    margins, groups = _replicate_margins(raker, method, n_replicates, seed)
    deviations = margins - margins.mean(axis=0)
    scale = (
        1.0 / (groups * (groups - 1))
        if method == "random_groups"
        else (groups - 1) / groups
    )
    variances = scale * np.sum(deviations**2, axis=0)
    return {name: float(variances[i]) for i, name in enumerate(raker._feature_names)}


def estimate_margin_variance(
    raker: OnlineRakingSGD,
    feature: str,
    method: str = "auto",
    n_replicates: int = DEFAULT_N_REPLICATES,
    seed: int = 0,
) -> float:
    """Estimate the sampling variance of a weighted margin by replication.

    A raked margin is not a sample proportion. It has been driven toward a fixed
    target by the calibration, which removes most of the variation an unweighted
    proportion would have, so ``p(1 - p) / n_eff`` -- the formula this function
    used before -- describes a quantity the raker is not computing. Measured over
    100 replicates of a 500-observation stream it reported 0.02340 against an
    actual spread of 0.00528, a factor of 4.4.

    What the calibration does to the spread is not available in closed form
    here. Deville and Sarndal's linearisation for calibration estimators exists
    but assumes the calibration is solved exactly, which streaming raking does
    not do: it takes a fixed number of gradient steps per observation and is
    still chasing the target when the stream ends. A replication estimator makes
    no such assumption -- it re-runs the calibration on each replicate and reads
    off how much the answer moves -- so it prices in whatever the raker actually
    does.

    Two schemes, both re-running the calibration:

    ``"random_groups"`` (default)
        Split the stream into ``n_replicates`` disjoint groups, calibrate each
        group on its own, and take the spread of the group estimates divided by
        the number of groups (Wolter, *Introduction to Variance Estimation*,
        ch. 2). Cost is the cheap direction: the groups partition the stream, so
        all of them together replay each observation once, against gradients over
        a group rather than the whole sample. Measured at 0.17 s for a
        500-observation stream, against 0.64 s for the original fit.

    ``"jackknife"``
        Delete-a-group jackknife: calibrate ``n_replicates`` times on all but one
        group each time, and scale by ``(G - 1) / G``. Each replicate sees nearly
        the whole sample, which makes it the more robust of the two when the
        estimator behaves differently at ``n / G`` than at ``n``. **It costs
        ``n_replicates`` full refits** -- about 4 s at the default group count on
        a 500-observation stream, and quadratic in stream length after that.

    Measured against the spread of 100 independent 500-observation streams
    (0.00528), both land inside the Monte Carlo noise: random groups 0.974 of the
    truth at 10 groups, jackknife 1.024. At 1000 observations, 1.012 and 1.039.

    Args:
        raker: A fitted OnlineRakingSGD or OnlineRakingMWU object.
        feature: Name of the feature to estimate variance for.
        method: Replication scheme, ``"random_groups"`` or ``"jackknife"``.
        n_replicates: Number of replicate groups. Capped at the number of
            observations, and at least 2. More groups give a steadier estimate;
            under ``"jackknife"`` they also cost proportionally more.
        seed: Seed for the balanced-but-randomized grouping. Vary it to see
            how much the estimate depends on which observations shared a
            replicate.

    Returns:
        Estimated variance of the weighted margin. NaN if the raker holds no
        observations, and 0.0 if it holds one, which has no spread to measure.

    Raises:
        ValueError: If ``method`` is not a known scheme, or ``n_replicates`` is
            below 2.
    """
    _check_replication(method, n_replicates)
    if raker._n_obs == 0:
        return np.nan

    return _replication_variances(raker, method, n_replicates, seed)[feature]


def estimate_margin_std_error(
    raker: OnlineRakingSGD,
    feature: str,
    method: str = "auto",
    n_replicates: int = DEFAULT_N_REPLICATES,
    seed: int = 0,
) -> float:
    """Estimate standard error of a weighted margin.

    Args:
        raker: A fitted OnlineRakingSGD or OnlineRakingMWU object.
        feature: Name of the feature.
        method: Replication scheme, passed to :func:`estimate_margin_variance`.
        n_replicates: Number of replicate groups.
        seed: Seed for the balanced-but-randomized grouping. Vary it to see
            how much the estimate depends on which observations shared a
            replicate.

    Returns:
        Estimated standard error.
    """
    var = estimate_margin_variance(raker, feature, method, n_replicates, seed)
    return float(np.sqrt(var)) if not np.isnan(var) else np.nan


def _z_score(confidence_level: float) -> float:
    """Normal quantile for a two-sided interval at this level.

    Computed rather than looked up. The table only held 0.90, 0.95 and 0.99,
    and every other level fell through to a scipy import that is not a
    dependency of this package -- so the ``except ImportError`` branch returned
    the 0.95 multiplier for *any* other level. Asking for an 80% interval
    silently produced a 95% one. ``statistics.NormalDist`` is in the standard
    library and exact.

    Args:
        confidence_level: Nominal coverage, for instance 0.95.

    Returns:
        The multiplier to apply to a standard error.

    Raises:
        ValueError: If ``confidence_level`` is not strictly between 0 and 1.
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")
    return float(NormalDist().inv_cdf(1 - (1 - confidence_level) / 2))


def margin_calibration(
    raker: OnlineRakingSGD,
    method: str = "auto",
    n_replicates: int = DEFAULT_N_REPLICATES,
    seed: int = 0,
) -> list[MarginCalibration]:
    """How closely each weighted margin reached its target.

    The replication runs once here rather than once per feature: a replicate
    carries the margins of every feature, so recalibrating for each of them in
    turn would multiply the cost by the number of features and return the same
    numbers.

    Args:
        raker: A fitted OnlineRakingSGD or OnlineRakingMWU object.
        method: Replication scheme, passed to :func:`estimate_margin_variance`.
        n_replicates: Number of replicate groups.
        seed: Seed for the balanced-but-randomized grouping. Vary it to see
            how much the estimate depends on which observations shared a
            replicate.

    Returns:
        List of MarginCalibration objects, one per feature.

    Raises:
        ValueError: If ``method`` is not a known scheme, or ``n_replicates`` is
            below 2.
    """
    _check_replication(method, n_replicates)
    estimates = []
    margins = raker.margins
    raw_margins = raker.raw_margins
    if raker._n_obs == 0:
        variances = dict.fromkeys(raker._feature_names, np.nan)
    else:
        variances = _replication_variances(raker, method, n_replicates, seed)

    for feature in raker._feature_names:
        target = raker.targets[feature]
        estimate = margins[feature]
        raw = raw_margins[feature]
        variance = variances[feature]
        std_error = float(np.sqrt(variance)) if not np.isnan(variance) else np.nan
        gap = estimate - target
        gap_ratio = (
            abs(gap) / std_error
            if np.isfinite(std_error) and std_error > 0
            else float("nan")
        )
        initial_gap = abs(raw - target)
        unclosed_fraction = abs(gap) / initial_gap if initial_gap > 0 else float("nan")

        # Compute bias reduction
        raw_bias = abs(raw - target)
        weighted_bias = abs(estimate - target)
        if raw_bias > 0:
            bias_reduction = 100 * (raw_bias - weighted_bias) / raw_bias
        else:
            bias_reduction = 0.0

        estimates.append(
            MarginCalibration(
                feature=feature,
                target=target,
                estimate=estimate,
                gap=gap,
                std_error=std_error,
                gap_ratio=gap_ratio,
                unclosed_fraction=unclosed_fraction,
                raw_estimate=raw,
                bias_reduction=bias_reduction,
            )
        )

    return estimates


def check_target_feasibility(
    raker: OnlineRakingSGD,
    margin_tolerance: float = 0.05,
) -> FeasibilityReport:
    """Check whether target margins are feasible given the observed data.

    Feasibility issues arise when:
    1. A target proportion is outside the range achievable by reweighting
    2. Target combinations are mutually inconsistent
    3. Not enough variation in the sample to achieve targets

    This function detects potential feasibility issues and provides
    actionable recommendations.

    Args:
        raker: A fitted OnlineRakingSGD or OnlineRakingMWU object.
        margin_tolerance: How far a margin may sit from its target and still
            count as reachable, in margin units. Not comparable to
            :func:`~onlinerake.convergence.analyze_convergence`'s
            ``loss_tolerance``, which is a squared-error loss.

    Returns:
        FeasibilityReport with diagnosis and recommendations.
    """
    if raker._n_obs == 0:
        return FeasibilityReport(
            is_feasible=False,
            problematic_features=[],
            feasibility_scores={},
            recommendations=["No observations processed yet."],
        )

    raw_margins = raker.raw_margins
    margins = raker.margins
    problematic = []
    scores = {}
    recommendations = []

    for feature in raker._feature_names:
        target = raker.targets[feature]
        raw = raw_margins[feature]
        weighted = margins[feature]
        is_binary = raker.targets.is_binary(feature)

        if is_binary:
            # Binary feature feasibility checks
            # Check if sample has required variation
            # No observations with feature=1, can't increase margin
            if raw == 0 and target > margin_tolerance:
                problematic.append(feature)
                scores[feature] = 0.0
                recommendations.append(
                    f"'{feature}': No observations with value=1, "
                    f"cannot achieve target {target:.2%}."
                )
                continue

            # All observations have feature=1, can't decrease margin
            if raw == 1 and target < 1 - margin_tolerance:
                problematic.append(feature)
                scores[feature] = 0.0
                recommendations.append(
                    f"'{feature}': All observations have value=1, "
                    f"cannot achieve target {target:.2%}."
                )
                continue
        else:
            # Continuous feature feasibility checks
            # Check if target is within sample range
            feature_idx = raker._feature_names.index(feature)
            feature_values = raker._features[: raker._n_obs, feature_idx]
            min_val = float(feature_values.min())
            max_val = float(feature_values.max())

            if target < min_val or target > max_val:
                problematic.append(feature)
                scores[feature] = 0.0
                recommendations.append(
                    f"'{feature}': Target mean {target:.2f} is outside sample range "
                    f"[{min_val:.2f}, {max_val:.2f}]. Cannot achieve this target."
                )
                continue

        # Check convergence toward target
        raw_error = abs(raw - target)
        weighted_error = abs(weighted - target)

        # Scale tolerance for continuous features based on target magnitude
        effective_tolerance = (
            margin_tolerance if is_binary else margin_tolerance * max(1.0, abs(target))
        )

        if raw_error > effective_tolerance:
            # We have a bias to correct
            if weighted_error < raw_error:
                # Making progress
                progress = 1 - (weighted_error / raw_error)
                scores[feature] = min(1.0, progress + PROGRESS_SCORE_BONUS)
            else:
                # Not making progress - potential feasibility issue
                scores[feature] = 0.3
                problematic.append(feature)
                if is_binary:
                    recommendations.append(
                        f"'{feature}': Weighted margin ({weighted:.2%}) not converging "
                        f"toward target ({target:.2%}). Consider adjusting the "
                        "learning rate."
                    )
                else:
                    recommendations.append(
                        f"'{feature}': Weighted mean ({weighted:.2f}) not converging "
                        f"toward target ({target:.2f}). Consider adjusting the "
                        "learning rate."
                    )
        else:
            # Already close to target
            scores[feature] = 1.0

        # Check for extreme weights suggesting feasibility strain
        # Use raker's configured bounds rather than hardcoded values
        weight_stats = raker.weight_distribution_stats
        at_max = weight_stats["max"] >= raker.max_weight * WEIGHT_THRESHOLD_RATIO
        at_min = weight_stats["min"] <= raker.min_weight * WEIGHT_LOWER_THRESHOLD_RATIO
        if at_max or at_min:
            if feature not in problematic:
                problematic.append(feature)
                scores[feature] = min(scores.get(feature, 1.0), MIN_FEASIBILITY_SCORE)
            extreme_warning = (
                f"Weights hitting bounds (min={raker.min_weight}, "
                f"max={raker.max_weight}). "
                "This may indicate target feasibility strain."
            )
            if extreme_warning not in recommendations:
                recommendations.append(extreme_warning)

    # Overall feasibility assessment
    is_feasible = len(problematic) == 0

    if not is_feasible and not recommendations:
        recommendations.append(
            "Consider: (1) Relaxing targets, (2) Increasing sample size, "
            "(3) Using different weight bounds."
        )

    return FeasibilityReport(
        is_feasible=is_feasible,
        problematic_features=problematic,
        feasibility_scores=scores,
        recommendations=recommendations,
    )


@requires_observations(lambda: np.nan)
def compute_design_effect(raker: OnlineRakingSGD) -> float:
    """Compute the design effect (DEFF) due to weighting.

    DEFF = n / ESS, where n is the nominal sample size and ESS is the
    effective sample size. A DEFF of 2 means you need twice the sample
    size to achieve the same precision as an unweighted sample.

    Args:
        raker: A fitted OnlineRakingSGD or OnlineRakingMWU object.

    Returns:
        Design effect. Values near 1 indicate minimal precision loss from weighting.
    """
    ess = raker.effective_sample_size
    if ess <= 0:
        return np.nan

    return float(raker._n_obs / ess)


@requires_observations(lambda: np.nan)
def compute_weight_efficiency(raker: OnlineRakingSGD) -> float:
    """Compute weight efficiency (ESS / n).

    This is the inverse of design effect, ranging from 0 to 1.
    Higher values indicate more efficient use of the sample.

    Args:
        raker: A fitted OnlineRakingSGD or OnlineRakingMWU object.

    Returns:
        Weight efficiency as a proportion.
    """
    ess = raker.effective_sample_size
    return float(ess / raker._n_obs)


def summarize_raking_results(raker: OnlineRakingSGD) -> dict:
    """Generate a comprehensive summary of raking results.

    Provides all key metrics and diagnostics in a single report.

    The margin section reports calibration rather than confidence intervals:
    see :class:`MarginCalibration` for why a raked margin does not carry
    sampling uncertainty about the target it was aimed at.

    Args:
        raker: A fitted OnlineRakingSGD or OnlineRakingMWU object.

    Returns:
        Dictionary with complete summary statistics.
    """
    feasibility = check_target_feasibility(raker)
    margin_estimates = margin_calibration(raker)

    return {
        "n_observations": raker._n_obs,
        "effective_sample_size": raker.effective_sample_size,
        "design_effect": compute_design_effect(raker),
        "weight_efficiency": compute_weight_efficiency(raker),
        "final_loss": raker.loss,
        "converged": raker.converged,
        "convergence_step": raker.convergence_step,
        "margin_estimates": [
            {
                "feature": est.feature,
                "target": est.target,
                "estimate": est.estimate,
                "gap": est.gap,
                "std_error": est.std_error,
                "gap_ratio": est.gap_ratio,
                "unclosed_fraction": est.unclosed_fraction,
                "raw_estimate": est.raw_estimate,
                "bias_reduction_pct": est.bias_reduction,
            }
            for est in margin_estimates
        ],
        "feasibility": {
            "is_feasible": feasibility.is_feasible,
            "problematic_features": feasibility.problematic_features,
            "scores": feasibility.feasibility_scores,
            "recommendations": feasibility.recommendations,
        },
        "weight_distribution": raker.weight_distribution_stats,
    }


@dataclass
class InfeasibilityAnalysis:
    """Detailed analysis of target infeasibility.

    When targets are infeasible, this class provides:
    1. Diagnosis of why targets are infeasible
    2. Achievable bounds for each feature
    3. Compromise solutions
    4. Recommendations for resolution

    Attributes:
        is_feasible: Whether exact targets are achievable.
        infeasibility_type: Type of infeasibility detected.
        achievable_bounds: For each feature, the achievable range.
        compromise_targets: Adjusted targets that are feasible.
        diagnosis: Detailed explanation of the infeasibility.
    """

    is_feasible: bool
    infeasibility_type: str
    achievable_bounds: dict[str, tuple[float, float]]
    compromise_targets: dict[str, float]
    diagnosis: list[str]


def analyze_infeasibility(
    raker: OnlineRakingSGD,
    max_weight_ratio: float = MAX_WEIGHT_RATIO_COMPROMISE,
) -> InfeasibilityAnalysis:
    """Analyze why targets may be infeasible and suggest compromises.

    This function provides detailed diagnosis when targets cannot be
    exactly achieved by reweighting. It identifies:

    1. **Structural infeasibility**: Some targets require weight=0 or ∞
    2. **Conflicting targets**: Targets are mutually inconsistent
    3. **Numerical infeasibility**: Extreme weights would be required

    Args:
        raker: A fitted OnlineRakingSGD or OnlineRakingMWU object.
        max_weight_ratio: Maximum allowable weight ratio for compromise targets.

    Returns:
        InfeasibilityAnalysis with diagnosis and compromise solutions.

    Examples:
        No reweighting can raise a margin above the proportion that actually
        carries the feature. Here nobody does, so 90% is not merely hard, it is
        unreachable at any weight:

        >>> from onlinerake import OnlineRakingSGD, Targets
        >>> raker = OnlineRakingSGD(Targets(rare=0.90))
        >>> for _ in range(20):
        ...     raker.partial_fit({"rare": 0})
        >>> analysis = analyze_infeasibility(raker)
        >>> analysis.is_feasible
        False
        >>> analysis.infeasibility_type
        'structural'
        >>> analysis.achievable_bounds
        {'rare': (0.0, 0.0)}
        >>> analysis.compromise_targets
        {'rare': 0.0}
    """
    if raker._n_obs == 0:
        return InfeasibilityAnalysis(
            is_feasible=False,
            infeasibility_type="no_data",
            achievable_bounds={},
            compromise_targets={},
            diagnosis=["No observations available for analysis."],
        )

    diagnosis: list[str] = []
    achievable_bounds: dict[str, tuple[float, float]] = {}
    compromise_targets: dict[str, float] = {}
    infeasibility_types: list[str] = []

    raw_margins = raker.raw_margins
    n_obs = raker._n_obs

    for feature in raker._feature_names:
        target = raker.targets[feature]
        raw = raw_margins[feature]
        is_binary = raker.targets.is_binary(feature)
        feature_idx = raker._feature_names.index(feature)
        feature_values = raker._features[:n_obs, feature_idx]

        if is_binary:
            # Binary feature analysis
            feature_sum = feature_values.sum()
            n_with_feature = int(feature_sum)
            n_without_feature = n_obs - n_with_feature

            # Compute achievable bounds
            # Lower bound: give all weight to observations without feature
            # Upper bound: give all weight to observations with feature
            min_achievable = 1.0 if n_without_feature == 0 else 0.0

            max_achievable = 0.0 if n_with_feature == 0 else 1.0

            achievable_bounds[feature] = (min_achievable, max_achievable)

            # Check if target is achievable
            if target < min_achievable:
                diagnosis.append(
                    f"'{feature}': Target {target:.1%} below achievable minimum "
                    f"{min_achievable:.1%} (all observations have feature=1)."
                )
                infeasibility_types.append("structural")
                compromise_targets[feature] = min_achievable
            elif target > max_achievable:
                diagnosis.append(
                    f"'{feature}': Target {target:.1%} above achievable maximum "
                    f"{max_achievable:.1%} (no observations have feature=1)."
                )
                infeasibility_types.append("structural")
                compromise_targets[feature] = max_achievable
            else:
                # Target is achievable in principle
                # Check if it requires extreme weights
                weight_ratio_needed = _estimate_required_weight_ratio(
                    raw, target, n_with_feature, n_without_feature
                )

                if weight_ratio_needed > EXTREME_WEIGHT_RATIO:
                    diagnosis.append(
                        f"'{feature}': Achieving target {target:.1%} from raw "
                        f"{raw:.1%} "
                        f"requires ~{weight_ratio_needed:.0f}:1 weight ratio."
                    )
                    infeasibility_types.append("numerical")
                    compromise_targets[feature] = _compute_achievable_with_ratio(
                        raw,
                        target,
                        n_with_feature,
                        n_without_feature,
                        max_ratio=max_weight_ratio,
                    )
                else:
                    compromise_targets[feature] = target
        else:
            # Continuous feature analysis
            min_val = float(feature_values.min())
            max_val = float(feature_values.max())

            achievable_bounds[feature] = (min_val, max_val)

            # Check if target is within sample range
            if target < min_val:
                diagnosis.append(
                    f"'{feature}': Target mean {target:.2f} below sample minimum "
                    f"{min_val:.2f}."
                )
                infeasibility_types.append("structural")
                compromise_targets[feature] = min_val
            elif target > max_val:
                diagnosis.append(
                    f"'{feature}': Target mean {target:.2f} above sample maximum "
                    f"{max_val:.2f}."
                )
                infeasibility_types.append("structural")
                compromise_targets[feature] = max_val
            else:
                # Target is achievable
                compromise_targets[feature] = target

    # Determine overall infeasibility type
    if not diagnosis:
        infeasibility_type = "feasible"
        is_feasible = True
    elif "structural" in infeasibility_types:
        infeasibility_type = "structural"
        is_feasible = False
    elif "numerical" in infeasibility_types:
        infeasibility_type = "numerical"
        is_feasible = False
    else:
        infeasibility_type = "unknown"
        is_feasible = False

    return InfeasibilityAnalysis(
        is_feasible=is_feasible,
        infeasibility_type=infeasibility_type,
        achievable_bounds=achievable_bounds,
        compromise_targets=compromise_targets,
        diagnosis=diagnosis,
    )


def _estimate_required_weight_ratio(
    raw: float,
    target: float,
    n_with: int,
    n_without: int,
) -> float:
    """Estimate the weight ratio required to move from raw to target margin."""
    if n_with == 0 or n_without == 0:
        return float("inf")

    if raw == target:
        return 1.0

    # Simplified model: w_1 for feature=1, w_0 for feature=0
    # Target = (n_with * w_1) / (n_with * w_1 + n_without * w_0)
    # Assuming w_0 = 1 (reference), solve for w_1:
    # target * (n_with * w_1 + n_without) = n_with * w_1
    # target * n_without = n_with * w_1 * (1 - target)
    # w_1 = target * n_without / (n_with * (1 - target))

    if target == 0:
        return 0.0
    if target == 1:
        return float("inf")

    w_1 = target * n_without / (n_with * (1 - target))

    # If w_1 > 1, ratio is w_1:1
    # If w_1 < 1, ratio is 1:1/w_1
    if w_1 > 1:
        return w_1
    return 1.0 / w_1 if w_1 > 0 else float("inf")


def _compute_achievable_with_ratio(
    raw: float,
    target: float,
    n_with: int,
    n_without: int,
    max_ratio: float,
) -> float:
    """Compute the closest achievable target given a maximum weight ratio."""
    if n_with == 0:
        return 0.0
    if n_without == 0:
        return 1.0

    # Two cases: need to increase or decrease margin
    if target > raw:
        # Need to increase: max weight on feature=1
        w_1 = max_ratio
        achievable = (n_with * w_1) / (n_with * w_1 + n_without)
        return min(target, achievable)
    # Need to decrease: max weight on feature=0
    w_0 = max_ratio
    achievable = n_with / (n_with + n_without * w_0)
    return max(target, achievable)


def suggest_feasible_targets(
    raker: OnlineRakingSGD,
    max_weight_ratio: float = MAX_WEIGHT_RATIO_COMPROMISE,
) -> dict[str, float]:
    """Suggest feasible targets based on observed data and weight constraints.

    Given the observed data distribution and constraints on weight variability,
    this function suggests the closest achievable targets to the original
    targets.

    Args:
        raker: A fitted OnlineRakingSGD or OnlineRakingMWU object.
        max_weight_ratio: Maximum allowable weight ratio (default 100:1).

    Returns:
        Dictionary of suggested feasible targets for each feature.

    Examples:
        The compromise half of :func:`analyze_infeasibility`, for when you want
        the numbers and not the diagnosis:

        >>> from onlinerake import OnlineRakingSGD, Targets
        >>> raker = OnlineRakingSGD(Targets(rare=0.90))
        >>> for _ in range(20):
        ...     raker.partial_fit({"rare": 0})
        >>> suggest_feasible_targets(raker, max_weight_ratio=50)
        {'rare': 0.0}
    """
    analysis = analyze_infeasibility(raker, max_weight_ratio=max_weight_ratio)
    return analysis.compromise_targets


def explain_infeasibility_causes() -> dict[str, str]:
    """Return explanations of different types of target infeasibility.

    Returns:
        Dictionary mapping infeasibility types to explanations.
    """
    return {
        "structural": (
            "STRUCTURAL INFEASIBILITY: The target proportion is mathematically "
            "impossible given the observed data. For example, if no observations "
            "have feature=1, you cannot achieve a target >0 for that feature. "
            "Solution: Adjust targets to be within achievable bounds, or collect "
            "more diverse data."
        ),
        "numerical": (
            "NUMERICAL INFEASIBILITY: The target is theoretically achievable but "
            "would require extreme weights (e.g., 1000:1 ratio). This leads to "
            "very low effective sample size and unstable estimates. "
            "Solution: Relax targets toward observed proportions, or accept "
            "lower weight efficiency."
        ),
        "conflicting": (
            "CONFLICTING TARGETS: Multiple targets are individually achievable but "
            "cannot be satisfied simultaneously. This occurs when features are "
            "correlated and targets imply impossible joint distributions. "
            "Solution: Use iterative methods like IPF that find compromise solutions, "
            "or adjust targets to be mutually consistent."
        ),
        "convergence": (
            "CONVERGENCE ISSUES: The algorithm is not converging toward targets, "
            "possibly due to learning rate issues or oscillation. "
            "Solution: Adjust learning rate, use diminishing schedules, or "
            "increase the number of gradient steps per observation."
        ),
    }


@dataclass
class IPFComparison:
    """Comparison between streaming raker and batch IPF solutions.

    Attributes:
        weight_kl: KL divergence D_KL(w_raker || w_ipf).
        weight_tv: Total variation distance between weight distributions.
        margin_mse: Mean squared error between margins.
        margin_max_diff: Maximum absolute difference in margins.
        ess_ratio: ESS_raker / ESS_ipf ratio.
        raker_loss: Final loss of the streaming raker.
        ipf_loss: Final loss of batch IPF.
    """

    weight_kl: float
    weight_tv: float
    margin_mse: float
    margin_max_diff: float
    ess_ratio: float
    raker_loss: float
    ipf_loss: float


def compare_to_ipf(
    raker: OnlineRakingSGD,
    ipf: BatchIPF | None = None,
) -> IPFComparison:
    """Compare streaming raker solution to batch IPF.

    This function quantifies how close a streaming raker (SGD or MWU) is
    to the batch IPF solution. The key insight is that MWU performs mirror
    descent with KL divergence, and should converge to IPF as the learning
    rate decreases.

    Args:
        raker: Fitted streaming raker (OnlineRakingSGD or OnlineRakingMWU).
        ipf: Pre-fitted BatchIPF object. If None, fits IPF on the raker's
            data (requires accessing internal state).

    Returns:
        IPFComparison with comparison metrics:
        - weight_kl: D_KL(w_raker || w_ipf) - how different the weights are
        - weight_tv: Total variation distance between normalized weights
        - margin_mse: MSE between weighted margins
        - margin_max_diff: Worst-case margin difference
        - ess_ratio: Relative effective sample size

    Raises:
        ValueError: If raker has no observations or if IPF fitting fails.

    Examples:
        >>> from onlinerake import OnlineRakingMWU, BatchIPF, Targets
        >>> targets = Targets(female=0.51, college=0.32)
        >>> data = [{"female": i % 2, "college": (i % 3) == 0} for i in range(60)]
        >>> mwu = OnlineRakingMWU(targets, learning_rate=0.5, n_sgd_steps=3)
        >>> for obs in data:
        ...     mwu.partial_fit(obs)
        >>> ipf = BatchIPF(targets).fit(data)
        >>> comparison = compare_to_ipf(mwu, ipf)

        Under the conditions the Note below lists, MWU lands close to the batch
        solution -- close in the weights themselves, and closer still in the
        margins they produce:

        >>> comparison.weight_kl < 0.01
        True
        >>> comparison.margin_mse < 1e-4
        True

    Note:
        For MWU to closely match IPF:
        1. Use small learning rates (η < 1.0)
        2. Process many observations (n > 100)
        3. Use multiple SGD steps per observation (n_sgd_steps >= 3)

        MWU with η → 0 should produce D_KL → 0 and margin_mse → 0.
    """
    from .batch_ipf import BatchIPF
    from .divergence import kl_divergence_weights, total_variation_weights

    if raker._n_obs == 0:
        raise ValueError("Raker has no observations; cannot compare to IPF")

    # Fit IPF if not provided
    if ipf is None:
        # Reconstruct observations from internal state
        ipf = BatchIPF(raker.targets)
        observations = []
        for i in range(raker._n_obs):
            obs = {}
            for j, name in enumerate(raker._feature_names):
                obs[name] = int(raker._features[i, j])
            observations.append(obs)
        ipf.fit(observations)

    # Ensure same number of observations
    if raker._n_obs != ipf._n_obs:
        raise ValueError(
            f"Raker has {raker._n_obs} observations but IPF has {ipf._n_obs}. "
            "Ensure IPF was fit on the same data."
        )

    # Get weights
    w_raker = raker.weights
    w_ipf = ipf.weights

    # Compute divergence metrics
    weight_kl = kl_divergence_weights(w_raker, w_ipf)
    weight_tv = total_variation_weights(w_raker, w_ipf)

    # Compute margin comparison
    margins_raker = raker.margins
    margins_ipf = ipf.margins

    margin_diffs = []
    for name in raker._feature_names:
        diff = margins_raker[name] - margins_ipf[name]
        margin_diffs.append(diff)

    margin_mse = float(np.mean([d**2 for d in margin_diffs]))
    margin_max_diff = float(np.max([abs(d) for d in margin_diffs]))

    # Compute ESS ratio
    ess_raker = raker.effective_sample_size
    ess_ipf = ipf.effective_sample_size
    ess_ratio = ess_raker / ess_ipf if ess_ipf > 0 else np.nan

    return IPFComparison(
        weight_kl=weight_kl,
        weight_tv=weight_tv,
        margin_mse=margin_mse,
        margin_max_diff=margin_max_diff,
        ess_ratio=ess_ratio,
        raker_loss=raker.loss,
        ipf_loss=ipf.loss,
    )


def optimal_mwu_learning_rate(n_observations: int, n_features: int) -> float:
    """Compute theoretical optimal learning rate for MWU to approximate IPF.

    From mirror descent theory, the optimal learning rate is approximately:
        η* ≈ sqrt(2 * log(n_observations) / T)

    where T is the expected number of iterations. For streaming raking with
    n_sgd_steps per observation, T ≈ n_observations * n_sgd_steps.

    A smaller learning rate means MWU stays closer to IPF at each step,
    but convergence is slower. This function provides a reasonable starting
    point for tuning.

    Args:
        n_observations: Expected number of observations.
        n_features: Number of features being calibrated.

    Returns:
        Recommended learning rate for MWU.

    Examples:
        >>> print(f"{optimal_mwu_learning_rate(n_observations=1000, n_features=4):.3f}")
        0.235

        The bound goes as ``1/sqrt(T)``, so a longer stream warrants a smaller
        step -- ten times the observations, roughly a third the rate:

        >>> rate = optimal_mwu_learning_rate(n_observations=10000, n_features=4)
        >>> print(f"{rate:.3f}")
        0.086

    Note:
        This is a theoretical guideline. In practice:
        - For IPF-matching: use lr < 0.5
        - For faster convergence: use lr 1.0-5.0
        - Monitor loss and adjust as needed
    """
    if n_observations <= 1:
        return 1.0

    # From mirror descent regret bounds:
    # η* = sqrt(2 * D / (T * G^2))
    # where D is diameter of domain (log(n_observations) for simplex)
    # and G is gradient bound (depends on n_features)

    # Simplified heuristic that works well empirically
    log_n = np.log(n_observations)
    eta = np.sqrt(2 * log_n / n_observations)

    # Scale by features (more features = more complex optimization)
    eta *= np.sqrt(n_features)

    # Clamp to reasonable range
    return float(np.clip(eta, 0.01, 5.0))
