"""Monte Carlo studies of the inference onlinerake ships.

Three claims in this package were statements about sampling behavior, and until
this file existed not one of them had been run over a replicate loop. **Two of
the three did not survive, and the functions carrying them have been removed.**
This file is what remains: the study of the one that held, plus the study of the
interval it did not have.

**What was withdrawn, and why the fix was removal rather than repair.**

``diagnostics.compute_confidence_interval`` put a normal-approximation interval
around a *weighted margin*. Its standard error was ``p(1-p)/ESS`` -- the variance
of an unweighted proportion -- measured at 4.43x the margin's actual spread, so
coverage read 1.000 because the interval was four times too wide. Replacing the
variance with an honest replication estimator dropped coverage to 0.210, which
is the first defect having concealed a second.

The second is not fixable by any variance estimator. A raked margin is
calibrated toward a target the caller supplied as *known*, so its estimand is
that target: R's ``survey`` reports a standard error of exactly **zero** for a
raking variable, and a real one for an outcome estimated under the same weights.
There is no sampling uncertainty about a number you supplied.

``streaming_inference.compute_confidence_sequence`` was the same quantity with a
Hoeffding-style width instead of a normal one, documented as "time-uniform ...
valid at all stopping times" and citing a betting construction it did not
implement. Measured anytime coverage 0.470. A margin does not acquire sampling
uncertainty by being watched over time.

Both are replaced by ``diagnostics.margin_calibration``, which reports what is
actually true: the gap between margin and target -- exact arithmetic, not an
estimate -- and its size relative to the margin's own run-to-run spread.

**Anytime validity is not fixed-n coverage, and testing it the fixed-n way is
worse than not testing it.** The withdrawn sequence read 1.000 at a fixed time
and 0.470 over the path, on the same 100 replicates of the same stream. Had the
study been written the obvious way it would have certified a broken function.

**Every tolerance comes from the replicate count.** The gates are simcheck's, so
raising ``SIMCHECK_REPS`` tightens every assertion here without editing a line.

**Every study has a case that fails.** ``assert_unbiased`` passing on the GREG
estimator would mean nothing if the gate could not fail, so the unweighted mean
of the same biased sample goes through the same gate and must be caught. The
GREG interval gets an interval shrunk to 80% width that its coverage floor must
reject -- and that control earned its place: it did not fire on first run, which
is how ``_z_score`` was found to be returning the 0.95 multiplier for every
level outside its three-entry table.
"""

from __future__ import annotations

import functools
import itertools

import numpy as np
import pytest
from simcheck import (
    GATE_SIGMAS,
    MonteCarloResult,
    assert_se_calibrated,
    assert_unbiased,
    binomial_band,
    reps_for,
)

from onlinerake import Targets
from onlinerake.diagnostics import (
    _z_score,
)
from onlinerake.model_assisted import (
    ModelAssistedRaker,
    ModelAssistedTargets,
    model_assisted_confidence_interval,
    model_assisted_std_error,
)
from onlinerake.models import LinearOutcomeModel

# ----------------------------------------------------------------------------
# The population, and the reason the sample needs raking at all
# ----------------------------------------------------------------------------

# True population proportions. These are also the targets handed to the raker:
# raking calibrates to margins that are known, so the estimand of a raked margin
# *is* the target, and a study of its interval is a study of whether the
# interval covers the number the raker was aimed at.
POPULATION = {"female": 0.52, "college": 0.35, "young": 0.30}

# Log-odds of being sampled, per feature. Without this the raw sample is already
# unbiased and the raker has nothing to correct, which would make every study
# below a study of an estimator that was never wrong to begin with.
SELECTION = {"female": 0.4, "college": 1.0, "young": -0.5}

# The feature every study reports on. It is the one the selection model pushes
# hardest (coefficient 1.0), so it is the margin raking has the most work to do
# on: the raw sample proportion runs near 0.50 against a target of 0.35.
FEATURE = "college"
TRUTH = POPULATION[FEATURE]

# Strength multiplier on SELECTION. Held fixed across every study in this file
# so that no result below can be attributed to a data-generating process tuned
# per test.
STRENGTH = 2.0

CONFIDENCE = 0.95


def _stream(rng: np.random.Generator, n: int) -> list[dict[str, int]]:
    """Draw ``n`` observations from the population under selection bias.

    Args:
        rng: Generator for this replicate.
        n: Number of accepted observations.

    Returns:
        list of dict: Observations in arrival order, each a mapping from
        feature name to 0/1.
    """
    accepted: list[dict[str, int]] = []
    while len(accepted) < n:
        obs = {name: int(rng.random() < p) for name, p in POPULATION.items()}
        logit = STRENGTH * sum(SELECTION[name] * obs[name] for name in POPULATION)
        if rng.random() < 1.0 / (1.0 + np.exp(-logit)):
            accepted.append(obs)
    return accepted


def assert_coverage_floor(
    covered: np.ndarray, nominal: float, label: str, sigmas: float = GATE_SIGMAS
) -> None:
    """Fail if a coverage rate falls below its binomial band, one-sided.

    ``assert_coverage`` bands the rate on *both* sides of nominal, which is the
    right gate for an interval claiming exactly its level and the wrong one for
    an anytime-valid sequence: a time-uniform bound is entitled to be
    conservative, and a two-sided gate would fail a correct sequence at the deep
    tier for covering too often. What the claim promises is the floor, so the
    floor is what is checked. The band still comes from ``binomial_band``, so
    the tolerance is the replicate count and not a number chosen here.

    Conservatism is not thereby exempted from scrutiny -- it is measured
    directly by ``test_the_reported_standard_error_matches_the_estimator_spread``
    and by the width study, which is what a one-sided coverage gate cannot see.

    Args:
        covered: One boolean per replicate.
        nominal: The level the intervals claim.
        label: Included in the failure message.
        sigmas: Slack, in binomial standard errors.

    Raises:
        AssertionError: If the observed rate is below the band.
    """
    reps = len(covered)
    observed = float(np.mean(covered))
    low, _ = binomial_band(nominal, reps, sigmas)
    if observed < low:
        raise AssertionError(
            f"{label}: coverage {observed:.3f} over {reps} replicates is below "
            f"the {sigmas:g}-sigma floor of {low:.3f} for a nominal "
            f"{nominal:.2f} interval"
        )


FEATURE_ORDER = sorted(POPULATION)


def _outcome_probability(obs: dict[str, int]) -> float:
    """Probability the outcome is 1 for this observation.

    Args:
        obs: Feature values.

    Returns:
        float: ``P(y = 1 | x)``.
    """
    return OUTCOME_BASE + sum(OUTCOME_COEF[name] * obs[name] for name in FEATURE_ORDER)


OUTCOME_COEF = {"female": 0.10, "college": 0.25, "young": -0.15}
OUTCOME_BASE = 0.40
FEATURE_ORDER = sorted(POPULATION)


OUTCOME_BASE = 0.40
FEATURE_ORDER = sorted(POPULATION)


def _population_cells() -> list[tuple[dict[str, int], float]]:
    """Every feature combination with its exact population probability.

    Three binary features give eight cells, so the population mean outcome and
    the population mean prediction are both computed exactly rather than
    estimated from a large draw. A GREG study whose ``truth`` and whose
    ``tau_pred`` were themselves simulated would confound the estimator's bias
    with the error in its own target.

    Returns:
        list of tuple: ``(features, probability)`` per cell.
    """
    cells = []
    for bits in itertools.product((0, 1), repeat=len(FEATURE_ORDER)):
        obs = dict(zip(FEATURE_ORDER, bits, strict=True))
        probability = 1.0
        for name in FEATURE_ORDER:
            probability *= POPULATION[name] if obs[name] else 1.0 - POPULATION[name]
        cells.append((obs, probability))
    return cells


POPULATION_MEAN_OUTCOME = sum(
    probability * _outcome_probability(obs) for obs, probability in _population_cells()
)


@functools.cache
def _fitted_model() -> LinearOutcomeModel:
    """A model fitted once, in batch, on an unbiased draw.

    The package's own framing is that model fitting is batch and calibration is
    streaming, so the model is fixed across replicates. That also makes
    ``tau_pred`` exact: with the model fixed, the population mean of its
    predictions is a sum over the eight cells.

    Returns:
        LinearOutcomeModel: The fitted model.
    """
    rng = np.random.default_rng(12345)
    features = np.array(
        [
            [int(rng.random() < POPULATION[name]) for name in FEATURE_ORDER]
            for _ in range(4000)
        ],
        dtype=float,
    )
    outcomes = np.array(
        [
            rng.random()
            < _outcome_probability(dict(zip(FEATURE_ORDER, row, strict=True)))
            for row in features
        ],
        dtype=float,
    )
    return LinearOutcomeModel().fit(features, outcomes)


@functools.cache
def run_greg_study(
    n: int = 600, reps: int | None = None, seed: int = 0
) -> tuple[MonteCarloResult, MonteCarloResult, MonteCarloResult]:
    """Run the GREG study and the two estimators it is compared against.

    Cached: three tests read the same study, and each replicate is a
    600-observation streaming fit.

    Args:
        n: Stream length.
        reps: Replicate count; defaults to the current simcheck tier.
        seed: Seed for the replicate stream.

    Returns:
        tuple of MonteCarloResult: The GREG estimate, the raked weighted mean
        outcome, and the unweighted sample mean, all against the same truth.
    """
    reps = reps_for() if reps is None else reps
    model = _fitted_model()
    tau_pred = float(
        sum(
            probability
            * model.predict(
                np.array([[obs[name] for name in FEATURE_ORDER]], dtype=float)
            )[0]
            for obs, probability in _population_cells()
        )
    )

    greg, weighted, unweighted = [], [], []
    for child in np.random.SeedSequence(seed).spawn(reps):
        rng = np.random.default_rng(child)
        raker = ModelAssistedRaker(
            ModelAssistedTargets(Targets(**POPULATION), {"y_hat": tau_pred}),
            model,
        )
        outcomes = []
        for obs in _stream(rng, n):
            outcome = int(rng.random() < _outcome_probability(obs))
            outcomes.append(outcome)
            raker.partial_fit(obs, outcome=outcome)
        greg.append(raker.model_assisted_estimate)
        weighted.append(raker.weighted_mean_outcome)
        unweighted.append(float(np.mean(outcomes)))

    def wrap(values: list[float]) -> MonteCarloResult:
        array = np.asarray(values, dtype=float)
        return MonteCarloResult(
            estimates=array,
            standard_errors=np.full(array.size, np.nan),
            covered=None,
            rejected=None,
            truth=POPULATION_MEAN_OUTCOME,
        )

    return wrap(greg), wrap(weighted), wrap(unweighted)


class TestModelAssistedEstimate:
    """Does the GREG estimator recover the population mean outcome?

    Measured verdict: **yes**. Bias +0.00112 over 100 replicates of a
    600-observation biased stream, 0.59 Monte Carlo standard errors from zero,
    against an unweighted sample mean of the same data that is off by +0.0489 at
    27.9 standard errors.

    This is the only one of the three claims in this file that survives its
    study, and the negative control is what makes that statement mean something:
    the same gate, over the same replicates, on the estimator the GREG
    adjustment is supposed to improve on.
    """

    def test_the_greg_estimate_is_unbiased(self):
        """The claim, measured: bias +0.00112 at 0.59 Monte Carlo SEs.

        ``model_assisted_estimate`` is
        ``weighted_mean_outcome + (tau_pred - weighted_mean_prediction)``, the
        difference estimator with the model's prediction as the auxiliary
        variable. With the auxiliary total known exactly, the adjustment removes
        what the calibration weights left behind.
        """
        greg, _, _ = run_greg_study()
        assert_unbiased(greg, "GREG model-assisted estimate")

    def test_the_unweighted_mean_of_the_same_sample_is_caught(self):
        """The falsification test. Without it the test above proves nothing.

        The sample over-represents college graduates, whose outcome probability
        is 0.25 higher, so the unweighted mean must be biased upward -- measured
        +0.0489, 27.9 Monte Carlo standard errors out. If this ever stops failing,
        the selection model has stopped biasing the sample and every study in
        this file is running on data that never needed raking.
        """
        _, _, unweighted = run_greg_study()
        with pytest.raises(AssertionError):
            assert_unbiased(unweighted, "unweighted sample mean")
        assert unweighted.bias > 0, (
            "over-sampling the high-outcome group must bias the unweighted mean "
            f"upward; measured {unweighted.bias:+.5f}"
        )

    def test_the_greg_adjustment_moves_the_estimate_toward_the_truth(self):
        """The adjustment is doing the work, not the calibration weights alone.

        The raked weighted mean outcome -- the same estimator without the GREG
        term -- is off by +0.0068, 3.60 Monte Carlo standard errors, against the
        +0.00112 and 0.59 the adjustment achieves on identical replicates. The
        assertion is on the ordering rather than on the weighted mean failing
        its own gate: 3.60 sits only just past the 3-sigma line, close enough
        that the verdict would turn on the seed.
        """
        greg, weighted, _ = run_greg_study()
        assert abs(greg.bias) < abs(weighted.bias), (
            "the GREG adjustment should reduce bias against the plain weighted "
            f"mean; measured {greg.bias:+.5f} against {weighted.bias:+.5f}"
        )


# ─── The GREG interval, which did not exist until now ──────────────────


@functools.cache
def run_greg_interval_study(
    method: str = "random_groups",
    n: int = 600,
    reps: int | None = None,
    seed: int = 0,
) -> MonteCarloResult:
    """Coverage of the interval around the GREG model-assisted estimate.

    The estimate itself was already studied by :func:`run_greg_study` and is the
    one claim in this file that survived. What had no study is its *interval*,
    because until now the package reported no standard error for it at all.

    This is the quantity that genuinely carries sampling uncertainty. A raked
    margin does not: after calibration it is the target it was aimed at, which
    is why R's ``survey`` reports a standard error of exactly zero for one and a
    real standard error for an outcome estimated under the same weights.

    Args:
        method: Replication scheme, ``"random_groups"`` or ``"jackknife"``.
        n: Stream length.
        reps: Replicate count; defaults to the current simcheck tier.
        seed: Seed for the replicate stream.

    Returns:
        MonteCarloResult: Estimates, reported standard errors and coverage flags.
    """
    reps = reps_for() if reps is None else reps
    model = _fitted_model()
    tau_pred = float(
        sum(
            probability
            * model.predict(
                np.array([[obs[name] for name in FEATURE_ORDER]], dtype=float)
            )[0]
            for obs, probability in _population_cells()
        )
    )

    estimates, errors, covered = [], [], []
    for replicate, child in enumerate(np.random.SeedSequence(seed).spawn(reps)):
        rng = np.random.default_rng(child)
        raker = ModelAssistedRaker(
            ModelAssistedTargets(Targets(**POPULATION), {"y_hat": tau_pred}),
            model,
        )
        for obs in _stream(rng, n):
            raker.partial_fit(
                obs, outcome=int(rng.random() < _outcome_probability(obs))
            )
        # Forward a per-replicate seed: the estimators group observations into
        # replicates randomly, and leaving them at their shared default meant
        # every Monte Carlo replicate reused one grouping, so the spread these
        # measure omitted the grouping's own contribution.
        lower, upper = model_assisted_confidence_interval(
            raker, CONFIDENCE, method=method, seed=seed + replicate
        )
        estimates.append(raker.model_assisted_estimate)
        errors.append(
            model_assisted_std_error(raker, method=method, seed=seed + replicate)
        )
        covered.append(bool(lower <= POPULATION_MEAN_OUTCOME <= upper))

    return MonteCarloResult(
        estimates=np.asarray(estimates, dtype=float),
        standard_errors=np.asarray(errors, dtype=float),
        covered=np.asarray(covered, dtype=bool),
        rejected=None,
        truth=POPULATION_MEAN_OUTCOME,
    )


class TestModelAssistedInterval:
    """Does the new GREG interval cover, and is its standard error honest?

    The estimate was already known to be unbiased. An interval can still fail in
    two ways around an unbiased centre -- a standard error that does not match
    the estimator's spread, and a coverage rate that misses -- so both are gated
    rather than one being inferred from the other.
    """

    def test_the_reported_standard_error_matches_the_estimator_spread(self) -> None:
        """The interval's width has to be the right size, not merely non-zero."""
        study = run_greg_interval_study()
        assert_se_calibrated(study, label="GREG standard error, random groups")

    def test_the_interval_covers_at_its_nominal_rate(self) -> None:
        """The registered claim for the quantity that deserves one."""
        study = run_greg_interval_study()
        assert_coverage_floor(study.covered, CONFIDENCE, "GREG interval, random groups")

    def test_the_jackknife_scheme_also_covers(self) -> None:
        """Both replication schemes are offered, so both are gated."""
        study = run_greg_interval_study(method="jackknife")
        assert_coverage_floor(study.covered, CONFIDENCE, "GREG interval, jackknife")

    def test_an_interval_shrunk_below_its_level_is_caught(self) -> None:
        """The negative control, without which the two gates above prove nothing.

        The same replicates, with each interval shrunk to the width an 80%
        interval would have. The ratio is derived from the normal quantiles
        rather than picked, so the control cannot drift.
        """
        study = run_greg_interval_study()
        shrink = _z_score(0.80) / _z_score(CONFIDENCE)
        half = shrink * _z_score(CONFIDENCE) * study.standard_errors
        narrow = np.abs(study.estimates - POPULATION_MEAN_OUTCOME) <= half
        with pytest.raises(AssertionError):
            assert_coverage_floor(narrow, CONFIDENCE, "deliberately narrow")


# ─── The premise behind the "auto" default ─────────────────────────────


@functools.cache
def run_paired_scheme_study(
    n: int = 300, streams: int = 120, seed: int = 11
) -> tuple[np.ndarray, np.ndarray, float]:
    """Both schemes' margin SE on identical streams, plus the margin's spread.

    Paired on the stream: the two schemes see the same raker, so
    stream-to-stream variation cancels in the difference instead of drowning it.

    ``n=300`` puts the replicates at ``n/G = 30``, the smallest cell in
    :func:`~onlinerake.diagnostics.resolve_replication_method`'s table.

    **The stream count is sized from a measurement in this population**, not
    from the two-margin one behind that table -- the gap is process-specific and
    comes out at +9.3% here against +14.4% there. Over 250 streams the paired
    difference measured +0.000605 +/- 0.000065, which is 9.3 sigma, implying 26
    streams for a 3-sigma gate. The default is 120 rather than 26 because the
    per-stream difference is heavy-tailed -- it is negative on 22% of streams --
    so its standard error is itself unstable at small counts. A first attempt at
    40 streams read 2.1 sigma and failed this gate, against 3.7 expected. 120
    gives about 6 sigma of headroom, so the gate reports the effect being gone
    rather than the run being unlucky.

    Args:
        n: Stream length.
        streams: Number of independent streams.
        seed: Base seed.

    Returns:
        tuple: Random-groups SEs, jackknife SEs, and the observed standard
        deviation of the margin itself across the same streams.
    """
    from onlinerake import OnlineRakingSGD
    from onlinerake.diagnostics import estimate_margin_std_error

    random_groups, jackknife, margins = [], [], []
    for s in range(streams):
        raker = OnlineRakingSGD(Targets(**POPULATION))
        for obs in _stream(np.random.default_rng(seed + s), n):
            raker.partial_fit(obs)
        margins.append(raker.margins[FEATURE])
        # Same seed for both schemes on a given stream -- that is what keeps
        # the comparison paired -- but varying across streams, so the measured
        # difference averages over groupings instead of resting on one.
        random_groups.append(
            estimate_margin_std_error(
                raker, FEATURE, method="random_groups", seed=seed + s
            )
        )
        jackknife.append(
            estimate_margin_std_error(raker, FEATURE, method="jackknife", seed=seed + s)
        )
    return (
        np.array(random_groups),
        np.array(jackknife),
        float(np.std(margins, ddof=1)),
    )


class TestTheAutoDefaultRestsOnAMeasuredGap:
    """``auto`` picks jackknife at small replicate sizes. This is why.

    A default justified only by a docstring is a default nothing checks. If the
    gap these tests measure ever closes -- a change to the raker, the scaling,
    or the grouping -- the reason for the rule is gone and it should be revisited
    rather than left standing on a stale measurement.
    """

    def test_jackknife_reports_a_larger_standard_error(self) -> None:
        """The registered direction, gated on the paired difference."""
        random_groups, jackknife, _ = run_paired_scheme_study()
        difference = jackknife - random_groups
        standard_error = difference.std(ddof=1) / np.sqrt(len(difference))
        assert difference.mean() > GATE_SIGMAS * standard_error, (
            f"paired jackknife - random_groups is {difference.mean():+.6f} "
            f"+/- {standard_error:.6f}, which does not clear {GATE_SIGMAS} "
            "sigma; the measurement behind the auto default no longer holds"
        )

    def test_random_groups_understates_the_margins_own_spread(self) -> None:
        """The direction that makes the gap a defect rather than a preference.

        Two schemes differing proves nothing on its own -- one of them has to be
        wrong. The margin's observed spread across the same streams is the
        arbiter, and it is random groups that falls short of it.
        """
        random_groups, jackknife, observed = run_paired_scheme_study()
        assert random_groups.mean() < observed, (
            f"random groups reported {random_groups.mean():.5f} against an "
            f"observed spread of {observed:.5f}; it is supposed to understate"
        )
        assert abs(jackknife.mean() - observed) < abs(
            random_groups.mean() - observed
        ), (
            f"jackknife ({jackknife.mean():.5f}) is supposed to sit closer to "
            f"the observed spread ({observed:.5f}) than random groups "
            f"({random_groups.mean():.5f})"
        )
