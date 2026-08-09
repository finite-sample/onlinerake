"""Monte Carlo studies of the intervals onlinerake ships.

Three claims in this package are statements about sampling behaviour, and until
this file existed not one of them had been run over a replicate loop:

* ``streaming_inference.compute_confidence_sequence`` -- "time-uniform ... remain
  valid at all stopping times". The strongest claim in the package.
* ``diagnostics.compute_confidence_interval`` -- a normal-approximation 95%
  interval for a weighted margin, built on
  ``diagnostics.estimate_margin_std_error``.
* ``model_assisted.ModelAssistedRaker.model_assisted_estimate`` -- the GREG
  estimator.

What was there before checked structure only. ``test_new_features.py`` verifies
that ``lower < upper``, that both ends lie in ``[0, 1]``, and that the point
estimate sits inside its own interval -- none of which can fail for any interval
of the form ``x +/- z * se``, whatever ``se`` is.

**Anytime validity is not fixed-n coverage, and testing it the fixed-n way is
worse than not testing it.** A confidence sequence promises the interval covers
*simultaneously at every step*, so the replicate outcome has to be "was the
sequence violated anywhere along the path", not "did it cover at some chosen
n". The difference is not academic here:
``test_evaluating_the_sequence_at_one_fixed_time_hides_the_violation`` runs both
readings over the same 100 replicates of the same stream and gets 1.000 from the
fixed-time reading and 0.470 from the path reading.

**Every tolerance comes from the replicate count.** The gates are simcheck's, so
raising ``SIMCHECK_REPS`` tightens every assertion here without editing a line,
and lowering it widens the band visibly in the failure message rather than
quietly weakening the test.

**Every study has a case that fails.** ``assert_unbiased`` *passing* on the GREG
estimator would mean nothing if the gate could not fail, so the unweighted mean
of the same biased sample goes through the same gate and is required to be
caught. The coverage floor and the width check each get the same treatment, and
the fixed-n interval study gets a textbook estimator that all three gates must
leave alone.
"""

from __future__ import annotations

import functools
import itertools

import numpy as np
import pytest
from simcheck import (
    GATE_SIGMAS,
    Estimate,
    MonteCarloResult,
    assert_se_calibrated,
    assert_unbiased,
    binomial_band,
    monte_carlo,
    reps_for,
)

from onlinerake import OnlineRakingMWU, OnlineRakingSGD, Targets
from onlinerake.diagnostics import (
    _z_score,
    compute_confidence_interval,
    estimate_margin_std_error,
)
from onlinerake.model_assisted import (
    ModelAssistedRaker,
    ModelAssistedTargets,
    model_assisted_confidence_interval,
    model_assisted_std_error,
)
from onlinerake.models import LinearOutcomeModel
from onlinerake.streaming_inference import compute_confidence_sequence

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


def _log_term(t: np.ndarray | float) -> np.ndarray | float:
    """The numerator of the shipped sequence's half-width, at time ``t``.

    Reproduces ``compute_confidence_sequence``'s ``log_term`` so the width study
    can predict the shrinkage rate from the formula rather than from a number
    chosen to match what the code happened to do.

    Args:
        t: Observation count.

    Returns:
        numpy.ndarray or float: ``log(log(t + 2) + 1) + log(2 / alpha)``.
    """
    return np.log(np.log(t + 2) + 1) + np.log(2.0 / (1.0 - CONFIDENCE))


# ----------------------------------------------------------------------------
# One replicate loop, reused by every sequence study
# ----------------------------------------------------------------------------


class SequenceStudy:
    """What one Monte Carlo study of a confidence sequence recorded.

    Holding the whole path per replicate rather than a single boolean is the
    point of the class: the same replicates then answer both the anytime
    question and the fixed-time question, so the gap between the two readings in
    ``test_evaluating_the_sequence_at_one_fixed_time_hides_the_violation`` cannot
    be blamed on the two studies having seen different data.

    Attributes:
        covered_anywhere: Per replicate, whether the truth stayed inside the
            sequence at *every* recorded step.
        covered_at_midpoint: Per replicate, whether it was inside at the single
            step halfway along the stream.
        first_miss: Observation number of the first violation, or -1.
        estimates: Weighted margin at the end of the stream.
        std_errors: ``estimate_margin_std_error`` at the end of the stream.
        fixed_interval_covered: Whether ``compute_confidence_interval`` covered
            at the end of the stream.
        width_early: Sequence width at the quarter point.
        width_late: Sequence width at the end.
        t_early: Observation number at the quarter point.
        t_late: Observation number at the end.
        reps: Number of replicates.
    """

    def __init__(self, records: list[dict[str, float]]) -> None:
        """Stack the per-replicate records into arrays.

        Args:
            records: One dict per replicate, as produced by ``run_sequence_study``.
        """
        self.reps = len(records)
        for field in (
            "covered_anywhere",
            "covered_at_midpoint",
            "fixed_interval_covered",
        ):
            setattr(self, field, np.array([r[field] for r in records], dtype=bool))
        for field in (
            "first_miss",
            "estimates",
            "std_errors",
            "width_early",
            "width_late",
            "t_early",
            "t_late",
        ):
            setattr(self, field, np.array([r[field] for r in records], dtype=float))

    @property
    def anytime_coverage(self) -> float:
        """Fraction of replicates never violated at any step."""
        return float(np.mean(self.covered_anywhere))

    @property
    def fixed_time_coverage(self) -> float:
        """Fraction covering at the single midpoint step."""
        return float(np.mean(self.covered_at_midpoint))

    @property
    def margin_result(self) -> MonteCarloResult:
        """The end-of-stream weighted margin and its reported standard error.

        Returns:
            MonteCarloResult: With ``covered`` taken from
            ``compute_confidence_interval``, so bias, standard-error
            calibration and fixed-n coverage are all read off one study.
        """
        return MonteCarloResult(
            estimates=self.estimates,
            standard_errors=self.std_errors,
            covered=self.fixed_interval_covered,
            rejected=None,
            truth=TRUTH,
        )


@functools.cache
def run_sequence_study(
    raker_kind: str,
    n: int,
    reps: int | None = None,
    seed: int = 0,
) -> SequenceStudy:
    """Stream ``n`` observations ``reps`` times and record the whole sequence.

    Every raker is built at its shipped defaults. That is deliberate and it is
    what makes ``test_the_sequence_is_violated_at_the_mwu_default`` a finding
    rather than a tuning exercise: nothing here is set to a value chosen to
    produce a failure.

    Cached, because several tests read different facets of the same study and
    each one costs O(n^2) raker updates. Caching is also what lets the anytime
    and fixed-time readings be taken over provably identical replicates.

    Args:
        raker_kind: ``"sgd"`` for :class:`OnlineRakingSGD` or ``"mwu"`` for
            :class:`OnlineRakingMWU`, each at its own default learning rate.
        n: Stream length. Kept at or below the rakers' default
            ``max_history=1000`` so that the sequence covers the whole path --
            past that the history is truncated and "at any time" would silently
            mean "over the last thousand".
        reps: Replicate count; defaults to the current simcheck tier.
        seed: Seed for the replicate stream.

    Returns:
        SequenceStudy: The recorded paths.

    Raises:
        ValueError: If ``raker_kind`` is not recognised, or if ``n`` exceeds the
            history the raker keeps.
    """
    if raker_kind not in {"sgd", "mwu"}:
        raise ValueError(f"raker_kind must be 'sgd' or 'mwu', got {raker_kind!r}")
    if n > 1000:
        raise ValueError(
            f"n={n} exceeds the default max_history of 1000, so the recorded "
            "history would be truncated and the path reading would silently "
            "become a reading over the last 1000 steps"
        )
    reps = reps_for() if reps is None else reps

    records: list[dict[str, float]] = []
    for child in np.random.SeedSequence(seed).spawn(reps):
        rng = np.random.default_rng(child)
        targets = Targets(**POPULATION)
        raker = (
            OnlineRakingSGD(targets)
            if raker_kind == "sgd"
            else OnlineRakingMWU(targets)
        )
        for obs in _stream(rng, n):
            raker.partial_fit(obs)

        sequence = compute_confidence_sequence(raker, FEATURE, CONFIDENCE)
        times = np.asarray(sequence.times, dtype=float)
        lower = np.asarray(sequence.lower_bounds, dtype=float)
        upper = np.asarray(sequence.upper_bounds, dtype=float)
        missed = (lower > TRUTH) | (upper < TRUTH)

        midpoint = len(times) // 2
        quarter = len(times) // 4
        low, high = compute_confidence_interval(raker, FEATURE, CONFIDENCE)

        records.append(
            {
                "covered_anywhere": not bool(missed.any()),
                "covered_at_midpoint": not bool(missed[midpoint]),
                "fixed_interval_covered": bool(low <= TRUTH <= high),
                "first_miss": float(times[missed][0]) if missed.any() else -1.0,
                "estimates": raker.margins[FEATURE],
                "std_errors": estimate_margin_std_error(raker, FEATURE),
                "width_early": float(upper[quarter] - lower[quarter]),
                "width_late": float(upper[-1] - lower[-1]),
                "t_early": float(times[quarter]),
                "t_late": float(times[-1]),
            }
        )
    return SequenceStudy(records)


# ----------------------------------------------------------------------------
# Gates that simcheck's released surface does not yet cover
# ----------------------------------------------------------------------------


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


def assert_widths_shrink(early: np.ndarray, late: np.ndarray, label: str) -> None:
    """Fail unless the interval is measurably narrower later in the stream.

    Coverage alone cannot see this. A sequence that never closes covers
    trivially and tells you nothing, and the floor gate above would pass it
    forever. The tolerance is the Monte Carlo standard error of the *paired*
    difference over the study, so it comes from the replicate count; the pairing
    matters because both widths come from the same path.

    This is a stand-in for simcheck's width gate, which is being added there in
    parallel and is not in the released surface this file imports. When it
    lands, this function and ``assert_width_ratio_at_most`` should be deleted in
    favour of it.

    Args:
        early: Width at the earlier time, one per replicate.
        late: Width at the later time, one per replicate.
        label: Included in the failure message.

    Raises:
        AssertionError: If the widths are not measurably ordered.
    """
    gap = early - late
    mc_se = float(np.std(gap, ddof=1) / np.sqrt(len(gap)))
    if mc_se == 0.0 or float(np.mean(gap)) <= GATE_SIGMAS * mc_se:
        raise AssertionError(
            f"{label}: mean width fell only {np.mean(gap):+.6g} between the two "
            f"times over {len(gap)} replicates, which is "
            f"{float('inf') if mc_se == 0 else np.mean(gap) / mc_se:.2f} Monte "
            f"Carlo standard errors, against a gate of {GATE_SIGMAS:g}"
        )


def assert_width_ratio_at_most(
    observed: np.ndarray, predicted: np.ndarray, label: str
) -> None:
    """Fail if the width shrank measurably *slower* than the formula allows.

    ``predicted`` is computed from the shipped half-width expression under the
    assumption the width claim rests on -- that the effective sample size keeps
    up with the stream, so ``ESS(t)`` is proportional to ``t``. Nothing is
    chosen: the ratio is ``sqrt(t_early / t_late) * sqrt(L(t_late) / L(t_early))``
    with ``L`` read straight off the implementation. What the assertion has bite
    on is the ESS, which is a property of the raker and can degrade: weights
    that concentrate stop buying information and the width stops closing at the
    documented rate.

    One-sided, because shrinking faster than predicted is not a defect of the
    claim "shrinks as O(1/sqrt(t))".

    Args:
        observed: Per-replicate ratio of late width to early width.
        predicted: Per-replicate ratio implied by the formula at those times.
        label: Included in the failure message.

    Raises:
        AssertionError: If the observed ratio exceeds the predicted one by more
            than the study's Monte Carlo standard error allows.
    """
    excess = observed - predicted
    mc_se = float(np.std(excess, ddof=1) / np.sqrt(len(excess)))
    if mc_se > 0 and float(np.mean(excess)) > GATE_SIGMAS * mc_se:
        raise AssertionError(
            f"{label}: width ratio {np.mean(observed):.4f} exceeds the "
            f"{np.mean(predicted):.4f} the formula predicts at these times by "
            f"{np.mean(excess) / mc_se:.2f} Monte Carlo standard errors over "
            f"{len(excess)} replicates, so the effective sample size is not "
            "keeping up with the stream"
        )


# ----------------------------------------------------------------------------
# compute_confidence_sequence
# ----------------------------------------------------------------------------


class TestConfidenceSequenceAnytimeValidity:
    """Is the confidence sequence valid at all stopping times, as documented?

    Measured verdict: **no**. It holds over a short stream under
    :class:`OnlineRakingSGD`'s default learning rate of 5.0 and fails outright
    under :class:`OnlineRakingMWU`'s default of 1.0, which
    ``compute_confidence_sequence`` documents itself as accepting. Anytime
    coverage at ``n = 1000`` over 100 replicates is **0.470** against a nominal
    0.95.

    The mechanism is arithmetic rather than a corner case, and the two tests
    after the headline one measure each half of it:

    * the interval is centred on ``state["weighted_margins"][feature]``, whose
      distance from the target is a **calibration residual that does not shrink
      with t**. Measured at the SGD default it is +0.0161 at n=125 and +0.0146
      at n=500, a difference the study cannot resolve;
    * the half-width is ``sqrt(L(t) / (2 * ESS))`` and **does** shrink, measured
      at almost exactly ``1 / sqrt(4)`` per quadrupling of t.

    A width going to zero around a centre that stays put crosses the truth at a
    computable time and never comes back. Under SGD at its default the residual
    is +0.0146 against a half-width of 0.0818 at n=500, so the crossing is out
    near n = 16,000, which is why the short-stream study passes and why passing
    it is not evidence of validity. Under MWU the residual is +0.0523 and the
    crossing lands at n ~= 500; the first violations in this study arrive at
    observations 477, 515, 531, 595, 602 and 622.
    """

    def test_the_sequence_holds_at_the_sgd_default_over_a_short_stream(self):
        """The case the claim survives, stated with its horizon attached.

        Anytime coverage 1.000 over 100 replicates of a 500-observation stream.
        The gate is a floor rather than a band because a time-uniform bound is
        allowed to be conservative -- and this one is very conservative, which
        the width and standard-error studies below quantify rather than hide.
        """
        study = run_sequence_study("sgd", n=500)
        assert_coverage_floor(
            study.covered_anywhere,
            CONFIDENCE,
            "confidence sequence, SGD default, n=500",
        )

    def test_the_sequence_is_violated_at_the_mwu_default(self):
        """The finding. Nothing here is tuned; both rakers run at their defaults.

        ``compute_confidence_sequence``'s own docstring says it takes "a fitted
        OnlineRakingSGD or OnlineRakingMWU object", and :class:`OnlineRakingMWU`
        ships ``learning_rate=1.0``. At that setting the calibration residual
        settles at +0.0523 and stays there -- the same non-shrinking behaviour
        ``test_the_calibration_residual_does_not_shrink_with_the_stream``
        measures under SGD -- while the half-width falls through it around
        observation 500.

        Measured anytime coverage: 0.470 over 100 replicates, against a nominal
        0.95 and a 3-sigma floor of 0.885. That is twenty-two binomial standard
        errors below nominal.

        If this test ever stops failing the gate, either the sequence gained a
        term that accounts for the residual or the raker started converging, and
        the numbers in this file need to be re-measured rather than the
        assertion relaxed.
        """
        study = run_sequence_study("mwu", n=1000)
        with pytest.raises(AssertionError):
            assert_coverage_floor(
                study.covered_anywhere,
                CONFIDENCE,
                "confidence sequence, MWU default, n=1000",
            )

        low, _ = binomial_band(CONFIDENCE, study.reps)
        assert study.anytime_coverage < low, (
            "the MWU default must violate the sequence; measured "
            f"{study.anytime_coverage:.3f} against a floor of {low:.3f}"
        )
        assert (study.first_miss[study.first_miss > 0] > 100).all(), (
            "the violations should be the width crossing the residual mid-stream, "
            "not a degenerate interval in the first few observations"
        )

    def test_evaluating_the_sequence_at_one_fixed_time_hides_the_violation(self):
        """The reason the obvious version of this test would be worse than none.

        Same 100 replicates, same stream, same intervals. Read the sequence at
        one chosen observation -- the halfway point, 502 -- and coverage is
        **1.000**, comfortably inside any band. Read the whole path, which is
        what "valid at all stopping times" means, and coverage is **0.470**.

        A fixed-n coverage test of this function would therefore have passed,
        and gone on passing, while the claim it was supposed to be checking was
        false by a factor of two.
        """
        study = run_sequence_study("mwu", n=1000)

        assert_coverage_floor(
            study.covered_at_midpoint,
            CONFIDENCE,
            "confidence sequence read at one fixed time, MWU default",
        )
        with pytest.raises(AssertionError):
            assert_coverage_floor(
                study.covered_anywhere,
                CONFIDENCE,
                "confidence sequence read over the whole path, MWU default",
            )
        assert study.fixed_time_coverage > study.anytime_coverage, (
            "the fixed-time reading must be the more forgiving one, or this "
            f"test is not demonstrating anything: {study.fixed_time_coverage:.3f} "
            f"vs {study.anytime_coverage:.3f}"
        )

    def test_the_calibration_residual_does_not_shrink_with_the_stream(self):
        """Half of the mechanism: the centre of the interval stays put.

        The weighted margin sits +0.0161 above the target after 125
        observations and +0.0146 after 500. The gap between the two is 1.3
        Monte Carlo standard errors, which this study cannot resolve from zero
        -- so as far as it can measure, four times the data buys no reduction in
        the residual at all.

        This is not a claim that the raker is broken. It is the observation that
        the quantity the sequence is centred on is not consistent for the
        target, which is what makes a width shrinking to zero around it fatal.
        """
        short = run_sequence_study("sgd", n=125)
        long_ = run_sequence_study("sgd", n=500)

        for study, label in ((short, "n=125"), (long_, "n=500")):
            with pytest.raises(AssertionError):
                assert_unbiased(study.margin_result, f"raked margin, SGD, {label}")

        gap = short.margin_result.bias - long_.margin_result.bias
        mc_se = float(np.hypot(short.margin_result.mc_se, long_.margin_result.mc_se))
        assert abs(gap) < GATE_SIGMAS * mc_se, (
            "quadrupling the stream should not measurably reduce the residual; "
            f"it moved {gap:+.6f}, which is {gap / mc_se:+.2f} Monte Carlo "
            "standard errors"
        )

    def test_the_width_shrinks_at_the_documented_rate(self):
        """The other half: the interval does close, at the rate it claims.

        A sequence that never closes is trivially valid and useless, and the
        coverage floor cannot tell the two apart. Between the quarter point
        (t=126) and the end (t=500) the mean width falls from 0.3279 to 0.1637,
        a drop of 157 Monte Carlo standard errors, and the ratio 0.4996 sits
        just inside the 0.5116 the shipped formula predicts at those two times
        given an effective sample size that keeps pace with the stream.

        Both numbers on the right come from the implementation and from the
        study; neither is a threshold.
        """
        study = run_sequence_study("sgd", n=500)

        assert_widths_shrink(
            study.width_early, study.width_late, "confidence sequence, SGD default"
        )
        predicted = np.sqrt(study.t_early / study.t_late) * np.sqrt(
            _log_term(study.t_late) / _log_term(study.t_early)
        )
        assert_width_ratio_at_most(
            study.width_late / study.width_early,
            predicted,
            "confidence sequence, SGD default",
        )

    def test_a_sequence_that_never_closes_is_caught(self):
        """The falsification test for the width gate.

        Without it, ``test_the_width_shrinks_at_the_documented_rate`` proves
        nothing: an assertion that cannot fail turns an unmeasured property into
        one that reports itself as measured. A constant width is exactly the
        vacuous sequence the width study exists to exclude.
        """
        constant = np.full(reps_for(), 0.4)
        with pytest.raises(AssertionError):
            assert_widths_shrink(constant, constant.copy(), "a width that never moves")

        rng = np.random.default_rng(7)
        noisy = 0.4 + 0.01 * rng.standard_normal(reps_for())
        with pytest.raises(AssertionError):
            assert_widths_shrink(
                noisy, noisy + 0.01 * rng.standard_normal(noisy.size), "noise only"
            )

    def test_a_broken_coverage_floor_is_caught(self):
        """The falsification test for the coverage floor.

        A gate that passes everything converts an untested claim into one that
        reports itself as tested, which is worse than leaving it untested. Two
        cases: an interval that never covers, and one that covers at 0.80 when
        it claims 0.95.

        The rates are constructed exactly rather than sampled, so the verdicts
        do not depend on a seed or on which tier the file is running at.
        """
        reps = reps_for()
        with pytest.raises(AssertionError):
            assert_coverage_floor(
                np.zeros(reps, dtype=bool), CONFIDENCE, "never covers"
            )

        def at_rate(rate: float) -> np.ndarray:
            hits = np.zeros(reps, dtype=bool)
            hits[: round(rate * reps)] = True
            return hits

        with pytest.raises(AssertionError):
            assert_coverage_floor(
                at_rate(0.80), CONFIDENCE, "an 80% interval sold as 95%"
            )

        # And it does not fire on a rate that really is nominal.
        assert_coverage_floor(at_rate(CONFIDENCE), CONFIDENCE, "a genuine 95% interval")


# ----------------------------------------------------------------------------
# diagnostics.compute_confidence_interval
# ----------------------------------------------------------------------------


class TestMarginConfidenceInterval:
    """Does the normal-approximation interval for a weighted margin hold up?

    Measured verdict: **the standard error does now; the interval still does
    not.** The two halves used to pull in opposite directions and partly cancel,
    which is why the structural tests never saw either.

    * ``estimate_margin_std_error`` re-runs the calibration on each of ten
      disjoint groups of the stream and reads the spread off the group
      estimates. Measured over 100 replicates at n=500: 0.00510 reported against
      an actual spread of 0.00528, a ratio of **0.966**. It used to return
      ``sqrt(p(1-p)/ESS)``, the standard error of an unweighted proportion at
      the effective sample size, which measured 0.02340 -- a ratio of **4.43**
      for a margin that had been *calibrated to a fixed target* and so varied
      far less than a proportion does.
    * the margin itself still carries a calibration residual of **+0.0146**,
      which is 2.8 times its own sampling spread. Nothing in ``diagnostics``
      can move that: it is the tracking lag of the raker's fixed-gain update,
      and it is not smaller after 1000 observations than after 125.

    A centre 2.8 sampling standard deviations from its estimand, inside an
    interval 4.4 times wider than it should be, covered **1.000** of the time --
    a measurement of the width rather than of the estimator. With the width
    honest the cancellation is gone and the same study measures **0.210**.
    ``test_the_interval_covers_at_least_its_nominal_rate`` is left exactly as it
    was written, and failing, because that is the finding: closing the gap means
    making the raker converge, which is a change to the raking algorithm and not
    to this module.
    """

    def test_the_interval_covers_at_least_its_nominal_rate(self):
        """Coverage 1.000 over 100 replicates, which is not the good news.

        The floor is the only part of this that can be asserted tier-independently:
        a two-sided band would pass at 100 replicates, where 1.000 is inside it,
        and fail at 400, where it is not. The two tests after this one are the
        ones that say what the 1.000 is made of.
        """
        study = run_sequence_study("sgd", n=500)
        assert_coverage_floor(
            study.fixed_interval_covered,
            CONFIDENCE,
            "diagnostics.compute_confidence_interval, SGD default, n=500",
        )
        assert study.fixed_interval_covered.all(), (
            "this study is expected to see no misses at all; if it now sees "
            "some, re-measure the numbers in this class rather than editing "
            "the assertion"
        )

    def test_the_reported_standard_error_matches_the_estimator_spread(self):
        """The gate the replication estimator was written to pass.

        ``assert_se_calibrated`` compares the standard error the package reports
        against the spread the estimator actually has across replicates.
        Measured ratio **0.966** -- 0.00510 reported against an observed 0.00528
        -- inside the 0.222 the replicate count allows.

        Before the replication estimator this test was written the other way
        round, with ``pytest.raises``, and recorded a ratio of **4.43**:
        ``estimate_margin_std_error`` returned ``sqrt(p(1-p)/ESS)``, the standard
        error of an unweighted proportion, for a margin that had been calibrated
        to a fixed target and so varied far less than one.

        The assertion here is simcheck's, so raising ``SIMCHECK_REPS`` narrows
        the band this has to sit in without editing a line.
        """
        study = run_sequence_study("sgd", n=500)
        assert_se_calibrated(study.margin_result, "diagnostics margin standard error")

    def test_the_calibrated_margin_is_not_centred_on_its_target(self):
        """The other half: a bias of +0.0146, 27 Monte Carlo standard errors out.

        Pinned rather than fixed, in the same way as the sequence result. The
        interval cannot be assessed without it: an interval whose centre is
        systematically off does not become correct by being wide.
        """
        study = run_sequence_study("sgd", n=500)
        with pytest.raises(AssertionError):
            assert_unbiased(study.margin_result, "raked margin, SGD default, n=500")
        assert study.margin_result.bias > 0, (
            "the residual should sit above the target, the direction the raw "
            f"sample is biased in; measured {study.margin_result.bias:+.5f}"
        )

    def test_the_gates_pass_on_a_textbook_estimator(self):
        """Positive control: none of the three gates above fires on a correct one.

        The sample proportion of a Bernoulli draw with its textbook standard
        error and normal interval. If ``assert_unbiased``, ``assert_se_calibrated``
        or the coverage floor fired here, the failures recorded above would be
        evidence about this file rather than about onlinerake.
        """
        n = 500

        def replicate(rng: np.random.Generator) -> Estimate:
            draws = rng.random(n) < TRUTH
            p_hat = float(np.mean(draws))
            se = float(np.sqrt(p_hat * (1 - p_hat) / n))
            return Estimate(p_hat, se, p_hat - 1.96 * se, p_hat + 1.96 * se)

        result = monte_carlo(replicate, truth=TRUTH, reps=reps_for(), seed=11)
        assert_unbiased(result, "unweighted sample proportion")
        assert_se_calibrated(result, "unweighted sample proportion")
        assert_coverage_floor(
            result.covered, CONFIDENCE, "unweighted sample proportion"
        )


# ----------------------------------------------------------------------------
# model_assisted.ModelAssistedRaker.model_assisted_estimate
# ----------------------------------------------------------------------------

# Outcome model: P(y = 1 | x) linear in the three binary features. Linear in the
# features means LinearOutcomeModel can represent it exactly, so nothing below
# turns on model misspecification -- the study is of the GREG adjustment, not of
# the model.
OUTCOME_COEF = {"female": 0.10, "college": 0.25, "young": -0.15}
OUTCOME_BASE = 0.40
FEATURE_ORDER = sorted(POPULATION)


def _outcome_probability(obs: dict[str, int]) -> float:
    """Probability the outcome is 1 for this observation.

    Args:
        obs: Feature values.

    Returns:
        float: ``P(y = 1 | x)``.
    """
    return OUTCOME_BASE + sum(OUTCOME_COEF[name] * obs[name] for name in FEATURE_ORDER)


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
    for child in np.random.SeedSequence(seed).spawn(reps):
        rng = np.random.default_rng(child)
        raker = ModelAssistedRaker(
            ModelAssistedTargets(Targets(**POPULATION), {"y_hat": tau_pred}),
            model,
        )
        for obs in _stream(rng, n):
            raker.partial_fit(
                obs, outcome=int(rng.random() < _outcome_probability(obs))
            )
        lower, upper = model_assisted_confidence_interval(
            raker, CONFIDENCE, method=method
        )
        estimates.append(raker.model_assisted_estimate)
        errors.append(model_assisted_std_error(raker, method=method))
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
