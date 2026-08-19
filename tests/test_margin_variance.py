"""Unit tests for the replication variance estimator.

Whether the numbers it produces are *right* is a question only a replicate loop
can answer, and ``test_inference_coverage.py`` answers it: over 100 streams of
500 observations the reported standard error is 0.966 of the estimator's
observed spread. What is left for this file is the contract around it -- that
running the replicates does not disturb the raker they were cloned from, that
both schemes are available, that the arguments are checked, and that the answer
does not move between two calls on the same raker.
"""

from __future__ import annotations

import numpy as np
import pytest

from onlinerake import OnlineRakingMWU, OnlineRakingSGD, Targets
from onlinerake.diagnostics import (
    AUTO_REPLICATE_SIZE,
    _group_assignment,
    _replicate_margins,
    _unfitted_copy,
    estimate_margin_std_error,
    estimate_margin_variance,
    margin_calibration,
    resolve_replication_method,
)

TARGETS = {"female": 0.52, "college": 0.35, "young": 0.30}


def _fitted(cls=OnlineRakingSGD, n: int = 80, seed: int = 0):
    """Fit a raker on a biased stream.

    Args:
        cls: Raker class to instantiate.
        n: Number of observations.
        seed: Seed for the stream.

    Returns:
        The fitted raker.
    """
    rng = np.random.default_rng(seed)
    raker = cls(Targets(**TARGETS))
    for _ in range(n):
        raker.partial_fit(
            {
                "female": int(rng.random() < 0.45),
                "college": int(rng.random() < 0.55),
                "young": int(rng.random() < 0.25),
            }
        )
    return raker


class TestReplicationLeavesTheRakerAlone:
    """The replicates are clones; the fitted raker must come back untouched.

    A replicate is built with ``copy.copy``, which shares every attribute with
    the original until the arrays are copied explicitly. Getting that wrong
    would silently overwrite the fitted weights with a replicate's, and every
    margin the caller read afterwards would be wrong -- without raising
    anything.
    """

    def test_state_is_identical_before_and_after(self):
        """Weights, features, count and margins all survive the call."""
        raker = _fitted()
        weights = raker.weights.copy()
        features = raker._features[: raker._n_obs].copy()
        n_obs = raker._n_obs
        margins = raker.margins
        history = len(raker.history)

        estimate_margin_variance(raker, "college")

        assert raker._n_obs == n_obs
        np.testing.assert_array_equal(raker.weights, weights)
        np.testing.assert_array_equal(raker._features[: raker._n_obs], features)
        assert raker.margins == margins
        assert len(raker.history) == history

    def test_the_estimate_does_not_move_between_calls(self):
        """Grouping is systematic, so there is no seed to depend on."""
        raker = _fitted()
        first = estimate_margin_variance(raker, "college")
        second = estimate_margin_variance(raker, "college")
        assert first == second


class TestBothSchemesRun:
    """Random groups and the delete-a-group jackknife are both available."""

    @pytest.mark.parametrize("method", ["random_groups", "jackknife"])
    @pytest.mark.parametrize("cls", [OnlineRakingSGD, OnlineRakingMWU])
    def test_a_positive_finite_variance_for_every_combination(self, method, cls):
        """Both schemes, on both rakers, for a raked binary margin."""
        raker = _fitted(cls)
        variance = estimate_margin_variance(raker, "college", method=method)
        assert np.isfinite(variance)
        assert variance > 0

    def test_a_continuous_margin_goes_through_the_same_path(self):
        """Replication needs no separate branch for a mean rather than a share.

        The old estimator had one: a weighted sample variance over ESS for
        continuous features and ``p(1-p)/ESS`` for binary ones. A replicate
        reports whatever the raker reports, so the two collapse into one path.
        """
        raker = OnlineRakingSGD(Targets(age=(35.0, "mean")))
        for age in (25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0):
            raker.partial_fit({"age": age})

        variance = estimate_margin_variance(raker, "age")
        assert np.isfinite(variance)
        assert variance > 0

    def test_more_groups_than_observations_are_capped(self):
        """A group per observation is the most the sample can supply."""
        raker = _fitted(n=4)
        variance = estimate_margin_variance(raker, "college", n_replicates=50)
        assert np.isfinite(variance)


class TestDegenerateSamples:
    """What comes back when there is no spread to measure."""

    def test_no_observations_gives_nan(self):
        """Nothing was estimated, so there is no variance to report."""
        raker = OnlineRakingSGD(Targets(**TARGETS))
        assert np.isnan(estimate_margin_variance(raker, "college"))
        assert np.isnan(estimate_margin_std_error(raker, "college"))

    def test_one_observation_gives_nan(self):
        """One observation cannot be split into two groups, so nothing is known.

        This asserted ``== 0.0`` and argued for it: the answer to "how much does
        this move across replicates" is that there are no replicates, not that
        the question is ill-posed. That reasoning does not survive the variance
        being used to build an interval. Zero variance is a claim that the
        estimator *cannot* vary, and it produced a zero-width 95% interval from
        a single observation -- which is exactly the
        ``proportion_confint(0, 20) -> (0.0, 0.0)`` defect this project logged
        as a finding against statsmodels.

        An unestimable variance is NaN.
        """
        raker = OnlineRakingSGD(Targets(**TARGETS))
        raker.partial_fit({"female": 1, "college": 0, "young": 1})
        assert np.isnan(estimate_margin_variance(raker, "college"))
        assert np.isnan(estimate_margin_std_error(raker, "college"))

    def test_one_observation_does_not_yield_a_zero_width_interval(self):
        """The consequence, asserted directly rather than left implied."""
        from onlinerake.model_assisted import (
            ModelAssistedRaker,
            ModelAssistedTargets,
            model_assisted_confidence_interval,
            model_assisted_variance,
        )
        from onlinerake.models import LinearOutcomeModel

        model = LinearOutcomeModel().fit(
            np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]), np.array([0.0, 1.0])
        )
        raker = ModelAssistedRaker(
            ModelAssistedTargets(Targets(**TARGETS), {"y_hat": 0.5}), model
        )
        raker.partial_fit({"female": 1, "college": 0, "young": 1}, outcome=1)

        assert np.isnan(model_assisted_variance(raker))
        low, high = model_assisted_confidence_interval(raker)
        message = (
            f"one observation produced the interval ({low}, {high}); a "
            "zero-width 95% interval claims perfect precision from a single "
            "draw"
        )
        assert np.isnan(low), message
        assert np.isnan(high), message


class TestArgumentsAreChecked:
    """Settings with no meaning are rejected rather than quietly reinterpreted."""

    def test_an_unknown_method_raises(self):
        """A typo must not fall through to whichever branch is the ``else``."""
        raker = _fitted()
        with pytest.raises(ValueError, match="method must be one of"):
            estimate_margin_variance(raker, "college", method="bootstrap")

    @pytest.mark.parametrize("n_replicates", [0, 1, -3])
    def test_fewer_than_two_groups_raises(self, n_replicates):
        """One group has no spread; zero and negative counts have no meaning."""
        raker = _fitted()
        with pytest.raises(ValueError, match="at least 2"):
            estimate_margin_variance(raker, "college", n_replicates=n_replicates)

    def test_margin_calibration_checks_them_too(self):
        """It reaches the replication directly, so it has to check for itself."""
        raker = _fitted()
        with pytest.raises(ValueError, match="method must be one of"):
            margin_calibration(raker, method="bootstrap")


class TestAutoPicksAScheme:
    """``"auto"`` is the default, so what it picks is part of the contract.

    A default that changes scheme with the data is harder to reason about than
    either fixed one. These tests are what makes it inspectable rather than
    merely documented.
    """

    def test_it_picks_jackknife_when_replicates_are_small(self):
        """Below the threshold, random groups understates; see the resolver."""
        assert resolve_replication_method(
            "auto", n_observations=300, n_replicates=10
        ) == ("jackknife")

    def test_it_picks_random_groups_when_replicates_are_large(self):
        """Above it the understatement has decayed and the cheap scheme wins."""
        assert resolve_replication_method(
            "auto", n_observations=4800, n_replicates=10
        ) == ("random_groups")

    def test_it_switches_at_the_documented_threshold(self):
        """The negative case for the two tests above.

        Without this, a resolver that ignored its arguments and returned a
        constant would pass one of them. Pinning both sides of the boundary is
        what makes them a measurement of the rule rather than of one branch.
        """
        groups = 10
        below = AUTO_REPLICATE_SIZE * groups - groups
        at = AUTO_REPLICATE_SIZE * groups
        assert resolve_replication_method("auto", below, groups) == "jackknife"
        assert resolve_replication_method("auto", at, groups) == "random_groups"

    @pytest.mark.parametrize("method", ["random_groups", "jackknife"])
    def test_an_explicit_scheme_passes_through_untouched(self, method):
        """``auto`` is opt-out. Naming a scheme must always get that scheme."""
        for n_obs in (50, 300, 4800):
            assert resolve_replication_method(method, n_obs, 10) == method

    def test_an_unknown_method_raises_here_too(self):
        """Validation happens as part of resolving, so it cannot be skipped."""
        with pytest.raises(ValueError, match="method must be one of"):
            resolve_replication_method("bootstrap", 300, 10)

    def test_the_default_dispatches_to_what_the_resolver_names(self):
        """The claim that makes the resolver honest.

        If ``auto`` computed one thing and reported another, every other test
        here would still pass. This is the one that ties them together.
        """
        raker = _fitted(n=300)
        chosen = resolve_replication_method("auto", raker._n_obs, 10)
        assert chosen == "jackknife"
        assert estimate_margin_variance(raker, "college") == (
            estimate_margin_variance(raker, "college", method=chosen)
        )
        assert estimate_margin_variance(raker, "college") != (
            estimate_margin_variance(raker, "college", method="random_groups")
        )

    def test_the_replicate_builder_refuses_an_unresolved_scheme(self):
        """The invariant the one-choke-point design rests on.

        ``_replicate_margins`` picks the subsets and ``_replication_variances``
        picks the scaling factor. If ``"auto"`` reached the first of those it
        would fall through to the jackknife subset rule while the second read
        the name again and could scale for random groups. The guard makes that
        a loud failure rather than a variance off by a factor of ``G**2``.
        """
        raker = _fitted(n=80)
        with pytest.raises(ValueError, match="expected a resolved scheme"):
            _replicate_margins(raker, "auto", 10, 0)


class TestMarginCalibration:
    """One replication pass has to give what per-feature calls would give."""

    def test_it_agrees_with_the_per_feature_call(self):
        """Running the replicates once must not change the answer.

        ``get_margin_estimates`` replicates once and reads every feature off the
        same set of refits, where calling ``estimate_margin_std_error`` per
        feature replicates once per feature. Both are deterministic and both
        replay the same subsamples, so they must agree exactly.
        """
        raker = _fitted()
        estimates = {est.feature: est for est in margin_calibration(raker)}
        for feature in raker._feature_names:
            assert estimates[feature].std_error == estimate_margin_std_error(
                raker, feature
            )


class TestTheDiagnosticsMoveOnTheRightAxes:
    """Each diagnostic is checked on BOTH axes, because one axis hid a defect.

    ``gap_ratio`` was originally documented as "below about one, the margin has
    arrived". That claim came from sweeping ``learning_rate * n_sgd_steps`` at a
    fixed stream length, where the ratio does fall neatly toward one. Sweeping
    the stream length instead shows the numerator flat and the denominator
    falling, so the ratio climbs -- 4.12, 8.11, 13.85 at n = 250, 1000, 4000 on
    an unchanged raker. A diagnostic validated on one axis is how that shipped.
    """

    @staticmethod
    def _fit(n, lr, steps, seed):
        rng = np.random.default_rng(seed)
        raker = OnlineRakingSGD(Targets(**TARGETS), learning_rate=lr, n_sgd_steps=steps)
        accepted = 0
        while accepted < n:
            obs = {
                "female": int(rng.random() < 0.51),
                "college": int(rng.random() < 0.35),
                "young": int(rng.random() < 0.40),
            }
            # Selection on college, so raking has real work to do.
            if rng.random() < (0.75 if obs["college"] else 0.35):
                raker.partial_fit(obs)
                accepted += 1
        return next(
            c
            for c in margin_calibration(raker, n_replicates=5)
            if c.feature == "college"
        )

    def test_both_diagnostics_fall_as_calibration_effort_rises(self):
        """Axis 1: more effort closes more of the gap, on either measure."""
        weak = self._fit(600, 5, 3, seed=11)
        strong = self._fit(600, 20, 10, seed=11)
        assert strong.unclosed_fraction < weak.unclosed_fraction
        assert strong.gap_ratio < weak.gap_ratio

    def test_unclosed_fraction_does_not_move_with_the_stream_length(self):
        """Axis 2: the scale-free measure stays put where the ratio does not.

        This is the property ``gap_ratio`` lacks, and the reason for adding a
        second measure rather than reinterpreting the first.
        """
        short = self._fit(400, 5, 3, seed=23)
        long = self._fit(2400, 5, 3, seed=23)

        # Same raker settings, six times the stream: the share of the initial
        # miscalibration left unclosed should be broadly unchanged.
        assert long.unclosed_fraction == pytest.approx(short.unclosed_fraction, rel=0.6)
        # while the ratio climbs, because its denominator shrinks with n.
        assert long.gap_ratio > short.gap_ratio


class TestReplicatesDoNotSeeParentData:
    """The copied arrays are cleared past the replicate's own observations.

    Every read in the package slices ``[:n]``, so leftover parent rows beyond
    that are harmless today. This pins the clearing anyway, because "harmless
    because I read every call site" stops being true the moment someone adds a
    vectorized operation that forgets the slice.
    """

    def test_the_tail_of_every_copied_array_is_zeroed(self):
        raker = OnlineRakingSGD(Targets(**TARGETS))
        rng = np.random.default_rng(5)
        for _ in range(60):
            raker.partial_fit({name: int(rng.random() < 0.5) for name in TARGETS})

        replicate = _unfitted_copy(raker)
        assert replicate._n_obs == 0
        # The parent's rows are non-zero, so a surviving tail would show up.
        assert np.any(raker._features[: raker._n_obs] != 0)
        assert not np.any(replicate._features)
        assert not np.any(replicate._weights)


def test_version_matches_pyproject():
    """One source of truth for the version.

    ``__init__.py`` used to restate the version as a literal beside
    ``pyproject.toml``, and the two drifted -- the literal still said 1.4.0
    after pyproject had moved. It now reads installed metadata, and this pins
    the two together so a bump in one without the other fails here.

    Worth knowing when this fails locally for no apparent reason: a stale
    ``onlinerake.egg-info/`` left by an older build shadows the real metadata
    and reports whatever version it was built at. Delete it and re-sync.
    """
    import tomllib
    from pathlib import Path

    import onlinerake

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    declared = tomllib.loads(pyproject.read_text())["project"]["version"]
    assert onlinerake.__version__ == declared


class TestGroupingDoesNotTrackArrivalOrder:
    """``random_groups`` must not assign by index.

    A systematic ``arange(n) % groups`` collapses onto any period in arrival
    order that shares a factor with ``groups``. With the default ten groups and
    every tenth observation positive, all of them land in one replicate: the
    spread across replicates then measures the index pattern rather than
    sampling variability, under a scheme whose name promises the opposite.
    """

    @staticmethod
    def _periodic(n=400, period=10):
        """Every ``period``-th observation is the positive one."""
        raker = OnlineRakingSGD(Targets(female=0.5))
        for i in range(n):
            raker.partial_fit({"female": 1 if i % period == 0 else 0})
        return raker

    def test_a_period_matching_the_group_count_does_not_split_perfectly(self):
        """The mechanism, asserted directly rather than through the variance.

        Under the old assignment group 0 held all 40 positives and the other
        nine held none. Any grouping that reproduces that is measuring the
        index.
        """
        raker = self._periodic()
        n = raker._n_obs
        assignment = _group_assignment(n, 10, seed=0)
        female = raker._features[:n, raker._feature_names.index("female")]

        per_group = [int(female[assignment == g].sum()) for g in range(10)]
        assert sum(per_group) == 40
        # Not all in one group, and not a single group empty of them either.
        assert max(per_group) < 40
        assert sum(1 for c in per_group if c > 0) >= 5

    def test_the_variance_no_longer_reports_the_index_pattern(self):
        """Systematic assignment reported 1.0e-2 here; randomized, ~1e-4.

        Bounded well above the randomized values and well below the systematic
        one, so it fails if the assignment ever goes back to tracking the
        index.
        """
        raker = self._periodic()
        variance = estimate_margin_variance(
            raker, "female", method="random_groups", n_replicates=10
        )
        assert variance < 5e-3, (
            f"variance {variance:.2e} is close to the 1.0e-2 the systematic "
            "assignment produced; grouping may be tracking arrival order again"
        )

    def test_the_answer_is_reproducible_but_the_seed_matters(self):
        """Randomized, not arbitrary: same seed same answer, different seed not.

        The second half is the falsifier. A grouping that ignored the seed
        would satisfy the first.
        """
        raker = self._periodic()
        first = estimate_margin_variance(raker, "female", seed=0)
        again = estimate_margin_variance(raker, "female", seed=0)
        other = estimate_margin_variance(raker, "female", seed=12345)
        assert first == again
        assert first != other


class TestMeasuringDoesNotChangeTheRaker:
    """A stateful learning-rate schedule must not be shared with replicates.

    ``_unfitted_copy`` uses ``copy.copy``, which shares every non-array
    attribute -- including ``_lr_schedule``. The three shipped schedules are
    stateless, computing from the ``t`` they are handed, so nothing showed. But
    ``LearningRateSchedule`` is public and a stateful implementation is legal,
    and sharing one meant every replicate advanced the *parent's* schedule.

    ``TestReplicationLeavesTheRakerAlone`` above asserts weights, features,
    counts and margins survive the call. It does not look at the schedule,
    which is why this went unnoticed.
    """

    @staticmethod
    def _counting_schedule():
        from onlinerake.learning_rate import LearningRateSchedule

        class Counting(LearningRateSchedule):
            """Stateful: the rate depends on how often it has been asked."""

            def __init__(self):
                self.calls = 0

            def __call__(self, t: int) -> float:
                self.calls += 1
                return 5.0 / (1 + 0.1 * self.calls)

            def get_params(self) -> dict:
                return {"type": "counting"}

        return Counting()

    def test_the_parents_schedule_is_not_advanced(self):
        """Measured before the fix: 31 calls became 301."""
        schedule = self._counting_schedule()
        raker = OnlineRakingSGD(Targets(a=0.5), learning_rate=schedule)
        for i in range(30):
            raker.partial_fit({"a": i % 2})

        calls_before = schedule.calls
        estimate_margin_variance(raker, "a")

        assert schedule.calls == calls_before, (
            f"measuring the variance advanced the raker's own schedule from "
            f"{calls_before} to {schedule.calls}"
        )

    def test_the_replicates_still_get_a_working_schedule(self):
        """The falsifier: handing replicates no schedule at all would satisfy
        the test above while changing what they fit.
        """
        schedule = self._counting_schedule()
        raker = OnlineRakingSGD(Targets(a=0.5), learning_rate=schedule)
        for i in range(30):
            raker.partial_fit({"a": i % 2})

        replicate = _unfitted_copy(raker)
        assert replicate._lr_schedule is not None
        assert replicate._lr_schedule is not schedule
        for i in range(5):
            replicate.partial_fit({"a": i % 2})
        assert replicate._lr_schedule.calls > 0
        assert schedule.calls == 30 + 1

    def test_a_stateless_schedule_is_unaffected(self):
        """The shipped schedules restart from ``t`` regardless, so the fix must
        not perturb the numbers they produce.
        """
        from onlinerake.learning_rate import PolynomialDecayLR

        def variance_with_schedule():
            raker = OnlineRakingSGD(
                Targets(a=0.5),
                learning_rate=PolynomialDecayLR(initial_lr=5.0, power=0.6),
            )
            for i in range(40):
                raker.partial_fit({"a": i % 2})
            return estimate_margin_variance(raker, "a")

        assert variance_with_schedule() == variance_with_schedule()
