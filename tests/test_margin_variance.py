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
    estimate_margin_std_error,
    estimate_margin_variance,
    margin_calibration,
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
        assert np.isnan(low) and np.isnan(high), (
            f"one observation produced the interval ({low}, {high}); a "
            "zero-width 95% interval claims perfect precision from a single "
            "draw"
        )


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
        return [
            c
            for c in margin_calibration(raker, n_replicates=5)
            if c.feature == "college"
        ][0]

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
