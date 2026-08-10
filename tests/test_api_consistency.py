"""One name per concept, one order per pair, across the public API.

These are cheap to write and cheap to run, and they catch the class of defect
that no behavioural test can: a function that works perfectly but disagrees with
its neighbours about what an argument is called or where it goes. That
disagreement is only visible if you read the whole API at once, which nobody
does.

The concrete thing this guards against was live before 2.0.0.
``optimal_mwu_learning_rate`` took ``(n_obs, n_features)`` while
``mwu_convergence_analysis`` and ``theoretical_convergence_bound`` took
``(n_features, n_observations)``. Both arguments are counts, so a positional
call with the pair reversed raised nothing and returned a plausible number:
``optimal_mwu_learning_rate(1000, 4)`` gives 0.2351 and the swap gives 5.0, a
21x different learning rate with no error.
"""

from __future__ import annotations

import inspect

import pytest

import onlinerake

PUBLIC_FUNCTIONS = [
    (name, obj)
    for name in onlinerake.__all__
    if inspect.isfunction(obj := getattr(onlinerake, name))
]


def _params(func):
    return list(inspect.signature(func).parameters)


class TestOneNamePerConcept:
    """The same quantity is spelled the same way everywhere."""

    @pytest.mark.parametrize("name,func", PUBLIC_FUNCTIONS)
    def test_observation_counts_are_called_n_observations(self, name, func):
        """``n_obs`` and ``n_observations`` both meant the count of rows."""
        assert "n_obs" not in _params(func), (
            f"{name} spells the observation count 'n_obs'; the rest of the "
            "package uses 'n_observations'"
        )

    @pytest.mark.parametrize("name,func", PUBLIC_FUNCTIONS)
    def test_no_single_letter_parameters(self, name, func):
        """``verify_robbins_monro`` took ``T``, alone in the package."""
        short = [p for p in _params(func) if len(p) == 1]
        assert not short, f"{name} has single-letter parameter(s) {short}"

    @pytest.mark.parametrize("name,func", PUBLIC_FUNCTIONS)
    def test_tolerance_says_what_it_is_a_tolerance_on(self, name, func):
        """A bare ``tolerance`` meant a squared-error loss in one function and a
        margin distance in another -- 1e-6 against 0.05, four orders apart under
        one name, so a value carried between them is nonsense.
        """
        assert "tolerance" not in _params(func), (
            f"{name} takes a bare 'tolerance'; name the quantity it bounds "
            "(loss_tolerance, margin_tolerance)"
        )


class TestOneOrderPerPair:
    """Interchangeable-looking arguments appear in one order everywhere."""

    @pytest.mark.parametrize("name,func", PUBLIC_FUNCTIONS)
    def test_observations_precede_features(self, name, func):
        """Both are ints, so the wrong order is silent rather than a TypeError.

        Fixing the order alone would not be enough on its own -- callers can
        still pass positionally -- but it removes the case where two sibling
        functions disagree, which is where the mistake actually comes from.
        """
        params = _params(func)
        if "n_observations" in params and "n_features" in params:
            assert params.index("n_observations") < params.index("n_features"), (
                f"{name} takes n_features before n_observations; every other "
                "function in the package takes the observation count first"
            )

    @pytest.mark.parametrize("name,func", PUBLIC_FUNCTIONS)
    def test_the_raker_is_always_first(self, name, func):
        """Nineteen functions take a fitted raker. It leads in all of them."""
        params = _params(func)
        if "raker" in params:
            assert params[0] == "raker", f"{name} does not take raker first"


class TestTheSchemesAgree:
    """The six replication entry points share one vocabulary."""

    REPLICATION = [
        "estimate_margin_variance",
        "estimate_margin_std_error",
        "margin_calibration",
        "model_assisted_variance",
        "model_assisted_std_error",
        "model_assisted_confidence_interval",
    ]

    @pytest.mark.parametrize("name", REPLICATION)
    def test_same_names_and_same_defaults(self, name):
        """Differing defaults across these would be a silent behaviour change
        depending on which one the caller reached for.
        """
        sig = inspect.signature(getattr(onlinerake, name))
        assert sig.parameters["method"].default == "auto"
        assert sig.parameters["n_replicates"].default == 10


class TestTheRakersOfferTheSameSurface:
    """``OnlineRakingMWU`` subclasses ``OnlineRakingSGD``; it must not narrow it."""

    def test_mwu_accepts_everything_sgd_does(self):
        """It used to drop ``max_history``, so the parent accepted a keyword the
        child raised ``TypeError`` for.
        """
        sgd = set(inspect.signature(onlinerake.OnlineRakingSGD.__init__).parameters)
        mwu = set(inspect.signature(onlinerake.OnlineRakingMWU.__init__).parameters)
        missing = sgd - mwu
        assert not missing, f"OnlineRakingMWU does not accept {sorted(missing)}"

    def test_mwu_does_not_narrow_a_parent_annotation(self):
        """Accepting the keyword is not enough; it must accept the same values.

        The first version of the ``max_history`` fix annotated it ``int`` where
        the parent has ``int | None``. Runtime was fine -- ``None`` worked --
        but a type-checked caller could not pass the value that disables the
        cap, so the surface was still narrower than the parent's. Comparing
        names alone cannot see that, which is why this test compares
        annotations.
        """
        sgd = inspect.signature(onlinerake.OnlineRakingSGD.__init__).parameters
        mwu = inspect.signature(onlinerake.OnlineRakingMWU.__init__).parameters
        narrowed = {
            name: (str(p.annotation), str(mwu[name].annotation))
            for name, p in sgd.items()
            if name in mwu
            and p.annotation is not inspect.Parameter.empty
            and mwu[name].annotation is not inspect.Parameter.empty
            and p.annotation != mwu[name].annotation
        }
        assert not narrowed, f"OnlineRakingMWU narrows: {narrowed}"
