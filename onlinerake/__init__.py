"""Top‑level namespace for the onlinerake package.

This package provides two streaming weight calibration algorithms—
stochastic gradient descent (SGD) and multiplicative weights update (MWU)—
for adjusting survey weights to match known population margins in real time.

The high‑level API mirrors the ``partial_fit`` pattern familiar to
scikit‑learn users. Each raker class accepts a sequence of demographic
observations (binary indicator vectors) and updates its internal weight
vector accordingly. Both implementations expose methods to compute
weighted margins, loss and effective sample size on demand.

Example::

    from onlinerake import OnlineRakingSGD, OnlineRakingMWU, Targets

    # target population proportions
    targets = Targets(age=0.5, gender=0.5, education=0.4, region=0.3)
    raker = OnlineRakingSGD(targets)

    # stream observations one at a time
    for obs in my_stream:
        raker.partial_fit(obs)
        # inspect current weighted margins
        print(raker.margins)

The :mod:`online_raking_sgd` and :mod:`online_raking_mwu` modules
implement the two update strategies. See their docstrings for details.
"""

from .targets import Targets  # noqa: F401
from .online_raking_sgd import OnlineRakingSGD  # noqa: F401
from .online_raking_mwu import OnlineRakingMWU  # noqa: F401

__all__ = [
    "Targets",
    "OnlineRakingSGD",
    "OnlineRakingMWU",
]
