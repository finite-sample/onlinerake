"""Measure online calibration and its weight-variation cost on simulated streams.

Run from the repository root: python examples/streaming_mwu.py
"""

import numpy as np

from onlinerake import OnlineRakingMWU, Targets


def evaluate(scenario: str, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    probabilities = (
        np.full(500, 0.8) if scenario == "stationary" else np.repeat([0.7, 0.95], 250)
    )
    binary = rng.binomial(1, probabilities)
    continuous = rng.normal(0.4, 0.5, 500)
    targets = Targets(a=0.5, z=(0.0, "mean")) if scenario == "mixed" else Targets(a=0.5)
    raker = OnlineRakingMWU(targets)
    raw_losses, weighted_losses = [], []
    for i, value in enumerate(binary):
        raker.partial_fit({"a": value, "z": continuous[i]})
        if i >= 99:
            raw_losses.append(
                sum(
                    (raker.raw_margins[name] - targets[name]) ** 2
                    for name in targets.feature_names
                )
            )
            weighted_losses.append(raker.loss)
    reduction = 1 - np.mean(weighted_losses) / np.mean(raw_losses)
    return float(reduction), raker.effective_sample_size


if __name__ == "__main__":
    print("Default MWU; 500 arrivals; average loss over arrivals 100-500; seeds 0-9")
    for scenario in ["stationary", "shift", "mixed"]:
        results = np.array([evaluate(scenario, seed) for seed in range(10)])
        reduction, ess = results.mean(axis=0)
        print(
            f"{scenario}: squared imbalance reduction={reduction:.1%}; "
            f"ESS={ess:.1f}/500"
        )
