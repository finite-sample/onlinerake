#!/usr/bin/env python3
"""Simulation harness for evaluating online raking algorithms.

This module contains tools for generating synthetic streaming data with
controlled bias patterns and for benchmarking the performance of
different online raking algorithms.  It reproduces and extends the
simulation suite described in the accompanying README.

The high‑level entry point is :func:`run_simulation_suite`, which
generates streams of observations under different bias dynamics,
applies both the stochastic gradient descent (SGD) and multiplicative
weights update (MWU) rakers, and aggregates a suite of diagnostic
metrics.  The returned DataFrame can be further analysed or printed
using the provided :func:`analyze_results` convenience function.

Example::

    python examples/simulation.py
    # or
    from examples.simulation import run_simulation_suite, analyze_results
    df = run_simulation_suite(n_seeds=5, n_obs=300)
    analyze_results(df)

Note that the simulations may be time consuming for large numbers of
observations or seeds.  Adjust ``n_obs`` and ``n_seeds`` as needed.
"""

import argparse
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from onlinerake.online_raking_mwu import OnlineRakingMWU
from onlinerake.online_raking_sgd import OnlineRakingSGD
from onlinerake.targets import Targets

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(message)s")


@dataclass
class DemographicObservation:
    """Simple container for a single set of demographic indicators."""

    age: int
    gender: int
    education: int
    region: int

    def as_dict(self) -> dict[str, int]:
        return {
            "age": self.age,
            "gender": self.gender,
            "education": self.education,
            "region": self.region,
        }


class BiasSimulator:
    """Simulate streams of demographic observations with evolving bias."""

    @staticmethod
    def linear_shift(
        n_obs: int, start_probs: dict[str, float], end_probs: dict[str, float]
    ) -> list[DemographicObservation]:
        """Generate a linear drift from ``start_probs`` to ``end_probs``.

        Each probability dict should map demographic names to the
        probability of a ``1`` in the binary indicator.  Probabilities
        interpolate linearly over time.
        """
        data: list[DemographicObservation] = []
        for i in range(n_obs):
            progress = i / (n_obs - 1) if n_obs > 1 else 0.0
            probs = {
                name: start_probs[name]
                + progress * (end_probs[name] - start_probs[name])
                for name in start_probs
            }
            obs = DemographicObservation(
                age=np.random.binomial(1, probs["age"]),
                gender=np.random.binomial(1, probs["gender"]),
                education=np.random.binomial(1, probs["education"]),
                region=np.random.binomial(1, probs["region"]),
            )
            data.append(obs)
        return data

    @staticmethod
    def sudden_shift(
        n_obs: int,
        shift_point: float,
        before_probs: dict[str, float],
        after_probs: dict[str, float],
    ) -> list[DemographicObservation]:
        """Generate a sudden shift at ``shift_point`` fraction of the stream."""
        data: list[DemographicObservation] = []
        shift_index = int(shift_point * n_obs)
        for i in range(n_obs):
            probs = before_probs if i < shift_index else after_probs
            obs = DemographicObservation(
                age=np.random.binomial(1, probs["age"]),
                gender=np.random.binomial(1, probs["gender"]),
                education=np.random.binomial(1, probs["education"]),
                region=np.random.binomial(1, probs["region"]),
            )
            data.append(obs)
        return data

    @staticmethod
    def oscillating_bias(
        n_obs: int, base_probs: dict[str, float], amplitude: float, period: int
    ) -> list[DemographicObservation]:
        """Generate an oscillating bias around ``base_probs``.

        Probabilities oscillate sinusoidally with amplitude ``amplitude``
        and period ``period``.  Probabilities are clipped to [0.1, 0.9]
        to avoid degeneracy.
        """
        data: list[DemographicObservation] = []
        for i in range(n_obs):
            phase = 2 * np.pi * i / period
            osc = amplitude * np.sin(phase)
            probs = {
                name: float(np.clip(base_probs[name] + osc, 0.1, 0.9))
                for name in base_probs
            }
            obs = DemographicObservation(
                age=np.random.binomial(1, probs["age"]),
                gender=np.random.binomial(1, probs["gender"]),
                education=np.random.binomial(1, probs["education"]),
                region=np.random.binomial(1, probs["region"]),
            )
            data.append(obs)
        return data


def run_simulation_suite(
    n_seeds: int = 10,
    n_obs: int = 300,
    targets: Targets | None = None,
    learning_rate_sgd: float = 5.0,
    learning_rate_mwu: float = 1.0,
    n_steps: int = 3,
) -> pd.DataFrame:
    """Run a comprehensive simulation suite across scenarios and seeds.

    Parameters
    ----------
    n_seeds : int
        Number of random seeds to simulate for each scenario.  More seeds
        provide smoother averages but increase runtime.
    n_obs : int
        Number of observations in each simulated stream.
    targets : :class:`~onlinerake.targets.Targets`, optional
        Target population margins.  If ``None``, defaults to
        ``Targets()``.
    learning_rate_sgd : float
        Learning rate for the SGD raker.
    learning_rate_mwu : float
        Learning rate for the MWU raker.
    n_steps : int
        Number of update steps to apply per observation for both
        algorithms.

    Returns
    -------
    pandas.DataFrame
        A tidy DataFrame containing summary metrics for each seed,
        scenario and method.  Each row corresponds to a (scenario,
        seed, method) triple.
    """
    if targets is None:
        targets = Targets()

    # define scenarios
    scenarios = {
        "linear": {
            "sim_fn": BiasSimulator.linear_shift,
            "params": {
                "n_obs": n_obs,
                "start_probs": {
                    "age": 0.2,
                    "gender": 0.3,
                    "education": 0.2,
                    "region": 0.1,
                },
                "end_probs": {
                    "age": 0.8,
                    "gender": 0.7,
                    "education": 0.6,
                    "region": 0.5,
                },
            },
        },
        "sudden": {
            "sim_fn": BiasSimulator.sudden_shift,
            "params": {
                "n_obs": n_obs,
                "shift_point": 0.5,
                "before_probs": {
                    "age": 0.2,
                    "gender": 0.2,
                    "education": 0.2,
                    "region": 0.2,
                },
                "after_probs": {
                    "age": 0.8,
                    "gender": 0.8,
                    "education": 0.6,
                    "region": 0.4,
                },
            },
        },
        "oscillating": {
            "sim_fn": BiasSimulator.oscillating_bias,
            "params": {
                "n_obs": n_obs,
                "base_probs": {
                    "age": 0.5,
                    "gender": 0.5,
                    "education": 0.4,
                    "region": 0.3,
                },
                "amplitude": 0.2,
                "period": max(50, n_obs // 4),
            },
        },
    }

    results: list[dict[str, Any]] = []
    for scenario_name, config in scenarios.items():
        sim_fn = config["sim_fn"]
        params = config["params"]
        for seed in range(n_seeds):
            np.random.seed(seed)
            # generate data
            stream = sim_fn(**params)
            # instantiate rakers
            rakers = {
                "SGD": OnlineRakingSGD(
                    targets=targets,
                    learning_rate=learning_rate_sgd,
                    n_sgd_steps=n_steps,
                    min_weight=1e-3,
                    max_weight=100.0,
                ),
                "MWU": OnlineRakingMWU(
                    targets=targets,
                    learning_rate=learning_rate_mwu,
                    n_steps=n_steps,
                    min_weight=1e-3,
                    max_weight=100.0,
                ),
            }
            # run both algorithms on the same stream
            for method_name, raker in rakers.items():
                for obs in stream:
                    raker.partial_fit(obs.as_dict())
                # compute summary metrics
                final_state = raker.history[-1]
                # compute temporal errors vs baseline
                temporal_errors = {}
                baseline_errors = {}
                # build arrays of errors for each time step
                for demo in ["age", "gender", "education", "region"]:
                    target_val = getattr(targets, demo)
                    weighted_errors = [
                        abs(h["weighted_margins"][demo] - target_val)
                        for h in raker.history
                    ]
                    raw_errors = [
                        abs(h["raw_margins"][demo] - target_val) for h in raker.history
                    ]
                    temporal_errors[f"{demo}_temporal_error"] = float(
                        np.mean(weighted_errors)
                    )
                    baseline_errors[f"{demo}_temporal_baseline_error"] = float(
                        np.mean(raw_errors)
                    )

                avg_temporal_loss = float(np.mean([h["loss"] for h in raker.history]))
                ess_final = float(final_state["ess"])
                weight_range = float(
                    final_state["weight_stats"]["max"]
                    - final_state["weight_stats"]["min"]
                )
                result = {
                    "scenario": scenario_name,
                    "seed": seed,
                    "method": method_name,
                    "avg_temporal_loss": avg_temporal_loss,
                    "final_loss": float(final_state["loss"]),
                    "final_ess": ess_final,
                    "final_weight_range": weight_range,
                }
                result.update(temporal_errors)
                result.update(baseline_errors)
                results.append(result)
    df = pd.DataFrame(results)
    return df


def analyze_results(df: pd.DataFrame) -> None:
    """Print a simple text summary of simulation results.

    This helper groups the results by scenario and method, reporting
    temporal margin errors, overall improvement relative to baseline
    errors, effective sample size and loss.  It is intended for quick
    inspection of DataFrame output.  For a more detailed or customised
    analysis, operate on the DataFrame directly.
    """
    if df.empty:
        logging.info("No results to analyse.")
        return
    demo_names = ["age", "gender", "education", "region"]
    for scenario in df["scenario"].unique():
        logging.info(f"\nScenario: {scenario}")
        scen_df = df[df["scenario"] == scenario]
        for method in scen_df["method"].unique():
            mdf = scen_df[scen_df["method"] == method]
            logging.info(f"  Method: {method}")
            # compute average errors
            for demo in demo_names:
                mean_w = mdf[f"{demo}_temporal_error"].mean()
                mean_b = mdf[f"{demo}_temporal_baseline_error"].mean()
                impr = (mean_b - mean_w) / mean_b * 100 if mean_b != 0 else 0.0
                logging.info(
                    f"    {demo:<10}: baseline {mean_b:.4f} -> weighted {mean_w:.4f} ({impr:+.1f}% imp)"
                )
            # aggregated improvement
            mean_w_overall = mdf[
                [f"{d}_temporal_error" for d in demo_names]
            ].values.mean()
            mean_b_overall = mdf[
                [f"{d}_temporal_baseline_error" for d in demo_names]
            ].values.mean()
            overall_impr = (
                (mean_b_overall - mean_w_overall) / mean_b_overall * 100
                if mean_b_overall != 0
                else 0.0
            )
            logging.info(f"    Overall improvement: {overall_impr:+.1f}%")
            logging.info(
                f"    Final ESS: mean {mdf['final_ess'].mean():.1f}, std {mdf['final_ess'].std():.1f}"
            )
            logging.info(
                f"    Final loss: mean {mdf['final_loss'].mean():.4f}, std {mdf['final_loss'].std():.4f}"
            )


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point for running simulations."""
    parser = argparse.ArgumentParser(description="Run online raking simulations.")
    parser.add_argument(
        "--seeds", type=int, default=3, help="number of random seeds per scenario"
    )
    parser.add_argument(
        "--n-obs",
        type=int,
        default=300,
        dest="n_obs",
        help="number of observations per seed",
    )
    parser.add_argument(
        "--sgd-rate", type=float, default=5.0, help="learning rate for SGD raker"
    )
    parser.add_argument(
        "--mwu-rate", type=float, default=1.0, help="learning rate for MWU raker"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=3,
        help="number of updates per observation for both algorithms",
    )
    args = parser.parse_args(argv)
    logging.info(
        f"Running simulation with {args.seeds} seeds, {args.n_obs} observations per seed..."
    )
    df = run_simulation_suite(
        n_seeds=args.seeds,
        n_obs=args.n_obs,
        learning_rate_sgd=args.sgd_rate,
        learning_rate_mwu=args.mwu_rate,
        n_steps=args.steps,
    )
    logging.info(df.head())
    analyze_results(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
