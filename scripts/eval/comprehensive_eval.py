#!/usr/bin/env python3
"""Comprehensive head-to-head evaluation of online raking algorithms.

This script runs a rigorous comparison of online raking algorithms (SGD, MWU)
against batch IPF baseline across multiple scenarios. It addresses key evaluation
criteria including:

- Convergence speed and final accuracy
- Effective sample size and weight efficiency
- Robustness under different conditions (easy/hard targets, drift, infeasibility)
- Numerical stability
- Computational performance

Usage:
    python scripts/comprehensive_eval.py --n_seeds 5   # Quick test
    python scripts/comprehensive_eval.py --n_seeds 50  # Full evaluation

Outputs:
    eval_outputs/results.csv       - Raw results for all runs
    eval_outputs/summary.csv       - Aggregated statistics
    eval_outputs/metadata.json     - Run configuration and environment
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from onlinerake import (
    BatchIPF,
    OnlineRakingMWU,
    OnlineRakingSGD,
    Targets,
    robbins_monro_schedule,
)


@dataclass
class EvalResult:
    """Results from evaluating a single algorithm on a single scenario/seed."""

    algorithm: str
    scenario: str
    seed: int
    final_loss: float
    ess: float
    weight_efficiency: float
    convergence_obs: int | None
    runtime_sec: float
    max_weight_ratio: float
    oscillation: bool
    mean_margin_error: float
    max_margin_error: float
    numerical_stable: bool
    n_obs: int
    n_features: int


def generate_scenario_data(
    scenario: str,
    n_obs: int,
    n_features: int,
    seed: int,
) -> tuple[Targets, list[dict[str, int]]]:
    """Generate targets and observations for a scenario.

    Args:
        scenario: One of 'easy', 'hard', 'infeasible', 'drift'
        n_obs: Number of observations to generate
        n_features: Number of binary features
        seed: Random seed for reproducibility

    Returns:
        Tuple of (Targets, list of observation dicts)
    """
    np.random.seed(seed)

    feature_names = [f"f{i}" for i in range(n_features)]

    if scenario == "easy":
        raw_probs = np.random.uniform(0.3, 0.7, n_features)
        target_probs = np.clip(
            raw_probs + np.random.uniform(-0.15, 0.15, n_features), 0.15, 0.85
        )
    elif scenario == "hard":
        raw_probs = np.random.uniform(0.4, 0.6, n_features)
        target_probs = np.array(
            [np.random.choice([0.08, 0.92]) for _ in range(n_features)]
        )
    elif scenario == "infeasible":
        raw_probs = np.random.uniform(0.3, 0.7, n_features)
        target_probs = np.array([0.90] * n_features)
    elif scenario == "drift":
        raw_probs = np.random.uniform(0.3, 0.7, n_features)
        target_probs = np.array([0.5] * n_features)
    else:
        raw_probs = np.random.uniform(0.3, 0.7, n_features)
        target_probs = np.array([0.5] * n_features)

    targets = Targets(
        **{name: float(target_probs[i]) for i, name in enumerate(feature_names)}
    )

    observations = []
    for t in range(n_obs):
        if scenario == "drift":
            drift_factor = t / n_obs
            probs = raw_probs * (1 - drift_factor) + (1 - raw_probs) * drift_factor
        else:
            probs = raw_probs

        obs = {
            name: int(np.random.random() < probs[i])
            for i, name in enumerate(feature_names)
        }
        observations.append(obs)

    return targets, observations


def evaluate_batch_ipf(
    targets: Targets,
    observations: list[dict[str, int]],
    seed: int,
    scenario: str,
) -> EvalResult:
    """Evaluate batch IPF baseline."""
    n_obs = len(observations)
    n_features = targets.n_features

    start_time = time.perf_counter()
    raker = BatchIPF(targets)
    raker.fit(observations)
    runtime = time.perf_counter() - start_time

    margins = raker.margins
    margin_errors = [abs(margins[f] - targets[f]) for f in margins]

    weights = raker.weights
    max_ratio = weights.max() / weights.min() if weights.min() > 0 else float("inf")

    return EvalResult(
        algorithm="batch_ipf",
        scenario=scenario,
        seed=seed,
        final_loss=raker.loss,
        ess=raker.effective_sample_size,
        weight_efficiency=raker.effective_sample_size / n_obs,
        convergence_obs=None,
        runtime_sec=runtime,
        max_weight_ratio=max_ratio,
        oscillation=False,
        mean_margin_error=float(np.mean(margin_errors)),
        max_margin_error=float(np.max(margin_errors)),
        numerical_stable=bool(np.all(np.isfinite(weights))),
        n_obs=n_obs,
        n_features=n_features,
    )


def evaluate_online_sgd_fixed(
    targets: Targets,
    observations: list[dict[str, int]],
    seed: int,
    scenario: str,
    learning_rate: float = 5.0,
    n_sgd_steps: int = 3,
) -> EvalResult:
    """Evaluate online SGD with fixed learning rate."""
    n_obs = len(observations)
    n_features = targets.n_features

    start_time = time.perf_counter()
    raker = OnlineRakingSGD(
        targets,
        learning_rate=learning_rate,
        n_sgd_steps=n_sgd_steps,
        track_convergence=True,
        compute_weight_stats=False,
    )

    convergence_obs = None
    for i, obs in enumerate(observations):
        raker.partial_fit(obs)
        if convergence_obs is None and raker.converged:
            convergence_obs = i + 1

    runtime = time.perf_counter() - start_time

    margins = raker.margins
    margin_errors = [abs(margins[f] - targets[f]) for f in margins]

    weights = raker.weights
    max_ratio = weights.max() / weights.min() if weights.min() > 0 else float("inf")

    return EvalResult(
        algorithm="sgd_fixed",
        scenario=scenario,
        seed=seed,
        final_loss=raker.loss,
        ess=raker.effective_sample_size,
        weight_efficiency=raker.effective_sample_size / n_obs,
        convergence_obs=convergence_obs,
        runtime_sec=runtime,
        max_weight_ratio=max_ratio,
        oscillation=raker.detect_oscillation(),
        mean_margin_error=float(np.mean(margin_errors)),
        max_margin_error=float(np.max(margin_errors)),
        numerical_stable=bool(np.all(np.isfinite(weights))),
        n_obs=n_obs,
        n_features=n_features,
    )


def evaluate_online_sgd_rm(
    targets: Targets,
    observations: list[dict[str, int]],
    seed: int,
    scenario: str,
    initial_lr: float = 5.0,
    power: float = 0.6,
    n_sgd_steps: int = 3,
) -> EvalResult:
    """Evaluate online SGD with Robbins-Monro diminishing schedule."""
    n_obs = len(observations)
    n_features = targets.n_features

    schedule = robbins_monro_schedule(initial_lr=initial_lr, power=power)

    start_time = time.perf_counter()
    raker = OnlineRakingSGD(
        targets,
        learning_rate=schedule,
        n_sgd_steps=n_sgd_steps,
        track_convergence=True,
        compute_weight_stats=False,
    )

    convergence_obs = None
    for i, obs in enumerate(observations):
        raker.partial_fit(obs)
        if convergence_obs is None and raker.converged:
            convergence_obs = i + 1

    runtime = time.perf_counter() - start_time

    margins = raker.margins
    margin_errors = [abs(margins[f] - targets[f]) for f in margins]

    weights = raker.weights
    max_ratio = weights.max() / weights.min() if weights.min() > 0 else float("inf")

    return EvalResult(
        algorithm="sgd_rm",
        scenario=scenario,
        seed=seed,
        final_loss=raker.loss,
        ess=raker.effective_sample_size,
        weight_efficiency=raker.effective_sample_size / n_obs,
        convergence_obs=convergence_obs,
        runtime_sec=runtime,
        max_weight_ratio=max_ratio,
        oscillation=raker.detect_oscillation(),
        mean_margin_error=float(np.mean(margin_errors)),
        max_margin_error=float(np.max(margin_errors)),
        numerical_stable=bool(np.all(np.isfinite(weights))),
        n_obs=n_obs,
        n_features=n_features,
    )


def evaluate_online_mwu(
    targets: Targets,
    observations: list[dict[str, int]],
    seed: int,
    scenario: str,
    learning_rate: float = 1.0,
    n_sgd_steps: int = 3,
) -> EvalResult:
    """Evaluate online MWU algorithm."""
    n_obs = len(observations)
    n_features = targets.n_features

    start_time = time.perf_counter()
    raker = OnlineRakingMWU(
        targets,
        learning_rate=learning_rate,
        n_sgd_steps=n_sgd_steps,
        track_convergence=True,
        compute_weight_stats=False,
    )

    convergence_obs = None
    for i, obs in enumerate(observations):
        raker.partial_fit(obs)
        if convergence_obs is None and raker.converged:
            convergence_obs = i + 1

    runtime = time.perf_counter() - start_time

    margins = raker.margins
    margin_errors = [abs(margins[f] - targets[f]) for f in margins]

    weights = raker.weights
    max_ratio = weights.max() / weights.min() if weights.min() > 0 else float("inf")

    return EvalResult(
        algorithm="mwu",
        scenario=scenario,
        seed=seed,
        final_loss=raker.loss,
        ess=raker.effective_sample_size,
        weight_efficiency=raker.effective_sample_size / n_obs,
        convergence_obs=convergence_obs,
        runtime_sec=runtime,
        max_weight_ratio=max_ratio,
        oscillation=raker.detect_oscillation(),
        mean_margin_error=float(np.mean(margin_errors)),
        max_margin_error=float(np.max(margin_errors)),
        numerical_stable=bool(np.all(np.isfinite(weights))),
        n_obs=n_obs,
        n_features=n_features,
    )


def evaluate_periodic_ipf(
    targets: Targets,
    observations: list[dict[str, int]],
    seed: int,
    scenario: str,
    batch_size: int = 100,
) -> EvalResult:
    """Evaluate periodic batch IPF (rerun IPF every batch_size observations)."""
    n_obs = len(observations)
    n_features = targets.n_features

    start_time = time.perf_counter()
    raker = BatchIPF(targets)

    for i in range(0, n_obs, batch_size):
        batch = observations[i : i + batch_size]
        raker.fit_incremental(batch)

    runtime = time.perf_counter() - start_time

    margins = raker.margins
    margin_errors = [abs(margins[f] - targets[f]) for f in margins]

    weights = raker.weights
    max_ratio = weights.max() / weights.min() if weights.min() > 0 else float("inf")

    return EvalResult(
        algorithm=f"periodic_ipf_{batch_size}",
        scenario=scenario,
        seed=seed,
        final_loss=raker.loss,
        ess=raker.effective_sample_size,
        weight_efficiency=raker.effective_sample_size / n_obs,
        convergence_obs=None,
        runtime_sec=runtime,
        max_weight_ratio=max_ratio,
        oscillation=False,
        mean_margin_error=float(np.mean(margin_errors)),
        max_margin_error=float(np.max(margin_errors)),
        numerical_stable=bool(np.all(np.isfinite(weights))),
        n_obs=n_obs,
        n_features=n_features,
    )


def run_comprehensive_eval(
    output_dir: str = "./eval_outputs",
    n_seeds: int = 50,
    verbose: bool = True,
    quick: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run full comprehensive evaluation.

    Args:
        output_dir: Directory to save results
        n_seeds: Number of random seeds per scenario
        verbose: Print progress information
        quick: If True, skip large scenario and use smaller sizes

    Returns:
        Tuple of (results_df, summary_df)
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    if quick:
        scenarios = {
            "small_easy": (200, 4, "easy"),
            "standard": (1000, 8, "easy"),
            "hard_targets": (1000, 8, "hard"),
            "infeasible": (1000, 8, "infeasible"),
            "high_dim": (500, 20, "easy"),
            "drift": (1000, 8, "drift"),
        }
    else:
        scenarios = {
            "small_easy": (500, 4, "easy"),
            "standard": (3000, 8, "easy"),
            "large": (50000, 8, "easy"),
            "hard_targets": (3000, 8, "hard"),
            "infeasible": (3000, 8, "infeasible"),
            "high_dim": (3000, 50, "easy"),
            "drift": (5000, 8, "drift"),
        }

    algorithms = [
        ("batch_ipf", evaluate_batch_ipf),
        ("sgd_fixed", evaluate_online_sgd_fixed),
        ("sgd_rm", evaluate_online_sgd_rm),
        ("mwu", evaluate_online_mwu),
        ("periodic_ipf_50", lambda t, o, s, sc: evaluate_periodic_ipf(t, o, s, sc, 50)),
        (
            "periodic_ipf_100",
            lambda t, o, s, sc: evaluate_periodic_ipf(t, o, s, sc, 100),
        ),
    ]

    results: list[EvalResult] = []

    for scenario_name, (n_obs, n_features, difficulty) in scenarios.items():
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Scenario: {scenario_name} (n={n_obs}, d={n_features})")
            print(f"{'=' * 60}")

        for seed in range(n_seeds):
            targets, observations = generate_scenario_data(
                difficulty, n_obs, n_features, seed
            )

            for algo_name, algo_func in algorithms:
                try:
                    result = algo_func(targets, observations, seed, scenario_name)
                    results.append(result)
                except Exception as e:
                    if verbose:
                        print(f"  Warning: {algo_name} failed on seed {seed}: {e}")

            if verbose and (seed + 1) % 10 == 0:
                print(f"  Completed seed {seed + 1}/{n_seeds}")

    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(output_path / "results.csv", index=False)

    agg_funcs: dict[str, Any] = {
        "final_loss": ["mean", "std", "median"],
        "ess": ["mean", "std"],
        "weight_efficiency": ["mean", "std"],
        "runtime_sec": ["mean", "std"],
        "mean_margin_error": ["mean", "std"],
        "max_margin_error": ["mean", "std"],
        "oscillation": ["sum"],
        "numerical_stable": ["sum", "count"],
    }

    summary = df.groupby(["scenario", "algorithm"]).agg(agg_funcs).round(6)
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    summary.to_csv(output_path / "summary.csv", index=False)

    metadata = {
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "n_seeds": n_seeds,
        "quick": quick,
        "scenarios": list(scenarios.keys()),
        "algorithms": [a[0] for a in algorithms],
    }
    (output_path / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    if verbose:
        print(f"\nResults saved to {output_path}")

    return df, summary


def print_summary_table(summary: pd.DataFrame) -> None:
    """Print a nicely formatted summary table."""
    print("\n" + "=" * 80)
    print("SUMMARY RESULTS")
    print("=" * 80)

    scenarios_order = [
        "small_easy",
        "standard",
        "large",
        "hard_targets",
        "infeasible",
        "high_dim",
        "drift",
    ]
    algo_order = [
        "batch_ipf",
        "sgd_fixed",
        "sgd_rm",
        "mwu",
        "periodic_ipf_50",
        "periodic_ipf_100",
    ]

    for scenario in scenarios_order:
        scenario_data = summary[summary["scenario"] == scenario]
        if len(scenario_data) == 0:
            continue

        print(f"\n--- {scenario.upper()} ---")
        print(
            f"{'Algorithm':<20} {'Loss (mean)':<12} {'ESS (mean)':<12} {'Efficiency':<12} {'Runtime (s)':<12}"
        )
        print("-" * 68)

        for algo in algo_order:
            algo_data = scenario_data[scenario_data["algorithm"] == algo]
            if len(algo_data) == 0:
                continue

            row = algo_data.iloc[0]
            print(
                f"{algo:<20} "
                f"{row['final_loss_mean']:>10.6f}  "
                f"{row['ess_mean']:>10.1f}  "
                f"{row['weight_efficiency_mean']:>10.1%}  "
                f"{row['runtime_sec_mean']:>10.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation of online raking algorithms"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_outputs",
        help="Directory to save results",
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=50,
        help="Number of random seeds per scenario",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick evaluation with smaller scenarios",
    )
    args = parser.parse_args()

    _df, summary = run_comprehensive_eval(
        output_dir=args.output_dir,
        n_seeds=args.n_seeds,
        verbose=not args.quiet,
        quick=args.quick,
    )

    print_summary_table(summary)


if __name__ == "__main__":
    main()
