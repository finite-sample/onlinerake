"""Command‑line entry point for running simulations and quick demos.

When run as ``python -m onlinerake`` from the repository root this
module executes a small simulation over the built‑in scenarios and
prints a summary of the results.  Use the ``--help`` option to see
available parameters.
"""

import argparse
from pprint import pprint

from .simulation import run_simulation_suite, analyze_results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run online raking simulations.")
    parser.add_argument("--seeds", type=int, default=3, help="number of random seeds per scenario")
    parser.add_argument("--n-obs", type=int, default=300, dest="n_obs", help="number of observations per seed")
    parser.add_argument("--sgd-rate", type=float, default=5.0, help="learning rate for SGD raker")
    parser.add_argument("--mwu-rate", type=float, default=1.0, help="learning rate for MWU raker")
    parser.add_argument("--steps", type=int, default=3, help="number of updates per observation for both algorithms")
    args = parser.parse_args(argv)
    print(f"Running simulation with {args.seeds} seeds, {args.n_obs} observations per seed...")
    df = run_simulation_suite(
        n_seeds=args.seeds,
        n_obs=args.n_obs,
        learning_rate_sgd=args.sgd_rate,
        learning_rate_mwu=args.mwu_rate,
        n_steps=args.steps,
    )
    print(df.head())
    analyze_results(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())