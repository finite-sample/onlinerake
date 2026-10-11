"""Fixed-memory streaming raking versus methods that store the stream.

Writes one CSV row per (scenario, seed, method) as soon as it is computed, so
an interrupted run keeps everything finished before the interruption.

Usage:
    uv run python scripts/eval/streaming_eval.py --n_seeds 2 --quick
    uv run python scripts/eval/streaming_eval.py --n_seeds 50
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comprehensive_eval import generate_scenario_data

from onlinerake import (
    BatchIPF,
    OnlineRakingMWU,
    OnlineRakingSGD,
    StreamingRaker,
    Targets,
    fit_penalized_raking,
)

SCENARIOS = {
    "standard": ("easy", 3000, 8),
    "large": ("easy", 50000, 8),
    "hard": ("hard", 3000, 8),
    "high_dim": ("easy", 3000, 50),
    "drift": ("drift", 5000, 8),
    "small": ("easy", 500, 4),
}
STRENGTHS = (0.01, 0.03, 0.1, 0.3, 1.0)
CHECKPOINTS = 20
FIELDS = [
    "scenario",
    "n",
    "p",
    "seed",
    "method",
    "final_err",
    "max_err",
    "anytime_err",
    "ess_frac",
    "bias",
    "sec_per_obs",
    "state_floats",
]


def outcome(x, seed, t):
    rng = np.random.default_rng(10_000 + seed)
    beta = rng.uniform(-1, 1, x.shape[1])
    y = 1 + x @ beta + x[:, 0] * x[:, 1] + rng.normal(size=len(x))
    # Population truth assumes independent features at the target margins.
    return y, float(1 + beta @ t + t[0] * t[1])


def summarize(w, x, t, y, truth):
    m = (w @ x) / w.sum()
    err = np.abs(m - t)
    return {
        "final_err": float(err.mean()),
        "max_err": float(err.max()),
        "ess_frac": float(w.sum() ** 2 / (w @ w) / len(w)),
        "bias": float((w @ y) / w.sum() - truth),
    }


def checkpoints(n):
    return sorted(
        {max(1, round(n * k / CHECKPOINTS)) for k in range(1, CHECKPOINTS + 1)}
    )


def ipf_weights(targets, dicts):
    raker = BatchIPF(targets)
    raker.fit(dicts)
    return raker.weights


def run_stream(targets, x, y, c):
    raker = StreamingRaker(
        targets, calibration_strength=c, outcomes=["y"], keep_weights=True
    )
    names = targets.feature_names
    marks = set(checkpoints(len(x)))
    errs = []
    start = time.perf_counter()
    for i, (row, yi) in enumerate(zip(x, y, strict=True), start=1):
        raker.partial_fit(
            dict(zip(names, row.tolist(), strict=True)), outcomes={"y": float(yi)}
        )
        if i in marks:
            errs.append(np.mean([abs(raker.margins[k] - targets[k]) for k in names]))
    elapsed = time.perf_counter() - start
    return raker.weights, float(np.mean(errs)), elapsed / len(x)


def ipf_factors(x, t, tol=1e-10, max_cycles=1000):
    """IPF on binary margins, keeping the per-feature factors in log space.

    IPF from unit weights multiplies each row by one factor per feature,
    chosen by whether the feature is 0 or 1, so a new row's weight under the
    last fit is the product of its factors and needs no refit. A feature
    that is constant so far cannot be raked yet and is skipped. Small early
    prefixes are often infeasible, and IPF then drives the factors apart
    without bound; logs keep that finite.
    """
    on = x.astype(bool)
    log_on, log_off = np.zeros(x.shape[1]), np.zeros(x.shape[1])
    movable = on.any(axis=0) & ~on.all(axis=0)

    def weights():
        log_w = np.where(on, log_on, log_off).sum(axis=1)
        return np.exp(log_w - log_w.max())

    w = weights()
    for _ in range(max_cycles):
        for j in np.flatnonzero(movable):
            m = (w @ x[:, j]) / w.sum()
            log_on[j] += np.log(t[j] / m)
            log_off[j] += np.log((1 - t[j]) / (1 - m))
            w = weights()
        if not movable.any() or np.abs((w @ x) / w.sum() - t)[movable].max() < tol:
            break
    log_scale = np.log(w.mean()) + np.where(on, log_on, log_off).sum(axis=1).max()
    return w / w.mean(), log_on, log_off, log_scale


def factor_weights(x, log_on, log_off, log_scale):
    log_w = np.where(x.astype(bool), log_on, log_off).sum(axis=1) - log_scale
    return np.exp(np.clip(log_w, -700, 700))


def refit_sizes(n, k):
    """Refit at 1, 2, 4, ... rows until k, then every k rows, and at n."""
    sizes = {min(2**i, n) for i in range(k.bit_length()) if 2**i < k}
    sizes |= set(range(k, n + 1, k)) | {n}
    return sorted(sizes)


def run_periodic(x, t, k):
    sizes = refit_sizes(len(x), k)
    start = time.perf_counter()
    fits = {size: ipf_factors(x[:size], t) for size in sizes}
    elapsed = time.perf_counter() - start
    errs = []
    for mark in checkpoints(len(x)):
        last = max(size for size in sizes if size <= mark)
        prior, ones, zeros, scale = fits[last]
        w = np.concatenate([prior, factor_weights(x[last:mark], ones, zeros, scale)])
        errs.append(np.abs((w @ x[:mark]) / w.sum() - t).mean())
    return fits[len(x)][0], float(np.mean(errs)), elapsed / len(x)


def run_all_weights(cls, lr, targets, dicts):
    raker = cls(targets, learning_rate=lr, track_convergence=False)
    start = time.perf_counter()
    for obs in dicts:
        raker.partial_fit(obs)
    return raker.weights, (time.perf_counter() - start) / len(dicts)


def evaluate_seed(name, scenario, n, p, seed, writer, handle):
    targets, dicts = generate_scenario_data(scenario, n, p, seed)
    names = targets.feature_names
    x = np.array([[o[k] for k in names] for o in dicts], dtype=float)
    t = np.array([targets[k] for k in names])
    y, truth = outcome(x, seed, t)
    base = {"scenario": name, "n": n, "p": p, "seed": seed}
    stored = n * (p + 1)

    def emit(method, w, anytime, sec, state):
        writer.writerow(
            {
                **base,
                "method": method,
                **summarize(w, x, t, y, truth),
                "anytime_err": anytime,
                "sec_per_obs": sec,
                "state_floats": state,
            }
        )
        handle.flush()

    raw_any = float(np.mean([np.abs(x[:m].mean(0) - t).mean() for m in checkpoints(n)]))
    emit("raw", np.ones(n), raw_any, 0.0, 0)

    start = time.perf_counter()
    w_ipf = ipf_weights(targets, dicts)
    sec = (time.perf_counter() - start) / n
    ipf_errs = []
    for m in checkpoints(n):
        w_m = ipf_weights(targets, dicts[:m])
        ipf_errs.append(np.abs((w_m @ x[:m]) / w_m.sum() - t).mean())
    emit("ipf", w_ipf, float(np.mean(ipf_errs)), sec, stored)

    # Each refit stores every row so far; timing covers all refits.
    for k in (100, 1000):
        if k < n:
            w, anytime, sec = run_periodic(x, t, k)
            emit(f"periodic_ipf_{k}", w, anytime, sec, stored)

    for c in STRENGTHS:
        w, anytime, sec = run_stream(targets, x, y, c)
        # Residual totals (p), outcome sum (1), total weight, sum of squared
        # weights, and count.
        emit(f"stream_c{c:g}", w, anytime, sec, p + 1 + 3)
        start = time.perf_counter()
        w = fit_penalized_raking(x, targets, calibration_strength=c)
        sec = (time.perf_counter() - start) / n
        final = summarize(w, x, t, y, truth)["final_err"]
        emit(f"batch_c{c:g}", w, final, sec, stored)

    if n <= 5000:
        for method, cls, lr in (
            ("sgd_all", OnlineRakingSGD, 5.0),
            ("mwu_all", OnlineRakingMWU, 1.0),
        ):
            w, sec = run_all_weights(cls, lr, targets, dicts)
            emit(method, w, float("nan"), sec, stored)


def order_dependence(out_dir, n_seeds, n_orders=10):
    path = out_dir / "streaming_order.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scenario", "seed", "method", "estimate_sd_across_orders"])
        for seed in range(n_seeds):
            targets, dicts = generate_scenario_data("easy", 3000, 8, seed)
            names = targets.feature_names
            x = np.array([[o[k] for k in names] for o in dicts], dtype=float)
            y, _ = outcome(x, seed, np.array([targets[k] for k in names]))
            rng = np.random.default_rng(seed)
            estimates = []
            for _ in range(n_orders):
                order = rng.permutation(len(x))
                w, _, _ = run_stream(targets, x[order], y[order], 0.1)
                estimates.append(float((w @ y[order]) / w.sum()))
            writer.writerow(
                ["standard", seed, "stream_c0.1", float(np.std(estimates, ddof=1))]
            )
            handle.flush()


def scaling(out_dir, sizes):
    path = out_dir / "streaming_scaling.csv"
    rng = np.random.default_rng(0)
    targets = Targets(**{f"f{i}": 0.5 for i in range(8)})
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["n", "method", "sec_per_obs"])
        for n in sizes:
            x = (rng.random((n, 8)) < rng.uniform(0.3, 0.7, 8)).astype(int)
            raker = StreamingRaker(targets)
            dicts = [
                dict(zip(targets.feature_names, r.tolist(), strict=True)) for r in x
            ]
            start = time.perf_counter()
            for obs in dicts:
                raker.partial_fit(obs)
            writer.writerow([n, "stream_c0.1", (time.perf_counter() - start) / n])
            start = time.perf_counter()
            ipf_weights(targets, dicts)
            writer.writerow(
                [
                    n,
                    "ipf_rerun_every_1000_amortized",
                    (time.perf_counter() - start) / 1000,
                ]
            )
            handle.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_seeds", type=int, default=50)
    parser.add_argument(
        "--quick", action="store_true", help="skip the large scenario and scaling run"
    )
    parser.add_argument("--output_dir", default="eval_outputs")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "streaming_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for name, (scenario, n, p) in SCENARIOS.items():
            if args.quick and name == "large":
                continue
            for seed in range(args.n_seeds):
                evaluate_seed(name, scenario, n, p, seed, writer, handle)
            print(f"done: {name}", flush=True)

    order_dependence(out_dir, min(args.n_seeds, 20))
    if not args.quick:
        scaling(out_dir, (10_000, 100_000, 1_000_000))


if __name__ == "__main__":
    main()
