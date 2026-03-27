#!/usr/bin/env python3
"""
Reproducible experiments for dual first-order entropy balancing.

This script solves entropy balancing via:
  1) Dual BFGS (SciPy minimize)
  2) Dual gradient descent (full batch)
  3) Dual gradient descent (mini-batch approximation, illustrative)
  4) Target revision: warm start vs cold start

Additions in this version:
  - "Hard targets / low ESS" regime using Dirichlet(alpha<1) targets.
  - BFGS evaluation counts (nfev, njev) with a separate LaTeX table.

Outputs (written to --out_dir):
  - results.csv
  - summary.csv
  - table_performance.tex
  - table_warm_start.tex
  - table_bfgs_evals.tex
  - fig_convergence_standard.pdf
  - fig_convergence_hard.pdf
  - fig_warm_start.pdf
  - fig_weight_rank.pdf
  - run_metadata.json

Usage:
  python run_experiments_v2.py --out_dir ./outputs --n_seeds 50 --shift_delta 0.05 --hard_alpha 0.1

Notes:
- Synthetic data are constructed so calibration targets are feasible by design:
    x_pop = X^T a, for simplex weights a.
- Targets use Dirichlet(alpha) sampling:
    alpha=1.0 is "easy" (diffuse weights), alpha<1 produces more concentrated a and lower ESS.
- Features are bounded using tanh so entries lie in (-1,1).
- Batch dual GD uses a data-driven step size:
    eta = 1 / lambda_max(Cov_u[x]) at lambda=0.
- The mini-batch method uses self-normalized weights within each mini-batch. This is an
  approximation (ratio estimator) and can plateau above stringent tolerances.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

# -----------------------------
# Synthetic problem generation
# -----------------------------


def _dirichlet_weights(rng: np.random.Generator, n: int, alpha: float) -> np.ndarray:
    g = rng.gamma(shape=alpha, scale=1.0, size=n)
    g = np.maximum(g, 1e-300)
    g /= g.sum()
    return g


def make_data(
    seed: int,
    n: int,
    d: int,
    alpha: float = 1.0,
    shift_delta: float | None = None,
    target_mode: str = "random",
    hard_k: int = 25,
):
    """
    Generate bounded features X and feasible targets.

    Features:
      X_ij = tanh(Z_ij), Z_ij ~ N(0,1), so X_ij in (-1,1).

    Targets (feasible by construction):
      - target_mode="random": draw a ~ Dirichlet(alpha * 1_n) over all n points.
      - target_mode="extreme": choose a direction v and place a on the hard_k points with
        largest projection X v, with a_sub ~ Dirichlet(alpha * 1_{hard_k}).

    If shift_delta is provided, also return x_pop2 computed from a convex combination
    of two independent draws under the same target_mode.
    """
    rng = np.random.default_rng(seed)
    X = np.tanh(rng.standard_normal((n, d)))  # bounded in (-1,1)
    u = np.full(n, 1.0 / n)

    def draw_a() -> np.ndarray:
        if target_mode == "random":
            return _dirichlet_weights(rng, n, alpha=alpha)

        if target_mode == "extreme":
            k = int(min(max(hard_k, 2), n))
            v = rng.standard_normal(d)
            v /= max(np.linalg.norm(v), 1e-12)
            scores = X @ v
            idx = np.argpartition(scores, -k)[-k:]
            a = np.zeros(n, dtype=float)
            a_sub = _dirichlet_weights(rng, k, alpha=alpha)
            a[idx] = a_sub
            a /= a.sum()
            return a

        raise ValueError(f"Unknown target_mode: {target_mode}")

    a1 = draw_a()
    x_pop1 = X.T @ a1

    if shift_delta is None:
        return X, u, x_pop1

    b = draw_a()
    a2 = (1.0 - shift_delta) * a1 + shift_delta * b
    a2 /= a2.sum()
    x_pop2 = X.T @ a2
    return X, u, x_pop1, x_pop2


# -----------------------------
# Dual objective# -----------------------------
# Dual objective and helpers
# -----------------------------


def dual_obj_grad(lam: np.ndarray, X: np.ndarray, u: np.ndarray, x_pop: np.ndarray):
    scores = np.log(u) - X.dot(lam)
    lse = logsumexp(scores)
    w = np.exp(scores - lse)
    D = float(lse + lam.dot(x_pop))
    grad = x_pop - X.T.dot(w)
    return D, grad, w


def moment_error(lam: np.ndarray, X: np.ndarray, u: np.ndarray, x_pop: np.ndarray):
    _, grad, w = dual_obj_grad(lam, X, u, x_pop)
    g = -grad
    return g, w


def effective_sample_size(w: np.ndarray) -> float:
    return float(1.0 / np.sum(w * w))


def estimate_L0(X: np.ndarray, u: np.ndarray) -> float:
    w = u / u.sum()
    mean = X.T @ w
    Xc = X - mean
    Cov = (Xc.T * w) @ Xc
    return float(np.linalg.eigvalsh(Cov).max())


# -----------------------------
# Solvers
# -----------------------------


@dataclass
class SolveResult:
    method: str
    scenario: str
    seed: int
    n: int
    d: int
    runtime_seconds: float
    iterations: float
    g2: float
    ginf: float
    ess: float
    converged: bool
    nfev: float
    njev: float
    notes: str


def solve_dual_gd_batch(
    X: np.ndarray,
    u: np.ndarray,
    x_pop: np.ndarray,
    tol: float,
    max_iter: int,
    eta: float | None = None,
    lam0: np.ndarray | None = None,
    track: bool = False,
):
    d = X.shape[1]
    lam = np.zeros(d) if lam0 is None else lam0.copy()

    if eta is None:
        eta = 1.0 / max(estimate_L0(X, u), 1e-12)

    history: list[tuple[int, float]] = []
    t0 = time.perf_counter()
    converged = False

    for t in range(max_iter):
        g, w = moment_error(lam, X, u, x_pop)
        g2 = float(np.linalg.norm(g))
        if track:
            history.append((t + 1, g2))
        if float(np.max(np.abs(g))) < tol:
            converged = True
            break
        lam = lam + eta * g

    runtime = time.perf_counter() - t0
    g, w = moment_error(lam, X, u, x_pop)
    return (
        lam,
        w,
        {
            "eta": float(eta),
            "nit": int(t + 1),
            "runtime": float(runtime),
            "converged": bool(converged),
            "history": history,
        },
    )


def solve_dual_bfgs(
    X: np.ndarray,
    u: np.ndarray,
    x_pop: np.ndarray,
    tol: float,
    max_iter: int,
    method: str = "BFGS",
    track: bool = False,
):
    lam0 = np.zeros(X.shape[1])
    history: list[tuple[int, float]] = []

    def fun(lam):
        D, _, _ = dual_obj_grad(lam, X, u, x_pop)
        return D

    def jac(lam):
        _, grad, _ = dual_obj_grad(lam, X, u, x_pop)
        return grad

    def callback(lamk):
        if not track:
            return
        g, _ = moment_error(lamk, X, u, x_pop)
        history.append((len(history) + 1, float(np.linalg.norm(g))))

    t0 = time.perf_counter()
    res = minimize(
        fun,
        lam0,
        jac=jac,
        method=method,
        callback=callback,
        options={"gtol": tol, "maxiter": max_iter, "disp": False},
    )
    runtime = time.perf_counter() - t0

    lam = res.x
    g, w = moment_error(lam, X, u, x_pop)
    return (
        lam,
        w,
        {
            "nit": int(getattr(res, "nit", 0)),
            "runtime": float(runtime),
            "converged": bool(getattr(res, "success", False)),
            "message": str(getattr(res, "message", "")),
            "history": history,
            "nfev": float(getattr(res, "nfev", float("nan"))),
            "njev": float(getattr(res, "njev", float("nan"))),
        },
    )


def solve_dual_gd_minibatch(
    X: np.ndarray,
    u: np.ndarray,
    x_pop: np.ndarray,
    eta: float,
    batch_size: int,
    epochs: int,
    seed: int,
):
    n, d = X.shape
    rng = np.random.default_rng(seed)
    indices = np.arange(n)

    lam = np.zeros(d)
    t0 = time.perf_counter()
    updates = 0

    for _ in range(epochs):
        rng.shuffle(indices)
        for start in range(0, n, batch_size):
            batch_idx = indices[start : start + batch_size]
            Xb = X[batch_idx]
            ub = u[batch_idx]

            scores = np.log(ub) - Xb.dot(lam)
            lse = logsumexp(scores)
            wb = np.exp(scores - lse)  # normalized over batch

            g_hat = Xb.T.dot(wb) - x_pop
            lam = lam + eta * g_hat
            updates += 1

    runtime = time.perf_counter() - t0
    g, w = moment_error(lam, X, u, x_pop)
    return lam, w, {"nit": float(updates), "runtime": float(runtime)}


# -----------------------------
# Reporting utilities
# -----------------------------


def mean_se(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    m = float(np.mean(x))
    se = float(np.std(x, ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0
    return m, se


def fmt_mean_se(m: float, se: float, digits: int = 3) -> str:
    fmt = f"{{:.{digits}f}}"
    return f"{fmt.format(m)} \\pm {fmt.format(se)}"


def fmt_sci_tex(m: float, se: float, sig: int = 2) -> str:
    if m == 0.0:
        return r"$0$"
    abs_m = abs(m)
    if 1e-3 <= abs_m < 1e3:
        return fmt_mean_se(m, se, digits=3)
    exp10 = int(math.floor(math.log10(abs_m)))
    scale = 10.0**exp10
    mant = m / scale
    mant_se = se / scale
    return rf"$({mant:.{sig}f} \pm {mant_se:.{sig}f}) \times 10^{{{exp10}}}$"


def fmt_metric(m: float, se: float, kind: str) -> str:
    if kind == "g2":
        return fmt_sci_tex(m, se, sig=2)
    if kind == "time":
        return fmt_mean_se(m, se, digits=3)
    if kind == "ess":
        return fmt_mean_se(m, se, digits=1)
    if kind in ("iter", "nfev", "njev"):
        return fmt_mean_se(m, se, digits=1)
    return fmt_mean_se(m, se, digits=3)


def write_performance_table(df: pd.DataFrame, out_path: Path) -> None:
    scenarios = [
        ("Standard (easy targets)", "standard_easy"),
        ("High-dimensional (easy targets)", "high_dim_easy"),
        ("Large-scale (easy targets)", "large_scale_easy"),
        ("Standard (hard targets)", "standard_hard"),
    ]
    methods_order = ["BFGS", "Dual GD (Batch)", "Dual GD (Mini-batch)"]

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Performance comparison (mean $\pm$ SE over seeds).}")
    lines.append(r"\label{tab:results}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Method} & \textbf{$\|g\|_2$} & \textbf{Time (s)} & \textbf{ESS} & \textbf{Iterations} \\"
    )
    lines.append(r"\midrule")

    for title, key in scenarios:
        sub = df[df["scenario"] == key].copy()
        if len(sub) == 0:
            continue
        n0 = int(sub["n"].iloc[0])
        d0 = int(sub["d"].iloc[0])
        desc = rf"$n={n0},\, d={d0}$"
        lines.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{title}: {desc}}}}} \\")
        for method in methods_order:
            subm = sub[sub["method"] == method]
            if len(subm) == 0:
                continue
            g2_m, g2_se = mean_se(subm["g2"].values)
            t_m, t_se = mean_se(subm["runtime_seconds"].values)
            ess_m, ess_se = mean_se(subm["ess"].values)
            it_m, it_se = mean_se(subm["iterations"].values)
            lines.append(
                rf"{method} & {fmt_metric(g2_m, g2_se, 'g2')} & {fmt_metric(t_m, t_se, 'time')} & {fmt_metric(ess_m, ess_se, 'ess')} & {fmt_metric(it_m, it_se, 'iter')} \\"
            )
        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    else:
        lines.append(r"\bottomrule")

    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_warmstart_table(df: pd.DataFrame, out_path: Path) -> None:
    sub = df[df["scenario"] == "warm_start"].copy()
    methods_order = ["Warm start", "Cold start"]

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Adapting to revised targets: warm start versus cold start (mean $\pm$ SE over seeds).}"
    )
    lines.append(r"\label{tab:warmstart}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Initialization} & \textbf{$\|g\|_2$} & \textbf{Time (s)} & \textbf{Iterations} \\"
    )
    lines.append(r"\midrule")

    for method in methods_order:
        subm = sub[sub["method"] == method]
        if len(subm) == 0:
            continue
        g2_m, g2_se = mean_se(subm["g2"].values)
        t_m, t_se = mean_se(subm["runtime_seconds"].values)
        it_m, it_se = mean_se(subm["iterations"].values)
        lines.append(
            rf"{method} & {fmt_metric(g2_m, g2_se, 'g2')} & {fmt_metric(t_m, t_se, 'time')} & {fmt_metric(it_m, it_se, 'iter')} \\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_bfgs_evals_table(df: pd.DataFrame, out_path: Path) -> None:
    sub = df[df["method"] == "BFGS"].copy()
    scenarios = [
        ("Standard (easy targets)", "standard_easy"),
        ("High-dimensional (easy targets)", "high_dim_easy"),
        ("Large-scale (easy targets)", "large_scale_easy"),
        ("Standard (hard targets)", "standard_hard"),
    ]

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{BFGS evaluation counts (mean $\pm$ SE over seeds).}")
    lines.append(r"\label{tab:bfgs_evals}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Scenario} & \textbf{Iterations} & \textbf{Obj evals (nfev)} & \textbf{Grad evals (njev)} \\"
    )
    lines.append(r"\midrule")

    for title, key in scenarios:
        subk = sub[sub["scenario"] == key]
        if len(subk) == 0:
            continue
        it_m, it_se = mean_se(subk["iterations"].values)
        nfev_m, nfev_se = mean_se(subk["nfev"].values)
        njev_m, njev_se = mean_se(subk["njev"].values)
        lines.append(
            rf"{title} & {fmt_metric(it_m, it_se, 'iter')} & {fmt_metric(nfev_m, nfev_se, 'nfev')} & {fmt_metric(njev_m, njev_se, 'njev')} \\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------
# Figures
# -----------------------------


def plot_convergence(
    out_path: Path,
    X: np.ndarray,
    u: np.ndarray,
    x_pop: np.ndarray,
    tol: float,
    title: str,
    bfgs_max_iter: int = 80,
) -> None:
    _, _, info_gd = solve_dual_gd_batch(
        X, u, x_pop, tol=tol, max_iter=500, eta=None, lam0=None, track=True
    )
    gd_x = [t for (t, _) in info_gd["history"]]
    gd_y = [g2 for (_, g2) in info_gd["history"]]

    bfgs_x, bfgs_y = [], []
    if bfgs_max_iter and bfgs_max_iter > 0:
        _, _, info_bfgs = solve_dual_bfgs(
            X, u, x_pop, tol=tol, max_iter=int(bfgs_max_iter), method="BFGS", track=True
        )
        bfgs_x = [t for (t, _) in info_bfgs["history"]]
        bfgs_y = [g2 for (_, g2) in info_bfgs["history"]]

    plt.figure()
    plt.plot(gd_x, gd_y, label="Dual GD (Batch)")
    if len(bfgs_x) > 0:
        plt.plot(bfgs_x, bfgs_y, label="BFGS (Dual)")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel(r"$\|g^{(t)}\|_2$")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()


def plot_warm_start(
    out_path: Path,
    X: np.ndarray,
    u: np.ndarray,
    x_pop1: np.ndarray,
    x_pop2: np.ndarray,
    tol: float,
) -> None:
    eta = 1.0 / max(estimate_L0(X, u), 1e-12)
    lam1, _, _ = solve_dual_gd_batch(
        X, u, x_pop1, tol=tol, max_iter=500, eta=eta, lam0=None, track=False
    )

    def track_from(lam0: np.ndarray, max_iter: int = 200) -> list[float]:
        lam = lam0.copy()
        vals: list[float] = []
        for _ in range(max_iter):
            g, _ = moment_error(lam, X, u, x_pop2)
            vals.append(float(np.linalg.norm(g)))
            if float(np.max(np.abs(g))) < tol:
                break
            lam = lam + eta * g
        return vals

    warm = track_from(lam1)
    cold = track_from(np.zeros(X.shape[1]))

    plt.figure()
    plt.plot(range(1, len(warm) + 1), warm, label="Warm start")
    plt.plot(range(1, len(cold) + 1), cold, label="Cold start")
    plt.yscale("log")
    plt.xlabel("Iteration after target revision")
    plt.ylabel(r"$\|g^{(t)}\|_2$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()


def plot_weight_rank(
    out_path: Path,
    X: np.ndarray,
    u: np.ndarray,
    x_pop_easy: np.ndarray,
    x_pop_hard: np.ndarray,
    tol: float,
    hard_label: str,
) -> None:
    _, w_easy, _ = solve_dual_gd_batch(
        X, u, x_pop_easy, tol=tol, max_iter=500, eta=None, lam0=None, track=False
    )
    _, w_hard, _ = solve_dual_gd_batch(
        X, u, x_pop_hard, tol=tol, max_iter=500, eta=None, lam0=None, track=False
    )

    we = np.sort(w_easy)[::-1]
    wh = np.sort(w_hard)[::-1]
    x = np.arange(1, len(we) + 1)

    plt.figure()
    plt.plot(x, we, label="Easy targets (alpha=1.0)")
    plt.plot(x, wh, label=hard_label)
    plt.yscale("log")
    plt.xlabel("Weight rank (largest to smallest)")
    plt.ylabel("Weight value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()


# -----------------------------
# Main runner
# -----------------------------


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        # Smaller and slightly looser settings for fast execution in constrained environments.
        args.large_n = min(args.large_n, 20000)
        args.large_d = min(args.large_d, 20)
        args.large_tol = max(args.large_tol, 1e-6)
        args.bfgs_maxiter = min(args.bfgs_maxiter, 15)
        args.bfgs_maxiter_large = min(args.bfgs_maxiter_large, 10)
        args.hard_n = min(getattr(args, "hard_n", 3000), 1000)
        args.hard_d = min(getattr(args, "hard_d", 10), 10)

    results: list[SolveResult] = []

    scenarios = [
        {
            "scenario": "standard_easy",
            "n": 3000,
            "d": 10,
            "tol": 1e-8,
            "max_iter": 500,
            "alpha": 1.0,
        },
        {
            "scenario": "high_dim_easy",
            "n": 3000,
            "d": 100,
            "tol": 1e-6,
            "max_iter": 500,
            "alpha": 1.0,
        },
        {
            "scenario": "large_scale_easy",
            "n": args.large_n,
            "d": args.large_d,
            "tol": args.large_tol,
            "max_iter": 200,
            "alpha": 1.0,
        },
        {
            "scenario": "standard_hard",
            "n": args.hard_n,
            "d": args.hard_d,
            "tol": 1e-6,
            "max_iter": 500,
            "alpha": args.hard_alpha,
            "target_mode": "extreme",
            "hard_k": args.hard_k,
        },
    ]

    if args.quick:
        # In quick mode we skip the large-scale scenario to keep runtime small.
        scenarios = [sc for sc in scenarios if sc["scenario"] != "large_scale_easy"]

    for sc in scenarios:
        for seed in range(args.n_seeds):
            X, u, x_pop = make_data(
                seed,
                sc["n"],
                sc["d"],
                alpha=sc["alpha"],
                target_mode=sc.get("target_mode", "random"),
                hard_k=sc.get("hard_k", 25),
            )

            lam, w, info = solve_dual_bfgs(
                X, u, x_pop, tol=sc["tol"], max_iter=200, method="BFGS", track=False
            )
            g = X.T @ w - x_pop
            results.append(
                SolveResult(
                    method="BFGS",
                    scenario=sc["scenario"],
                    seed=seed,
                    n=sc["n"],
                    d=sc["d"],
                    runtime_seconds=float(info["runtime"]),
                    iterations=float(info["nit"]),
                    g2=float(np.linalg.norm(g)),
                    ginf=float(np.max(np.abs(g))),
                    ess=effective_sample_size(w),
                    converged=bool(info["converged"]),
                    nfev=float(info["nfev"]),
                    njev=float(info["njev"]),
                    notes="SciPy minimize(BFGS)",
                )
            )

            lam, w, info = solve_dual_gd_batch(
                X,
                u,
                x_pop,
                tol=sc["tol"],
                max_iter=sc["max_iter"],
                eta=None,
                lam0=None,
                track=False,
            )
            g = X.T @ w - x_pop
            results.append(
                SolveResult(
                    method="Dual GD (Batch)",
                    scenario=sc["scenario"],
                    seed=seed,
                    n=sc["n"],
                    d=sc["d"],
                    runtime_seconds=float(info["runtime"]),
                    iterations=float(info["nit"]),
                    g2=float(np.linalg.norm(g)),
                    ginf=float(np.max(np.abs(g))),
                    ess=effective_sample_size(w),
                    converged=bool(info["converged"]),
                    nfev=float("nan"),
                    njev=float("nan"),
                    notes=f"eta={info['eta']:.3g}",
                )
            )

            if sc["scenario"] == "standard_easy":
                lam, w, info = solve_dual_gd_minibatch(
                    X, u, x_pop, eta=0.02, batch_size=50, epochs=10, seed=seed
                )
                g = X.T @ w - x_pop
                results.append(
                    SolveResult(
                        method="Dual GD (Mini-batch)",
                        scenario=sc["scenario"],
                        seed=seed,
                        n=sc["n"],
                        d=sc["d"],
                        runtime_seconds=float(info["runtime"]),
                        iterations=float(info["nit"]),
                        g2=float(np.linalg.norm(g)),
                        ginf=float(np.max(np.abs(g))),
                        ess=effective_sample_size(w),
                        converged=False,
                        nfev=float("nan"),
                        njev=float("nan"),
                        notes="batch_size=50, epochs=10, eta=0.02",
                    )
                )

    # Warm-start experiment (standard easy)
    for seed in range(args.n_seeds):
        X, u, x_pop1, x_pop2 = make_data(
            seed, n=3000, d=10, alpha=1.0, shift_delta=args.shift_delta
        )
        eta = 1.0 / max(estimate_L0(X, u), 1e-12)

        lam1, _, _ = solve_dual_gd_batch(
            X, u, x_pop1, tol=1e-8, max_iter=500, eta=eta, lam0=None, track=False
        )
        lam_w, w_w, info_w = solve_dual_gd_batch(
            X, u, x_pop2, tol=1e-8, max_iter=500, eta=eta, lam0=lam1, track=False
        )
        g = X.T @ w_w - x_pop2
        results.append(
            SolveResult(
                method="Warm start",
                scenario="warm_start",
                seed=seed,
                n=3000,
                d=10,
                runtime_seconds=float(info_w["runtime"]),
                iterations=float(info_w["nit"]),
                g2=float(np.linalg.norm(g)),
                ginf=float(np.max(np.abs(g))),
                ess=effective_sample_size(w_w),
                converged=bool(info_w["converged"]),
                nfev=float("nan"),
                njev=float("nan"),
                notes=f"shift_delta={args.shift_delta}",
            )
        )

        lam_c, w_c, info_c = solve_dual_gd_batch(
            X, u, x_pop2, tol=1e-8, max_iter=500, eta=eta, lam0=None, track=False
        )
        g = X.T @ w_c - x_pop2
        results.append(
            SolveResult(
                method="Cold start",
                scenario="warm_start",
                seed=seed,
                n=3000,
                d=10,
                runtime_seconds=float(info_c["runtime"]),
                iterations=float(info_c["nit"]),
                g2=float(np.linalg.norm(g)),
                ginf=float(np.max(np.abs(g))),
                ess=effective_sample_size(w_c),
                converged=bool(info_c["converged"]),
                nfev=float("nan"),
                njev=float("nan"),
                notes=f"shift_delta={args.shift_delta}",
            )
        )

    df = pd.DataFrame([r.__dict__ for r in results])
    df.to_csv(out_dir / "results.csv", index=False)

    summary = (
        df[
            df["scenario"].isin(
                ["standard_easy", "high_dim_easy", "large_scale_easy", "standard_hard"]
            )
        ]
        .groupby(["scenario", "method"])
        .agg(
            g2_mean=("g2", "mean"),
            g2_se=(
                "g2",
                lambda x: float(np.std(x, ddof=1) / math.sqrt(len(x)))
                if len(x) > 1
                else 0.0,
            ),
            ess_mean=("ess", "mean"),
            time_mean=("runtime_seconds", "mean"),
            it_mean=("iterations", "mean"),
        )
        .reset_index()
        .sort_values(["scenario", "method"])
    )
    summary.to_csv(out_dir / "summary.csv", index=False)

    write_performance_table(df, out_dir / "table_performance.tex")
    write_warmstart_table(df, out_dir / "table_warm_start.tex")
    write_bfgs_evals_table(df, out_dir / "table_bfgs_evals.tex")

    # Figures (representative seed=0)
    X, u, x_pop = make_data(0, n=3000, d=10, alpha=1.0)
    plot_convergence(
        out_dir / "fig_convergence_standard.pdf",
        X,
        u,
        x_pop,
        tol=1e-8,
        title="Standard (easy targets)",
        bfgs_max_iter=(0 if args.quick else 80),
    )

    Xh, uh, x_pop_hard = make_data(
        0,
        n=args.hard_n,
        d=args.hard_d,
        alpha=args.hard_alpha,
        target_mode="extreme",
        hard_k=args.hard_k,
    )
    plot_convergence(
        out_dir / "fig_convergence_hard.pdf",
        Xh,
        uh,
        x_pop_hard,
        tol=1e-6,
        title="Standard (hard targets)",
        bfgs_max_iter=(0 if args.quick else 40),
    )

    Xw, uw, x_pop1, x_pop2 = make_data(
        0, n=3000, d=10, alpha=1.0, shift_delta=args.shift_delta
    )
    plot_warm_start(out_dir / "fig_warm_start.pdf", Xw, uw, x_pop1, x_pop2, tol=1e-8)

    # Weight rank plot on shared X for easy vs hard targets
    Xr, ur, x_easy = make_data(0, n=args.hard_n, d=args.hard_d, alpha=1.0)
    _, _, x_hard = make_data(
        0,
        n=args.hard_n,
        d=args.hard_d,
        alpha=args.hard_alpha,
        target_mode="extreme",
        hard_k=args.hard_k,
    )
    plot_weight_rank(
        out_dir / "fig_weight_rank.pdf",
        Xr,
        ur,
        x_easy,
        x_hard,
        tol=1e-8,
        hard_label=f"Hard targets (alpha={args.hard_alpha}, k={args.hard_k})",
    )

    meta = {
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "pandas": pd.__version__,
        "matplotlib": __import__("matplotlib").__version__,
        "n_seeds": args.n_seeds,
        "shift_delta": args.shift_delta,
        "quick": bool(args.quick),
        "hard_alpha": args.hard_alpha,
        "hard_k": getattr(args, "hard_k", None),
        "hard_n": getattr(args, "hard_n", None),
        "hard_d": getattr(args, "hard_d", None),
        "bfgs_maxiter": getattr(args, "bfgs_maxiter", None),
        "bfgs_maxiter_large": getattr(args, "bfgs_maxiter_large", None),
        "large_n": getattr(args, "large_n", None),
        "large_d": getattr(args, "large_d", None),
        "large_tol": getattr(args, "large_tol", None),
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print("Wrote outputs to:", out_dir)
    print(summary)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="./outputs")
    p.add_argument("--n_seeds", type=int, default=50)

    # Target revision for warm-start experiment
    p.add_argument("--shift_delta", type=float, default=0.05)

    # Hard-target regime: extreme feasible targets to induce more concentrated solutions
    p.add_argument("--hard_alpha", type=float, default=0.1)
    p.add_argument("--hard_k", type=int, default=25)
    p.add_argument("--hard_n", type=int, default=3000)
    p.add_argument("--hard_d", type=int, default=10)

    # Large-scale scenario size (default matches the paper draft)
    p.add_argument("--large_n", type=int, default=50000)
    p.add_argument("--large_d", type=int, default=20)
    p.add_argument("--large_tol", type=float, default=1e-8)

    # Cap BFGS iterations in large-scale scenario (useful for quick runs)
    # Cap BFGS iterations (used for non-large scenarios)
    p.add_argument("--bfgs_maxiter", type=int, default=200)

    p.add_argument("--bfgs_maxiter_large", type=int, default=200)

    # Quick mode: reduces large-scale size and tightness to fit into constrained environments
    p.add_argument(
        "--quick",
        action="store_true",
        help="Use smaller settings for faster execution.",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
