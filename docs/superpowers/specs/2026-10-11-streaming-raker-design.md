# Streaming raking in fixed memory: design

## Problem

The manuscript `ms/onlinerake.tex` describes a streaming algorithm that updates
only the arriving observation's weight from running totals, at O(Kp) cost.
Three findings (2026-10-11) invalidate it as written:

1. Its experiments ran the package's `OnlineRakingSGD`/`OnlineRakingMWU`, which
   re-optimize every retained weight on each arrival (O(n·d·K) per arrival).
2. The algorithm as written (fixed eta, K=3) does not calibrate: the gradient
   with respect to the new weight is (1/W)·sum_j (m_j - t_j)(x_j - m_j), roughly
   1/t^2, so new weights stay near one. Prototype: easy scenario, n=3000, p=8,
   final mean |margin error| 0.068 versus 0.068 unweighted.
3. Rerunning batch IPF is cheap (numpy, p=8: 0.02 s at n=50k, 1.8 s at n=5M), so
   periodic IPF beats the all-weights online methods on accuracy and cost.

What periodic IPF cannot do is calibrate a stream that is not stored. That is
the contribution this work targets: raking with O(p + q) memory, no past rows
retained, plus weighted estimates of q outcomes.

## Method

Notation: arriving row x in R^p, targets t in R^p, design weight d > 0,
weight w. Running state after previous arrivals: residual totals
r = sum_i w_i (x_i - t) (equal to W·(m - t)), total weight W, sum of squared
weights W2, weighted outcome sums Y in R^q, count n.

Batch objective (penalized calibration with the raking distance; Deville and
Sarndal 1992 for the distance, Bardsley and Chambers 1984 for penalizing the
constraint residual instead of enforcing it):

    F(w) = sum_i D(w_i, d_i) + (c/2) * ||sum_i w_i (x_i - t)||^2
    D(w, d) = w log(w/d) - w + d

c > 0 is the calibration strength. As c -> infinity, the minimizer of F tends
to the raking (IPF) solution when the margins are feasible.

Arrival update: minimize F over the arriving weight only, past weights fixed.
With delta = x - t and r0 the residual before the arrival,

    f(w) = D(w, d) + (c/2) ||r0 + w delta||^2      (strictly convex, w > 0)
    f'(w) = log(w/d) + c delta.r0 + c q w,  q = ||delta||^2

Setting f'(w) = 0 and substituting u = c q w gives u + log u = y with
y = log(c q d) - c delta.r0, so u = omega(y), the Wright omega function
(`scipy.special.wrightomega`), and w = omega(y) / (c q). If q = 0, w = d.
Because f is convex, clipping w to [min_weight, max_weight] gives the
constrained minimizer. Cost O(p) per arrival; no learning rate.

Batch counterpart: the minimizer of F satisfies w_i = d_i exp(-lambda.delta_i)
with lambda = c·r, so it is an exponential tilt found from the p-dimensional
strictly convex dual

    G(lambda) = [sum_i d_i exp(-lambda.delta_i) + ||lambda||^2 / (2c)] / sum_i d_i

solved with `scipy.optimize.minimize(method="trust-exact")` using the analytic
gradient and Hessian. Dividing by sum_i d_i keeps the gradient on a scale where
SciPy's `gtol` is meaningful (unscaled, it reported failure at convergence).
This is entropy balancing (Hainmueller 2012) with a ridge on the multipliers;
as c -> infinity it reproduces IPF (prototype: 2e-8 relative at c = 1e6).
Comparing it with the single pass, at the same c, isolates the cost of not
storing the data. Cyclic coordinate descent with the arrival update was tried
and rejected: it fixes the margins quickly but had not reached the
minimum-distance weights after 500 passes (weights up to 9x off IPF).

Each arrival's weight is the same tilt with the running multiplier:
w = d exp(-lambda.delta) with lambda = c·r after the update. The single pass is
therefore a stochastic approximation of the batch multiplier, which is how the
paper should explain it.

## Outputs

1. `onlinerake.StreamingRaker` and `onlinerake.fit_penalized_raking`
   (module `onlinerake/streaming_raker.py`). Default calibration strength
   c = 0.1, chosen from the prototype frontier and rechecked against the
   50-seed results before the paper is written.
2. `scripts/eval/streaming_eval.py`, which writes per-seed results to CSV,
   and `scripts/eval/plot_streaming_eval.py` for figures and a LaTeX table.
3. Rewritten `ms/onlinerake.tex` and rebuilt `ms/onlinerake.pdf`.
4. README, docs, CHANGELOG, and repository CLAUDE.md that recommend periodic
   `BatchIPF` when rows can be stored and `StreamingRaker` when they cannot.
5. Updated research-site catalog entry and synced PDF.

## Success criteria

- The arrival update solves f'(w) = 0 to |f'(w)| < 1e-9 (relative) on random
  inputs, checked against a bisection that shares no code with it.
- Running state equals a full recomputation from stored weights to 1e-9.
- State size is independent of n when weights are not kept (structural test),
  and time per arrival at n = 40,000 is under 2x that at n = 2,500.
- `fit_penalized_raking` at c = 1e6 matches `BatchIPF` weights (mean-normalized)
  to 1e-6 relative on feasible binary data. The two share no code.
- `StreamingRaker` with c = 0.1 on the easy scenario (5 seeds) reaches mean
  |margin error| below 0.005 with ESS/n above 0.7 (prototype: 0.0018, 0.76).
- The superseded fixed-eta update leaves the mean margin error within 2% of the
  unweighted error on the easy scenario (pins finding 2).
- 50-seed evaluation reports, per scenario and method: final and anytime margin
  error, ESS/n, bias and RMSE of a weighted outcome mean, time per arrival,
  and retained state. The paper's claims are written from those numbers.
- `ruff check`, `ruff format --check`, `pyright`, the full pytest suite, and
  `sphinx-build -W` pass.

## Out of scope

- Proving convergence or bias bounds for the single pass; the paper reports
  them empirically and says so.
- Removing `OnlineRakingSGD`/`OnlineRakingMWU`; other modules depend on them.
  Their documentation changes, not their behavior.
- Continuous features in `fit_penalized_raking`'s IPF comparison (`BatchIPF`
  is binary-only). `StreamingRaker` itself accepts continuous mean targets.
- Rolling-window or forgetting variants for drift.
