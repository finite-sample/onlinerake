# Online calibration: implementation and checks

`OnlineRakingMWU` adjusts survey weights as records arrive. It is useful when
the accumulated sample differs from known population proportions or means and
you need updated weights during collection. Call `partial_fit(observation)` and
inspect `margins`, `loss`, `weights`, and `effective_sample_size` at any point.

## What is implemented

Each arriving row starts with weight one. For the accumulated feature matrix
X and positive weights w, the weighted means are m = Xᵀw / Σw and the loss is
L = Σⱼ(mⱼ − targetⱼ)². The weight gradient is
gᵢ = 2 Σⱼ(mⱼ − targetⱼ)(Xᵢⱼ − mⱼ) / Σw.

For each configured step, MWU computes
wᵢ ← exp(clip(log(wᵢ) − learning_rate × gᵢ, log(min_weight), log(max_weight))).
The SGD class uses an additive, bounded step on the same loss. Binary and
continuous targets share this calculation. Scale continuous inputs and their
targets consistently: squared errors depend on measurement units.

These are incremental fitting interfaces over retained data. Every arrival
revisits all rows; memory is O(n × d), work per arrival O(n × d × n_sgd_steps).
This implementation is appropriate for moderate streams, not a constant-memory
algorithm for unlimited data. Changing the arrival order can change the weights.

For comparison, **batch entropy balancing** chooses weights after collecting
the whole dataset, minimizing their departure from a reference distribution
subject to matching the target means. Online MWU instead takes a fixed number
of steps toward those targets after each arrival. `BatchIPF` supplies a
full-sample reference for binary margins. Matching margins does not imply that
two methods chose the same weights.

## Corrected failures

The audit reproduced these failures on default-branch commit `31b5d5a`:

- Online convergence meant the loss had stopped changing, even when every
  observed value was zero and the target was 0.8. It also remained true after
  later arrivals worsened calibration. It now checks current squared error and
  resets when the target is missed. History agrees with the current flag.
- IPF stopped on a small change in loss instead of checking actual margin
  errors. Both initial and incremental fitting now require those errors to
  meet the configured tolerance. Empty incremental input no longer divides by zero.
- Nonfinite continuous values and invalid binary values could silently enter
  the calculation. Observation values, targets, and core numeric settings now
  have finite/domain checks. Invalid observations leave fitted data intact.
- Clipping an exponent before multiplying by the old weight could still
  overflow. MWU now bounds the updated log weight before exponentiation.
- Requested weight statistics could describe an earlier sample. They now
  describe current weights; optional automatic history collection remains
  controlled by `compute_weight_stats`.
- Extreme fitted weights were treated as proof that targets were infeasible.
  They now generate tuning warnings. The feasibility helper is explicitly a
  marginal support screen, not a certificate of joint feasibility under bounds.
- Positive learning-rate floors were ignored when assessing series convergence.
  They now correctly fail square summability. The default polynomial and
  inverse-time floors are zero.
- The numerical regret/convergence-bound helpers had no established bound for
  this growing, clipped problem. They were removed. The learning-rate heuristic
  is named `suggest_mwu_learning_rate`; schedule checks do not certify balance.

`convergence_window` is the minimum number of observations before the online
convergence flag can become true. The default loss tolerance is 1e-6;
`check_convergence(tolerance=...)` allows an explicit threshold. IPF uses its
`tolerance` for maximum absolute margin error instead.

## Independent checks and usefulness

The correctness tests compare the gradient with finite differences of the
objective, one MWU step with a separately calculated update, and IPF with a
known 2-by-2 entropy projection. They cover infeasible targets, changing
streams, invalid inputs, extreme learning rates, current diagnostics, and
learning-rate floors. These checks test computations, not documentation claims.

`examples/streaming_mwu.py` evaluates stationary bias, a sampling shift at row
251, and mixed binary/continuous calibration. It fixes seeds 0–9, 500 arrivals,
default MWU settings, and evaluation at every prefix from 100 through 500.
Targets remain fixed; all weighting uses only observations already received.
The example reports reduction in average squared imbalance and final effective
sample size. The latter exposes the cost of unequal weights. It does not
measure outcome-estimation error or establish a general convergence theorem.

The useful online implementation and its regression tests live in `onlinerake`.
The separate `adaptive-eb` prototype is not a runtime dependency and does not
need to be maintained to provide this functionality. Its fixed-sample dual
solver is a different optimization problem; it has not been copied here or
claimed as a new online algorithm.
