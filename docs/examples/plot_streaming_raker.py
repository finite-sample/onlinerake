"""
Raking a biased stream in fixed memory
======================================

Rows arrive from a sample that has too many women, young people and college
graduates, and an outcome that depends on all three. ``StreamingRaker``
weights each row as it arrives and keeps only running totals; ``BatchIPF``
rakes the full stored sample at the end, which is the benchmark.
"""

# %%
# A biased stream
# ---------------
import matplotlib.pyplot as plt
import numpy as np

from onlinerake import BatchIPF, StreamingRaker, Targets

targets = Targets(female=0.5, young=0.3, college=0.35)
names = targets.feature_names
population = np.array([targets[name] for name in names])
sample_shares = {"college": 0.45, "female": 0.65, "young": 0.45}

rng = np.random.default_rng(0)
n = 20_000
rows = np.column_stack([rng.random(n) < sample_shares[name] for name in names]).astype(
    int
)
coef = {"college": 2.0, "female": 1.0, "young": -1.5}
y = rows @ np.array([coef[name] for name in names]) + rng.normal(0, 1, n)
true_mean = float(population @ np.array([coef[name] for name in names]))

# %%
# Weight the stream as it arrives
# -------------------------------
raker = StreamingRaker(targets, outcomes=["y"])
checkpoints = np.unique(np.geomspace(10, n, 40).astype(int))
stream_error, raw_error = [], []
for i, (row, yi) in enumerate(zip(rows, y, strict=True), start=1):
    raker.partial_fit(dict(zip(names, row.tolist(), strict=True)), outcomes={"y": yi})
    if i in checkpoints:
        margins = np.array([raker.margins[name] for name in names])
        stream_error.append(np.abs(margins - population).max())
        raw_error.append(np.abs(rows[:i].mean(axis=0) - population).max())

ipf = BatchIPF(targets).fit([dict(zip(names, r.tolist(), strict=True)) for r in rows])
w = ipf.weights

print(f"true mean of y: {true_mean:.3f}")
print(f"{'':<16} {'estimate':>8} {'ESS':>7} {'numbers kept':>13}")
print(f"{'unweighted':<16} {y.mean():8.3f} {n:7d} {'':>13}")
print(
    f"{'StreamingRaker':<16} {raker.estimate('y'):8.3f} "
    f"{raker.effective_sample_size:7.0f} {len(names) + 5:13d}"
)
print(
    f"{'BatchIPF':<16} {w @ y / w.sum():8.3f} "
    f"{w.sum() ** 2 / (w @ w):7.0f} {rows.size + n:13d}"
)

# %%
# Margin error as the stream grows
# --------------------------------
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(checkpoints, raw_error, label="unweighted")
ax.plot(checkpoints, stream_error, label="StreamingRaker")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("rows seen")
ax.set_ylabel("largest margin error")
ax.legend()
fig.tight_layout()
plt.show()

# %%
# Design weights and trimming
# ---------------------------
# Rows can arrive with design weights in any units. Heavy-tailed design
# weights make the estimate noisy; ``trim`` caps each weight at a multiple of
# the mean weight so far, trading a little bias for a larger effective sample.
design = 5000 * rng.lognormal(0.0, 1.5, n)
for trim in (None, 3.0):
    weighted = StreamingRaker(targets, trim=trim, outcomes=["y"])
    for row, yi, d in zip(rows, y, design, strict=True):
        weighted.partial_fit(
            dict(zip(names, row.tolist(), strict=True)),
            design_weight=float(d),
            outcomes={"y": yi},
        )
    print(
        f"trim={trim}: estimate {weighted.estimate('y'):.3f}, "
        f"ESS {weighted.effective_sample_size:.0f}"
    )
