"""Figures and table for the streaming raking manuscript.

Usage:
    uv run python scripts/eval/plot_streaming_eval.py --input_dir eval_outputs \
        --output_dir ms
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import NullFormatter

ORDER = ["standard", "large", "hard", "high_dim", "drift", "small"]
TABLE_METHODS = [
    "raw",
    "ipf",
    "periodic_ipf_100",
    "periodic_ipf_1000",
    "periodic_ipf_doubling",
    "batch_c0.3",
    "stream_c0.3",
    "stream_c1",
    "sgd_all",
    "mwu_all",
]
PERIODIC = {
    "periodic_ipf_100": "IPF every 100",
    "periodic_ipf_1000": "IPF every 1000",
    "periodic_ipf_doubling": "IPF at doubling",
}


def frontier(results, out):
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for ax, scenario in zip(axes.flat, ORDER, strict=True):
        d = (
            results[results.scenario == scenario]
            .groupby("method")[["final_err", "ess_frac"]]
            .mean()
        )
        for family, style, label in (
            ("stream_c", "o-", "stream"),
            ("batch_c", "s--", "batch"),
        ):
            rows = d[d.index.str.startswith(family)]
            rows = rows.assign(
                c=rows.index.str.removeprefix(family).astype(float)
            ).sort_values("c")
            ax.plot(rows.ess_frac, rows.final_err, style, label=label)
        for method, style in (("ipf", "k*"), ("raw", "kx")):
            if method in d.index:
                ax.plot(
                    d.loc[method, "ess_frac"],
                    d.loc[method, "final_err"],
                    style,
                    label=method,
                )
        ax.set_title(scenario)
        ax.set_xlabel("ESS / n")
        ax.set_ylabel("mean |margin error|")
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_frontier.pdf")
    plt.close(fig)


def tradeoff(results, out):
    """During-stream error against compute; memory is in the caption."""
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for ax, scenario in zip(axes.flat, ORDER, strict=True):
        d = (
            results[results.scenario == scenario]
            .groupby("method")[["anytime_err", "sec_per_obs"]]
            .mean()
        )
        d["us"] = d.sec_per_obs * 1e6
        stream = d[d.index.str.startswith("stream_c")]
        stream = stream.assign(
            c=stream.index.str.removeprefix("stream_c").astype(float)
        ).sort_values("c")
        ax.plot(stream.us, stream.anytime_err, "o-", label="stream (c varies)")
        for method, label in PERIODIC.items():
            if method in d.index:
                ax.plot(
                    d.loc[method, "us"], d.loc[method, "anytime_err"], "s", label=label
                )
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(scenario)
        ax.set_xlabel("microseconds per arrival")
        ax.set_ylabel("anytime mean |margin error|")
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_tradeoff.pdf")
    plt.close(fig)


def scaling(scale, out):
    wide = scale.pivot(index="n", columns="method", values="sec_per_obs") * 1e6
    ax = wide.plot(logx=True, logy=True, marker="o", figsize=(5, 3.5))
    ax.set_ylabel("microseconds per arrival")
    ax.figure.tight_layout()
    ax.figure.savefig(out / "fig_scaling.pdf")
    plt.close(ax.figure)


def table(results, order, out):
    d = results[results.method.isin(TABLE_METHODS)].copy()
    d["abs_bias"] = d.bias.abs()
    d["anytime_abs_bias"] = d.anytime_bias
    columns = [
        "final_err",
        "anytime_err",
        "ess_frac",
        "abs_bias",
        "anytime_abs_bias",
        "sec_per_obs",
    ]
    g = d.groupby(["scenario", "method"])[columns].mean()
    g["sec_per_obs"] *= 1e6
    g = g.rename(columns={"sec_per_obs": "us_per_obs"})
    g = g.reindex([(s, m) for s in ORDER for m in TABLE_METHODS if (s, m) in g.index])
    g.to_latex(
        out / "table_streaming.tex", na_rep="--", float_format="%.4f", escape=True
    )
    order_sd = order.estimate_sd_across_orders.mean()
    standard = results[
        (results.scenario == "standard") & (results.method == "stream_c0.3")
    ]
    print(
        f"order SD {order_sd:.4f} vs across-seed SD {standard.bias.std():.4f}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", default="eval_outputs")
    parser.add_argument("--output_dir", default="ms")
    args = parser.parse_args()
    src, out = Path(args.input_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(src / "streaming_results.csv")
    frontier(results, out)
    tradeoff(results, out)
    scaling(pd.read_csv(src / "streaming_scaling.csv"), out)
    table(results, pd.read_csv(src / "streaming_order.csv"), out)


if __name__ == "__main__":
    main()
