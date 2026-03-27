#!/usr/bin/env python3
"""Generate plots from comprehensive evaluation results.

This script reads the results from comprehensive_eval.py and generates
publication-quality visualizations.

Usage:
    python scripts/plot_eval_results.py
    python scripts/plot_eval_results.py --input_dir ./eval_outputs --output_dir ./eval_plots

Outputs:
    - fig_convergence_curves.pdf     - Loss vs observations for each algorithm
    - fig_efficiency_accuracy.pdf    - Weight efficiency vs accuracy tradeoff
    - fig_weight_distributions.pdf   - Violin plots of weight distributions
    - fig_algorithm_comparison.pdf   - Bar chart comparing algorithms
    - fig_robustness.pdf            - Performance under infeasibility/drift
    - table_main_results.tex        - LaTeX table for paper
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def setup_matplotlib_style() -> None:
    """Configure matplotlib for publication-quality plots."""
    if not MATPLOTLIB_AVAILABLE:
        return

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 12,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )


def plot_algorithm_comparison(
    summary: pd.DataFrame,
    output_path: Path,
    scenario: str = "standard",
) -> None:
    """Create bar chart comparing algorithms on key metrics."""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping plot")
        return

    scenario_data = summary[summary["scenario"] == scenario].copy()
    if len(scenario_data) == 0:
        print(f"No data for scenario: {scenario}")
        return

    algo_order = [
        "batch_ipf",
        "sgd_fixed",
        "sgd_rm",
        "mwu",
        "periodic_ipf_50",
        "periodic_ipf_100",
    ]
    algo_labels = [
        "Batch IPF",
        "SGD (fixed)",
        "SGD (R-M)",
        "MWU",
        "Periodic IPF\n(50)",
        "Periodic IPF\n(100)",
    ]

    scenario_data = scenario_data[scenario_data["algorithm"].isin(algo_order)]
    scenario_data = scenario_data.set_index("algorithm").loc[
        [a for a in algo_order if a in scenario_data["algorithm"].values]
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    x = np.arange(len(scenario_data))
    width = 0.6

    colors = plt.cm.Set2(np.linspace(0, 1, len(scenario_data)))

    ax = axes[0]
    loss_vals = scenario_data["final_loss_mean"].values
    loss_err = scenario_data["final_loss_std"].values / np.sqrt(50)
    ax.bar(x, loss_vals, width, yerr=loss_err, color=colors, capsize=3)
    ax.set_ylabel("Final Loss (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [algo_labels[algo_order.index(a)] for a in scenario_data.index],
        rotation=45,
        ha="right",
    )
    ax.set_title("Accuracy")
    ax.set_yscale("log")

    ax = axes[1]
    eff_vals = scenario_data["weight_efficiency_mean"].values * 100
    eff_err = scenario_data["weight_efficiency_std"].values * 100 / np.sqrt(50)
    ax.bar(x, eff_vals, width, yerr=eff_err, color=colors, capsize=3)
    ax.set_ylabel("Weight Efficiency (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [algo_labels[algo_order.index(a)] for a in scenario_data.index],
        rotation=45,
        ha="right",
    )
    ax.set_title("Efficiency")
    ax.set_ylim(0, 100)

    ax = axes[2]
    time_vals = scenario_data["runtime_sec_mean"].values
    time_err = scenario_data["runtime_sec_std"].values / np.sqrt(50)
    ax.bar(x, time_vals, width, yerr=time_err, color=colors, capsize=3)
    ax.set_ylabel("Runtime (seconds)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [algo_labels[algo_order.index(a)] for a in scenario_data.index],
        rotation=45,
        ha="right",
    )
    ax.set_title("Speed")

    fig.suptitle(f"Algorithm Comparison: {scenario.replace('_', ' ').title()} Scenario")
    plt.tight_layout()
    plt.savefig(output_path / "fig_algorithm_comparison.pdf")
    plt.close()


def plot_efficiency_accuracy_tradeoff(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create scatter plot of efficiency vs accuracy tradeoff."""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    algo_colors = {
        "batch_ipf": "tab:blue",
        "sgd_fixed": "tab:orange",
        "sgd_rm": "tab:green",
        "mwu": "tab:red",
        "periodic_ipf_50": "tab:purple",
        "periodic_ipf_100": "tab:brown",
    }

    scenario_markers = {
        "small_easy": "o",
        "standard": "s",
        "large": "^",
        "hard_targets": "D",
        "high_dim": "v",
    }

    for algo in algo_colors:
        for scenario, marker in scenario_markers.items():
            data = summary[
                (summary["algorithm"] == algo) & (summary["scenario"] == scenario)
            ]
            if len(data) == 0:
                continue

            ax.scatter(
                data["final_loss_mean"],
                data["weight_efficiency_mean"] * 100,
                c=algo_colors[algo],
                marker=marker,
                s=80,
                alpha=0.7,
                label=f"{algo} ({scenario})" if scenario == "standard" else "",
            )

    ax.set_xlabel("Final Loss (log scale)")
    ax.set_ylabel("Weight Efficiency (%)")
    ax.set_xscale("log")
    ax.set_title("Efficiency vs Accuracy Tradeoff")

    algo_patches = [
        mpatches.Patch(color=c, label=a.replace("_", " ").title())
        for a, c in algo_colors.items()
    ]
    ax.legend(handles=algo_patches, loc="lower left")

    plt.tight_layout()
    plt.savefig(output_path / "fig_efficiency_accuracy.pdf")
    plt.close()


def plot_scenario_heatmap(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create heatmap showing algorithm performance across scenarios."""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping plot")
        return

    scenarios = ["small_easy", "standard", "large", "hard_targets", "high_dim", "drift"]
    algorithms = ["batch_ipf", "sgd_fixed", "sgd_rm", "mwu"]

    loss_matrix = np.zeros((len(scenarios), len(algorithms)))
    eff_matrix = np.zeros((len(scenarios), len(algorithms)))

    for i, scenario in enumerate(scenarios):
        for j, algo in enumerate(algorithms):
            data = summary[
                (summary["scenario"] == scenario) & (summary["algorithm"] == algo)
            ]
            if len(data) > 0:
                loss_matrix[i, j] = data["final_loss_mean"].values[0]
                eff_matrix[i, j] = data["weight_efficiency_mean"].values[0] * 100
            else:
                loss_matrix[i, j] = np.nan
                eff_matrix[i, j] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    im = ax.imshow(np.log10(loss_matrix + 1e-10), cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(algorithms)))
    ax.set_xticklabels([a.replace("_", "\n") for a in algorithms])
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels([s.replace("_", "\n") for s in scenarios])
    ax.set_title("Final Loss (log10)")
    fig.colorbar(im, ax=ax)

    for i in range(len(scenarios)):
        for j in range(len(algorithms)):
            val = loss_matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2e}", ha="center", va="center", fontsize=7)

    ax = axes[1]
    im = ax.imshow(eff_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(algorithms)))
    ax.set_xticklabels([a.replace("_", "\n") for a in algorithms])
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels([s.replace("_", "\n") for s in scenarios])
    ax.set_title("Weight Efficiency (%)")
    fig.colorbar(im, ax=ax)

    for i in range(len(scenarios)):
        for j in range(len(algorithms)):
            val = eff_matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path / "fig_scenario_heatmap.pdf")
    plt.close()


def plot_robustness_comparison(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Compare algorithm performance on challenging scenarios."""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    challenging_scenarios = ["hard_targets", "infeasible", "drift"]
    algorithms = ["batch_ipf", "sgd_fixed", "sgd_rm", "mwu"]
    algo_labels = ["Batch IPF", "SGD (fixed)", "SGD (R-M)", "MWU"]

    x = np.arange(len(challenging_scenarios))
    width = 0.2

    ax = axes[0]
    for i, (algo, label) in enumerate(zip(algorithms, algo_labels, strict=True)):
        losses = []
        for scenario in challenging_scenarios:
            data = summary[
                (summary["scenario"] == scenario) & (summary["algorithm"] == algo)
            ]
            if len(data) > 0:
                losses.append(data["final_loss_mean"].values[0])
            else:
                losses.append(np.nan)

        ax.bar(x + i * width, losses, width, label=label)

    ax.set_ylabel("Final Loss")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([s.replace("_", "\n") for s in challenging_scenarios])
    ax.set_title("Loss on Challenging Scenarios")
    ax.legend()
    ax.set_yscale("log")

    ax = axes[1]
    for i, (algo, label) in enumerate(zip(algorithms, algo_labels, strict=True)):
        effs = []
        for scenario in challenging_scenarios:
            data = summary[
                (summary["scenario"] == scenario) & (summary["algorithm"] == algo)
            ]
            if len(data) > 0:
                effs.append(data["weight_efficiency_mean"].values[0] * 100)
            else:
                effs.append(np.nan)

        ax.bar(x + i * width, effs, width, label=label)

    ax.set_ylabel("Weight Efficiency (%)")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([s.replace("_", "\n") for s in challenging_scenarios])
    ax.set_title("Efficiency on Challenging Scenarios")
    ax.legend()
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_path / "fig_robustness.pdf")
    plt.close()


def generate_latex_table(
    summary: pd.DataFrame,
    output_path: Path,
    n_seeds: int = 50,
) -> None:
    """Generate LaTeX table for paper."""
    available_scenarios = summary["scenario"].unique().tolist()
    scenarios = [
        s for s in ["standard", "large", "hard_targets"] if s in available_scenarios
    ]

    algorithms = ["batch_ipf", "sgd_fixed", "sgd_rm", "mwu", "periodic_ipf_100"]
    algo_labels = {
        "batch_ipf": "Batch IPF",
        "sgd_fixed": "SGD (fixed LR)",
        "sgd_rm": "SGD (Robbins-Monro)",
        "mwu": "MWU",
        "periodic_ipf_100": "Periodic IPF (100)",
    }

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Comparison of raking algorithms (mean $\pm$ SE over {n_seeds} seeds).}}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"\textbf{Scenario} & \textbf{Algorithm} & \textbf{Loss} & \textbf{ESS} & \textbf{Efficiency} & \textbf{Runtime (s)} \\",
        r"\midrule",
    ]

    scenario_had_data = False

    for scenario in scenarios:
        scenario_label = scenario.replace("_", " ").title()
        first_row = True

        for algo in algorithms:
            data = summary[
                (summary["scenario"] == scenario) & (summary["algorithm"] == algo)
            ]
            if len(data) == 0:
                continue

            row = data.iloc[0]
            loss_mean = row["final_loss_mean"]
            loss_se = row["final_loss_std"] / np.sqrt(n_seeds)
            ess_mean = row["ess_mean"]
            ess_se = row["ess_std"] / np.sqrt(n_seeds)
            eff_mean = row["weight_efficiency_mean"] * 100
            eff_se = row["weight_efficiency_std"] * 100 / np.sqrt(n_seeds)
            time_mean = row["runtime_sec_mean"]
            time_se = row["runtime_sec_std"] / np.sqrt(n_seeds)

            if first_row:
                scenario_col = scenario_label
                first_row = False
                scenario_had_data = True
            else:
                scenario_col = ""

            if loss_mean < 0.001:
                loss_str = f"${loss_mean:.2e}$"
            else:
                loss_str = f"${loss_mean:.4f} \\pm {loss_se:.4f}$"

            lines.append(
                f"{scenario_col} & {algo_labels[algo]} & "
                f"{loss_str} & "
                f"${ess_mean:.0f} \\pm {ess_se:.0f}$ & "
                f"${eff_mean:.1f}\\% \\pm {eff_se:.1f}\\%$ & "
                f"${time_mean:.3f} \\pm {time_se:.3f}$ \\\\"
            )

        if scenario_had_data and scenario != scenarios[-1]:
            lines.append(r"\midrule")
            scenario_had_data = False

    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    else:
        lines.append(r"\bottomrule")
    lines.extend(
        [
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    (output_path / "table_main_results.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate plots from evaluation results"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./eval_outputs",
        help="Directory containing evaluation results",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_outputs",
        help="Directory to save plots",
    )
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    summary_file = input_path / "summary.csv"
    metadata_file = input_path / "metadata.json"

    if not summary_file.exists():
        print(f"Error: {summary_file} not found. Run comprehensive_eval.py first.")
        return

    summary = pd.read_csv(summary_file)

    n_seeds = 50
    if metadata_file.exists():
        import json

        with open(metadata_file) as f:
            metadata = json.load(f)
            n_seeds = metadata.get("n_seeds", 50)

    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available. Only generating LaTeX table.")
        generate_latex_table(summary, output_path, n_seeds=n_seeds)
        return

    setup_matplotlib_style()

    print("Generating algorithm comparison plot...")
    plot_algorithm_comparison(summary, output_path)

    print("Generating efficiency vs accuracy plot...")
    plot_efficiency_accuracy_tradeoff(summary, output_path)

    print("Generating scenario heatmap...")
    plot_scenario_heatmap(summary, output_path)

    print("Generating robustness comparison...")
    plot_robustness_comparison(summary, output_path)

    print("Generating LaTeX table...")
    generate_latex_table(summary, output_path, n_seeds=n_seeds)

    print(f"\nAll outputs saved to {output_path}")


if __name__ == "__main__":
    main()
