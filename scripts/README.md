# Scripts

This directory contains scripts for evaluation, benchmarking, and figure generation.

## Directory Structure

```
scripts/
├── eval/                              # Evaluation and benchmarking
│   ├── comprehensive_eval.py          # Head-to-head algorithm comparison
│   ├── plot_eval_results.py           # Generate plots from evaluation results
│   └── run_experiments.py             # Entropy balancing experiments
├── figures/                           # Manuscript figure generation
│   └── generate_fig_age_margin_error.py
└── README.md
```

## Usage

### Evaluation Scripts

Run comprehensive algorithm comparison (quick mode):
```bash
uv run python scripts/eval/comprehensive_eval.py --n_seeds 5 --quick
```

Run full evaluation (50 seeds):
```bash
uv run python scripts/eval/comprehensive_eval.py --n_seeds 50
```

Generate plots from evaluation results:
```bash
uv run python scripts/eval/plot_eval_results.py
```

### Figure Generation

Generate manuscript figures:
```bash
uv run python scripts/figures/generate_fig_age_margin_error.py
```

## Output

- Evaluation results are saved to `eval_outputs/` (gitignored)
- Generated figures in `scripts/figures/` are gitignored (`.png`, `.pdf`)
