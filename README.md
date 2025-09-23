## onlinerake: Streaming Survey Raking Via MWU and SGD

[![PyPI version](https://img.shields.io/pypi/v/onlinerake.svg)](https://pypi.org/project/onlinerake/)
[![PyPI Downloads](https://static.pepy.tech/badge/onlinerake)](https://pepy.tech/projects/onlinerake)
[![Documentation](https://img.shields.io/badge/docs-github.io-blue)](https://finite-sample.github.io/onlinerake/)
[![Python application](https://github.com/finite-sample/onlinerake/actions/workflows/tests.yml/badge.svg)](https://github.com/finite-sample/onlinerake/actions/workflows/tests.yml)

Modern online surveys and passive data collection streams generate
responses one record at a time.  Classic weighting methods such as
iterative proportional fitting (IPF, or “raking”) and calibration
weighting are inherently *batch* procedures: they reprocess the entire
dataset whenever a new case arrives.  The `onlinerake` package
provides **incremental**, per‑observation updates to survey weights so
that weighted margins track known population totals in real time.

The package implements two complementary algorithms:

* **SGD raking** – an additive update that performs stochastic
  gradient descent on a squared–error loss over the margins.  It
  produces smooth weight trajectories and maintains high effective
  sample size (ESS).
* **MWU raking** – a multiplicative update inspired by the
  multiplicative‑weights update rule.  It corresponds to mirror
  descent under the Kullback–Leibler divergence and yields weight
  distributions reminiscent of classic IPF.  However, it can produce
  heavier tails when the learning rate is large.

Both methods share the same API: call `.partial_fit(obs)` for each
incoming observation and inspect properties such as `.margins`, `.loss`
and `.effective_sample_size` to monitor progress.

## Installation

Install from PyPI:

```bash
pip install onlinerake
```

For development, clone the repository and install in editable mode:

```bash
git clone https://github.com/finite-sample/onlinerake.git
cd onlinerake
pip install -e .
```

No external dependencies are required beyond `numpy` and `pandas`.

## Usage

```python
from onlinerake import OnlineRakingSGD, OnlineRakingMWU, Targets

# define target population margins (proportion of the population with indicator = 1)
targets = Targets(age=0.5, gender=0.5, education=0.4, region=0.3)

# instantiate a raker
raker = OnlineRakingSGD(targets, learning_rate=5.0)

# stream demographic observations
for obs in stream_of_dicts:
    raker.partial_fit(obs)
    print(raker.margins)  # current weighted margins

print("final effective sample size", raker.effective_sample_size)
```

To use the multiplicative‑weights version, replace
`OnlineRakingSGD` with `OnlineRakingMWU` and adjust the
`learning_rate` (a typical default is `1.0`).  See the docstrings
for full parameter descriptions.

## Simulation results

To understand the behaviour of the two update rules we simulated
three typical non‑stationary bias patterns: a **linear drift** in
demographic composition, a **sudden shift** halfway through the stream,
and an **oscillation** around the target frame.  For each scenario we
generated 300 observations per seed and averaged results over five
random seeds.  SGD used a learning rate of 5.0 and MWU used a
learning rate of 1.0 with three update steps per observation.  The
table below summarises the mean improvement in absolute margin error
relative to the unweighted baseline (positive values indicate an
improvement), the final effective sample size (ESS) and the mean final
loss (squared‑error on margins).  Higher ESS and larger improvements
are better.

| Scenario | Method | Age Imp (%) | Gender Imp (%) | Education Imp (%) | Region Imp (%) | Overall Imp (%) | Final ESS | Final Loss |
|---------|--------|-------------|---------------|------------------|---------------|----------------|---------:|-----------:|
| linear | SGD | 82.8 | 78.6 | 76.8 | 67.5 | 77.0 | 251.8 | 0.00147 |
| linear | MWU | 57.2 | 53.6 | 46.9 | 34.6 | 48.8 | 240.9 | 0.00676 |
| sudden | SGD | 82.9 | 82.3 | 79.6 | 63.5 | 79.5 | 225.5 | 0.00102 |
| sudden | MWU | 52.6 | 51.2 | 46.3 | 26.3 | 47.3 | 175.9 | 0.01235 |
| oscillating | SGD | 69.7 | 78.5 | 65.6 | 72.0 | 72.2 | 278.7 | 0.00023 |
| oscillating | MWU | 49.6 | 57.3 | 48.3 | 50.1 | 52.0 | 276.0 | 0.00048 |

**Interpretation**

* In all scenarios the online rakers dramatically reduce the margin
  errors relative to the unweighted baseline.  For example, in the
  sudden‑shift scenario the SGD raker reduces the average age error
  from 0.20 to about 0.03 (a 83% improvement).
* The SGD update consistently yields *higher* improvements and lower
  final loss than the MWU update, albeit at the cost of choosing a
  more aggressive learning rate.
* The MWU update, while less accurate in these settings, maintains
  comparable effective sample sizes and might be preferable when
  multiplicative adjustments are desired (e.g., when starting from
  unequal base weights).

You can reproduce these results or design new experiments by running

```bash
python examples/simulation.py
```

from the repository root.  See the source of
`examples/simulation.py` for details.

## Examples

Realistic usage examples are provided in `examples/realistic_examples.py`, including:
- Correcting gender bias in online surveys
- Real-time polling with demographic shifts  
- Performance comparison between SGD and MWU algorithms

Run the examples:

```bash
python examples/realistic_examples.py
```

## Testing

Run the comprehensive test suite:

```bash
pytest tests/test_onlinerake.py -v
```

## Contributing

Pull requests are welcome!  Feel free to open issues if you find bugs
or have suggestions for new features, such as support for multi‑level
controls or adaptive learning‑rate schedules.
