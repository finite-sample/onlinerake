Quick Start Guide
=================

Streaming Raking
----------------

``StreamingRaker`` weights each row as it arrives and keeps only running
totals, so memory does not grow with the stream:

.. code-block:: python

   from onlinerake import StreamingRaker, Targets

   raker = StreamingRaker(Targets(female=0.5), outcomes=["spend"])
   for female, spend in [(1, 10.0), (1, 12.0), (1, 11.0), (0, 20.0)] * 50:
       weight = raker.partial_fit({"female": female}, outcomes={"spend": spend})

   print(round(raker.margins["female"], 2))  # 0.52
   print(round(raker.estimate("spend"), 1))  # 15.3

**calibration_strength**
   Penalty on the running gap between weighted and target totals (default
   0.3). Larger values match the targets more closely and make weights more
   variable.

**design_weight**
   Pass ``design_weight=`` to ``partial_fit`` for rows with base weights.
   Rescale them to mean 1 first; the bounds ``min_weight`` and ``max_weight``
   apply to the ratio of a row's weight to its design weight.

**Units**
   The default strength assumes continuous features in units where their
   spread is about 1, for example income in tens of thousands of dollars.

**keep_weights**
   ``StreamingRaker(..., keep_weights=True)`` stores each assigned weight in
   ``.weights``. Memory then grows with the stream.

When the rows can be stored, refitting ``BatchIPF`` periodically is exact at
each refit and is the better choice for survey-sized streams.

Online Classes That Store Every Row
-----------------------------------

``OnlineRakingSGD`` and ``OnlineRakingMWU`` store every row and re-optimize
all weights at each arrival, so their cost per row grows with the stream:

1. **SGD Raking** - Stochastic gradient descent with smooth updates
2. **MWU Raking** - Multiplicative weights with exponential updates

Both follow the same API pattern:

.. code-block:: python

   from onlinerake import OnlineRakingSGD, OnlineRakingMWU, Targets

   # Define target population proportions
   targets = Targets(
       age=0.52,      # 52% over 35 years old
       gender=0.51,   # 51% female
       education=0.35, # 35% college educated
       region=0.19    # 19% rural
   )

   raker = OnlineRakingSGD(targets, learning_rate=3.0)

   observations = [
       {"age": 1, "gender": 0, "education": 1, "region": 0},
       {"age": 0, "gender": 1, "education": 0, "region": 1},
   ]

   for obs in observations:
       raker.partial_fit(obs)

   print(f"Weighted margins: {raker.margins}")
   print(f"Effective sample size: {raker.effective_sample_size}")
   print(f"Loss: {raker.loss}")

Key Concepts
------------

**Targets**
   Population proportions you want to match. Each field represents the 
   proportion with indicator value 1 (e.g., female=1, male=0).

**Observations**
   Binary demographic indicators, provided as dictionaries or objects
   with ``age``, ``gender``, ``education``, ``region`` attributes.

**Margins**
   Current weighted proportions after processing all observations so far.

**Effective Sample Size**
   Measure of how "concentrated" the weights are. Higher is better.

**Loss**
   Squared error between current margins and targets. Lower is better.

Algorithm Choice
----------------

**Use SGD when:**
- You prefer additive weight adjustments
- Smooth weight trajectories are important
- You can tune learning rates appropriately

**Use MWU when:**
- You prefer multiplicative (percentage-based) adjustments
- You want multiplicative updates with explicit weight bounds

Parameter Tuning
----------------

**Learning Rate**
- SGD: Start with 3.0-5.0, increase if convergence is slow
- MWU: Start with 1.0-1.5, decrease if weights become unstable

**Weight Bounds**
- ``min_weight``: Prevents weights from collapsing (default: 1e-3)
- ``max_weight``: Prevents runaway weights (default: 100.0)

**Update Steps**
- ``n_sgd_steps`` (both methods): More steps = more aggressive updates (default: 3)

Next Steps
----------

- See :doc:`notebooks` for worked, executable tutorials
- Check :doc:`api_reference` for complete parameter descriptions
- Browse `examples/ <https://github.com/finite-sample/onlinerake/tree/main/examples>`_
  for realistic end-to-end use cases
