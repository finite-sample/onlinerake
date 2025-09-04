Quick Start Guide
=================

Basic Usage
-----------

The ``onlinerake`` package provides streaming survey raking with two algorithms:

1. **SGD Raking** - Stochastic gradient descent with smooth updates
2. **MWU Raking** - Multiplicative weights with exponential updates

Both algorithms follow the same API pattern:

.. code-block:: python

   from onlinerake import OnlineRakingSGD, OnlineRakingMWU, Targets

   # Define target population proportions
   targets = Targets(
       age=0.52,      # 52% over 35 years old
       gender=0.51,   # 51% female  
       education=0.35, # 35% college educated
       region=0.19    # 19% rural
   )

   # Initialize raker
   raker = OnlineRakingSGD(targets, learning_rate=3.0)

   # Process observations one at a time
   observations = [
       {"age": 1, "gender": 0, "education": 1, "region": 0},
       {"age": 0, "gender": 1, "education": 0, "region": 1},
       # ... more observations
   ]

   for obs in observations:
       raker.partial_fit(obs)
       
   # Inspect results
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
- You want the most accurate margin tracking
- Smooth weight trajectories are important
- You can tune learning rates appropriately

**Use MWU when:**
- You prefer multiplicative (percentage-based) adjustments
- You want weight distributions similar to classic IPF
- You're starting from unequal base weights

Parameter Tuning
----------------

**Learning Rate**
- SGD: Start with 3.0-5.0, increase if convergence is slow
- MWU: Start with 1.0-1.5, decrease if weights become unstable

**Weight Bounds**
- ``min_weight``: Prevents weights from collapsing (default: 1e-3)
- ``max_weight``: Prevents runaway weights (default: 100.0)

**Update Steps**
- ``n_sgd_steps`` (SGD): More steps = smoother convergence (default: 3)
- ``n_steps`` (MWU): More steps = more aggressive updates (default: 3)

Next Steps
----------

- See :doc:`tutorials/index` for detailed examples
- Check :doc:`api_reference` for complete parameter descriptions
- Try :doc:`examples` for realistic use cases