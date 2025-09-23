onlinerake: Streaming Survey Raking
===================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   tutorials/index
   diagnostics
   api_reference
   examples
   contributing
   changelog

Modern online surveys and passive data collection streams generate
responses one record at a time. Classic weighting methods such as
iterative proportional fitting (IPF, or "raking") and calibration
weighting are inherently *batch* procedures: they reprocess the entire
dataset whenever a new case arrives. The ``onlinerake`` package
provides **incremental**, per‑observation updates to survey weights so
that weighted margins track known population totals in real time.

Key Features
------------

- **Real-time weight calibration** for streaming survey data
- **Two complementary algorithms**: SGD and multiplicative weights update (MWU)
- **scikit-learn style API** with ``partial_fit`` method
- **Minimal dependencies**: only numpy and pandas
- **Comprehensive testing** with realistic examples

Quick Start
-----------

.. code-block:: python

   from onlinerake import OnlineRakingSGD, Targets

   # Define target population margins
   targets = Targets(age=0.5, gender=0.5, education=0.4, region=0.3)
   
   # Create raker
   raker = OnlineRakingSGD(targets, learning_rate=5.0)
   
   # Process streaming observations
   for obs in stream_of_observations:
       raker.partial_fit(obs)
       print(f"Current margins: {raker.margins}")

Algorithms
----------

**SGD Raking**
   Stochastic gradient descent on squared-error loss over margins.
   Produces smooth weight trajectories and maintains high effective sample size.

**MWU Raking** 
   Multiplicative weights update inspired by mirror descent under KL divergence.
   Yields weight distributions similar to classic IPF but can produce heavier tails.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`