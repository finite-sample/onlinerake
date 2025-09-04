API Reference
=============

This page provides detailed documentation for all public classes and functions
in the ``onlinerake`` package.

Core Classes
------------

.. currentmodule:: onlinerake

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   Targets
   OnlineRakingSGD
   OnlineRakingMWU

Targets
-------

.. autoclass:: Targets
   :members:
   :undoc-members:
   :show-inheritance:

OnlineRakingSGD
---------------

.. autoclass:: OnlineRakingSGD
   :members:
   :undoc-members:
   :show-inheritance:

OnlineRakingMWU
---------------

.. autoclass:: OnlineRakingMWU
   :members:
   :undoc-members:
   :show-inheritance:

Simulation Module
-----------------

The simulation module provides tools for benchmarking and testing the algorithms.

.. currentmodule:: onlinerake.simulation

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   run_simulation_suite
   analyze_results
   DemographicObservation

.. autofunction:: run_simulation_suite

.. autofunction:: analyze_results

.. autoclass:: DemographicObservation
   :members:
   :undoc-members:
   :show-inheritance: