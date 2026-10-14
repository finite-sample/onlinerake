Installation
============

Requirements
------------

- Python 3.12 or later
- NumPy >= 1.21
- Pandas >= 1.3

From PyPI (Recommended)
-----------------------

.. code-block:: bash

   pip install onlinerake

From Source
-----------

Clone the repository and install in development mode:

.. code-block:: bash

   git clone https://github.com/finite-sample/onlinerake.git
   cd onlinerake
   uv sync

Development Installation
------------------------

For development work, install with additional dependencies:

.. code-block:: bash

   uv sync --all-groups

Verify Installation
-------------------

Test that the package is working correctly:

.. doctest::

   >>> from onlinerake import OnlineRakingMWU, Targets
   >>> raker = OnlineRakingMWU(Targets(female=0.5))
   >>> raker.partial_fit({"female": 1})
   >>> raker.partial_fit({"female": 0})
   >>> raker.margins
   {'female': 0.5}

The examples are scripts in ``docs/examples`` in the repository. Run one with:

.. code-block:: bash

   uv run --group docs python docs/examples/plot_streaming_raker.py
