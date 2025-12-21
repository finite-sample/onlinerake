Installation
============

Requirements
------------

- Python 3.11 or later
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

   uv sync --group dev --group test

Verify Installation
-------------------

Test that the package is working correctly:

.. code-block:: python

   import onlinerake
   from onlinerake import OnlineRakingSGD, OnlineRakingMWU, Targets
   
   # Run a quick test
   targets = Targets()
   raker = OnlineRakingSGD(targets)
   obs = {"age": 1, "gender": 0, "education": 1, "region": 0}
   raker.partial_fit(obs)
   print(f"Success! Margins: {raker.margins}")

You can also explore the interactive tutorials:

.. code-block:: bash

   # Install with documentation dependencies
   pip install onlinerake[docs]
   
   # Launch Jupyter notebooks
   jupyter notebook docs/notebooks/