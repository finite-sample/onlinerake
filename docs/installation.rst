Installation
============

Requirements
------------

- Python 3.10 or later
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

   git clone https://github.com/your-username/onlinerake.git
   cd onlinerake
   pip install -e .

Development Installation
------------------------

For development work, install with additional dependencies:

.. code-block:: bash

   pip install -e .
   pip install pytest black flake8

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

You can also run the simulation suite:

.. code-block:: bash

   python examples/simulation.py