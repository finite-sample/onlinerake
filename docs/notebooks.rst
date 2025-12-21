Interactive Notebooks
====================

Get hands-on experience with OnlineRake through our comprehensive Jupyter notebooks!

These interactive notebooks provide complete tutorials with visualizations, real-time monitoring,
and comprehensive examples that demonstrate the power of streaming weight calibration.

🚀 Getting Started
------------------

Perfect for newcomers to OnlineRake! Learn the basics with clear examples and visual proof
that the algorithms work.

.. toctree::
   :maxdepth: 1

   notebooks/01_getting_started

**What you'll learn:**

* Basic OnlineRake usage with SGD and MWU algorithms
* Correcting feature bias in real-time survey data  
* Handling time-varying patterns in streaming data
* Visual validation with comprehensive plots
* Clear before/after comparisons showing success

⚡ Performance Comparison 
------------------------

Deep dive into algorithm performance across different bias scenarios.

.. toctree::
   :maxdepth: 1

   notebooks/02_performance_comparison

**What you'll learn:**

* Comprehensive SGD vs MWU comparison
* Testing across multiple bias patterns (linear, sudden, oscillating)
* Performance metrics and statistical analysis
* Algorithm selection guidance
* Parameter tuning insights

🔬 Advanced Diagnostics
-----------------------

Master the monitoring and diagnostic capabilities for production deployments.

.. toctree::
   :maxdepth: 1

   notebooks/03_advanced_diagnostics

**What you'll learn:**

* Automatic convergence detection
* Oscillation monitoring and problem diagnosis  
* Weight distribution evolution analysis
* Real-time performance tracking
* Production monitoring best practices

🎯 Quick Start Guide
--------------------

1. **Install dependencies**: ``pip install onlinerake[docs]``
2. **Start with Getting Started**: Master the basics first
3. **Compare algorithms**: Understand when to use SGD vs MWU  
4. **Learn diagnostics**: Essential for production deployments

💡 Tips for Success
-------------------

* **Run notebooks locally** for the best interactive experience
* **Experiment with parameters** to see their effects
* **Try your own data** after completing the tutorials
* **Check diagnostics regularly** in production environments

Each notebook is self-contained and includes all necessary imports and setup code.
The visualizations clearly demonstrate that OnlineRake successfully corrects bias
in streaming data!