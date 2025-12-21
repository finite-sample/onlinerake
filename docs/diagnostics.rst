Diagnostics and Monitoring
==========================

The ``onlinerake`` package provides comprehensive diagnostics and monitoring capabilities to help you understand the behavior of your streaming raking algorithms. These features are particularly useful for debugging convergence issues, tuning parameters, and monitoring the quality of your weight calibration.

The algorithms are designed with numerical stability in mind, automatically handling extreme cases that could cause overflow, underflow, or convergence detection failures.

Enhanced Monitoring Features
-----------------------------

Both ``OnlineRakingSGD`` and ``OnlineRakingMWU`` support the following diagnostic features:

Convergence Monitoring
~~~~~~~~~~~~~~~~~~~~~~

**Automatic Convergence Detection**

The algorithms can automatically detect when they have converged based on loss stability:

.. code-block:: python

   from onlinerake import OnlineRakingSGD, Targets
   
   targets = Targets(age=0.5, gender=0.5, education=0.4, region=0.3)
   raker = OnlineRakingSGD(
       targets,
       learning_rate=3.0,
       track_convergence=True,    # Enable convergence detection
       convergence_window=20      # Use last 20 observations for convergence check
   )
   
   # Process observations
   for obs in observations:
       raker.partial_fit(obs)
       
       if raker.converged:
           print(f"Converged at observation {raker.convergence_step}")
           break

**Gradient Norm Tracking**

Monitor the magnitude of gradient updates to understand convergence behavior:

.. code-block:: python

   # After processing observations
   gradient_norms = raker.gradient_norm_history
   
   # Plot convergence
   import matplotlib.pyplot as plt
   plt.plot(gradient_norms)
   plt.xlabel('Observation')
   plt.ylabel('Gradient Norm')
   plt.title('Convergence Behavior')

**Loss Moving Average**

Track smoothed loss over a configurable window:

.. code-block:: python

   print(f"Current loss: {raker.loss:.6f}")
   print(f"Moving average: {raker.loss_moving_average:.6f}")

Oscillation Detection
~~~~~~~~~~~~~~~~~~~~~

Detect when algorithms are oscillating rather than converging:

.. code-block:: python

   # Check if algorithm is oscillating
   oscillating = raker.detect_oscillation(threshold=0.1)
   
   if oscillating:
       print("Warning: Algorithm may be oscillating")
       print("Consider reducing learning rate")

Weight Distribution Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Monitor the distribution of weights to detect outliers and understand effective sample size:

.. code-block:: python

   weight_stats = raker.weight_distribution_stats
   
   print(f"Weight range: [{weight_stats['min']:.3f}, {weight_stats['max']:.3f}]")
   print(f"Median weight: {weight_stats['median']:.3f}")
   print(f"Outliers detected: {weight_stats['outliers_count']}")
   print(f"Effective sample size: {raker.effective_sample_size:.1f}")

Verbose Mode
~~~~~~~~~~~~

Enable verbose output for real-time monitoring:

.. code-block:: python

   raker = OnlineRakingSGD(
       targets,
       learning_rate=3.0,
       verbose=True  # Print progress every 100 observations
   )
   
   # Output will show:
   # Obs 100: loss=0.001234, grad_norm=0.005678, ess=85.3

Configuration Options
---------------------

The diagnostic features can be configured via constructor parameters:

.. code-block:: python

   raker = OnlineRakingSGD(
       targets,
       learning_rate=5.0,
       verbose=False,               # Disable verbose output
       track_convergence=True,      # Enable convergence detection  
       convergence_window=20        # Window size for convergence check
   )

**Parameters:**

- ``verbose`` (bool): Enable progress output every 100 observations
- ``track_convergence`` (bool): Enable automatic convergence detection
- ``convergence_window`` (int): Number of recent observations to use for convergence analysis

Comprehensive History Tracking
-------------------------------

All diagnostic information is automatically stored in the ``history`` attribute:

.. code-block:: python

   # Access full history
   for i, state in enumerate(raker.history):
       print(f"Step {i+1}:")
       print(f"  Loss: {state['loss']:.6f}")
       print(f"  Gradient norm: {state['gradient_norm']:.6f}")
       print(f"  ESS: {state['ess']:.1f}")
       print(f"  Converged: {state['converged']}")
       print(f"  Oscillating: {state['oscillating']}")

Each history entry contains:

- ``loss``: Current squared-error loss on margins
- ``gradient_norm``: L2 norm of the gradient vector
- ``loss_moving_avg``: Moving average of loss over convergence window
- ``ess``: Effective sample size
- ``converged``: Whether convergence has been detected
- ``oscillating``: Whether oscillation is detected
- ``weight_stats``: Comprehensive weight distribution statistics
- ``weighted_margins``: Current weighted demographic margins
- ``raw_margins``: Unweighted demographic margins

Practical Examples
------------------

**Debugging Convergence Issues**

.. code-block:: python

   import numpy as np
   from onlinerake import OnlineRakingSGD, Targets
   
   targets = Targets(age=0.5, gender=0.5, education=0.4, region=0.3)
   raker = OnlineRakingSGD(
       targets,
       learning_rate=10.0,  # Potentially too high
       verbose=True,
       track_convergence=True
   )
   
   # Simulate data
   for i in range(200):
       obs = {
           "age": np.random.binomial(1, 0.3),
           "gender": np.random.binomial(1, 0.4), 
           "education": np.random.binomial(1, 0.6),
           "region": np.random.binomial(1, 0.2)
       }
       raker.partial_fit(obs)
       
       # Check for problems
       if i > 50 and raker.detect_oscillation():
           print(f"Oscillation detected at step {i+1}")
           print("Consider reducing learning rate")
           break
           
       if raker.converged:
           print(f"Successfully converged at step {i+1}")
           break

**Monitoring Real-time Performance**

.. code-block:: python

   # Set up monitoring
   raker = OnlineRakingSGD(targets, learning_rate=3.0, verbose=True)
   
   for obs in data_stream:
       raker.partial_fit(obs)
       
       # Monitor every 100 observations
       if raker._n_obs % 100 == 0:
           stats = raker.weight_distribution_stats
           print(f"\\nDiagnostics at observation {raker._n_obs}:")
           print(f"  Loss: {raker.loss:.6f}")
           print(f"  ESS: {raker.effective_sample_size:.1f}")
           print(f"  Weight outliers: {stats['outliers_count']}")
           
           if stats['outliers_count'] > raker._n_obs * 0.1:
               print("  Warning: High proportion of weight outliers")

Numerical Stability and Robustness
-----------------------------------

The algorithms include several built-in safeguards for numerical stability:

**MWU Exponent Clipping**

The multiplicative weights update algorithm automatically clips exponential arguments to prevent overflow:

.. code-block:: python

   # Internally, MWU clips extreme exponents
   expo = np.clip(-learning_rate * grad, -50.0, 50.0)
   update = np.exp(expo)

This prevents NaN/Inf values even with extreme learning rates or gradients.

**Robust Convergence Detection**

Convergence detection handles edge cases gracefully:

.. code-block:: python

   # Convergence when loss approaches zero
   raker = OnlineRakingSGD(targets, track_convergence=True)
   
   # Algorithm automatically detects:
   # 1. Perfect convergence (loss ≈ 0)
   # 2. Relative stability (low variance)
   
   if raker.converged:
       print(f"Converged at step {raker.convergence_step}")

**Extreme Parameter Handling**

Both algorithms are robust to extreme parameter settings:

.. code-block:: python

   # High learning rates with extreme targets
   extreme_targets = Targets(age=0.1, gender=0.9, education=0.1, region=0.9)
   raker = OnlineRakingMWU(extreme_targets, learning_rate=50.0)
   
   # Algorithm remains stable despite extreme settings
   for obs in challenging_data:
       raker.partial_fit(obs)
       assert np.all(np.isfinite(raker.weights))  # Always finite

For complete interactive examples demonstrating all diagnostic features, see the ``03_advanced_diagnostics.ipynb`` notebook in ``docs/notebooks/``.