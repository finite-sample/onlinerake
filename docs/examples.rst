Examples
========

This page contains complete, realistic examples demonstrating how to use
``onlinerake`` in various scenarios.

Example 1: Correcting Gender Bias in Tech Survey
------------------------------------------------

Online tech surveys often over-represent young males. Here's how to correct this bias:

.. code-block:: python

   import numpy as np
   from onlinerake import OnlineRakingSGD, Targets

   # US population targets (approximate)
   targets = Targets(
       age=0.52,      # 52% over 35 years old
       gender=0.51,   # 51% female
       education=0.35, # 35% college educated
       region=0.19    # 19% rural
   )

   # Initialize raker with higher learning rate for quick correction
   raker = OnlineRakingSGD(targets, learning_rate=4.0)

   # Simulate biased tech survey responses
   np.random.seed(42)
   n_responses = 500
   raw_totals = {"age": 0, "gender": 0, "education": 0, "region": 0}

   for i in range(n_responses):
       # Bias: 70% young males, 60% college educated
       age = 1 if np.random.random() < 0.3 else 0      # 30% older
       gender = 1 if np.random.random() < 0.35 else 0  # 35% female
       education = 1 if np.random.random() < 0.6 else 0 # 60% college
       region = 1 if np.random.random() < 0.15 else 0   # 15% rural
       
       obs = {"age": age, "gender": gender, "education": education, "region": region}
       raker.partial_fit(obs)
       
       # Track raw proportions
       for key in raw_totals:
           raw_totals[key] += obs[key]

   # Compare results
   raw_margins = {k: v/n_responses for k, v in raw_totals.items()}
   weighted_margins = raker.margins

   print("Results after", n_responses, "responses:")
   print("Characteristic | Target | Raw    | Weighted")
   print("-" * 40)
   for char in ['gender', 'age', 'education', 'region']:
       target = targets.as_dict()[char]
       raw = raw_margins[char]
       weighted = weighted_margins[char]
       print(f"{char:<12} | {target:.3f} | {raw:.3f} | {weighted:.3f}")

   print(f"\\nEffective Sample Size: {raker.effective_sample_size:.1f}")
   print(f"Final Loss: {raker.loss:.6f}")

**Expected Output:**

.. code-block:: text

   Results after 500 responses:
   Characteristic | Target | Raw    | Weighted
   ----------------------------------------
   gender       | 0.510 | 0.344 | 0.491
   age          | 0.520 | 0.330 | 0.491
   education    | 0.350 | 0.602 | 0.378
   region       | 0.190 | 0.134 | 0.167

   Effective Sample Size: 294.1
   Final Loss: 0.002512

Example 2: Real-time Election Polling
-------------------------------------

Handle streaming poll responses with changing demographics:

.. code-block:: python

   from onlinerake import OnlineRakingSGD, Targets

   # 2024 US voter demographics
   targets = Targets(
       age=0.48,      # 48% over 50 years old
       gender=0.53,   # 53% female voters  
       education=0.32, # 32% college degree
       region=0.17    # 17% rural voters
   )

   raker = OnlineRakingSGD(targets, learning_rate=3.0)

   # Simulate poll responses with time-varying bias
   import numpy as np
   np.random.seed(789)
   n_polls = 1000

   # Track evolution of margins
   checkpoints = [200, 400, 600, 800, 1000]
   
   for i in range(n_polls):
       # Demographics change over time as different groups respond
       time_factor = i / n_polls
       
       # Early: social media recruitment (younger)
       # Later: phone polling kicks in (older)
       p_older = 0.2 + 0.4 * time_factor
       age = 1 if np.random.random() < p_older else 0
       
       # Education bias decreases over time
       p_educated = 0.6 - 0.3 * time_factor
       education = 1 if np.random.random() < p_educated else 0
       
       # Other demographics relatively stable
       gender = 1 if np.random.random() < 0.52 else 0
       region = 1 if np.random.random() < 0.18 else 0
       
       obs = {"age": age, "gender": gender, "education": education, "region": region}
       raker.partial_fit(obs)
       
       # Print progress at checkpoints
       if (i + 1) in checkpoints:
           margins = raker.margins
           print(f"After {i+1:4d} responses: Age={margins['age']:.3f}, "
                 f"Gender={margins['gender']:.3f}, Education={margins['education']:.3f}")

   print(f"\\nFinal ESS: {raker.effective_sample_size:.1f} / {n_polls}")

Example 3: Comparing SGD vs MWU
-------------------------------

Side-by-side comparison of both algorithms:

.. code-block:: python

   from onlinerake import OnlineRakingSGD, OnlineRakingMWU, Targets
   import numpy as np

   targets = Targets(age=0.45, gender=0.52, education=0.38, region=0.22)

   # Different learning rates optimized for each method
   sgd_raker = OnlineRakingSGD(targets, learning_rate=5.0)
   mwu_raker = OnlineRakingMWU(targets, learning_rate=1.0)

   # Simulate sudden demographic shift
   np.random.seed(2024)
   n_obs = 800

   for i in range(n_obs):
       if i < n_obs // 2:
           # First half: younger, more educated
           age = 1 if np.random.random() < 0.25 else 0
           education = 1 if np.random.random() < 0.65 else 0
       else:
           # Second half: older, less educated  
           age = 1 if np.random.random() < 0.70 else 0
           education = 1 if np.random.random() < 0.15 else 0
       
       gender = 1 if np.random.random() < 0.50 else 0
       region = 1 if np.random.random() < 0.20 else 0
       
       obs = {"age": age, "gender": gender, "education": education, "region": region}
       
       sgd_raker.partial_fit(obs)
       mwu_raker.partial_fit(obs)

   # Compare final results
   print("Final Results:")
   print("Metric               | Target | SGD    | MWU")
   print("-" * 45)

   sgd_final = sgd_raker.margins
   mwu_final = mwu_raker.margins

   for char in ['age', 'gender', 'education', 'region']:
       target = targets.as_dict()[char]
       sgd_val = sgd_final[char]
       mwu_val = mwu_final[char]
       print(f"{char:<20} | {target:.3f} | {sgd_val:.3f} | {mwu_val:.3f}")

   print("-" * 45)
   print(f"Loss (squared error) |        | {sgd_raker.loss:.5f} | {mwu_raker.loss:.5f}")
   print(f"Effective Sample Size|        | {sgd_raker.effective_sample_size:.1f} | {mwu_raker.effective_sample_size:.1f}")

Running the Examples
-------------------

All examples are available in the repository as ``realistic_examples.py``:

.. code-block:: bash

   python realistic_examples.py

You can also run the built-in simulation suite:

.. code-block:: bash

   python -m onlinerake.simulation