"""Example applications of onlinerake for ML and data science use cases.

This package contains practical examples demonstrating streaming weight
calibration for various real-world applications:

- **ad_targeting_calibration.py**: Real-time demographic reweighting for ad delivery
- **ab_test_calibration.py**: Covariate balancing for A/B test analysis
- **recommendation_balancing.py**: Diversity calibration for recommendation systems

Each example is self-contained and can be run directly:

    python examples/ad_targeting_calibration.py

The examples demonstrate:
1. How to set up targets and process streaming observations
2. Comparison of SGD, MWU, and batch IPF algorithms
3. Use of convergence analysis and diagnostics
4. Streaming inference with confidence sequences
"""
