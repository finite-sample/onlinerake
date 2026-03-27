#!/usr/bin/env python3
"""Recommendation System Balancing: Real-time calibration for diverse recommendations.

This example demonstrates streaming weight calibration for recommendation systems,
where served recommendations need to be reweighted to achieve diversity targets
across different item categories.

Use Case:
---------
A streaming platform's recommendation algorithm tends to over-recommend
popular content, creating filter bubbles. The platform wants to ensure
recommendations are balanced across genres, freshness, and creator diversity.
Online raking adjusts item weights in real-time to meet diversity targets
while preserving personalization signal.

This is a streaming problem because:
1. Recommendations are generated continuously
2. User feedback (clicks, watch time) arrives as a stream
3. Diversity metrics need real-time monitoring
4. Can't rebatch recommendations already served

Key insight: This is "raking for items" rather than "raking for users" -
we reweight which items get recommended to match category targets.

Example output shows:
- How recommendation diversity improves with weighting
- Trade-off between diversity and relevance
- Algorithm comparison for this use case
"""

import numpy as np

from onlinerake import BatchIPF, OnlineRakingMWU, OnlineRakingSGD, Targets
from onlinerake.diagnostics import summarize_raking_results


def simulate_recommendations(
    n_recommendations: int = 3000,
    seed: int = 42,
) -> list[dict]:
    """Simulate a stream of item recommendations with popularity bias.

    Real recommendation systems exhibit several biases:
    - Popularity bias: popular items recommended more
    - Recency bias: new content under-explored
    - Creator concentration: few creators dominate
    - Genre imbalance: some genres over-represented

    Args:
        n_recommendations: Number of recommendations to simulate.
        seed: Random seed.

    Returns:
        List of recommendation records.
    """
    np.random.seed(seed)

    # Item categories (mutually exclusive for simplicity)
    # In reality, these could be multi-label

    recommendations = []
    for i in range(n_recommendations):
        # Popularity bias: 70% of recs go to top content (is_popular=1)
        # Target: want 40% popular, 60% long-tail
        is_popular = 1 if np.random.random() < 0.70 else 0

        # Recency bias: 80% of recs go to older content
        # Target: want 50% new content (< 30 days old)
        is_new_content = 1 if np.random.random() < 0.20 else 0

        # Creator diversity: 60% from top creators
        # Target: want 35% top creator content
        is_top_creator = 1 if np.random.random() < 0.60 else 0

        # Genre: drama over-represented at 40%
        # Target: want 25% drama
        is_drama = 1 if np.random.random() < 0.40 else 0

        # User engagement (affected by popularity and recency)
        base_ctr = 0.05
        ctr_boost = (
            is_popular * 0.03  # Popular items have higher CTR
            + (1 - is_new_content) * 0.01  # Old content slightly higher CTR
            + is_top_creator * 0.02  # Known creators get more clicks
        )
        clicked = 1 if np.random.random() < (base_ctr + ctr_boost) else 0

        # Watch time for clicked items (seconds)
        if clicked:
            base_watch = 120
            watch_time = base_watch * (1 + np.random.exponential(0.5))
        else:
            watch_time = 0

        recommendations.append(
            {
                "is_popular": is_popular,
                "is_new_content": is_new_content,
                "is_top_creator": is_top_creator,
                "is_drama": is_drama,
                "clicked": clicked,
                "watch_time": watch_time,
                "slot_position": i % 10,  # Position in recommendation slate
            }
        )

    return recommendations


def compute_engagement_metrics(
    recs: list[dict],
    weights: np.ndarray | None = None,
) -> dict:
    """Compute engagement metrics with optional weighting.

    Args:
        recs: List of recommendation records.
        weights: Per-recommendation weights (None = equal).

    Returns:
        Dictionary of engagement metrics.
    """
    if weights is None:
        weights = np.ones(len(recs))

    # Normalize weights
    weights = weights / weights.sum() * len(recs)

    weighted_clicks = sum(w * r["clicked"] for w, r in zip(weights, recs, strict=False))
    weighted_watch = sum(
        w * r["watch_time"] for w, r in zip(weights, recs, strict=False)
    )

    return {
        "total_clicks": weighted_clicks,
        "ctr": weighted_clicks / len(recs),
        "total_watch_time": weighted_watch,
        "avg_watch_per_click": weighted_watch / weighted_clicks
        if weighted_clicks > 0
        else 0,
    }


def run_recommendation_example() -> None:
    """Run the recommendation balancing example."""
    print("=" * 70)
    print("RECOMMENDATION BALANCING: Diversity Through Streaming Calibration")
    print("=" * 70)

    # Diversity targets for recommendations
    targets = Targets(
        is_popular=0.40,  # 40% popular content (reduce from 70%)
        is_new_content=0.50,  # 50% new content (increase from 20%)
        is_top_creator=0.35,  # 35% top creators (reduce from 60%)
        is_drama=0.25,  # 25% drama (reduce from 40%)
    )

    print("\n🎯 Diversity Targets (what we want to achieve):")
    for feature, target in targets.as_dict().items():
        print(f"   {feature}: {target:.0%}")

    # Simulate biased recommendations
    print("\n📺 Simulating 3,000 recommendations with typical biases...")
    recommendations = simulate_recommendations(n_recommendations=3000)

    # Raw (biased) distribution
    raw_distribution = {
        feature: sum(r[feature] for r in recommendations) / len(recommendations)
        for feature in ["is_popular", "is_new_content", "is_top_creator", "is_drama"]
    }

    print("\n❌ Raw (Biased) Distribution:")
    for feature, value in raw_distribution.items():
        target = targets[feature]
        bias = value - target
        direction = "↑" if bias > 0 else "↓"
        print(
            f"   {feature}: {value:.1%} (target: {target:.0%}, {direction} {abs(bias):.0%})"
        )

    # Raw engagement metrics
    raw_engagement = compute_engagement_metrics(recommendations)
    print("\n📊 Raw Engagement (biased recommendations):")
    print(f"   CTR: {raw_engagement['ctr']:.2%}")
    print(f"   Avg Watch Time per Click: {raw_engagement['avg_watch_per_click']:.1f}s")

    # METHOD 1: Online SGD
    print("\n" + "-" * 70)
    print("METHOD 1: Online SGD (Fast convergence)")
    print("-" * 70)

    sgd_raker = OnlineRakingSGD(
        targets,
        learning_rate=5.0,
        n_sgd_steps=3,
        track_convergence=True,
    )

    for rec in recommendations:
        sgd_raker.partial_fit(rec)

    sgd_margins = sgd_raker.margins
    print("\n✅ SGD Weighted Distribution:")
    for feature in ["is_popular", "is_new_content", "is_top_creator", "is_drama"]:
        weighted = sgd_margins[feature]
        target = targets[feature]
        error = abs(weighted - target)
        print(
            f"   {feature}: {weighted:.1%} (target: {target:.0%}, error: {error:.1%})"
        )

    print(f"\n   Final Loss: {sgd_raker.loss:.6f}")
    print(
        f"   ESS: {sgd_raker.effective_sample_size:.1f} ({sgd_raker.effective_sample_size / 3000:.1%} efficiency)"
    )

    # METHOD 2: Online MWU
    print("\n" + "-" * 70)
    print("METHOD 2: Online MWU (More stable weights)")
    print("-" * 70)

    mwu_raker = OnlineRakingMWU(
        targets,
        learning_rate=1.0,
        n_steps=3,
        track_convergence=True,
    )

    for rec in recommendations:
        mwu_raker.partial_fit(rec)

    mwu_margins = mwu_raker.margins
    print("\n✅ MWU Weighted Distribution:")
    for feature in ["is_popular", "is_new_content", "is_top_creator", "is_drama"]:
        weighted = mwu_margins[feature]
        target = targets[feature]
        error = abs(weighted - target)
        print(
            f"   {feature}: {weighted:.1%} (target: {target:.0%}, error: {error:.1%})"
        )

    print(f"\n   Final Loss: {mwu_raker.loss:.6f}")
    print(
        f"   ESS: {mwu_raker.effective_sample_size:.1f} ({mwu_raker.effective_sample_size / 3000:.1%} efficiency)"
    )

    # METHOD 3: Batch IPF
    print("\n" + "-" * 70)
    print("METHOD 3: Batch IPF (Baseline)")
    print("-" * 70)

    ipf = BatchIPF(targets)
    ipf.fit(recommendations)

    print(f"   Iterations: {ipf.n_iterations}")
    print(f"   Final Loss: {ipf.loss:.6f}")
    print(
        f"   ESS: {ipf.effective_sample_size:.1f} ({ipf.effective_sample_size / 3000:.1%} efficiency)"
    )

    # ENGAGEMENT vs DIVERSITY TRADEOFF
    print("\n" + "=" * 70)
    print("ENGAGEMENT vs DIVERSITY TRADEOFF")
    print("=" * 70)

    # Compute weighted engagement for each method
    sgd_engagement = compute_engagement_metrics(recommendations, sgd_raker.weights)
    mwu_engagement = compute_engagement_metrics(recommendations, mwu_raker.weights)
    ipf_engagement = compute_engagement_metrics(recommendations, ipf.weights)

    print(f"\n{'Metric':<25} {'Raw':>10} {'SGD':>10} {'MWU':>10} {'IPF':>10}")
    print("-" * 65)
    print(
        f"{'CTR':<25} {raw_engagement['ctr']:>10.2%} {sgd_engagement['ctr']:>10.2%} {mwu_engagement['ctr']:>10.2%} {ipf_engagement['ctr']:>10.2%}"
    )
    print(
        f"{'Avg Watch (sec)':<25} {raw_engagement['avg_watch_per_click']:>10.1f} {sgd_engagement['avg_watch_per_click']:>10.1f} {mwu_engagement['avg_watch_per_click']:>10.1f} {ipf_engagement['avg_watch_per_click']:>10.1f}"
    )

    # Diversity improvement
    print(f"\n{'Diversity Score':<25} {'Raw':>10} {'SGD':>10} {'MWU':>10} {'IPF':>10}")
    print("-" * 65)
    for feature in ["is_popular", "is_new_content", "is_top_creator", "is_drama"]:
        raw_error = abs(raw_distribution[feature] - targets[feature])
        sgd_error = abs(sgd_margins[feature] - targets[feature])
        mwu_error = abs(mwu_margins[feature] - targets[feature])
        ipf_error = abs(ipf.margins[feature] - targets[feature])
        print(
            f"{'  ' + feature + ' error':<25} {raw_error:>10.1%} {sgd_error:>10.1%} {mwu_error:>10.1%} {ipf_error:>10.1%}"
        )

    # Summary statistics
    print("\n" + "-" * 70)
    print("WEIGHT DISTRIBUTION COMPARISON")
    print("-" * 70)

    for name, raker in [("SGD", sgd_raker), ("MWU", mwu_raker)]:
        stats = raker.weight_distribution_stats
        print(f"\n{name}:")
        print(f"   Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"   Median: {stats['median']:.3f}")
        print(f"   Max/Min Ratio: {stats['max'] / stats['min']:.1f}")

    # Full summary
    print("\n" + "-" * 70)
    print("FULL ANALYSIS SUMMARY (SGD)")
    print("-" * 70)

    summary = summarize_raking_results(sgd_raker)
    print(f"\nObservations: {summary['n_observations']:,}")
    print(f"Design Effect: {summary['design_effect']:.2f}")
    print(f"Weight Efficiency: {summary['weight_efficiency']:.1%}")

    print("\nFeasibility Assessment:")
    feas = summary["feasibility"]
    print(f"   All targets feasible: {feas['is_feasible']}")
    if feas["recommendations"]:
        for rec in feas["recommendations"][:3]:
            print(f"   - {rec}")

    print("\n" + "=" * 70)
    print("KEY INSIGHTS FOR RECOMMENDATION BALANCING:")
    print("=" * 70)
    print("""
1. DIVERSITY IS ACHIEVABLE: Online raking successfully rebalances
   recommendations to meet diversity targets.

2. ENGAGEMENT TRADEOFF: Weighting toward diversity slightly reduces
   aggregate engagement metrics (CTR, watch time) because popular
   content genuinely has higher engagement rates.

3. LONG-TERM BENEFITS: The engagement "cost" of diversity may be
   offset by reduced filter bubbles, user retention, and content
   ecosystem health (not captured in this simulation).

4. ALGORITHM CHOICE:
   - SGD: Faster convergence, slightly more extreme weights
   - MWU: More stable weight distribution, smoother convergence
   - Both achieve similar diversity outcomes

5. WEIGHT EFFICIENCY: Higher efficiency = less variance inflation.
   MWU often achieves better efficiency than SGD.

6. PRACTICAL APPLICATION:
   - Use weights as probability multipliers for item sampling
   - Or as bid modifiers in ad-supported recommendations
   - Or as ranking score adjustments
   - Monitor ESS to ensure sufficient effective sample size

7. REAL-TIME MONITORING: Streaming approach enables continuous
   monitoring and adjustment as content mix changes.
""")


if __name__ == "__main__":
    run_recommendation_example()
