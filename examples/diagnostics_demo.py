"""Demonstration of enhanced diagnostics and monitoring features."""

import numpy as np

from onlinerake import OnlineRakingSGD, Targets


def demo_convergence_monitoring():
    """Demonstrate convergence monitoring and diagnostics."""
    print("=" * 60)
    print("DIAGNOSTICS DEMO: Convergence Monitoring")
    print("=" * 60)

    # Set up targets and raker with diagnostics enabled
    targets = Targets(age=0.5, gender=0.5, education=0.4, region=0.3)
    raker = OnlineRakingSGD(
        targets,
        learning_rate=3.0,
        verbose=True,  # Enable verbose output
        track_convergence=True,
        convergence_window=10,
    )

    print(f"Target margins: {targets.as_dict()}")
    print(f"Convergence window: {raker.convergence_window}")
    print()

    # Generate converging data stream
    np.random.seed(42)
    n_obs = 200

    # Start with biased data, gradually approaching targets
    for i in range(n_obs):
        # Gradually shift probabilities toward targets
        progress = min(i / 100.0, 1.0)  # Reach targets after ~100 observations

        age_prob = 0.3 + progress * (0.5 - 0.3)  # 0.3 -> 0.5
        gender_prob = 0.2 + progress * (0.5 - 0.2)  # 0.2 -> 0.5
        education_prob = 0.6 + progress * (0.4 - 0.6)  # 0.6 -> 0.4
        region_prob = 0.1 + progress * (0.3 - 0.1)  # 0.1 -> 0.3

        obs = {
            "age": np.random.binomial(1, age_prob),
            "gender": np.random.binomial(1, gender_prob),
            "education": np.random.binomial(1, education_prob),
            "region": np.random.binomial(1, region_prob),
        }

        raker.partial_fit(obs)

        # Print diagnostic info every 50 observations
        if (i + 1) % 50 == 0:
            print(f"\nStep {i + 1}:")
            print(
                f"  Loss: {raker.loss:.6f} (moving avg: {raker.loss_moving_average:.6f})"
            )
            print(f"  Gradient norm: {raker.gradient_norm_history[-1]:.6f}")
            print(f"  ESS: {raker.effective_sample_size:.1f}")
            print(f"  Converged: {raker.converged}")
            print(f"  Oscillating: {raker.detect_oscillation()}")

            # Weight distribution stats
            weight_stats = raker.weight_distribution_stats
            print(
                f"  Weight range: [{weight_stats['min']:.3f}, {weight_stats['max']:.3f}]"
            )
            print(f"  Weight outliers: {weight_stats['outliers_count']}")

    print("\nFINAL RESULTS:")
    print(f"Converged: {raker.converged}")
    if raker.converged:
        print(f"Convergence detected at observation: {raker.convergence_step}")

    print(f"Final loss: {raker.loss:.6f}")
    print(f"Final margins: {raker.margins}")
    print(f"Target margins: {targets.as_dict()}")

    # Analyze convergence history
    losses = [state["loss"] for state in raker.history]
    gradient_norms = raker.gradient_norm_history

    print("\nCONVERGENCE ANALYSIS:")
    print(f"Loss range: [{min(losses):.6f}, {max(losses):.6f}]")
    print(
        f"Gradient norm range: [{min(gradient_norms):.6f}, {max(gradient_norms):.6f}]"
    )
    print(f"Final 10 losses: {losses[-10:]}")


def demo_oscillation_detection():
    """Demonstrate oscillation detection."""
    print("\n\n" + "=" * 60)
    print("DIAGNOSTICS DEMO: Oscillation Detection")
    print("=" * 60)

    targets = Targets(age=0.5, gender=0.5, education=0.5, region=0.5)

    # High learning rate to induce oscillation
    raker = OnlineRakingSGD(
        targets,
        learning_rate=15.0,  # Aggressively high
        track_convergence=True,
        convergence_window=15,
    )

    print(f"Using high learning rate: {raker.learning_rate}")
    print("Generating alternating extreme observations...")

    # Generate alternating extreme observations
    for i in range(50):
        if i % 2 == 0:
            obs = {"age": 1, "gender": 1, "education": 1, "region": 1}
        else:
            obs = {"age": 0, "gender": 0, "education": 0, "region": 0}

        raker.partial_fit(obs)

        if (i + 1) % 10 == 0:
            oscillating = raker.detect_oscillation()
            print(f"Step {i + 1}: Loss={raker.loss:.6f}, Oscillating={oscillating}")

    print(f"\nFinal oscillation status: {raker.detect_oscillation()}")
    print(f"Converged: {raker.converged}")

    # Show recent loss variance
    recent_losses = [
        state["loss"] for state in raker.history[-raker.convergence_window :]
    ]
    print(f"Recent loss variance: {np.var(recent_losses):.6f}")
    print(f"Recent loss mean: {np.mean(recent_losses):.6f}")


def demo_weight_distribution():
    """Demonstrate weight distribution analysis."""
    print("\n\n" + "=" * 60)
    print("DIAGNOSTICS DEMO: Weight Distribution Analysis")
    print("=" * 60)

    targets = Targets(age=0.3, gender=0.7, education=0.2, region=0.8)  # Extreme targets
    raker = OnlineRakingSGD(targets, learning_rate=5.0)

    print(f"Extreme target margins: {targets.as_dict()}")
    print("Generating uniform random observations (will require extreme weights)...")

    np.random.seed(123)
    for i in range(100):
        # Uniform random observations (prob=0.5 for each indicator)
        obs = {
            "age": np.random.binomial(1, 0.5),
            "gender": np.random.binomial(1, 0.5),
            "education": np.random.binomial(1, 0.5),
            "region": np.random.binomial(1, 0.5),
        }
        raker.partial_fit(obs)

        if (i + 1) % 25 == 0:
            weight_stats = raker.weight_distribution_stats
            print(f"\nStep {i + 1} weight distribution:")
            print(f"  Range: [{weight_stats['min']:.3f}, {weight_stats['max']:.3f}]")
            print(f"  Mean±SD: {weight_stats['mean']:.3f}±{weight_stats['std']:.3f}")
            print(
                f"  Median (IQR): {weight_stats['median']:.3f} "
                f"({weight_stats['q25']:.3f}-{weight_stats['q75']:.3f})"
            )
            print(f"  Outliers: {weight_stats['outliers_count']}")
            print(f"  ESS: {raker.effective_sample_size:.1f}/{i + 1}")

    print(f"\nFinal margins achieved: {raker.margins}")
    print(f"Target margins: {targets.as_dict()}")


if __name__ == "__main__":
    demo_convergence_monitoring()
    demo_oscillation_detection()
    demo_weight_distribution()

    print("\n" + "=" * 60)
    print("All diagnostics demonstrations completed!")
    print("=" * 60)
