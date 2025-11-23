"""Realistic examples demonstrating onlinerake in action."""

import numpy as np

from onlinerake import OnlineRakingMWU, OnlineRakingSGD, Targets


def example_1_gender_bias():
    """Example 1: Correcting gender bias in an online survey."""
    print("=" * 60)
    print("EXAMPLE 1: Correcting Gender Bias in Online Survey")
    print("=" * 60)

    # US population targets (approximate)
    targets = Targets(
        age=0.52,  # 52% over 35 years old
        gender=0.51,  # 51% female
        education=0.35,  # 35% college educated
        region=0.19,  # 19% rural
    )

    print(f"Target margins: {targets.as_dict()}")

    # Initialize both rakers
    sgd_raker = OnlineRakingSGD(targets, learning_rate=4.0)
    mwu_raker = OnlineRakingMWU(targets, learning_rate=1.2)

    # Simulate biased online responses (tech survey = more young males)
    np.random.seed(42)
    n_responses = 500

    print(f"\nSimulating {n_responses} biased survey responses...")
    print("Bias pattern: Tech survey with overrepresentation of young males")

    raw_totals = {"age": 0, "gender": 0, "education": 0, "region": 0}

    for _i in range(n_responses):
        # Bias: 70% young (age=0), 65% male (gender=0), 60% college (education=1)
        age = 1 if np.random.random() < 0.3 else 0  # 30% older
        gender = 1 if np.random.random() < 0.35 else 0  # 35% female
        education = 1 if np.random.random() < 0.6 else 0  # 60% college
        region = 1 if np.random.random() < 0.15 else 0  # 15% rural

        obs = {"age": age, "gender": gender, "education": education, "region": region}

        # Update both rakers
        sgd_raker.partial_fit(obs)
        mwu_raker.partial_fit(obs)

        # Track raw proportions
        for key in raw_totals:
            raw_totals[key] += obs[key]

    # Calculate raw proportions
    raw_margins = {k: v / n_responses for k, v in raw_totals.items()}

    # Get final results
    sgd_margins = sgd_raker.margins
    mwu_margins = mwu_raker.margins

    print(f"\nRESULTS after {n_responses} responses:")
    print("-" * 40)
    print(f"{'Characteristic':<12} {'Target':<8} {'Raw':<8} {'SGD':<8} {'MWU':<8}")
    print("-" * 40)

    for char in ["age", "gender", "education", "region"]:
        target = targets.as_dict()[char]
        raw = raw_margins[char]
        sgd = sgd_margins[char]
        mwu = mwu_margins[char]
        print(f"{char:<12} {target:<8.3f} {raw:<8.3f} {sgd:<8.3f} {mwu:<8.3f}")

    print("\nEffective Sample Size:")
    print(f"SGD: {sgd_raker.effective_sample_size:.1f}")
    print(f"MWU: {mwu_raker.effective_sample_size:.1f}")

    print("\nFinal Loss (squared error on margins):")
    print(f"SGD: {sgd_raker.loss:.6f}")
    print(f"MWU: {mwu_raker.loss:.6f}")


def example_2_streaming_polls():
    """Example 2: Real-time polling with changing demographics."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Real-time Polling with Demographic Shifts")
    print("=" * 60)

    # 2024 US voter demographics (approximate)
    targets = Targets(
        age=0.48,  # 48% over 50 years old
        gender=0.53,  # 53% female voters
        education=0.32,  # 32% college degree
        region=0.17,  # 17% rural voters
    )

    print(f"Target voter margins: {targets.as_dict()}")

    raker = OnlineRakingSGD(targets, learning_rate=3.0)

    # Simulate polling responses with time-varying bias
    np.random.seed(789)
    n_polls = 1000

    print(f"\nSimulating {n_polls} poll responses with time-varying bias...")

    # Track margins over time
    margin_history = []

    for i in range(n_polls):
        # Early responses skew younger (social media recruitment)
        # Later responses skew older (phone polling kicks in)
        time_factor = i / n_polls

        # Probability of older voter increases over time
        p_older = 0.2 + 0.4 * time_factor  # 20% -> 60%
        age = 1 if np.random.random() < p_older else 0

        # Gender relatively stable
        gender = 1 if np.random.random() < 0.52 else 0

        # Education: early respondents more educated
        p_educated = 0.6 - 0.3 * time_factor  # 60% -> 30%
        education = 1 if np.random.random() < p_educated else 0

        # Region relatively stable
        region = 1 if np.random.random() < 0.18 else 0

        obs = {"age": age, "gender": gender, "education": education, "region": region}
        raker.partial_fit(obs)

        # Record margins every 100 responses
        if (i + 1) % 200 == 0:
            margins = raker.margins
            margin_history.append((i + 1, margins.copy()))

    print("\nMargin evolution over time:")
    print("-" * 50)
    print(f"{'N':<6} {'Age':<8} {'Gender':<8} {'Edu':<8} {'Region':<8}")
    print("-" * 50)

    for n, margins in margin_history:
        print(
            f"{n:<6} {margins['age']:<8.3f} {margins['gender']:<8.3f} "
            f"{margins['education']:<8.3f} {margins['region']:<8.3f}"
        )

    final_margins = raker.margins
    print("\nFinal weighted margins vs targets:")
    print("-" * 40)
    for char in ["age", "gender", "education", "region"]:
        target = targets.as_dict()[char]
        final = final_margins[char]
        error = abs(final - target)
        print(f"{char:<12}: {final:.3f} (target: {target:.3f}, error: {error:.3f})")

    print(f"\nFinal ESS: {raker.effective_sample_size:.1f} / {n_polls}")


def example_3_comparative_analysis():
    """Example 3: Side-by-side comparison of SGD vs MWU."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: SGD vs MWU Performance Comparison")
    print("=" * 60)

    targets = Targets(age=0.45, gender=0.52, education=0.38, region=0.22)

    # Different learning rates optimized for each method
    sgd_raker = OnlineRakingSGD(targets, learning_rate=5.0)
    mwu_raker = OnlineRakingMWU(targets, learning_rate=1.0)

    # Simulate sudden demographic shift scenario
    np.random.seed(2024)
    n_obs = 800

    print(f"Simulating sudden demographic shift after {n_obs // 2} observations...")

    for i in range(n_obs):
        if i < n_obs // 2:
            # First half: younger, more educated sample
            age = 1 if np.random.random() < 0.25 else 0  # 25% older
            education = 1 if np.random.random() < 0.65 else 0  # 65% educated
        else:
            # Second half: older, less educated sample
            age = 1 if np.random.random() < 0.70 else 0  # 70% older
            education = 1 if np.random.random() < 0.15 else 0  # 15% educated

        gender = 1 if np.random.random() < 0.50 else 0
        region = 1 if np.random.random() < 0.20 else 0

        obs = {"age": age, "gender": gender, "education": education, "region": region}

        sgd_raker.partial_fit(obs)
        mwu_raker.partial_fit(obs)

    print(f"\nFinal Results after {n_obs} observations:")
    print("=" * 50)
    print(f"{'Metric':<20} {'Target':<10} {'SGD':<10} {'MWU':<10}")
    print("=" * 50)

    sgd_final = sgd_raker.margins
    mwu_final = mwu_raker.margins

    for char in ["age", "gender", "education", "region"]:
        target = targets.as_dict()[char]
        sgd_val = sgd_final[char]
        mwu_val = mwu_final[char]
        print(f"{char:<20} {target:<10.3f} {sgd_val:<10.3f} {mwu_val:<10.3f}")

    print("-" * 50)
    print(
        f"{'Loss (sq error)':<20} {'':<10} {sgd_raker.loss:<10.6f} {mwu_raker.loss:<10.6f}"
    )
    print(
        f"{'Effective Sample Size':<20} {'':<10} {sgd_raker.effective_sample_size:<10.1f} {mwu_raker.effective_sample_size:<10.1f}"
    )

    # Calculate improvement over raw data
    raw_totals = {"age": 0, "gender": 0, "education": 0, "region": 0}
    for i in range(n_obs):
        if i < n_obs // 2:
            age = 1 if np.random.random() < 0.25 else 0
            education = 1 if np.random.random() < 0.65 else 0
        else:
            age = 1 if np.random.random() < 0.70 else 0
            education = 1 if np.random.random() < 0.15 else 0
        gender = 1 if np.random.random() < 0.50 else 0
        region = 1 if np.random.random() < 0.20 else 0

        raw_totals["age"] += age
        raw_totals["gender"] += gender
        raw_totals["education"] += education
        raw_totals["region"] += region

    print("\nImprovement vs Raw Data:")
    print("-" * 30)
    for char in ["age", "gender", "education", "region"]:
        target = targets.as_dict()[char]
        raw_prop = raw_totals[char] / n_obs
        raw_error = abs(raw_prop - target)
        sgd_error = abs(sgd_final[char] - target)
        mwu_error = abs(mwu_final[char] - target)

        sgd_improvement = (1 - sgd_error / raw_error) * 100 if raw_error > 0 else 0
        mwu_improvement = (1 - mwu_error / raw_error) * 100 if raw_error > 0 else 0

        print(
            f"{char}: SGD {sgd_improvement:.1f}% improvement, MWU {mwu_improvement:.1f}% improvement"
        )


if __name__ == "__main__":
    example_1_gender_bias()
    example_2_streaming_polls()
    example_3_comparative_analysis()
