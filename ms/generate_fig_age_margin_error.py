from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


# Define data structures
@dataclass
class Demographics:
    """Container for demographic data: binary indicators"""

    age: int
    gender: int
    education: int
    region: int


@dataclass
class Targets:
    """Target population proportions for each demographic variable"""

    age: float = 0.5
    gender: float = 0.5
    education: float = 0.4
    region: float = 0.3


class OnlineRakingSGD:
    """Online raking algorithm using stochastic gradient descent (additive updates)."""

    def __init__(
        self,
        targets: Targets,
        learning_rate: float = 5.0,
        min_weight: float = 0.01,
        max_weight: float = 10.0,
    ):
        self.targets = targets
        self.learning_rate = learning_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.weights: list[float] = []
        self.demographics: list[Demographics] = []

    def _compute_margins(
        self, weights: np.ndarray, demographics: list[Demographics]
    ) -> dict[str, float]:
        if len(demographics) == 0:
            return {}
        total_weight = weights.sum()
        # stack demographic arrays
        age_arr = np.array([d.age for d in demographics])
        gender_arr = np.array([d.gender for d in demographics])
        edu_arr = np.array([d.education for d in demographics])
        region_arr = np.array([d.region for d in demographics])
        return {
            "age": (weights * age_arr).sum() / total_weight,
            "gender": (weights * gender_arr).sum() / total_weight,
            "education": (weights * edu_arr).sum() / total_weight,
            "region": (weights * region_arr).sum() / total_weight,
        }

    def _compute_gradients(
        self, weights: np.ndarray, demographics: list[Demographics]
    ) -> np.ndarray:
        if len(weights) == 0:
            return np.array([])
        n = len(weights)
        total_weight = weights.sum()
        self._compute_margins(weights, demographics)
        # stack arrays
        demo_arrays = {
            "age": np.array([d.age for d in demographics]),
            "gender": np.array([d.gender for d in demographics]),
            "education": np.array([d.education for d in demographics]),
            "region": np.array([d.region for d in demographics]),
        }
        targets_dict = {
            "age": self.targets.age,
            "gender": self.targets.gender,
            "education": self.targets.education,
            "region": self.targets.region,
        }
        gradients = np.zeros(n)
        for demo_name, demo_array in demo_arrays.items():
            weighted_sum = (weights * demo_array).sum()
            current_margin = weighted_sum / total_weight
            target = targets_dict[demo_name]
            # gradient of margin w.r.t each weight
            margin_grads = (demo_array * total_weight - weighted_sum) / (
                total_weight**2
            )
            # chain rule for loss gradient
            loss_grads = 2 * (current_margin - target) * margin_grads
            gradients += loss_grads
        return gradients

    def add_observation(self, demo: Demographics, n_steps: int = 3):
        # add new observation
        self.demographics.append(demo)
        self.weights.append(1.0)
        weights_array = np.array(self.weights)
        for _ in range(n_steps):
            gradients = self._compute_gradients(weights_array, self.demographics)
            weights_array -= self.learning_rate * gradients
            weights_array = np.clip(weights_array, self.min_weight, self.max_weight)
        self.weights = weights_array.tolist()
        return self._compute_margins(weights_array, self.demographics)


class OnlineRakingMWU(OnlineRakingSGD):
    """Online raking algorithm using multiplicative weights updates."""

    def __init__(
        self,
        targets: Targets,
        learning_rate: float = 1.0,
        min_weight: float = 0.01,
        max_weight: float = 10.0,
    ):
        super().__init__(targets, learning_rate, min_weight, max_weight)

    def add_observation(self, demo: Demographics, n_steps: int = 3):
        self.demographics.append(demo)
        self.weights.append(1.0)
        weights_array = np.array(self.weights)
        for _ in range(n_steps):
            gradients = self._compute_gradients(weights_array, self.demographics)
            weights_array *= np.exp(-self.learning_rate * gradients)
            weights_array = np.clip(weights_array, self.min_weight, self.max_weight)
        self.weights = weights_array.tolist()
        return self._compute_margins(weights_array, self.demographics)


class BiasSimulator:
    """Simulate evolving bias patterns in streaming data."""

    @staticmethod
    def linear_shift(
        n_obs: int, start_probs: dict[str, float], end_probs: dict[str, float]
    ) -> list[Demographics]:
        data = []
        for i in range(n_obs):
            progress = i / (n_obs - 1) if n_obs > 1 else 0
            probs = {
                k: start_probs[k] + progress * (end_probs[k] - start_probs[k])
                for k in start_probs
            }
            demo = Demographics(
                age=np.random.binomial(1, probs["age"]),
                gender=np.random.binomial(1, probs["gender"]),
                education=np.random.binomial(1, probs["education"]),
                region=np.random.binomial(1, probs["region"]),
            )
            data.append(demo)
        return data


def simulate_age_error_linear_drift(n_obs: int = 300, n_seeds: int = 5):
    targets = Targets()
    start_probs = {"age": 0.2, "gender": 0.3, "education": 0.2, "region": 0.1}
    end_probs = {"age": 0.8, "gender": 0.7, "education": 0.6, "region": 0.5}

    errors_sgd = []  # list of error arrays per seed
    errors_mwu = []
    errors_base = []
    for seed in range(n_seeds):
        np.random.seed(seed)
        data = BiasSimulator.linear_shift(n_obs, start_probs, end_probs)
        raker_sgd = OnlineRakingSGD(targets, learning_rate=5.0)
        raker_mwu = OnlineRakingMWU(targets, learning_rate=1.0)
        age_errors_sgd = []
        age_errors_mwu = []
        age_errors_base = []
        for demo in data:
            # unweighted margin error
            # append demo to rakers first, since baseline uses raw margins after updating list
            raker_sgd.add_observation(demo)
            raker_mwu.add_observation(demo)
            # compute baseline raw age margin (without weighting)
            # use raker_sgd.demographics for same list
            demogs = raker_sgd.demographics
            raw_age_mean = np.mean([d.age for d in demogs])
            age_errors_base.append(abs(raw_age_mean - targets.age))
            age_margins_sgd = raker_sgd._compute_margins(
                np.array(raker_sgd.weights), demogs
            )
            age_margins_mwu = raker_mwu._compute_margins(
                np.array(raker_mwu.weights), demogs
            )
            age_errors_sgd.append(abs(age_margins_sgd["age"] - targets.age))
            age_errors_mwu.append(abs(age_margins_mwu["age"] - targets.age))
        errors_sgd.append(age_errors_sgd)
        errors_mwu.append(age_errors_mwu)
        errors_base.append(age_errors_base)
    # convert to arrays and average across seeds
    errors_sgd = np.mean(np.array(errors_sgd), axis=0)
    errors_mwu = np.mean(np.array(errors_mwu), axis=0)
    errors_base = np.mean(np.array(errors_base), axis=0)
    return errors_sgd, errors_mwu, errors_base


def main():
    n_obs = 300
    n_seeds = 5
    errors_sgd, errors_mwu, errors_base = simulate_age_error_linear_drift(
        n_obs, n_seeds
    )
    x = np.arange(1, n_obs + 1)
    plt.figure(figsize=(8, 4.5))
    plt.plot(x, errors_base, label="Baseline (raw)", linestyle="--")
    plt.plot(x, errors_sgd, label="SGD online raking")
    plt.plot(x, errors_mwu, label="MWU online raking")
    plt.xlabel("Observation index")
    plt.ylabel("Absolute age margin error")
    plt.title("Absolute age margin error over time in linear drift scenario")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_age_margin_error.png")
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("Figure saved as fig_age_margin_error.png")


if __name__ == "__main__":
    main()
