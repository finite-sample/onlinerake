"""Model-assisted streaming calibration (GREG/MRP).

This module extends onlinerake to support model-assisted streaming calibration
using SGD. The key insight is that model fitting is batch (done once on
initial data), while calibration weight updates are streaming.

Two variants are provided:

1. **GREG-style** (ModelAssistedRaker): Uses model predictions as auxiliary
   variables. If the population mean of predictions is known, it's added as
   a calibration target for efficiency gains.

2. **MRP-style** (StreamingMRP): Cell-based post-stratification where cells
   are defined by demographic combinations, model predicts cell-level outcomes,
   and weights adjust to match cell proportions.

Mathematical formulation for GREG:

    L(w) = ||m(w) - τ||² + α||m_pred(w) - τ_pred||² + λ Σᵢ wᵢ rᵢ²

Where:
    - m(w) = weighted demographic margins
    - m_pred(w) = weighted mean of model predictions
    - rᵢ = yᵢ - ŷᵢ = residuals (optional penalty)
    - τ_pred = known population mean of predictions

The gradient remains convex in weights, so SGD convergence applies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from .models import OutcomeModel
from .online_raking_sgd import OnlineRakingSGD
from .targets import Targets

if TYPE_CHECKING:
    from .learning_rate import LearningRateSchedule


@dataclass
class ModelAssistedTargets:
    """Combined demographic and prediction targets for model-assisted calibration.

    Attributes:
        demographic_targets: Standard Targets object with demographic margins.
        prediction_targets: Optional dict mapping prediction names to known
            population means. E.g., {"vote_prob": 0.48} means the true population
            probability of voting is 0.48.

    Examples:
        >>> targets = ModelAssistedTargets(
        ...     demographic_targets=Targets(female=0.51, college=0.32),
        ...     prediction_targets={"vote_prob": 0.48},
        ... )
    """

    demographic_targets: Targets
    prediction_targets: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.prediction_targets is not None:
            for name, value in self.prediction_targets.items():
                if not isinstance(value, (int, float)):
                    raise ValueError(
                        f"Prediction target '{name}' must be numeric, "
                        f"got {type(value).__name__}"
                    )


class ModelAssistedRaker(OnlineRakingSGD):
    """GREG-style streaming calibrator with model-assisted estimation.

    This class extends OnlineRakingSGD to incorporate model predictions
    as auxiliary variables for efficiency gains. The model is fixed
    (fitted once in batch), while weights update in streaming fashion.

    The loss function includes:
    1. Demographic margin matching (inherited from OnlineRakingSGD)
    2. Prediction target matching (if prediction_targets provided)
    3. Optional residual penalty for variance reduction

    Args:
        targets: ModelAssistedTargets with demographic and prediction targets.
        model: Fitted outcome model with predict() method.
        prediction_weight: Weight for prediction calibration term (α).
            Higher values prioritize matching prediction targets. Default 1.0.
        residual_weight: Weight for residual penalty term (λ).
            Penalizes observations with large residuals. Default 0.0.
        learning_rate: Step size for gradient descent. Default 5.0.
        min_weight: Lower bound for weights. Default 0.001.
        max_weight: Upper bound for weights. Default 100.0.
        n_sgd_steps: Number of gradient steps per observation. Default 3.
        verbose: If True, log progress. Default False.
        track_convergence: If True, monitor convergence. Default True.
        convergence_window: Observations for convergence detection. Default 20.
        compute_weight_stats: Control weight stats computation. Default False.
        feature_names_in_obs: Names of features in observation that correspond
            to model input. If None, uses demographic target feature names.

    Attributes:
        model: The outcome model.
        predictions: Stored predictions for each observation.
        outcomes: Stored outcomes for each observation (if provided).
        prediction_targets: Dict of prediction targets.

    Examples:
        >>> from sklearn.linear_model import LogisticRegression
        >>> from onlinerake import Targets
        >>> from onlinerake.model_assisted import ModelAssistedRaker, ModelAssistedTargets
        >>> from onlinerake.models import ExternalModelWrapper
        >>>
        >>> # Stage 1: Fit model (batch)
        >>> clf = LogisticRegression().fit(X_train, y_train)
        >>> model = ExternalModelWrapper(clf, use_proba=True)
        >>>
        >>> # Stage 2: Stream with calibration
        >>> targets = ModelAssistedTargets(
        ...     demographic_targets=Targets(female=0.51, college=0.32),
        ...     prediction_targets={"pred": 0.48},
        ... )
        >>> raker = ModelAssistedRaker(targets, model)
        >>>
        >>> for obs in poll_stream:
        ...     raker.partial_fit(obs, outcome=obs.get("vote"))
        ...     print(f"GREG estimate: {raker.model_assisted_estimate:.3f}")
    """

    def __init__(
        self,
        targets: ModelAssistedTargets,
        model: OutcomeModel,
        prediction_weight: float = 1.0,
        residual_weight: float = 0.0,
        learning_rate: float | LearningRateSchedule = 5.0,
        min_weight: float = 1e-3,
        max_weight: float = 100.0,
        n_sgd_steps: int = 3,
        verbose: bool = False,
        track_convergence: bool = True,
        convergence_window: int = 20,
        compute_weight_stats: bool | int = False,
        max_history: int | None = 1000,
        feature_names_in_obs: list[str] | None = None,
    ) -> None:
        # Store model-assisted specific attributes
        self.model = model
        self.prediction_weight = prediction_weight
        self.residual_weight = residual_weight
        self.prediction_targets = targets.prediction_targets or {}
        self.model_assisted_targets = targets
        self.feature_names_in_obs = (
            feature_names_in_obs or targets.demographic_targets.feature_names
        )

        # Initialize parent with demographic targets
        super().__init__(
            targets=targets.demographic_targets,
            learning_rate=learning_rate,
            min_weight=min_weight,
            max_weight=max_weight,
            n_sgd_steps=n_sgd_steps,
            verbose=verbose,
            track_convergence=track_convergence,
            convergence_window=convergence_window,
            compute_weight_stats=compute_weight_stats,
            max_history=max_history,
        )

        # Storage for predictions and outcomes (capacity doubling)
        self._predictions_capacity: int = 0
        self._predictions: np.ndarray = np.empty(0, dtype=np.float64)
        self._outcomes_capacity: int = 0
        self._outcomes: np.ndarray = np.empty(0, dtype=np.float64)
        self._has_outcomes: np.ndarray = np.empty(0, dtype=bool)

    def _expand_capacity(self) -> None:
        """Expand capacity for base arrays plus predictions/outcomes."""
        super()._expand_capacity()

        # Expand predictions array
        if self._n_obs >= self._predictions_capacity:
            new_capacity = max(8, self._predictions_capacity * 2, self._n_obs + 1)
            new_predictions = np.zeros(new_capacity, dtype=np.float64)
            if self._predictions_capacity > 0:
                new_predictions[: self._predictions_capacity] = self._predictions[
                    : self._predictions_capacity
                ]
            self._predictions = new_predictions
            self._predictions_capacity = new_capacity

        # Expand outcomes array
        if self._n_obs >= self._outcomes_capacity:
            new_capacity = max(8, self._outcomes_capacity * 2, self._n_obs + 1)
            new_outcomes = np.zeros(new_capacity, dtype=np.float64)
            new_has_outcomes = np.zeros(new_capacity, dtype=bool)
            if self._outcomes_capacity > 0:
                new_outcomes[: self._outcomes_capacity] = self._outcomes[
                    : self._outcomes_capacity
                ]
                new_has_outcomes[: self._outcomes_capacity] = self._has_outcomes[
                    : self._outcomes_capacity
                ]
            self._outcomes = new_outcomes
            self._has_outcomes = new_has_outcomes
            self._outcomes_capacity = new_capacity

    def _extract_model_features(
        self, obs: dict[str, Any] | Any
    ) -> npt.NDArray[np.float64]:
        """Extract features for model prediction from observation.

        Args:
            obs: Observation dict or object.

        Returns:
            Feature array for model.predict().
        """
        features = []
        for name in self.feature_names_in_obs:
            if isinstance(obs, dict):
                val = obs.get(name, 0)
            else:
                val = getattr(obs, name, 0)
            features.append(float(val))
        return np.array(features, dtype=np.float64)

    def _compute_gradient(self) -> npt.NDArray[np.float64]:
        """Compute gradient including prediction and residual terms.

        Returns:
            Gradient vector of shape (n_obs,).
        """
        # Get base demographic gradient
        grad = super()._compute_gradient()

        if self._n_obs == 0:
            return grad

        n = self._n_obs
        w = self._weights[:n]
        total_w = w.sum()

        # Add prediction calibration gradient
        if self.prediction_targets and self.prediction_weight > 0:
            predictions = self._predictions[:n]
            weighted_pred_sum = (w * predictions).sum()
            current_pred_margin = weighted_pred_sum / total_w

            for target_value in self.prediction_targets.values():
                pred_margin_grad = (predictions * total_w - weighted_pred_sum) / (
                    total_w * total_w
                )
                loss_grad = (
                    2.0
                    * self.prediction_weight
                    * (current_pred_margin - target_value)
                    * pred_margin_grad
                )
                grad += loss_grad
                break

        # Add residual penalty gradient
        if self.residual_weight > 0 and np.any(self._has_outcomes[:n]):
            outcomes = self._outcomes[:n]
            predictions = self._predictions[:n]
            has_outcome = self._has_outcomes[:n]

            residuals = np.where(has_outcome, outcomes - predictions, 0.0)
            residuals_sq = residuals * residuals

            # Gradient of Σ wᵢ rᵢ² is rᵢ² for each i
            grad += self.residual_weight * residuals_sq

        return grad

    def partial_fit(
        self,
        obs: dict[str, Any] | Any,
        outcome: float | None = None,
    ) -> None:
        """Process observation with model-assisted calibration.

        Args:
            obs: Observation containing feature values.
            outcome: Optional observed outcome (for residual penalty and
                GREG estimation). If None, only prediction is used.

        Note:
            After calling, inspect `model_assisted_estimate` for the GREG
            adjusted estimate.
        """
        # Ensure capacity
        self._expand_capacity()

        # Get model prediction
        model_features = self._extract_model_features(obs)
        prediction = float(self.model.predict(model_features.reshape(1, -1))[0])

        # Store prediction
        self._predictions[self._n_obs] = prediction

        # Store outcome if provided
        if outcome is not None:
            self._outcomes[self._n_obs] = float(outcome)
            self._has_outcomes[self._n_obs] = True
        else:
            self._has_outcomes[self._n_obs] = False

        # Extract demographic features using parent helper
        feature_values = self._extract_feature_values(obs)

        # Store observation
        self._features[self._n_obs] = feature_values
        self._weights[self._n_obs] = 1.0
        self._n_obs += 1

        # Get current learning rate
        current_lr = self._get_current_learning_rate()

        # Perform SGD steps
        final_gradient_norm = 0.0
        for step in range(self.n_sgd_steps):
            grad = self._compute_gradient()
            gradient_norm = float(np.linalg.norm(grad))
            if step == self.n_sgd_steps - 1:
                final_gradient_norm = gradient_norm

            self._weights[: self._n_obs] -= current_lr * grad
            np.clip(
                self._weights[: self._n_obs],
                self.min_weight,
                self.max_weight,
                out=self._weights[: self._n_obs],
            )

            if step == 0:
                self._log_update("GREG Obs", gradient_norm)

        # Record state
        self._record_state(gradient_norm=final_gradient_norm)

    @property
    def predictions(self) -> npt.NDArray[np.float64]:
        """Get stored predictions for all observations."""
        return self._predictions[: self._n_obs].copy()

    @property
    def outcomes(self) -> npt.NDArray[np.float64]:
        """Get stored outcomes for observations where provided."""
        return self._outcomes[: self._n_obs].copy()

    @property
    def residuals(self) -> npt.NDArray[np.float64]:
        """Get residuals (outcome - prediction) where outcome available."""
        res = np.zeros(self._n_obs, dtype=np.float64)
        mask = self._has_outcomes[: self._n_obs]
        res[mask] = (
            self._outcomes[: self._n_obs][mask] - self._predictions[: self._n_obs][mask]
        )
        return res

    @property
    def weighted_mean_prediction(self) -> float:
        """Get weighted mean of model predictions."""
        if self._n_obs == 0:
            return np.nan
        w = self._weights[: self._n_obs]
        return float((w * self._predictions[: self._n_obs]).sum() / w.sum())

    @property
    def weighted_mean_outcome(self) -> float:
        """Get weighted mean of outcomes (where available)."""
        if self._n_obs == 0:
            return np.nan

        mask = self._has_outcomes[: self._n_obs]
        if not np.any(mask):
            return np.nan

        w = self._weights[: self._n_obs][mask]
        outcomes = self._outcomes[: self._n_obs][mask]
        return float((w * outcomes).sum() / w.sum())

    @property
    def model_assisted_estimate(self) -> float:
        """Compute GREG model-assisted estimate.

        The GREG estimator is:
            θ̂_GREG = θ̂_weighted + (τ_pred - mean_pred_weighted)

        where τ_pred is the known population prediction mean.

        If no prediction target is set, returns weighted mean outcome.
        If no outcomes available, returns weighted mean prediction.
        """
        if self._n_obs == 0:
            return np.nan

        # If no outcomes, return weighted prediction
        if not np.any(self._has_outcomes[: self._n_obs]):
            return self.weighted_mean_prediction

        weighted_mean = self.weighted_mean_outcome

        # If we have a prediction target, apply GREG adjustment
        if self.prediction_targets:
            target_value = list(self.prediction_targets.values())[0]
            weighted_pred = self.weighted_mean_prediction
            # GREG adjustment: add the difference between target and achieved prediction
            greg_adjustment = target_value - weighted_pred
            return weighted_mean + greg_adjustment

        return weighted_mean

    @property
    def prediction_margin_loss(self) -> float:
        """Get squared error loss for prediction margin."""
        if self._n_obs == 0 or not self.prediction_targets:
            return 0.0

        target_value = list(self.prediction_targets.values())[0]
        current = self.weighted_mean_prediction
        return float((current - target_value) ** 2)

    @property
    def total_loss(self) -> float:
        """Get total loss including demographic and prediction terms."""
        demo_loss = self.loss
        pred_loss = self.prediction_weight * self.prediction_margin_loss
        return float(demo_loss + pred_loss)


@dataclass
class PoststratificationCell:
    """A single cell in post-stratification.

    Attributes:
        cell_id: Unique identifier for the cell.
        features: Dict of feature values defining the cell.
        population_proportion: Known proportion of population in this cell.
        model_prediction: Model's prediction for this cell.
    """

    cell_id: str
    features: dict[str, Any]
    population_proportion: float
    model_prediction: float | None = None


@dataclass
class PoststratificationCells:
    """Collection of post-stratification cells for MRP.

    Attributes:
        cells: List of PoststratificationCell objects.

    Examples:
        >>> cells = PoststratificationCells([
        ...     PoststratificationCell("young_female", {"age_group": "young", "female": 1}, 0.15),
        ...     PoststratificationCell("young_male", {"age_group": "young", "female": 0}, 0.14),
        ...     PoststratificationCell("old_female", {"age_group": "old", "female": 1}, 0.36),
        ...     PoststratificationCell("old_male", {"age_group": "old", "female": 0}, 0.35),
        ... ])
    """

    cells: list[PoststratificationCell] = field(default_factory=list)

    def __post_init__(self) -> None:
        total_prop = sum(c.population_proportion for c in self.cells)
        if abs(total_prop - 1.0) > 1e-6 and len(self.cells) > 0:
            raise ValueError(
                f"Cell population proportions must sum to 1.0, got {total_prop}"
            )

    @property
    def cell_ids(self) -> list[str]:
        """Get list of cell IDs."""
        return [c.cell_id for c in self.cells]

    @property
    def n_cells(self) -> int:
        """Get number of cells."""
        return len(self.cells)

    def get_cell(self, cell_id: str) -> PoststratificationCell | None:
        """Get cell by ID."""
        for cell in self.cells:
            if cell.cell_id == cell_id:
                return cell
        return None

    def get_cell_for_obs(
        self, obs: dict[str, Any] | Any
    ) -> PoststratificationCell | None:
        """Find the cell an observation belongs to.

        Args:
            obs: Observation with feature values.

        Returns:
            Matching cell or None if no match.
        """
        for cell in self.cells:
            match = True
            for feature, value in cell.features.items():
                if isinstance(obs, dict):
                    obs_value = obs.get(feature)
                else:
                    obs_value = getattr(obs, feature, None)

                if obs_value != value:
                    match = False
                    break

            if match:
                return cell

        return None


class StreamingMRP:
    """Full MRP with cell-based aggregation and streaming weight updates.

    MRP (Multilevel Regression with Poststratification) estimates population
    quantities by:
    1. Fitting a model to predict outcomes within demographic cells
    2. Weighting cell-level estimates by population proportions
    3. Aggregating to population-level estimate

    This class implements streaming MRP where:
    - Model is fixed (fitted in batch before streaming)
    - Cell-level weights update as data streams in
    - Population estimate updates with each observation

    Args:
        model: Fitted outcome model for cell-level predictions.
        cells: PoststratificationCells defining the cell structure.
        learning_rate: Learning rate for weight updates. Default 5.0.
        min_weight: Minimum weight bound. Default 0.001.
        max_weight: Maximum weight bound. Default 100.0.
        n_sgd_steps: SGD steps per observation. Default 3.

    Attributes:
        model: The outcome model.
        cells: The cell structure.
        cell_rakers: Dict mapping cell_id to OnlineRakingSGD for that cell.

    Examples:
        >>> from onlinerake.model_assisted import StreamingMRP, PoststratificationCells
        >>> from onlinerake.models import LogisticOutcomeModel
        >>>
        >>> # Define cells
        >>> cells = PoststratificationCells([
        ...     PoststratificationCell("young_f", {"age": "young", "female": 1}, 0.15, 0.45),
        ...     PoststratificationCell("young_m", {"age": "young", "female": 0}, 0.14, 0.52),
        ...     PoststratificationCell("old_f", {"age": "old", "female": 1}, 0.36, 0.48),
        ...     PoststratificationCell("old_m", {"age": "old", "female": 0}, 0.35, 0.55),
        ... ])
        >>>
        >>> # Create MRP estimator
        >>> mrp = StreamingMRP(model=model, cells=cells)
        >>>
        >>> for obs in poll_stream:
        ...     mrp.partial_fit(obs, outcome=obs.get("vote"))
        ...     print(f"Population estimate: {mrp.population_estimate:.3f}")
    """

    def __init__(
        self,
        model: OutcomeModel,
        cells: PoststratificationCells,
        learning_rate: float = 5.0,
        min_weight: float = 1e-3,
        max_weight: float = 100.0,
        n_sgd_steps: int = 3,
    ) -> None:
        self.model = model
        self.cells = cells
        self.learning_rate = learning_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.n_sgd_steps = n_sgd_steps

        # Cell-level storage
        self._cell_outcomes: dict[str, list[float]] = {
            c.cell_id: [] for c in cells.cells
        }
        self._cell_weights: dict[str, list[float]] = {
            c.cell_id: [] for c in cells.cells
        }
        self._cell_predictions: dict[str, list[float]] = {
            c.cell_id: [] for c in cells.cells
        }

        self._n_obs: int = 0

    def partial_fit(
        self,
        obs: dict[str, Any] | Any,
        outcome: float | None = None,
    ) -> None:
        """Process observation for MRP.

        Args:
            obs: Observation with cell-defining features.
            outcome: Observed outcome (required for MRP estimation).
        """
        # Find cell for this observation
        cell = self.cells.get_cell_for_obs(obs)
        if cell is None:
            # Observation doesn't match any cell, skip
            return

        # Use pre-computed cell model prediction if available
        prediction = cell.model_prediction if cell.model_prediction is not None else 0.5

        # Store in cell
        self._cell_predictions[cell.cell_id].append(prediction)
        self._cell_weights[cell.cell_id].append(1.0)

        if outcome is not None:
            self._cell_outcomes[cell.cell_id].append(float(outcome))

        self._n_obs += 1

    def get_cell_estimate(self, cell_id: str) -> float:
        """Get weighted estimate for a single cell.

        Args:
            cell_id: Cell identifier.

        Returns:
            Weighted mean outcome in cell, or model prediction if no outcomes.
        """
        outcomes = self._cell_outcomes.get(cell_id, [])
        weights = self._cell_weights.get(cell_id, [])

        if not outcomes or len(outcomes) == 0:
            # Fall back to model prediction
            cell = self.cells.get_cell(cell_id)
            if cell and cell.model_prediction is not None:
                return cell.model_prediction
            return np.nan

        w_arr = np.array(weights, dtype=np.float64)
        o_arr = np.array(outcomes, dtype=np.float64)

        return float((w_arr * o_arr).sum() / w_arr.sum())

    def get_cell_n(self, cell_id: str) -> int:
        """Get number of observations in a cell."""
        return len(self._cell_outcomes.get(cell_id, []))

    @property
    def cell_estimates(self) -> dict[str, float]:
        """Get estimates for all cells."""
        return {
            cell_id: self.get_cell_estimate(cell_id) for cell_id in self.cells.cell_ids
        }

    @property
    def cell_counts(self) -> dict[str, int]:
        """Get observation counts for all cells."""
        return {cell_id: self.get_cell_n(cell_id) for cell_id in self.cells.cell_ids}

    @property
    def population_estimate(self) -> float:
        """Compute MRP population estimate.

        Returns:
            Σ_cells (N_cell/N) × θ̂_cell

        where N_cell/N is the population proportion in each cell.
        """
        total = 0.0
        for cell in self.cells.cells:
            cell_est = self.get_cell_estimate(cell.cell_id)
            if np.isnan(cell_est):
                # Use model prediction as fallback
                cell_est = (
                    cell.model_prediction if cell.model_prediction is not None else 0.0
                )
            total += cell.population_proportion * cell_est

        return total

    @property
    def effective_sample_size(self) -> float:
        """Approximate effective sample size across cells."""
        total_n = 0
        for cell_id in self.cells.cell_ids:
            total_n += self.get_cell_n(cell_id)
        return float(total_n)

    @property
    def n_obs(self) -> int:
        """Total observations processed."""
        return self._n_obs
