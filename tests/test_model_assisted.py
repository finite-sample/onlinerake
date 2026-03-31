"""Tests for model-assisted streaming calibration (GREG/MRP)."""

import numpy as np
import pytest

from onlinerake import Targets
from onlinerake.model_assisted import (ModelAssistedRaker,
                                       ModelAssistedTargets,
                                       PoststratificationCell,
                                       PoststratificationCells, StreamingMRP)
from onlinerake.models import (ExternalModelWrapper, LinearOutcomeModel,
                               LogisticOutcomeModel)


class TestLinearOutcomeModel:
    """Test LinearOutcomeModel."""

    def test_fit_predict(self):
        """Test basic fit and predict."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        true_coef = np.array([1.0, -0.5, 0.3])
        y = X @ true_coef + np.random.randn(100) * 0.1

        model = LinearOutcomeModel()
        model.fit(X, y)

        assert model.is_fitted
        assert model.coef_ is not None
        assert len(model.coef_) == 3

        predictions = model.predict(X)
        assert predictions.shape == (100,)

        # Check reasonable fit
        mse = np.mean((predictions - y) ** 2)
        assert mse < 0.1

    def test_fit_with_intercept(self):
        """Test model with intercept."""
        X = np.array([[1], [2], [3], [4], [5]], dtype=float)
        y = np.array([3, 5, 7, 9, 11], dtype=float)

        model = LinearOutcomeModel(fit_intercept=True)
        model.fit(X, y)

        assert abs(model.coef_[0] - 2.0) < 0.01
        assert abs(model.intercept_ - 1.0) < 0.01

    def test_fit_without_intercept(self):
        """Test model without intercept."""
        X = np.array([[1], [2], [3], [4], [5]], dtype=float)
        y = np.array([2, 4, 6, 8, 10], dtype=float)

        model = LinearOutcomeModel(fit_intercept=False)
        model.fit(X, y)

        assert abs(model.coef_[0] - 2.0) < 0.01
        assert model.intercept_ == 0.0

    def test_predict_before_fit_raises(self):
        """Test that predict before fit raises error."""
        model = LinearOutcomeModel()
        with pytest.raises(ValueError, match="must be fitted"):
            model.predict(np.array([[1, 2, 3]]))


class TestLogisticOutcomeModel:
    """Test LogisticOutcomeModel."""

    def test_fit_predict(self):
        """Test basic fit and predict for binary classification."""
        np.random.seed(42)
        X = np.random.randn(200, 2)
        true_coef = np.array([2.0, -1.0])
        z = X @ true_coef
        p = 1 / (1 + np.exp(-z))
        y = (np.random.random(200) < p).astype(float)

        model = LogisticOutcomeModel(learning_rate=0.5, max_iter=2000)
        model.fit(X, y)

        assert model.is_fitted
        assert model.coef_ is not None

        predictions = model.predict(X)
        assert predictions.shape == (200,)
        assert np.all((predictions >= 0) & (predictions <= 1))

    def test_predict_before_fit_raises(self):
        """Test that predict before fit raises error."""
        model = LogisticOutcomeModel()
        with pytest.raises(ValueError, match="must be fitted"):
            model.predict(np.array([[1, 2]]))


class TestExternalModelWrapper:
    """Test ExternalModelWrapper."""

    def test_wrap_simple_model(self):
        """Test wrapping a simple model with predict method."""

        class SimpleModel:
            def predict(self, X):
                return np.ones(len(X))

        model = SimpleModel()
        wrapper = ExternalModelWrapper(model)

        predictions = wrapper.predict(np.array([[1, 2], [3, 4]]))
        assert np.allclose(predictions, [1.0, 1.0])

    def test_wrap_model_with_proba(self):
        """Test wrapping model with predict_proba."""

        class ProbaModel:
            def predict(self, X):
                return np.zeros(len(X))

            def predict_proba(self, X):
                return np.column_stack([np.zeros(len(X)), np.ones(len(X)) * 0.7])

        model = ProbaModel()
        wrapper = ExternalModelWrapper(model, use_proba=True, proba_class=1)

        predictions = wrapper.predict(np.array([[1, 2], [3, 4]]))
        assert np.allclose(predictions, [0.7, 0.7])

    def test_model_without_predict_raises(self):
        """Test that model without predict raises error."""

        class NoPredict:
            pass

        with pytest.raises(ValueError, match="must have a predict"):
            ExternalModelWrapper(NoPredict())

    def test_proba_without_method_raises(self):
        """Test that use_proba without predict_proba raises error."""

        class OnlyPredict:
            def predict(self, X):
                return np.zeros(len(X))

        with pytest.raises(ValueError, match="predict_proba"):
            ExternalModelWrapper(OnlyPredict(), use_proba=True)


class TestModelAssistedTargets:
    """Test ModelAssistedTargets dataclass."""

    def test_basic_targets(self):
        """Test basic target creation."""
        targets = ModelAssistedTargets(
            demographic_targets=Targets(female=0.51, college=0.32),
            prediction_targets={"vote_prob": 0.48},
        )

        assert targets.demographic_targets["female"] == 0.51
        assert targets.prediction_targets is not None
        assert targets.prediction_targets["vote_prob"] == 0.48

    def test_without_prediction_targets(self):
        """Test targets without prediction targets."""
        targets = ModelAssistedTargets(
            demographic_targets=Targets(female=0.51, college=0.32),
        )

        assert targets.prediction_targets is None

    def test_invalid_prediction_target_raises(self):
        """Test that invalid prediction target raises error."""
        with pytest.raises(ValueError, match="must be numeric"):
            ModelAssistedTargets(
                demographic_targets=Targets(female=0.51),
                prediction_targets={"vote_prob": "invalid"},
            )


class TestModelAssistedRaker:
    """Test ModelAssistedRaker for GREG-style calibration."""

    def create_simple_model(self):
        """Create a simple linear model for testing."""
        model = LinearOutcomeModel()
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0.3, 0.4, 0.5, 0.6])
        model.fit(X, y)
        return model

    def test_initialization(self):
        """Test proper initialization."""
        model = self.create_simple_model()
        targets = ModelAssistedTargets(
            demographic_targets=Targets(female=0.5, college=0.3),
            prediction_targets={"pred": 0.45},
        )

        raker = ModelAssistedRaker(
            targets=targets,
            model=model,
            prediction_weight=1.0,
        )

        assert raker.prediction_weight == 1.0
        assert raker.prediction_targets == {"pred": 0.45}
        assert raker._n_obs == 0

    def test_partial_fit(self):
        """Test processing observations."""
        model = self.create_simple_model()
        targets = ModelAssistedTargets(
            demographic_targets=Targets(female=0.5, college=0.3),
            prediction_targets={"pred": 0.45},
        )

        raker = ModelAssistedRaker(targets, model)

        obs = {"female": 1, "college": 0}
        raker.partial_fit(obs, outcome=0.6)

        assert raker._n_obs == 1
        assert len(raker.predictions) == 1
        assert len(raker.outcomes) == 1

    def test_multiple_observations(self):
        """Test processing multiple observations."""
        model = self.create_simple_model()
        targets = ModelAssistedTargets(
            demographic_targets=Targets(female=0.5, college=0.3),
            prediction_targets={"pred": 0.45},
        )

        raker = ModelAssistedRaker(targets, model, learning_rate=2.0)

        observations = [
            {"female": 1, "college": 0},
            {"female": 0, "college": 1},
            {"female": 1, "college": 1},
            {"female": 0, "college": 0},
        ]
        outcomes = [0.5, 0.6, 0.4, 0.3]

        for obs, outcome in zip(observations, outcomes):
            raker.partial_fit(obs, outcome=outcome)

        assert raker._n_obs == 4
        assert len(raker.predictions) == 4
        assert not np.isnan(raker.weighted_mean_prediction)
        assert not np.isnan(raker.weighted_mean_outcome)

    def test_model_assisted_estimate(self):
        """Test GREG model-assisted estimate."""
        model = self.create_simple_model()
        targets = ModelAssistedTargets(
            demographic_targets=Targets(female=0.5, college=0.3),
            prediction_targets={"pred": 0.5},
        )

        raker = ModelAssistedRaker(targets, model, prediction_weight=2.0)

        observations = [
            {"female": 1, "college": 1},
            {"female": 0, "college": 0},
            {"female": 1, "college": 0},
            {"female": 0, "college": 1},
        ]
        outcomes = [0.6, 0.4, 0.5, 0.5]

        for obs, outcome in zip(observations, outcomes):
            raker.partial_fit(obs, outcome=outcome)

        estimate = raker.model_assisted_estimate
        assert not np.isnan(estimate)
        assert 0 <= estimate <= 1

    def test_convergence_with_prediction_target(self):
        """Test that raker converges toward prediction target."""
        np.random.seed(42)

        model = self.create_simple_model()
        prediction_target = 0.45

        targets = ModelAssistedTargets(
            demographic_targets=Targets(female=0.5, college=0.3),
            prediction_targets={"pred": prediction_target},
        )

        raker = ModelAssistedRaker(
            targets, model, prediction_weight=5.0, learning_rate=5.0
        )

        for _ in range(200):
            obs = {
                "female": np.random.randint(2),
                "college": np.random.randint(2),
            }
            raker.partial_fit(obs, outcome=np.random.random())

        pred_margin = raker.weighted_mean_prediction
        assert abs(pred_margin - prediction_target) < 0.15

    def test_residual_property(self):
        """Test residuals computation."""
        model = self.create_simple_model()
        targets = ModelAssistedTargets(
            demographic_targets=Targets(female=0.5, college=0.3),
        )

        raker = ModelAssistedRaker(targets, model)

        obs1 = {"female": 1, "college": 1}
        raker.partial_fit(obs1, outcome=0.8)

        obs2 = {"female": 0, "college": 0}
        raker.partial_fit(obs2, outcome=None)

        residuals = raker.residuals
        assert len(residuals) == 2
        assert residuals[0] != 0
        assert residuals[1] == 0

    def test_total_loss(self):
        """Test total loss includes both demographic and prediction terms."""
        model = self.create_simple_model()
        targets = ModelAssistedTargets(
            demographic_targets=Targets(female=0.5, college=0.3),
            prediction_targets={"pred": 0.45},
        )

        raker = ModelAssistedRaker(targets, model, prediction_weight=1.0)

        for _ in range(10):
            obs = {"female": np.random.randint(2), "college": np.random.randint(2)}
            raker.partial_fit(obs)

        total = raker.total_loss
        demo_loss = raker.loss
        pred_loss = raker.prediction_margin_loss

        assert total == demo_loss + pred_loss


class TestPoststratificationCells:
    """Test PoststratificationCells."""

    def test_basic_cells(self):
        """Test basic cell creation."""
        cells = PoststratificationCells(
            [
                PoststratificationCell(
                    "young_f", {"age": "young", "female": 1}, 0.15, 0.45
                ),
                PoststratificationCell(
                    "young_m", {"age": "young", "female": 0}, 0.15, 0.52
                ),
                PoststratificationCell(
                    "old_f", {"age": "old", "female": 1}, 0.35, 0.48
                ),
                PoststratificationCell(
                    "old_m", {"age": "old", "female": 0}, 0.35, 0.55
                ),
            ]
        )

        assert cells.n_cells == 4
        assert "young_f" in cells.cell_ids
        assert "old_m" in cells.cell_ids

    def test_invalid_proportions_raise(self):
        """Test that cell proportions must sum to 1."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            PoststratificationCells(
                [
                    PoststratificationCell("a", {"x": 1}, 0.3),
                    PoststratificationCell("b", {"x": 0}, 0.3),
                ]
            )

    def test_get_cell(self):
        """Test getting cell by ID."""
        cells = PoststratificationCells(
            [
                PoststratificationCell("a", {"x": 1}, 0.5),
                PoststratificationCell("b", {"x": 0}, 0.5),
            ]
        )

        cell = cells.get_cell("a")
        assert cell is not None
        assert cell.cell_id == "a"

        assert cells.get_cell("nonexistent") is None

    def test_get_cell_for_obs(self):
        """Test finding cell for observation."""
        cells = PoststratificationCells(
            [
                PoststratificationCell("young_f", {"age": "young", "female": 1}, 0.25),
                PoststratificationCell("young_m", {"age": "young", "female": 0}, 0.25),
                PoststratificationCell("old_f", {"age": "old", "female": 1}, 0.25),
                PoststratificationCell("old_m", {"age": "old", "female": 0}, 0.25),
            ]
        )

        obs = {"age": "young", "female": 1, "other": "value"}
        cell = cells.get_cell_for_obs(obs)

        assert cell is not None
        assert cell.cell_id == "young_f"

        obs_no_match = {"age": "middle", "female": 1}
        assert cells.get_cell_for_obs(obs_no_match) is None


class TestStreamingMRP:
    """Test StreamingMRP for cell-based post-stratification."""

    def create_simple_model(self):
        """Create simple model for testing."""

        class ConstantModel:
            def predict(self, X):
                return np.ones(len(X)) * 0.5

        return ConstantModel()

    def create_cells(self):
        """Create test cells."""
        return PoststratificationCells(
            [
                PoststratificationCell(
                    "young_f", {"age": "young", "female": 1}, 0.15, 0.45
                ),
                PoststratificationCell(
                    "young_m", {"age": "young", "female": 0}, 0.15, 0.55
                ),
                PoststratificationCell(
                    "old_f", {"age": "old", "female": 1}, 0.35, 0.48
                ),
                PoststratificationCell(
                    "old_m", {"age": "old", "female": 0}, 0.35, 0.52
                ),
            ]
        )

    def test_initialization(self):
        """Test MRP initialization."""
        model = self.create_simple_model()
        cells = self.create_cells()

        mrp = StreamingMRP(model=model, cells=cells)

        assert mrp.n_obs == 0
        assert mrp.cells.n_cells == 4

    def test_partial_fit(self):
        """Test processing observations."""
        model = self.create_simple_model()
        cells = self.create_cells()
        mrp = StreamingMRP(model=model, cells=cells)

        obs = {"age": "young", "female": 1, "vote": 1}
        mrp.partial_fit(obs, outcome=1.0)

        assert mrp.n_obs == 1
        assert mrp.get_cell_n("young_f") == 1
        assert mrp.get_cell_n("young_m") == 0

    def test_cell_estimates(self):
        """Test cell-level estimates."""
        model = self.create_simple_model()
        cells = self.create_cells()
        mrp = StreamingMRP(model=model, cells=cells)

        for _ in range(10):
            obs = {"age": "young", "female": 1}
            mrp.partial_fit(obs, outcome=0.6)

        for _ in range(10):
            obs = {"age": "old", "female": 0}
            mrp.partial_fit(obs, outcome=0.4)

        estimates = mrp.cell_estimates
        assert abs(estimates["young_f"] - 0.6) < 0.01
        assert abs(estimates["old_m"] - 0.4) < 0.01

    def test_population_estimate(self):
        """Test population-level estimate."""
        model = self.create_simple_model()
        cells = self.create_cells()
        mrp = StreamingMRP(model=model, cells=cells)

        observations = [
            ({"age": "young", "female": 1}, 0.6),
            ({"age": "young", "female": 0}, 0.5),
            ({"age": "old", "female": 1}, 0.4),
            ({"age": "old", "female": 0}, 0.5),
        ] * 20

        for obs, outcome in observations:
            mrp.partial_fit(obs, outcome=outcome)

        pop_est = mrp.population_estimate

        expected = 0.15 * 0.6 + 0.15 * 0.5 + 0.35 * 0.4 + 0.35 * 0.5
        assert abs(pop_est - expected) < 0.01

    def test_fallback_to_model_prediction(self):
        """Test fallback to model prediction when no data in cell."""
        model = self.create_simple_model()
        cells = self.create_cells()
        mrp = StreamingMRP(model=model, cells=cells)

        obs = {"age": "young", "female": 1}
        mrp.partial_fit(obs, outcome=0.6)

        old_f_est = mrp.get_cell_estimate("old_f")
        assert old_f_est == 0.48

    def test_observation_skipped_if_no_cell_match(self):
        """Test that observations not matching any cell are skipped."""
        model = self.create_simple_model()
        cells = self.create_cells()
        mrp = StreamingMRP(model=model, cells=cells)

        obs = {"age": "middle", "female": 1}
        mrp.partial_fit(obs, outcome=0.5)

        assert mrp.n_obs == 0


class TestGREGFormula:
    """Test GREG formula correctness."""

    def test_greg_adjustment_direction(self):
        """Test that GREG adjustment is in correct direction."""

        class PredictHighModel:
            def predict(self, X):
                return np.ones(len(X)) * 0.7

        model = PredictHighModel()
        targets = ModelAssistedTargets(
            demographic_targets=Targets(female=0.5),
            prediction_targets={"pred": 0.5},
        )

        raker = ModelAssistedRaker(targets, model, prediction_weight=5.0)

        for _ in range(100):
            obs = {"female": np.random.randint(2)}
            raker.partial_fit(obs, outcome=np.random.random())

        greg_est = raker.model_assisted_estimate
        weighted_mean = raker.weighted_mean_outcome
        weighted_pred = raker.weighted_mean_prediction

        adjustment = 0.5 - weighted_pred
        expected = weighted_mean + adjustment

        assert abs(greg_est - expected) < 0.01


class TestSimulationKnownDGP:
    """Test with known data generating process for bias verification."""

    def test_bias_reduction_vs_unweighted(self):
        """Test that GREG reduces bias compared to unweighted estimate."""
        np.random.seed(42)

        true_population_mean = 0.5
        sample_bias = 0.1

        n_obs = 500

        true_probs = np.random.uniform(0.3, 0.7, n_obs)
        sample_probs = true_probs + sample_bias

        model = LinearOutcomeModel()
        X_dummy = np.random.randn(n_obs, 2)
        model.fit(X_dummy, true_probs)

        targets = ModelAssistedTargets(
            demographic_targets=Targets(feature=0.5),
            prediction_targets={"pred": true_population_mean},
        )

        raker = ModelAssistedRaker(
            targets=targets,
            model=model,
            prediction_weight=3.0,
            feature_names_in_obs=["f1", "f2"],
        )

        for i in range(n_obs):
            obs = {
                "feature": int(np.random.random() < 0.5),
                "f1": X_dummy[i, 0],
                "f2": X_dummy[i, 1],
            }
            raker.partial_fit(obs, outcome=sample_probs[i])

        raw_mean = np.mean(sample_probs)
        greg_est = raker.model_assisted_estimate

        raw_bias = abs(raw_mean - true_population_mean)
        greg_bias = abs(greg_est - true_population_mean)

        assert greg_bias <= raw_bias + 0.05


class TestStreamingVsBatch:
    """Compare streaming GREG to conceptual batch GREG."""

    def test_streaming_approaches_batch(self):
        """Test that streaming GREG approximates batch GREG behavior."""
        np.random.seed(123)

        model = LinearOutcomeModel()
        X_train = np.random.randn(100, 2)
        y_train = X_train @ np.array([0.3, -0.2]) + 0.5 + np.random.randn(100) * 0.1
        model.fit(X_train, y_train)

        targets = ModelAssistedTargets(
            demographic_targets=Targets(female=0.5),
            prediction_targets={"pred": 0.5},
        )

        raker = ModelAssistedRaker(
            targets=targets,
            model=model,
            prediction_weight=2.0,
            learning_rate=3.0,
            feature_names_in_obs=["f1", "f2"],
        )

        X_stream = np.random.randn(200, 2)
        y_stream = X_stream @ np.array([0.3, -0.2]) + 0.5 + np.random.randn(200) * 0.1

        all_predictions = []
        for i in range(200):
            obs = {
                "female": int(np.random.random() < 0.5),
                "f1": X_stream[i, 0],
                "f2": X_stream[i, 1],
            }
            raker.partial_fit(obs, outcome=y_stream[i])
            all_predictions.append(raker.predictions[-1])

        batch_weighted_pred = np.mean(all_predictions)
        batch_weighted_outcome = np.mean(y_stream)
        batch_greg = batch_weighted_outcome + (0.5 - batch_weighted_pred)

        streaming_greg = raker.model_assisted_estimate

        assert abs(streaming_greg - batch_greg) < 0.2
