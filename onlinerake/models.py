"""Outcome models for model-assisted streaming calibration.

This module provides protocols and implementations for outcome models
used in GREG-style and MRP model-assisted calibration. Models provide
predictions that serve as auxiliary variables for efficiency gains.

The key insight is that model fitting is batch (done once on initial data),
while calibration weight updates are streaming. This separation allows
standard GREG/MRP workflows with online weight adjustment.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


@runtime_checkable
class OutcomeModel(Protocol):
    """Protocol for outcome models used in model-assisted calibration.

    Any model with a predict() method satisfies this protocol, including
    scikit-learn estimators, statsmodels fitted models, or custom implementations.

    Examples:
        The protocol is ``runtime_checkable``, so ``predict`` is the whole
        requirement -- no inheritance and no dependency on any modeling library:

        >>> class ConstantModel:
        ...     def predict(self, X):
        ...         return np.zeros(len(np.atleast_2d(X)))
        >>> isinstance(ConstantModel(), OutcomeModel)
        True

        And a class without it is rejected, which is what makes the check above
        mean something:

        >>> class NotAModel:
        ...     def score(self, X):
        ...         return 0.0
        >>> isinstance(NotAModel(), OutcomeModel)
        False
    """

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.floating[Any]]:
        """Generate predictions for input features.

        Args:
            X: Feature matrix of shape (n_samples, n_features) or
               (n_features,) for single sample.

        Returns:
            Predictions of shape (n_samples,) or scalar for single sample.
        """
        ...


class LinearOutcomeModel:
    """Simple linear regression model for outcome prediction.

    Fits y = X @ beta using ordinary least squares.
    Useful when you want a lightweight model without sklearn dependency.

    Args:
        fit_intercept: Whether to fit an intercept term. Default True.

    Attributes:
        coef_: Fitted coefficients of shape (n_features,).
        intercept_: Fitted intercept (0.0 if fit_intercept=False).
        is_fitted: Whether the model has been fitted.

    Examples:
        On a noiseless design OLS recovers the generating coefficients exactly,
        which is the cheapest way to see the fit is doing what it claims. Here
        ``y = 1 + 2*x1 + 4*x2``:

        >>> X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        >>> y = np.array([1.0, 3.0, 5.0, 7.0])
        >>> model = LinearOutcomeModel().fit(X, y)
        >>> model.coef_.round(6)
        array([2., 4.])
        >>> round(model.intercept_, 6)
        1.0
        >>> model.predict(np.array([[1.0, 1.0]]))
        array([7.])
    """

    def __init__(self, fit_intercept: bool = True) -> None:
        self.fit_intercept = fit_intercept
        self.coef_: npt.NDArray[np.float64] | None = None
        self.intercept_: float = 0.0
        self.is_fitted: bool = False

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> LinearOutcomeModel:
        """Fit the linear model using ordinary least squares.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target values of shape (n_samples,).

        Returns:
            Self for method chaining.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        if self.fit_intercept:
            X_arr = np.column_stack([np.ones(len(X_arr)), X_arr])

        # OLS: beta = (X'X)^{-1} X'y
        coef, _, _, _ = np.linalg.lstsq(X_arr, y_arr, rcond=None)

        if self.fit_intercept:
            self.intercept_ = float(coef[0])
            self.coef_ = np.asarray(coef[1:], dtype=np.float64)
        else:
            self.intercept_ = 0.0
            self.coef_ = np.asarray(coef, dtype=np.float64)

        self.is_fitted = True
        return self

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Generate predictions.

        Args:
            X: Feature matrix of shape (n_samples, n_features) or
               (n_features,) for single sample.

        Returns:
            Predictions of shape (n_samples,).

        Raises:
            ValueError: If model has not been fitted.
        """
        if not self.is_fitted or self.coef_ is None:
            raise ValueError("Model must be fitted before calling predict()")

        X_arr = np.asarray(X, dtype=np.float64)
        single_sample = X_arr.ndim == 1
        if single_sample:
            X_arr = X_arr.reshape(1, -1)

        return X_arr @ self.coef_ + self.intercept_


class LogisticOutcomeModel:
    """Simple logistic regression model for binary outcome prediction.

    Fits logistic regression using gradient descent.
    Useful when you want a lightweight model without sklearn dependency.

    Args:
        fit_intercept: Whether to fit an intercept term. Default True.
        learning_rate: Step size for gradient descent. Default 0.1.
        max_iter: Maximum iterations. Default 1000.
        tol: Convergence tolerance. Default 1e-6.

    Attributes:
        coef_: Fitted coefficients of shape (n_features,).
        intercept_: Fitted intercept (0.0 if fit_intercept=False).
        is_fitted: Whether the model has been fitted.

    Examples:
        ``predict`` returns probabilities, not labels -- that is what
        model-assisted calibration needs. Here the second column separates the
        classes perfectly, so the fitted probabilities order with it:

        >>> X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        >>> y = np.array([0, 0, 1, 1])
        >>> model = LogisticOutcomeModel().fit(X, y)
        >>> model.predict(X).round(3)
        array([0.075, 0.05 , 0.967, 0.95 ])
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        learning_rate: float = 0.1,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> None:
        self.fit_intercept = fit_intercept
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.coef_: npt.NDArray[np.float64] | None = None
        self.intercept_: float = 0.0
        self.is_fitted: bool = False

    def _sigmoid(self, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Numerically stable sigmoid function."""
        return np.where(
            z >= 0,
            1 / (1 + np.exp(-z)),
            np.exp(z) / (1 + np.exp(z)),
        )

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> LogisticOutcomeModel:
        """Fit logistic regression using gradient descent.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Binary target values of shape (n_samples,).

        Returns:
            Self for method chaining.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        n_samples, n_features = X_arr.shape

        if self.fit_intercept:
            X_arr = np.column_stack([np.ones(n_samples), X_arr])
            n_features += 1

        # Initialize weights
        weights = np.zeros(n_features, dtype=np.float64)

        # Gradient descent
        for _ in range(self.max_iter):
            z = X_arr @ weights
            p = self._sigmoid(z)

            # Gradient: X'(p - y) / n
            gradient = X_arr.T @ (p - y_arr) / n_samples

            # Update
            weights_new = weights - self.learning_rate * gradient

            # Check convergence
            if np.linalg.norm(weights_new - weights) < self.tol:
                weights = weights_new
                break

            weights = weights_new

        if self.fit_intercept:
            self.intercept_ = float(weights[0])
            self.coef_ = weights[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = weights

        self.is_fitted = True
        return self

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Generate probability predictions.

        Args:
            X: Feature matrix of shape (n_samples, n_features) or
               (n_features,) for single sample.

        Returns:
            Predicted probabilities of shape (n_samples,).

        Raises:
            ValueError: If model has not been fitted.
        """
        if not self.is_fitted or self.coef_ is None:
            raise ValueError("Model must be fitted before calling predict()")

        X_arr = np.asarray(X, dtype=np.float64)
        single_sample = X_arr.ndim == 1
        if single_sample:
            X_arr = X_arr.reshape(1, -1)

        z = X_arr @ self.coef_ + self.intercept_
        return self._sigmoid(z)


class ExternalModelWrapper:
    """Wrapper for external models (sklearn, statsmodels, etc.).

    Provides a uniform interface for models that may have different
    prediction methods (predict, predict_proba, etc.).

    Args:
        model: Fitted model with predict() or predict_proba() method.
        use_proba: If True and model has predict_proba(), use it to get
            probability of class 1. Default False.
        proba_class: Class index for probability extraction. Default 1.

    Attributes:
        model: The wrapped model.

    Examples:
        Any object with the sklearn method names will do -- here a stand-in
        classifier, so the example needs no modeling library to run:

        >>> class Classifier:
        ...     def predict(self, X):
        ...         return np.array([0, 1])
        ...     def predict_proba(self, X):
        ...         return np.array([[0.7, 0.3], [0.1, 0.9]])
        >>> X = np.array([[0.0], [1.0]])

        ``use_proba`` is the whole point of the wrapper: hard labels are the
        wrong input to a model-assisted estimate, and this is what switches
        them for the probability of class 1.

        >>> ExternalModelWrapper(Classifier()).predict(X)
        array([0., 1.])
        >>> ExternalModelWrapper(Classifier(), use_proba=True).predict(X)
        array([0.3, 0.9])

        Asking for probabilities from a model that cannot give them is refused
        at construction rather than at predict time:

        >>> class LabelsOnly:
        ...     def predict(self, X):
        ...         return np.zeros(len(X))
        >>> ExternalModelWrapper(LabelsOnly(), use_proba=True)
        Traceback (most recent call last):
            ...
        ValueError: use_proba=True requires model with predict_proba() method
    """

    def __init__(
        self,
        model: Any,
        use_proba: bool = False,
        proba_class: int = 1,
    ) -> None:
        self.model = model
        self.use_proba = use_proba
        self.proba_class = proba_class

        # Validate model has predict method
        if not hasattr(model, "predict"):
            raise ValueError("Model must have a predict() method")

        if use_proba and not hasattr(model, "predict_proba"):
            raise ValueError(
                "use_proba=True requires model with predict_proba() method"
            )

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Generate predictions using wrapped model.

        Args:
            X: Feature matrix of shape (n_samples, n_features) or
               (n_features,) for single sample.

        Returns:
            Predictions of shape (n_samples,).
        """
        X_arr = np.asarray(X, dtype=np.float64)
        single_sample = X_arr.ndim == 1
        if single_sample:
            X_arr = X_arr.reshape(1, -1)

        if self.use_proba:
            proba = self.model.predict_proba(X_arr)
            predictions = proba[:, self.proba_class]
        else:
            predictions = self.model.predict(X_arr)

        return np.asarray(predictions, dtype=np.float64)
