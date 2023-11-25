import numpy as np
from typing import Tuple
from copy import copy

class FactorizationMachines:
    """
    Factorization Machines model for regression and classification tasks.

    Args:
        n_features (int): Number of features in the input data.
        n_factors (int): Number of factors to use in the pairwise interaction term.
        init_sigma (float, optional): Standard deviation of the Gaussian initialization for the factor matrix.
            Defaults to 1.0.
        seed (int, optional): Seed to ensure reproducibility of the model. Defaults to None.

    Attributes:
        rng (numpy.random.Generator):
            Random number generator used for initialization.
        b (float):
            Bias term.
        w (np.ndarray[n_features,]):
            Linear weights for each feature.
        V (np.ndarray[n_features, n_factors]):
            Factor matrix for pairwise interactions.
    """

    def __init__(self, n_features, n_factors, init_sigma=1., seed=None):
        self.rng = np.random.default_rng(seed)
        self.b = 0.
        self.w = np.zeros(n_features)
        self.V = self.rng.normal(0, init_sigma**2, size=(n_features, n_factors))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the target variable for the input data.

        Args:
            X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            y_pred (np.ndarray): Predicted target variable of shape (n_samples,).
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        bias = self.b
        linear = X @ self.w
        interaction = 0.5 * np.sum(
            (X @ self.V) ** 2 - (X ** 2) @ (self.V ** 2),
            axis=1,
        )
        return (bias + linear + interaction).flatten()
