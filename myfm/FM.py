import numpy as np
from typing import Tuple
from copy import copy

class Sampler:
    def sample(self, model, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError

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

    def __init__(self, n_features, n_factors, sampler: Sampler, init_sigma=1., seed=None):
        self.rng = np.random.default_rng(seed)
        self.params = {}
        self.params['b'] = self.rng.normal(0, init_sigma**2)
        self.params['w'] = self.rng.normal(0, init_sigma**2, size=n_features)
        self.params['V'] = self.rng.normal(0, init_sigma**2, size=(n_features, n_factors))
        self.sampler = sampler

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

        b = self.params['b']
        w = self.params['w']
        V = self.params['V']

        bias = b
        linear = X @ w
        interaction = 0.5 * np.sum(
            (X @ V) ** 2 - (X ** 2) @ (V ** 2),
            axis=1,
        )
        return (bias + linear + interaction).flatten()

    def initialize(self):
        self.params['b'] = 0.
        self.params['w'] = np.zeros_like(self.params['w'])
        self.params['V'] = self.rng.normal(0, 1., size=self.params['V'].shape)

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the model on the given data.
        """

        b, w, V, MSE_hist = self.sampler.sample(self, X, y)

        self.params['b'] = b
        self.params['w'] = w
        self.params['V'] = V
        return MSE_hist
