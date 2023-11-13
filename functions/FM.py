import numpy as np
from typing import Tuple
from copy import copy
from julia import Main # type: ignore

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

def train_als(
    X_data:np.ndarray,
    Y_data:np.ndarray,
    Y_pred:np.ndarray,
    b_init:float,
    w_init:np.ndarray,
    V_init:np.ndarray,
    max_iter:int = 100,
    lamb_b:float = 1.,
    lamb_w:float = 1.,
    lamb_v:float = 1.,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    b = copy(b_init)
    w = copy(w_init)
    V = copy(V_init)

    N = w.shape[0]
    K = V.shape[1]

    # precompute error and quadratic
    error = Y_data - Y_pred # (D,)
    quad = X_data @ V       # (D, K)
    error_hist = np.empty(max_iter + 1, dtype=float)
    error_hist[0] = np.mean(error**2)

    for iter in range(max_iter):
        # update b: 1 * O(D) = O(D)
        b_new = np.sum(b - error) / (lamb_b + X_data.shape[1])
        error = error + b_new - b
        b = b_new

        # update w: N * O(D) = O(D * N)
        for i in range(N):
            w_i_new = np.sum(X_data[:,i] * (w[i] * X_data[:,i] - error)) / (lamb_w + np.sum(X_data[:,i] ** 2))
            error = error + (w_i_new - w[i]) * X_data[:,i]
            w[i] = w_i_new

        # update V: N * K * O(D) = O(D * N * K)
        for i in range(N):
            for k in range(K):
                G_ik = X_data[:,i] * (quad[:,k] - V[i,k] * X_data[:,i])
                V_ik_new = np.sum(G_ik * (V[i,k] * G_ik - error)) / (lamb_v + np.sum(G_ik ** 2))
                error = error + (V_ik_new - V[i,k]) * G_ik
                quad[:,k] = quad[:,k] + (V_ik_new - V[i,k]) * X_data[:,i]

        error_hist[iter+1] = np.mean(error**2)
        print(f"iter: {iter+1}, error: {np.mean(error**2)}")
    return b, w, V, error_hist

def train_als_julia(
    X_data:np.ndarray,
    Y_data:np.ndarray,
    Y_pred:np.ndarray,
    b_init:float,
    w_init:np.ndarray,
    V_init:np.ndarray,
    max_iter:int = 100,
    lamb_b:float = 1.,
    lamb_w:float = 1.,
    lamb_v:float = 1.,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    b = copy(b_init)
    w = copy(w_init)
    V = copy(V_init)

    Main.include("functions/trainer.jl")
    X_data_float = X_data.astype(np.float64)
    b, w, V, error_hist = Main.train_als(X_data_float, Y_data, Y_pred, b, w, V, max_iter, lamb_b, lamb_w, lamb_v)
    return b, w, V, error_hist
