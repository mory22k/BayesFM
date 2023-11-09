import numpy as np

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

def als(
    model: FactorizationMachines, X_data: np.ndarray, Y_data: np.ndarray,
    max_iter: int = 100, use_true_error: bool = False,
    lamb_b = 0., lamb_w = 0., lamb_v = 0.
) -> np.ndarray:
    """
    Fast Alternating Least Squares (ALS) algorithm for training a Factorization Machines model.

    Args:
        model (FactorizationMachines): The Factorization Machines model to be trained.
        X_data (np.ndarray): The input data matrix of shape (D, N), where D is the number of features and N is the number of samples.
        Y_data (np.ndarray): The output data matrix of shape (N,).
        max_iter (int): The maximum number of iterations to run the ALS algorithm. Default is 100.
        use_true_error (bool): Whether to calculate the true error (Y_data - Y_pred) in each iteration. Default is False.
        lamb_b (float): The regularization parameter for the bias term b. Default is 0.
        lamb_w (float): The regularization parameter for the weight vector w. Default is 0.
        lamb_v (float): The regularization parameter for the factor matrix V. Default is 0.

    Returns:
        np.ndarray: The error history of the ALS algorithm, of shape (max_iter + 1,).

    References:
        S. Rendle, Z. Gantner, C. Freudenthaler, and L. Schmidt-Thieme.
        Fast Context-Aware Recommendations with Factorization Machines,
        in Proceedings of the 34th International ACM SIGIR Conference on Research and Development in Information Retrieval (2011), pp. 635–644.
    """
    # extract dimensions
    D = X_data.shape[0]
    N = model.w.shape[0]
    K = model.V.shape[1]

    # precompute: O(DN) + O(K) = O(DN)
    X_squared = X_data ** 2
    error = Y_data - model.predict(X_data)
    V_X = np.empty((D, K))
    for k in range(K):
        V_X[:, k] = np.sum(model.V[:, k] * X_data, axis=1)

    # error history
    error_hist = np.empty(max_iter + 1, dtype=float)
    if use_true_error:
        error_true = Y_data - model.predict(X_data)
        error_hist[0] = (np.mean(error_true**2))
    else:
        error_hist[0] = (np.mean(error**2))
    for iter in range(max_iter):

        # update b: 1 * O(D) = O(D)
        delta_param = np.sum(error) / (D + lamb_b)
        delta_error = -np.sum(delta_param)
        model.b += delta_param
        error += delta_error

        # update w: N * O(D) = O(N * D)
        for n in range(N):
            G = X_data[:, n]
            delta_param = G.T @ error / (G.T @ G + lamb_v)
            delta_error = -delta_param * G
            model.w[n] += delta_param
            error += delta_error

        # update V: N * K * O(D) = O(N * D * K)
        for n in range(N):
            for k in range(K):
                # G = X_data[:, n] * np.sum(model.V[:, k] * X_data, axis=1) - model.V[n, k] * X_data[:, n] ** 2
                G = X_data[:, n] * V_X[:, k] - model.V[n, k] * X_squared[:, n]
                delta_param = G.T @ error / (G.T @ G + lamb_v)
                delta_V_X_k = delta_param * X_data[:, n]
                delta_error = -delta_param * G
                model.V[n, k] += delta_param
                V_X[:, k] += delta_V_X_k
                error += delta_error

        if use_true_error:
            error = Y_data - model.predict(X_data)
            error_hist[iter+1] = (np.mean(error**2))
        else:
            error_hist[iter+1] = (np.mean(error**2))

        print(f"iter: {iter}, error: {error_hist[iter]}")

    return error_hist
