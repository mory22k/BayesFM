import numpy as np

def fm_naive(x, b, w, v):
    bias = b
    linear_term = x @ w
    interaction_term = 0
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            interaction_term += v[i] @ v[j] * x[i] * x[j]

    return bias + linear_term + interaction_term

def fm_fast(x, b, w, v, q=None):
    if q is None:
        q = x @ v
    bias = b
    linear_term = x @ w
    interaction_term = (q**2 - (x**2) @ (v**2)).sum() / 2

    return bias + linear_term + interaction_term

def fm_grad_b(X):
    N, _ = X.shape
    return np.ones(N)

def fm_grad_w(X, i):
    return X[:, i]

class FactorizationMachineRegressor:
    def __init__(self, dim_hidden, seed=None):
        self.rng = np.random.default_rng(seed)
        self.bias_ = None
        self.linear_coef_ = None
        self.hidden_vector_ = None
        self.dim_hidden_ = dim_hidden

    def initialize_params(self, X, y):
        N, d = X.shape
        self.bias_ = np.mean(y)
        self.linear_coef_ = np.zeros(d)
        self.hidden_vector_ = self.rng.standard_normal(size=(d, self.dim_hidden_))

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        return np.apply_along_axis(
            lambda x: fm_fast(x, self.bias_, self.linear_coef_, self.hidden_vector_), axis=1, arr=X
        )

    def error(self, X, y):
        return np.mean((self.predict(X) - y)**2)

    def get_params(self):
        return self.bias_, self.linear_coef_, self.hidden_vector_

    def set_params(self, bias, linear_coef, hidden_vector):
        self.bias_ = bias
        self.linear_coef_ = linear_coef
        self.hidden_vector_ = hidden_vector
