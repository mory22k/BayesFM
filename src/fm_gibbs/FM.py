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
