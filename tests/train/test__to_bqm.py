from bayesfm import BayesFMRegression
import numpy as np


def test__to_bqm():
    k = 15
    d = 16
    n = 500

    X = np.random.choice([0, 1], (n, d))
    Q = np.random.standard_normal((d, d))
    y_true = np.einsum("ij, di, dj -> d", Q, X, X)

    bayes_fm_regression = BayesFMRegression(n_factors=k, n_mcs=100, n_warmup=10, seed=42, show_progress_bar=True)
    bayes_fm_regression.fit(X, y_true)

    bqm = bayes_fm_regression.to_bqm()

    linear_biases, quadratic_biases, offset = bqm.to_numpy_vectors()

    qubo_matrix = np.zeros((d, d))
    qubo_matrix[np.diag_indices(d)] = linear_biases

    row_idx, col_idx, biases = quadratic_biases
    qubo_matrix[row_idx, col_idx] = biases

    qubo_matrix = qubo_matrix.T

    x = np.random.choice([0, 1], d)
    y1 = bayes_fm_regression.predict(x)
    y2 = x @ qubo_matrix @ x + offset
    assert np.allclose(y1, y2)
    print(y1, y2)
