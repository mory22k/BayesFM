import logging
import numpy as np
from bayesfm.logging import set_verbosity
from bayesfm import BayesFMRegression

set_verbosity(logging.DEBUG)


def test__sample():
    d = 16
    k = 15
    n = 500
    X = np.random.choice([0, 1], (n, d))
    Q = np.random.standard_normal((d, d))
    y_true = np.einsum("ij, di, dj -> d", Q, X, X)

    n_warmup = 1000
    n_mcs = 100

    bfm = BayesFMRegression(
        n_factors=k,
        n_mcs=n_mcs,
        n_warmup=n_warmup,
        seed=42,
        show_progress_bar=True,
    )

    bfm.fit(X, y_true)


if __name__ == "__main__":
    test__sample()
