import numpy as np
import logging
from fmgibbs.train import als

def test_als():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    seed = 72
    rng = np.random.default_rng(seed)
    N = 500
    d = 16
    X = rng.choice([0, 1], (N, d))
    y = rng.standard_normal(N)

    fm_als = als.FactorizationMachineALSRegressor(seed=seed, max_iter=5)

    fm_als.fit(X, y, logger=logger, record_error=True)

if __name__ == '__main__':
    test_als()
