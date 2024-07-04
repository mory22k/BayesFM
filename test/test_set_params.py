import numpy as np
import logging
from fmgibbs.train import als, gibbs, grad

def test_set_params():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    seed = 72
    rng = np.random.default_rng(seed)
    N = 500
    d = 16
    X = rng.choice([0, 1], (N, d))
    y = rng.standard_normal(N)

    fm_grad = grad.FactorizationMachineGradRegressor(seed=seed, max_iter=5)
    fm_als = als.FactorizationMachineALSRegressor(seed=seed, max_iter=5)
    fm_gibbs = gibbs.FactorizationMachineGibbsSampler(seed=seed, max_iter=5, max_pretrain_iter=0)

    fm_grad.fit(X, y, logger=logger, record_error=True)
    fm_als.fit(X, y, logger=logger, record_error=True)
    fm_gibbs.fit(X, y, logger=logger, record_error=True)

    print("AFTER FIT I")
    print("|", fm_grad.get_params()[1][-1])
    print("|", fm_als.get_params()[1][-1])
    print("|", fm_gibbs.get_params()[1][-1])

    b, w, v = fm_grad.get_params()
    print(b, w.shape, v.shape)

    fm_grad.set_params(b, w, v)
    fm_als.set_params(b, w, v)
    fm_gibbs.set_params(b, w, v)

    print("BEFORE FIT II")
    print("|", fm_grad.get_params()[1][-1])
    print("|", fm_als.get_params()[1][-1])
    print("|", fm_gibbs.get_params()[1][-1])

    fm_grad.fit(X, y, logger=logger, record_error=True)
    fm_als.fit(X, y, logger=logger, record_error=True)
    fm_gibbs.fit(X, y, logger=logger, record_error=True)

    print("AFTER FIT II")
    print("|", fm_grad.get_params()[1][-1])
    print("|", fm_als.get_params()[1][-1])
    print("|", fm_gibbs.get_params()[1][-1])

if __name__ == '__main__':
    test_set_params()
