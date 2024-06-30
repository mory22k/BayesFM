import numpy as np
import logging
from fm_gibbs.Train.FMALS import FactorizationMachineALSRegressor
from fm_gibbs.Train.FMGibbs import FactorizationMachineGibbsSampler
import matplotlib.pyplot as plt
from pathlib import Path

def test_bfm():
    seed = 72
    n_iter = 1000

    N = 500
    d = 16
    k = 5

    rng = np.random.default_rng(seed)

    X = rng.choice([0, 1], (N, d))
    Q = rng.standard_normal((d, d))
    y_true = np.apply_along_axis(
        lambda x: x @ Q @ x, axis=1, arr=X
    )

    fm_als = FactorizationMachineALSRegressor(
        dim_hidden=k, alpha_b=1e-3, alpha_w=1e-3, alpha_v=1e-3, max_iter=n_iter, warm_start=False, seed=seed
    )
    fm_gibbs = FactorizationMachineGibbsSampler(
        dim_hidden=k, max_iter=n_iter, warm_start=False, seed=seed
    )

    logger = logging.getLogger('test')
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)

    num_data = 100

    fm_als.fit(X[:num_data], y_true[:num_data], logger=logger, record_error=True)
    fm_gibbs.fit(X[:num_data], y_true[:num_data], logger=logger, record_error=True)

    ax.plot(fm_als.error_history_, label=rf'N=${num_data}$, ALS')
    ax.plot(fm_gibbs.error_history_, label=rf'N=${num_data}$, Gibbs sampler')



    num_data = 1000

    fm_als.fit(X[:num_data], y_true[:num_data], logger=logger, record_error=True)
    fm_gibbs.fit(X[:num_data], y_true[:num_data], logger=logger, record_error=True)

    ax.plot(fm_als.error_history_, label=rf'N=${num_data}$, ALS')
    ax.plot(fm_gibbs.error_history_, label=rf'N=${num_data}$, Gibbs sampler')

    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'RMSE $\sqrt{\frac{1}{N}\sum_{i=1}^N (y^{(i)} - \hat f(x^{(i)}))^2}$')
    ax.set_yscale('log')
    ax.legend()
    plt.tight_layout()
    (Path(__file__).parent / 'out').mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(__file__).parent / 'out' / 'bfm_vs_als.png')

if __name__ == '__main__':
    test_bfm()
