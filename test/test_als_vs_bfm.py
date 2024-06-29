import numpy as np
import logging
from fm_gibbs.FM import fm_fast
from fm_gibbs.Train import FMGibbs, FMALS
import matplotlib.pyplot as plt
from pathlib import Path

def test_bfm():
    n_iter = 2000

    N = 1000
    d = 16
    k = 5

    X = np.random.choice([0, 1], (N, d))
    b_true = np.random.standard_normal()
    w_true = np.random.standard_normal(d)
    v_true = np.random.standard_normal((d, k))
    y = np.apply_along_axis(
        lambda x: fm_fast(x, b_true, w_true, v_true), axis=1, arr=X
    )

    b_init = 0.0
    w_init = np.random.standard_normal(d)
    v_init = np.random.standard_normal((d, k))

    l_b = 1e-3
    l_w = 1e-3
    l_v = 1e-3

    logger = logging.getLogger('test')
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    fig = plt.figure()
    ax = fig.add_subplot(111)



    num_data = 100

    b, w, v, error_hist = FMALS.alternate_least_squares(
        X[:num_data], y[:num_data], b_init, w_init, v_init, l_b, l_w, l_v, n_iter=n_iter, logger=logger, return_errors=True
    )
    ax.plot(error_hist, label=rf'N=${num_data}$, ALS')

    b, w, v, error_hist = FMGibbs.gibbs_sampling(
        X[:num_data], y[:num_data], b_init, w_init, v_init, n_iter=n_iter, logger=logger, return_errors=True
    )
    ax.plot(error_hist, label=rf'N=${num_data}$, Gibbs sampler')



    num_data = 1000

    b, w, v, error_hist = FMALS.alternate_least_squares(
        X[:num_data], y[:num_data], b_init, w_init, v_init, l_b, l_w, l_v, n_iter=n_iter, logger=logger, return_errors=True
    )
    ax.plot(error_hist, label=rf'N=${num_data}$, ALS')

    b, w, v, error_hist = FMGibbs.gibbs_sampling(
        X[:num_data], y[:num_data], b_init, w_init, v_init, n_iter=n_iter, logger=logger, return_errors=True
    )
    ax.plot(error_hist, label=rf'N=${num_data}$, Gibbs sampler')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Error')
    ax.set_yscale('log')
    ax.legend()
    (Path(__file__).parent / 'out').mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(__file__).parent / 'out' / 'bfm_vs_als.png')

if __name__ == '__main__':
    test_bfm()
