import numpy as np
import torch.nn as nn
import torch.optim as optim
import logging
from fmgibbs.train.als import FactorizationMachineALSRegressor
from fmgibbs.train.gibbs import FactorizationMachineGibbsSampler
from fmgibbs.train.grad import FactorizationMachineGradRegressor
import matplotlib.pyplot as plt
from pathlib import Path

def test_train():
    seed = 72
    n_iter = 5000

    N = 1000
    d = 16
    k = 5

    rng = np.random.default_rng(seed)

    X = rng.choice([0, 1], (N, d))
    Q = rng.standard_normal((d, d))
    y_true = np.apply_along_axis(
        lambda x: x @ Q @ x, axis=1, arr=X
    )

    fm_grad = FactorizationMachineGradRegressor(
        dim_hidden=k, max_iter=n_iter, optimizer_class=optim.AdamW, criterion=nn.MSELoss(), warm_start=False, seed=seed
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

    num_data = 10
    fm_grad.fit(X[:num_data], y_true[:num_data], logger=logger, record_error=True)
    fm_als.fit(X[:num_data], y_true[:num_data], logger=logger, record_error=True)
    fm_gibbs.fit(X[:num_data], y_true[:num_data], logger=logger, record_error=True)
    history_grad_10 = fm_grad.error_history_
    history_als_10 = fm_als.error_history_
    history_gibbs_10 = fm_gibbs.error_history_

    num_data = 1000
    fm_grad.fit(X[:num_data], y_true[:num_data], logger=logger, record_error=True)
    fm_als.fit(X[:num_data], y_true[:num_data], logger=logger, record_error=True)
    fm_gibbs.fit(X[:num_data], y_true[:num_data], logger=logger, record_error=True)
    history_grad_1000 = fm_grad.error_history_
    history_als_1000 = fm_als.error_history_
    history_gibbs_1000 = fm_gibbs.error_history_

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)

    ax.plot(history_grad_10, label=r'Grad, N=$10$', linestyle=':')
    ax.plot(history_grad_1000, label=r'Grad, N=$1000$', linestyle=':')
    ax.plot(history_als_10, label=r'ALS, N=$10$', linestyle='--')
    ax.plot(history_als_1000, label=r'ALS, N=$1000$', linestyle='--')
    ax.plot(history_gibbs_10, label=r'Gibbs sampler, N=$10$', linestyle='-')
    ax.plot(history_gibbs_1000, label=r'Gibbs sampler, N=$1000$', linestyle='-')

    ax.set_xlabel(r'Iteration $t$')
    ax.set_ylabel(r'RMSE $\sqrt{\frac{1}{N}\sum_{i=1}^N (y^{(i)} - \hat f(x^{(i)}))^2}$')
    ax.set_yscale('log')
    ax.legend()
    plt.tight_layout()
    (Path(__file__).parent / 'out').mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(__file__).parent / 'out' / 'train.png')

if __name__ == '__main__':
    test_train()
