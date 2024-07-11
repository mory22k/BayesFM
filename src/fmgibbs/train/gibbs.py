import numpy as np
import logging
from fmgibbs.fm import fm_fast, FactorizationMachineRegressor

logger_local = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] - %(message)s')
console_handler.setFormatter(formatter)
logger_local.addHandler(console_handler)

def gibbs_sampling(
    X, y, b_init, w_init, v_init,
    mu_w, mu_v, sigma2_w, sigma2_v,
    mu_b, sigma2_b, alpha, beta, alpha_0, beta_0, m_0, tau2_0,
    n_iter=100, logger=None, record_error=False, check_error=False, rng=None
):
    if rng is None:
        rng = np.random.default_rng()

    b = np.copy(b_init)
    w = w_init.copy()
    v = v_init.copy()

    N, d = X.shape
    k = v.shape[1]

    f = np.apply_along_axis(
        lambda x: fm_fast(x, b, w, v), axis=1, arr=X
    )

    q = X @ v # (N, k)
    e = f - y # (N,)

    error_history = []
    if record_error:
        error_history.append(np.mean(e**2))

    # set static hyperparams
    alpha_star = alpha + N / 2
    alpha_theta_star = alpha_0 + (d + 1) / 2
    tau2_theta_star = 1 / (1 / tau2_0 + d)

    for mcs in range(n_iter):
        # update sigma2
        beta_star = beta + e @ e / 2
        if beta_star <= 0:
            beta_star = 1e-6
            logger_local.warning(f'beta_star <= 0 at mcs={mcs}')
        sigma2 = 1 / rng.gamma(alpha_star, 1 / beta_star)

        # update sigma2_w, mu_w
        beta_w_star = beta_0 + ( ((w[:] - mu_w)**2).sum() - (mu_w - m_0)**2 ) / 2
        m_w_star = tau2_theta_star * (w[:].sum() + m_0 / tau2_0)
        if beta_w_star <= 0:
            logger_local.warning(f'beta_w_star <= 0 at mcs={mcs}')
            beta_w_star = 1e-6
        sigma2_w = 1 / rng.gamma(alpha_theta_star, 1 / beta_w_star)
        mu_w = rng.normal(m_w_star, np.sqrt(tau2_theta_star * sigma2_w))

        # update sigma2_v, mu_v
        for f in range(k):
            beta_v_f_star = beta_0 + ( ((v[:,f] - mu_v[f])**2).sum() - (mu_v[f] - m_0)**2 ) / 2
            m_v_f_star = tau2_theta_star * (v[:,f].sum() + m_0 / tau2_0)
            sigma2_v[f] = 1 / rng.gamma(alpha_theta_star, 1 / beta_v_f_star)
            mu_v[f] = rng.normal(m_v_f_star, np.sqrt(tau2_theta_star * sigma2_v[f]))

        # update b
        h = np.ones(N)
        nu = b - e
        sigma2_b_star = 1 / (1 / sigma2 * (h @ h) + 1 / sigma2_b)
        mu2_b_star = sigma2_b_star * ((h @ nu) / sigma2 + mu_b / sigma2_b)
        b_new = rng.normal(mu2_b_star, np.sqrt(sigma2_b_star))
        e = e + b_new - b
        b = b_new

        for i in range(d):
            h = X[:,i]
            nu = w[i] * h - e
            sigma2_w_i_star = 1 / (1 / sigma2 * (h @ h) + 1 / sigma2_w)
            mu2_w_i_star = sigma2_w_i_star * ((h @ nu) / sigma2 + mu_w / sigma2_w)
            w_i_new = rng.normal(mu2_w_i_star, np.sqrt(sigma2_w_i_star))
            e = e + (w_i_new - w[i]) * h
            w[i] = w_i_new

            for f in range(k):
                h = (q[:,f] - v[i,f] * X[:,i]) * X[:,i]
                nu = v[i,f] * h - e
                sigma2_v_if_star = 1 / (1 / sigma2 * (h @ h) + 1 / sigma2_v[f])
                mu2_v_if_star = sigma2_v_if_star * ((h @ nu) / sigma2 + mu_v[f] / sigma2_v[f])
                v_if_new = rng.normal(mu2_v_if_star, np.sqrt(sigma2_v_if_star))
                q[:,f] = q[:,f] + (v_if_new - v[i,f]) * X[:,i]
                e = e + (v_if_new - v[i,f]) * h
                v[i,f] = v_if_new

        if check_error:
            f_pred = np.apply_along_axis(
                lambda x: fm_fast(x, b, w, v), axis=1, arr=X
            )
            assert np.isclose(e, f_pred - y).all()

        if logger is not None or record_error:
            error = np.sqrt(np.mean(e**2))
        if logger is not None:
            logger.info(f'{mcs:4d} | {error:10.3e}')
        if record_error:
            error_history.append(error)

    return b, w, v, mu_w, mu_v, sigma2_w, sigma2_v, error_history

class FactorizationMachineGibbsSampler(FactorizationMachineRegressor):
    def __init__(
        self,
        dim_hidden=5,
        max_iter=100,
        warm_start=True,
        seed=None,
        mu_b=0.0,
        sigma2_b=1.0,
        alpha=1.0,
        beta=1.0,
        alpha_0=1.0,
        beta_0=1.0,
        m_0=0.0,
        tau2_0=1.0,
    ):
        super().__init__(dim_hidden, seed)

        self.n_iter_ = max_iter
        self.warm_start = warm_start
        self.error_history_ = None
        self.rng = np.random.default_rng(seed)

        # fixed params
        self.mu_b_     = mu_b
        self.sigma2_b_ = sigma2_b
        self.alpha_    = alpha
        self.beta_     = beta
        self.alpha_0_  = alpha_0
        self.beta_0_   = beta_0
        self.m_0_      = m_0
        self.tau2_0_   = tau2_0

        # hidden params
        self.mu_w_ = None
        self.mu_v_ = None
        self.sigma2_w_ = None
        self.sigma2_v_ = None

    def initialize_hidden_params(self, X, y):
        N, d = X.shape
        self.mu_w_ = 0.0
        self.mu_v_ = np.zeros(self.dim_hidden_ )
        self.sigma2_w_ = 1.0
        self.sigma2_v_ = np.ones(self.dim_hidden_)

    def fit(self, X, y, logger=None, record_error=False, check_error=False):
        if not self.warm_start or self.bias_ is None:
            self.initialize_params(X, y)
            self.initialize_hidden_params(X, y)

        b, w, v = self.get_params()
        mu_w, mu_v, sigma2_w, sigma2_v = self.get_hidden_params()

        b, w, v, mu_w, mu_v, sigma2_w, sigma2_v, error_history = gibbs_sampling(
            X, y, b, w, v,
            mu_w, mu_v, sigma2_w, sigma2_v,
            self.mu_b_, self.sigma2_b_, self.alpha_, self.beta_,
            self.alpha_0_, self.beta_0_, self.m_0_, self.tau2_0_,
            n_iter=self.n_iter_, logger=logger,
            record_error=record_error, check_error=check_error, rng=self.rng
        )

        self.set_params(b, w, v)
        self.set_hidden_params(mu_w, mu_v, sigma2_w, sigma2_v)
        self.error_history_ = error_history

    def get_hidden_params(self):
        return self.mu_w_, self.mu_v_, self.sigma2_w_, self.sigma2_v_

    def set_hidden_params(self, mu_w, mu_v, sigma2_w, sigma2_v):
        self.mu_w_ = mu_w
        self.mu_v_ = mu_v
        self.sigma2_w_ = sigma2_w
        self.sigma2_v_ = sigma2_v
