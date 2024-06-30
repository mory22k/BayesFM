import numpy as np
from fm_gibbs.FM import fm_fast
from fm_gibbs.Train.FMALS import alternate_least_squares

def gibbs_sampling(
    X, y, b_init, w_init, v_init,
    mu_b=0, sigma2_b=1, alpha=1, beta=1, alpha_0=1, beta_0=1, m_0=0, tau2_0=1,
    n_iter=100, n_pretrain=5, logger=None, return_errors=False, check_error=False
):
    b = np.copy(b_init)
    w = w_init.copy()
    v = v_init.copy()

    assert 0 <= n_pretrain <= n_iter
    n_iter -= n_pretrain
    b, w, v, error_history = alternate_least_squares(
        X, y, b, w, v, 1, 1, 1, n_iter=n_pretrain, return_errors=True
    )
    N, d = X.shape
    k = v.shape[1]

    f = np.apply_along_axis(
        lambda x: fm_fast(x, b, w, v), axis=1, arr=X
    )

    q = X @ v # (N, k)
    e = f - y # (N,)

    sigma2_w = 1.0
    sigma2_v = np.ones(k)
    mu_w = 0.0
    mu_v = np.zeros(k)

    if return_errors:
        error_history.append(np.mean(e**2))

    for mcs in range(n_iter):
        # update sigma2
        alpha_star = alpha + N / 2
        beta_star = beta + e @ e / 2
        sigma2 = 1 / np.random.gamma(alpha_star, 1 / beta_star)

        # update hyperparams for w, v
        alpha_theta_star = alpha_0 + (d + 1) / 2
        tau2_theta_star = 1 / (1 / tau2_0 + d)

        # update hyperparams for w
        beta_w_star = beta_0 + ( ((w[:] - mu_w)**2).sum() - (mu_w - m_0)**2 ) / 2
        m_w_star = tau2_theta_star * (w[:].sum() + m_0 / tau2_0)
        sigma2_w = 1 / np.random.gamma(alpha_theta_star, 1 / beta_w_star)
        mu_w = np.random.normal(m_w_star, tau2_theta_star * sigma2_w)

        # update hyperparams for v
        for f in range(k):
            beta_v_f_star = beta_0 + ( ((v[:,f] - mu_v[f])**2).sum() - (mu_v[f] - m_0)**2 ) / 2
            m_v_f_star = tau2_theta_star * (v[:,f].sum() + m_0 / tau2_0)
            sigma2_v[f] = 1 / np.random.gamma(alpha_theta_star, 1 / beta_v_f_star)
            mu_v[f] = np.random.normal(m_v_f_star, tau2_theta_star * sigma2_v[f])

        # update b
        h = np.ones(N)
        nu = b - e
        sigma2_b_star = 1 / (1 / sigma2 * (h @ h) + 1 / sigma2_b)
        mu2_b_star = sigma2_b_star * ((h @ nu) / sigma2 + mu_b / sigma2_b)
        b_new = np.random.normal(mu2_b_star, sigma2_b_star)
        e = e + b_new - b
        b = b_new

        for i in range(d):
            h = X[:,i]
            nu = w[i] * h - e
            sigma2_w_i_star = 1 / (1 / sigma2 * (h @ h) + 1 / sigma2_w)
            mu2_w_i_star = sigma2_w_i_star * ((h @ nu) / sigma2 + mu_w / sigma2_w)
            w_i_new = np.random.normal(mu2_w_i_star, sigma2_w_i_star)
            e = e + (w_i_new - w[i]) * h
            w[i] = w_i_new

            for f in range(k):
                h = (q[:,f] - v[i,f] * X[:,i]) * X[:,i]
                nu = v[i,f] * h - e
                sigma2_v_if_star = 1 / (1 / sigma2 * (h @ h) + 1 / sigma2_v[f])
                mu2_v_if_star = sigma2_v_if_star * ((h @ nu) / sigma2 + mu_v[f] / sigma2_v[f])
                v_if_new = np.random.normal(mu2_v_if_star, sigma2_v_if_star)
                q[:,f] = q[:,f] + (v_if_new - v[i,f]) * X[:,i]
                e = e + (v_if_new - v[i,f]) * h
                v[i,f] = v_if_new

        if check_error:
            f_pred = np.apply_along_axis(
                lambda x: fm_fast(x, b, w, v), axis=1, arr=X
            )
            assert np.isclose(e, f_pred - y).all()

        if logger is not None or return_errors:
            error = np.mean(e**2)
        if logger is not None:
            logger.info(f'{mcs:4d} | {error:10.3e}')
        if return_errors:
            error_history.append(error)

    if return_errors:
        return b, w, v, error_history
    return b, w, v
