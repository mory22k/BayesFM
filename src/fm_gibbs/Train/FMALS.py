import numpy as np
from fm_gibbs.FM import fm_fast, FactorizationMachineRegressor

def alternate_least_squares(
    X, y, b_init, w_init, v_init,
    l_b, l_w, l_v,
    n_iter=100, logger=None, record_error=False, check_error=False
):
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

    for mcs in range(n_iter):
        h = np.ones(N)
        # nu = b * h - e
        nu = b - e
        b_new = h @ nu / (h @ h + l_b)
        e = e + b_new - b
        b = b_new

        for i in range(d):
            h = X[:,i]
            nu = w[i] * h - e
            w_i_new = h @ nu / (h @ h + l_w)
            e = e + (w_i_new - w[i]) * h
            w[i] = w_i_new

            for f in range(k):
                h = (q[:,f] - v[i,f] * X[:,i]) * X[:,i]
                nu = v[i,f] * h - e
                v_if_new = h @ nu / (h @ h + l_v)
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

    return b, w, v, error_history

class FactorizationMachineALSRegressor(FactorizationMachineRegressor):
    def __init__(
        self,
        dim_hidden=5,
        max_iter=100,
        warm_start=True,
        seed=None,
        alpha_b=1.0,
        alpha_w=1.0,
        alpha_v=1.0,
    ):
        super().__init__(dim_hidden, seed)

        self.n_iter_ = max_iter
        self.warm_start = warm_start
        self.error_history_ = None

        self.alpha_b_ = alpha_b
        self.alpha_w_ = alpha_w
        self.alpha_v_ = alpha_v

    def fit(self, X, y, logger=None, record_error=False):
        if not self.warm_start or self.coef_ is None:
            self.initialize_params(X, y)

        b, w, v = self.get_params()

        b, w, v, error_history = alternate_least_squares(
            X, y, b, w, v,
            self.alpha_b_, self.alpha_w_, self.alpha_v_,
            n_iter=self.n_iter_, logger=logger, record_error=record_error
        )

        self.set_params(b, w, v)
        self.error_history_ = error_history
