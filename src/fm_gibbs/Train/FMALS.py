import numpy as np
from fm_gibbs.FM import fm_fast

def alternate_least_squares(
    X, y, b_init, w_init, v_init, l_b, l_w, l_v, n_iter=100, logger=None, return_errors=False, check_error=False
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
    if return_errors:
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

        if logger is not None or return_errors:
            error = np.mean(e**2)
        if logger is not None:
            logger.info(f'{mcs:4d} | {error:10.3e}')
        if return_errors:
            error_history.append(error)

    if return_errors:
        return b, w, v, error_history
    return b, w, v
