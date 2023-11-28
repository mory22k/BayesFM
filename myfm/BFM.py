import numpy as np
from copy import copy
from typing import Tuple, Optional
import os
from . import FM

from julia import Main # type: ignore

script_dir = os.path.dirname(os.path.abspath(__file__))

def sample_noise_var(
    x_hist: np.ndarray,
    y_hist: np.ndarray,
    model_param: float,
    hyper_param: dict
):
    assert x_hist.shape == y_hist.shape

    alpha_noise = hyper_param["alpha_noise"]
    beta_noise = hyper_param["beta_noise"]
    D = x_hist.shape[0]
    alpha_noise_post = alpha_noise + D / 2
    beta_noise_post = beta_noise + np.sum((y_hist - x_hist * model_param)**2) / 2
    var_noise = 1/np.random.gamma(alpha_noise_post, 1/beta_noise_post)
    return var_noise

def sample_parameter(
    x_hist: np.ndarray,
    y_hist: np.ndarray,
    var_noise: float,
    mean_param: float,
    var_param: float
):
    var_param_post = 1 / ( np.sum(x_hist ** 2) / var_noise + 1/var_param )
    mean_param_post = var_param_post * (np.sum(x_hist*y_hist) / var_noise + mean_param/var_param)
    return np.random.normal(mean_param_post, np.sqrt(var_param_post))

def sample_mean_parameter(
    model_params:np.ndarray,
    var_param:float,
    hyper_param:dict
):
    mean_mean_param = hyper_param["mean_mean_param"]
    precision_mean_param = hyper_param["precision_mean_param"]
    N = model_params.shape[0]
    precision_mean_param_post = precision_mean_param + N
    mean_mean_param_post = (np.sum(model_params) + precision_mean_param * mean_mean_param) / precision_mean_param_post
    return np.random.normal(mean_mean_param_post, np.sqrt(var_param/precision_mean_param_post))

def sample_var_parameter(
    model_params: np.ndarray,
    mean_param: float,
    hyper_param: dict
):
    mean_mean_param = hyper_param["mean_mean_param"]
    precision_mean_param = hyper_param["precision_mean_param"]
    alpha_param = hyper_param["alpha_param"]
    beta_param = hyper_param["beta_param"]

    N = model_params.shape[0]
    alpha_param_post = alpha_param + (N + 1) / 2
    beta_param_post = beta_param + (np.sum((model_params - mean_param)**2) + precision_mean_param * (mean_param - mean_mean_param)**2) / 2
    return 1/np.random.gamma(alpha_param_post, 1/beta_param_post)

def train_bayes(
    X_data: np.ndarray,
    Y_data: np.ndarray,
    Y_pred: np.ndarray,
    b_init: float,
    w_init: np.ndarray,
    V_init: np.ndarray,
    max_iter: int,
    show_progress: bool,
    hyper_param: dict,
    seed: int
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)

    b = copy(b_init)
    w = copy(w_init)
    V = copy(V_init)

    D = X_data.shape[0]
    N = X_data.shape[1]
    K = V.shape[1]

    # place holder
    var_w = 1.
    mean_w = 0.
    var_v = np.ones(K)
    mean_v = np.zeros(K)

    # precompute error and quadratic
    error = Y_pred - Y_data # (D,)
    quad = X_data @ V       # (D, K)
    error_hist = np.empty(max_iter + 1, dtype=float)
    error_hist[0] = np.mean(error**2)

    for iter in range(max_iter):
        # update hyperparameters
        # sample noise variance: O(D)
        var_noise = sample_noise_var(np.ones(D), b - error, float(b), hyper_param)

        # sample mean and variance of w: O(1)
        var_w = sample_var_parameter(w, mean_w, hyper_param)
        mean_w = sample_mean_parameter(w, var_w, hyper_param)
        for k in range(K):
            var_v[k] = sample_var_parameter(V[:,k], mean_v[k], hyper_param)
            mean_v[k] = sample_mean_parameter(V[:,k], var_v[k], hyper_param)

        # update parameters
        # sample b: 1 * O(D) = O(D)
        x_hist = np.ones(D)
        y_hist = b - error
        b_new = sample_parameter(x_hist, y_hist, var_noise, hyper_param["mean_b"], hyper_param["var_b"])
        error = error + b_new - b
        b = b_new

        # sample w: N * O(D) = O(D * N)
        for i in range(N):
            x_hist = X_data[:,i]
            y_hist = X_data[:,i] * (w[i] * X_data[:,i] - error)
            w_i_new = sample_parameter(x_hist, y_hist, var_noise, mean_w, var_w)
            error = error + (w_i_new - w[i]) * X_data[:,i]
            w[i] = w_i_new

        # sample V: N * K * O(D) = O(D * N * K)
        for k in range(K):
            for i in range(N):
                G_ik = X_data[:,i] * (quad[:,k] - V[i,k] * X_data[:,i])
                x_hist = G_ik
                y_hist = (V[i,k] * G_ik - error)
                V_ik_new = sample_parameter(x_hist, y_hist, var_noise, mean_v[k], var_v[k])
                error = error + (V_ik_new - V[i,k]) * G_ik
                quad[:,k] = quad[:,k] + (V_ik_new - V[i,k]) * X_data[:,i]
                V[i,k] = V_ik_new

        error_hist[iter+1] = np.mean(error**2)
        if show_progress and (iter+1) % 50 == 0:
            print(f"iter: {iter+1}, error: {np.mean(error**2)}")
    return b, w, V, error_hist # type: ignore

def train_bayes_julia(
    X_data:np.ndarray,
    Y_data:np.ndarray,
    Y_pred:np.ndarray,
    b_init:float,
    w_init:np.ndarray,
    V_init:np.ndarray,
    max_iter:int,
    show_progress:bool,
    hyper_param:dict,
    seed:int
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:

    Main.include(os.path.join(script_dir, "julia/BFM.jl"))
    b, w, V, error_hist = Main.train_bayes(
        X_data.astype(np.float64),
        Y_data.astype(np.float64),
        Y_pred.astype(np.float64),
        b_init,
        w_init,
        V_init,
        max_iter,
        show_progress,
        hyper_param,
        seed
    )

    return b, w, V, error_hist

class BayesianFMSampler(FM.Sampler):
    def __init__(self, max_iter:int, seed:Optional[int] = None, show_progress:bool = True, use_julia:bool = True):
        self.max_iter = max_iter
        self.rng = np.random.default_rng(seed)
        self.show_progress = show_progress
        self.use_julia = use_julia
        self.hyper_param = {
            "alpha_noise": 1.,
            "beta_noise": 1e-3,
            "mean_b": 0.,
            "var_b": 1.,
            "mean_mean_param": 0.,
            "precision_mean_param": 1.,
            "alpha_param": 1.,
            "beta_param": 1.,
        }

    def sample(self, model: FM.FactorizationMachines, X_data: np.ndarray, Y_data: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        seed = int(self.rng.integers(0, 2**32-1))
        b_init = model.params["b"]
        w_init = model.params["w"]
        V_init = model.params["V"]

        if self.use_julia:
            b, w, V, MSE_hist = train_bayes_julia(
                X_data,
                Y_data,
                model.predict(X_data),
                b_init,
                w_init,
                V_init,
                self.max_iter,
                self.show_progress,
                self.hyper_param,
                seed = seed
            )
        else:
            b, w, V, MSE_hist = train_bayes(
                X_data,
                Y_data,
                model.predict(X_data),
                b_init,
                w_init,
                V_init,
                self.max_iter,
                self.show_progress,
                self.hyper_param,
                seed = self.rng.integers(0, 2**32-1)
            )

        return b, w, V, MSE_hist
