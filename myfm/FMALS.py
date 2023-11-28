import numpy as np
from copy import copy
from typing import Tuple, Optional
from julia import Main # type: ignore
import os
from . import FM

script_dir = os.path.dirname(os.path.abspath(__file__))

def least_squares(
    x_hist: np.ndarray,
    y_hist: np.ndarray,
    precision_param: float = 1.,
):
    assert x_hist.shape[0] == y_hist.shape[0]
    if np.all(x_hist == 1):
        w_pred = np.sum(y_hist) / (x_hist.shape[0] + precision_param)
    else:
        w_pred = np.sum(x_hist * y_hist) / (np.sum(x_hist ** 2) + precision_param)

    return w_pred

def train_als(
    model,
    X_data:np.ndarray,
    Y_data:np.ndarray,
    Y_pred:np.ndarray,
    b_init:float,
    w_init:np.ndarray,
    V_init:np.ndarray,
    max_iter:int,
    show_progress:bool,
    hyper_param:dict
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    b = copy(b_init)
    w = copy(w_init)
    V = copy(V_init)

    D = X_data.shape[0]
    N = X_data.shape[1]
    K = V.shape[1]

    # precompute error and quadratic
    error = Y_pred - Y_data # (D,)
    quad = X_data @ V       # (D, K)
    error_hist = np.empty(max_iter + 1, dtype=float)
    error_hist[0] = np.mean(error**2)

    for iter in range(max_iter):
        # update b: 1 * O(D) = O(D)
        x_hist = np.ones(D)
        y_hist = b - error
        b_new = least_squares(x_hist, y_hist, hyper_param["precision_param"])
        error = error + b_new - b
        b = b_new

        # update w: N * O(D) = O(D * N)
        for i in range(N):
            x_hist = X_data[:,i]
            y_hist = X_data[:,i] * (w[i] * X_data[:,i] - error)
            w_i_new = least_squares(x_hist, y_hist, hyper_param["precision_param"])
            error = error + (w_i_new - w[i]) * X_data[:,i]
            w[i] = w_i_new

        # update V: N * K * O(D) = O(D * N * K)
        for i in range(N):
            for k in range(K):
                G_ik = X_data[:,i] * (quad[:,k] - V[i,k] * X_data[:,i])
                x_hist = G_ik
                y_hist = (V[i,k] * G_ik - error)
                V_ik_new = least_squares(x_hist, y_hist, hyper_param["precision_param"])
                error = error + (V_ik_new - V[i,k]) * G_ik
                quad[:,k] = quad[:,k] + (V_ik_new - V[i,k]) * X_data[:,i]
                V[i,k] = V_ik_new

        error_hist[iter+1] = np.mean(error**2)
        if show_progress and (iter+1) % 50 == 0:
            print(f"iter: {iter+1}, error: {np.mean(error**2)}")
    return b, w, V, error_hist # type: ignore

def train_als_julia(
    X_data:np.ndarray,
    Y_data:np.ndarray,
    Y_pred:np.ndarray,
    b_init:float,
    w_init:np.ndarray,
    V_init:np.ndarray,
    max_iter:int,
    show_progress:bool,
    hyper_param:dict
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:

    Main.include(os.path.join(script_dir, "julia/FMALS.jl"))
    b, w, V, error_hist = Main.train_als(
        X_data.astype(np.float64),
        Y_data.astype(np.float64),
        Y_pred.astype(np.float64),
        b_init,
        w_init,
        V_init,
        max_iter,
        show_progress,
        hyper_param
    )
    return b, w, V, error_hist

class FMAlternateLeastSquares(FM.Sampler):
    def __init__(self, max_iter:int, show_progress:bool = True, use_julia:bool = True):
        self.max_iter = max_iter
        self.show_progress = show_progress
        self.use_julia = use_julia
        self.hyper_param = {
            "precision_param": 1e-3,
        }

    def sample(self, model: FM.FactorizationMachines, X_data: np.ndarray, Y_data: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        b_init = model.params["b"]
        w_init = model.params["w"]
        V_init = model.params["V"]

        if self.use_julia:
            b, w, V, MSE_hist = train_als_julia(
                X_data,
                Y_data,
                model.predict(X_data),
                b_init,
                w_init,
                V_init,
                self.max_iter,
                self.show_progress,
                self.hyper_param
            )
        else:
            b, w, V, MSE_hist = train_als(
                model,
                X_data,
                Y_data,
                model.predict(X_data),
                b_init,
                w_init,
                V_init,
                self.max_iter,
                self.show_progress,
                self.hyper_param
            )

        return b, w, V, MSE_hist
