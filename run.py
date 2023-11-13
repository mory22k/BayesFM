from functions import FM

import matplotlib.pyplot as plt
import numpy as np
import os

from julia import Main # type: ignore

D = 16
N = 16
K = 8
var_y = .1

seed = 0
rng = np.random.default_rng(seed)

Q: np.ndarray = 2*rng.random((N, N))-1
X_data: np.ndarray = rng.choice([0, 1], size=(D,N))
Y_data: np.ndarray = np.einsum("di,ij,dj->d", X_data, Q, X_data) + rng.normal(0, var_y, size=D)
print("X_data:", X_data)
print("Y_data:", Y_data)

model = FM.FactorizationMachines(N, K, seed=seed)
Y_pred = model.predict(X_data)
b, w, V, MSE_hist_als = FM.train_als(X_data, Y_data, Y_pred, model.b, model.w, model.V, max_iter=1000)
b, w, V, MSE_hist_als_julia = FM.train_als_julia(X_data, Y_data, Y_pred, model.b, model.w, model.V, max_iter=1000)

plt.plot(MSE_hist_als, label="ALS (Python)")
plt.plot(MSE_hist_als_julia, label="ALS (Julia)")
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.yscale("log")
plt.legend()

if not os.path.exists("figures"):
    os.makedirs("figures")
plt.savefig("figures/MSE.png", dpi=300, bbox_inches="tight")
