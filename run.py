from functions import FM

import matplotlib.pyplot as plt
import numpy as np
import os

D = 64
N = 16
K = 8
var_y = .1

Q: np.ndarray = np.random.rand(N, N)
X_data: np.ndarray = np.random.choice([0, 1], size=(D,N))
Y_data: np.ndarray = np.einsum("ij,ij->i", X_data @ Q, X_data @ Q) + np.random.normal(0, var_y, size=D)

model = FM.FactorizationMachines(n_features=N, n_factors=K, seed=0)
MSE_hist_als = FM.train_als(model, X_data, Y_data, max_iter=1000, use_true_error=True)

model = FM.FactorizationMachines(n_features=N, n_factors=K, seed=0)
MSE_hist_bayes = FM.train_bayes(model, X_data, Y_data, max_iter=1000, var_y=var_y, use_true_error=True)
# use_true_error: Use the true error value to ensure that the MSE value is accurate (not recommended

plt.plot(MSE_hist_als, label="ALS")
plt.plot(MSE_hist_bayes, label="Bayes")
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.yscale("log")
plt.ylim(bottom=1e-10)
plt.legend()

if not os.path.exists("figures"):
    os.makedirs("figures")
plt.savefig("figures/MSE.png", dpi=300, bbox_inches="tight")
