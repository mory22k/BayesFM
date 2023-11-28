#import julia
#julia.install()

import myfm

import matplotlib.pyplot as plt
import numpy as np
import os

D = 512
N = 32
K = 16
var_y = .1

seed = 0
rng = np.random.default_rng(seed)

Q: np.ndarray = 2*rng.random((N, N))-1
X_data: np.ndarray = rng.choice([0, 1], size=(D,N))
Y_data: np.ndarray = np.sum(X_data, axis=1).astype(float) #np.einsum("di,ij,dj->d", X_data, Q, X_data) + rng.normal(0, var_y, size=D)

sampler = myfm.FMALS.FMAlternateLeastSquares(max_iter=5000)
model = myfm.FM.FactorizationMachines(N, K, sampler, seed=seed)
MSE_hist_als = model.train(X_data, Y_data)

sampler = myfm.BFM.BayesianFMSampler(max_iter=5000, seed=seed)
model = myfm.FM.FactorizationMachines(N, K, sampler, seed=seed)
MSE_hist_bayes = model.train(X_data, Y_data)

plt.plot(MSE_hist_als, label="Alternate Least Squares")
plt.plot(MSE_hist_bayes, label="Bayes")

plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.xscale("linear")
plt.yscale("log")
plt.legend()

if not os.path.exists("figures"):
    os.makedirs("figures")
plt.savefig("figures/MSE.png", dpi=300, bbox_inches="tight")
