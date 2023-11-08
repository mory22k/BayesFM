from functions import FMALS

import matplotlib.pyplot as plt
import numpy as np
import os

D = 100
N = 10
K = 5

model = FMALS.FactorizationMachines(n_features=N, n_factors=K)
X_data = np.random.choice([0, 1], size=(D,N))
Y_data = np.random.randn(D)

MSE_hist = FMALS.als(model, X_data, Y_data, max_iter=1000, use_true_error=False)

plt.plot(MSE_hist)
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.yscale("log")

if not os.path.exists("figures"):
    os.makedirs("figures")
plt.savefig("figures/ALS-FM.png", dpi=300, bbox_inches="tight")
