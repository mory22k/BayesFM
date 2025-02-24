# Copyright 2025 KM

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional
from numpy.typing import NDArray
import numpy as np


def fm_naive(
    X: NDArray,
    b: float,
    w: NDArray,
    v: NDArray,
) -> NDArray | float:
    is_X_1d = X.ndim == 1
    if is_X_1d:
        X = X.reshape(1, -1)

    n_samples = X.shape[0]
    bias = b
    linear_term = X @ w
    interaction_term = np.zeros(n_samples)

    for idx, x in enumerate(X):
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                interaction_term[idx] += v[i] @ v[j] * x[i] * x[j]

    value: NDArray | float
    if is_X_1d:
        value = bias + linear_term[0] + interaction_term[0]
        return value
    else:
        value = bias + linear_term + interaction_term
        return value


def fm_fast(
    X: NDArray,
    b: float,
    w: NDArray,
    v: NDArray,
    q: Optional[NDArray] = None,
) -> NDArray | float:
    is_X_1d = X.ndim == 1
    if is_X_1d:
        X = X.reshape(1, -1)

    if q is None:
        q = X @ v
    linear_term = X @ w
    interaction_term = np.sum(q**2 - (X**2) @ (v**2), axis=1) / 2

    value: NDArray | float
    if is_X_1d:
        value = b + linear_term[0] + interaction_term[0]
        return value
    else:
        value = b + linear_term + interaction_term
        return value


def fm_variables_to_qubo(w: NDArray, V: NDArray) -> NDArray:
    n = len(w)
    qubo = np.zeros((n, n))
    i, j = np.diag_indices(n)
    qubo[i, j] = w
    qubo += np.triu(np.einsum("if, jf -> ij", V, V), k=1)
    return qubo
