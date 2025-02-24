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

from typing import Any
import abc
from numpy.typing import NDArray


class BaseFMRegression(abc.ABC):
    """Abstract base class for quadratic factorization machine regression models.

    Attributes:
        intercept_ (float): Intercept term.
        w_ (NDArray): Linear coefficients of shape (n,).
        V_ (NDArray): Latent factors of shape (n, n_factors).
        n_factors (int): Number of latent factors.
        is_fitted (bool): Flag indicating whether the model has been fitted.
    """

    intercept_: float
    w_: NDArray
    V_: NDArray
    n_factors: int

    @abc.abstractmethod
    def fit(self, X: NDArray, y: NDArray, **kwargs: Any) -> "BaseFMRegression":
        """Fits the factorization machine regression model.

        Args:
            X (NDArray): Feature matrix of shape (num_samples, n).
            y (NDArray): Target vector of shape (num_samples,).

        Returns:
            BaseFMRegression: The fitted regression model.
        """
        pass
