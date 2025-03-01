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

import logging
import warnings
from typing import Any

from numpy.typing import NDArray
import numpy as np
from tqdm import tqdm
import dimod

from ._base import BaseFMRegression
from ._fm import fm_fast
from .logging import get_logger

LOGGER = get_logger()


def _sample_from_inverse_gamma(
    alpha: float | NDArray, beta: float | NDArray, rng: np.random.Generator
) -> float | NDArray:
    return 1 / rng.gamma(alpha, 1 / beta)


def _sample_sigma2(
    e: NDArray,
    n: int,
    alpha_sigma: float,
    beta_sigma: float,
    rng: np.random.Generator,
) -> float:
    alpha_sigma_star = alpha_sigma + n / 2
    beta_sigma_star = beta_sigma + e @ e / 2
    new_sigma2: float = _sample_from_inverse_gamma(
        alpha_sigma_star, beta_sigma_star, rng
    )  # type: ignore
    return new_sigma2


def _sample_sigma2_w(
    d: int,
    w: NDArray,
    mu_w: float,
    m_0: float,
    tau2_0: float,
    alpha_0: float,
    beta_0: float,
    rng: np.random.Generator,
) -> float:
    alpha_w_star: float = alpha_0 + (d + 1) / 2
    beta_w_star: float = (
        beta_0 + (((w[:] - mu_w) ** 2).sum() + (mu_w - m_0) ** 2 / tau2_0) / 2
    )
    new_sigma2_w: float = _sample_from_inverse_gamma(
        alpha_w_star,
        beta_w_star,
        rng,
    )  # type: ignore
    return new_sigma2_w


def _sample_mu_w(
    d: int,
    w: NDArray,
    sigma2_w: float,
    m_0: float,
    tau2_0: float,
    rng: np.random.Generator,
) -> float:
    tau2_w_star: float = 1 / (1 / sigma2_w + d / tau2_0)
    m_w_star: float = tau2_w_star * (w[:].sum() + m_0 / tau2_0)
    new_mu_w: float = rng.normal(m_w_star, np.sqrt(tau2_w_star))
    return new_mu_w


def _sample_sigma2_V(
    d: int,
    k: int,
    V: NDArray,
    mu_V: NDArray,
    m_0: float,
    tau2_0: float,
    alpha_0: float,
    beta_0: float,
    rng: np.random.Generator,
) -> NDArray:
    alpha_V_f_star = alpha_0 + (d + 1) / 2
    new_sigma2_V = np.zeros(k)
    for f in range(k):
        beta_Vf_star = (
            beta_0
            + (((V[:, f] - mu_V[f]) ** 2).sum() + (mu_V[f] - m_0) ** 2 / tau2_0) / 2
        )
        new_sigma2_V[f] = _sample_from_inverse_gamma(alpha_V_f_star, beta_Vf_star, rng)

    return new_sigma2_V


def _sample_mu_V(
    d: int,
    k: int,
    V: NDArray,
    sigma2_V: NDArray,
    m_0: float,
    tau2_0: float,
    rng: np.random.Generator,
) -> NDArray:
    tau2_theta_star = 1 / (d + 1 / tau2_0)
    new_mu_V = np.zeros(k)
    for f in range(k):
        m_Vf_star = tau2_theta_star * (V[:, f].sum() + m_0 / tau2_0)
        new_mu_V[f] = rng.normal(m_Vf_star, np.sqrt(tau2_theta_star * sigma2_V[f]))

    return new_mu_V


def _sample_b_update_e(
    e: NDArray,
    n: int,
    b: float,
    sigma2: float,
    mu_b: float,
    sigma2_b: float,
    rng: np.random.Generator,
) -> tuple[float, NDArray]:
    h = np.ones(n)
    y_minus_g_b = e + b
    sigma2_b_star = 1 / (n / sigma2 + 1 / sigma2_b)
    mu_b_star = sigma2_b_star * (y_minus_g_b @ h / sigma2 + mu_b / sigma2_b)
    new_b: float = rng.normal(mu_b_star, np.sqrt(sigma2_b_star))  # type: ignore

    e -= new_b - b

    return new_b, e


def _sample_w_update_e(
    X: NDArray,
    e: NDArray,
    d: int,
    w: NDArray,
    sigma2: float,
    mu_w: float,
    sigma2_w: float,
    rng: np.random.Generator,
) -> tuple[NDArray, NDArray]:
    new_w = np.zeros(d)

    for i in range(d):
        h = X[:, i]
        y_minus_g_wi = e + w[i] * h
        sigma2_wi_star = 1 / (h @ h / sigma2 + 1 / sigma2_w)
        mu_wi_star = sigma2_wi_star * (y_minus_g_wi @ h / sigma2 + mu_w / sigma2_w)
        new_w[i] = rng.normal(mu_wi_star, np.sqrt(sigma2_wi_star))

        e -= (new_w[i] - w[i]) * h

    return new_w, e


def _sample_V_update_e_q(
    X: NDArray,
    e: NDArray,
    q: NDArray,
    d: int,
    k: int,
    V: NDArray,
    sigma2: float,
    mu_V: NDArray,
    sigma2_V: NDArray,
    rng: np.random.Generator,
) -> tuple[NDArray, NDArray, NDArray]:
    new_V = np.zeros((d, k))

    for i in range(d):
        for f in range(k):
            h = (q[:, f] - V[i, f] * X[:, i]) * X[:, i]
            y_minus_g_vif = e + V[i, f] * h
            sigma2_Vif_star = 1 / (h @ h / sigma2 + 1 / sigma2_V[f])
            mu_Vif_star = sigma2_Vif_star * (
                y_minus_g_vif @ h / sigma2 + mu_V[f] / sigma2_V[f]
            )
            new_V[i, f] = rng.normal(mu_Vif_star, np.sqrt(sigma2_Vif_star))

            e -= (new_V[i, f] - V[i, f]) * h
            q[:, f] += (new_V[i, f] - V[i, f]) * X[:, i]

    return new_V, e, q


def _markov_trainsition(
    # dataset
    X: NDArray,
    e: NDArray,
    q: NDArray,
    n: int,
    d: int,
    k: int,
    # params
    b: float,
    w: NDArray,
    V: NDArray,
    # hyperparams
    mu_w: float,
    mu_V: NDArray,
    # fixed hyperparams
    mu_b: float,
    sigma2_b: float,
    alpha_sigma: float,
    beta_sigma: float,
    m_0: float,
    tau2_0: float,
    alpha_0: float,
    beta_0: float,
    # generator
    rng: np.random.Generator,
) -> tuple[NDArray, NDArray, float, NDArray, NDArray, float, float, NDArray]:
    new_sigma2 = _sample_sigma2(e, n, alpha_sigma, beta_sigma, rng)
    new_sigma2_w = _sample_sigma2_w(d, w, mu_w, m_0, tau2_0, alpha_0, beta_0, rng)
    new_mu_w = _sample_mu_w(d, w, new_sigma2_w, m_0, tau2_0, rng)
    new_sigma2_V = _sample_sigma2_V(d, k, V, mu_V, m_0, tau2_0, alpha_0, beta_0, rng)
    new_mu_V = _sample_mu_V(d, k, V, new_sigma2_V, m_0, tau2_0, rng)

    new_b, e = _sample_b_update_e(e, n, b, new_sigma2, mu_b, sigma2_b, rng)
    new_w, e = _sample_w_update_e(X, e, d, w, new_sigma2, new_mu_w, new_sigma2_w, rng)
    new_V, e, q = _sample_V_update_e_q(
        X, e, q, d, k, V, new_sigma2, new_mu_V, new_sigma2_V, rng
    )

    return e, q, new_b, new_w, new_V, new_sigma2, new_mu_w, new_mu_V


def _sample(
    n_iter: int,
    X: NDArray,
    y: NDArray,
    b_init: float,
    w_init: NDArray,
    V_init: NDArray,
    mu_w_init: float,
    mu_V_init: NDArray,
    mu_b: float,
    sigma2_b: float,
    alpha_sigma: float,
    beta_sigma: float,
    m_0: float,
    tau2_0: float,
    alpha_0: float,
    beta_0: float,
    show_progress_bar: bool,
    progress_bar_desc: str,
    log_per: int,
    rng: np.random.Generator,
) -> tuple[NDArray, NDArray, NDArray, NDArray, float, NDArray]:
    if n_iter < 1:
        warnings.warn("Number of iterations must be at least 1.")
        return np.array([]), np.array([]), np.array([]), np.array([]), 0.0, np.array([])

    n = X.shape[0]
    d = X.shape[1]
    k = V_init.shape[1]

    b = b_init
    w = w_init
    V = V_init
    mu_w = mu_w_init
    mu_V = mu_V_init

    q = X @ V
    e = y - fm_fast(X, b, w, V, q)

    b_samples = np.zeros(n_iter)
    w_samples = np.zeros((n_iter, d))
    V_samples = np.zeros((n_iter, d, k))
    sigma2_samples = np.zeros(n_iter)

    for i in tqdm(range(n_iter), desc=progress_bar_desc, disable=not show_progress_bar):
        e, q, b, w, V, sigma2, mu_w, mu_V = _markov_trainsition(
            X,
            e,
            q,
            n,
            d,
            k,
            b,
            w,
            V,
            mu_w,
            mu_V,
            mu_b,
            sigma2_b,
            alpha_sigma,
            beta_sigma,
            m_0,
            tau2_0,
            alpha_0,
            beta_0,
            rng,
        )

        b_samples[i] = b
        w_samples[i] = w
        V_samples[i] = V
        sigma2_samples[i] = sigma2

        if i % log_per == 0:
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.debug(f"[{i:4}]: {np.sqrt(np.mean(e**2)):10.3e}")

    return b_samples, w_samples, V_samples, sigma2_samples, mu_w, mu_V


class BayesFMRegression(BaseFMRegression):
    """Bayesian Factorization Machine Regression model using MCMC sampling.

    This model performs regression via a Factorization Machine with Bayesian parameter estimation.
    It uses MCMC sampling to update the parameters (intercept, linear weights, and latent factors).

    Before fitting, the target y is zero-centered; after sampling, y’s mean is added back to the
    learned intercept. If warm_start is True and the model is already fitted, previous parameters
    are used as initial values.

    Attributes:
        n_mcs (int): Number of MCMC sampling iterations.
        n_warmup (int): Number of warm-up iterations.
        warm_start (bool): Whether to use previous parameter values when fitting if already fitted.
        n_factors (int): Number of latent factors.

        The following hyperparameters are used in the sampling process:
            mu_b (float): Prior mean for the intercept.
            sigma2_b (float): Prior variance for the intercept.
            alpha_sigma (float): Shape parameter for the noise variance.
            beta_sigma (float): Scale parameter for the noise variance.
            m_0 (float): Prior mean for weights and latent factors.
            tau2_0 (float): Prior variance for weights and latent factors.
            alpha_0 (float): Shape parameter for the inverse-gamma prior on weight variance.
            beta_0 (float): Scale parameter for the inverse-gamma prior on weight variance.

        Additionally, the learned parameters are stored in:
            intercept_ (float)
            w (NDArray): Linear weights of shape (d,).
            V (NDArray): Latent factor matrix of shape (d, n_factors).
    """

    def __init__(
        self,
        n_factors: int = 5,
        n_mcs: int = 1000,
        n_warmup: int = 1000,
        warm_start: bool = False,
        show_progress_bar: bool = True,
        save_samples: bool = False,
        seed: int | None = None,
        log_per: int = 10,
        mu_b: float = 0.0,
        sigma2_b: float = 1.0,
        alpha_sigma: float = 1.0,
        beta_sigma: float = 1.0,
        m_0: float = 0.0,
        tau2_0: float = 1.0,
        alpha_0: float = 1.0,
        beta_0: float = 1.0,
    ) -> None:
        """
        Initializes the BayesFMRegression instance.

        Args:
            n_factors (int, optional): Number of latent factors. Defaults to 5.
            n_mcs (int, optional): Number of MCMC sampling iterations. Defaults to 1000.
            n_warmup (int, optional): Number of warm-up iterations to discard. Defaults to 500.
            warm_start (bool, optional): Whether to reuse previous parameters if already fitted. Defaults to False.
            show_progress_bar (bool, optional): Whether to show a progress bar during sampling. Defaults to True.
            save_samples (bool, optional): Whether to save samples for each parameter. If True, samples are stored in the samples attribute. Defaults to False.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            log_per (int, optional): Log progress every log_per iterations. Defaults to 10.
            mu_b (float, optional): Prior mean for the intercept. Defaults to 0.0.
            sigma2_b (float, optional): Prior variance for the intercept. Defaults to 1.0.
            alpha_sigma (float, optional): Shape parameter for the noise variance. Defaults to 1.0.
            beta_sigma (float, optional): Scale parameter for the noise variance. Defaults to 1.0.
            m_0 (float, optional): Prior mean for weights and latent factors. Defaults to 0.0.
            tau2_0 (float, optional): Prior variance for weights and latent factors. Defaults to 1.0.
            alpha_0 (float, optional): Shape parameter for the inverse-gamma prior on weight variance. Defaults to 1.0.
            beta_0 (float, optional): Scale parameter for the inverse-gamma prior on weight variance. Defaults to 1.0.
        """
        self.n_factors = n_factors
        self.n_mcs = n_mcs
        self.n_warmup = n_warmup
        self.warm_start = warm_start
        self.show_progress_bar = show_progress_bar
        self.save_samples = save_samples
        self.rng = np.random.default_rng(seed)
        self.log_per = log_per

        # Sampling hyperparameters.
        self.mu_b = mu_b
        self.sigma2_b = sigma2_b
        self.alpha_sigma = alpha_sigma
        self.beta_sigma = beta_sigma
        self.m_0 = m_0
        self.tau2_0 = tau2_0
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0

        # Initialize model parameters (will be of shape based on input dimension).
        self.intercept_ = 0.0
        self.w_ = np.array([])  # To be initialized upon fitting.
        self.V_ = np.array([[]])  # To be initialized upon fitting.

        # Initialize storage for samples.
        self.samples: dict[str, NDArray] = {}

        self.is_fitted: bool = False

    def fit(
        self,
        X: NDArray,
        y: NDArray,
        **kwargs: Any,
    ) -> "BayesFMRegression":
        """Fits the Bayesian Factorization Machine Regression model via MCMC sampling.

        The target vector y is zero-centered prior to sampling; after sampling, the mean is added
        back to the learned intercept.

        If warm_start is True and the model is already fitted, the current parameters are used as
        initial values.

        Args:
            X (NDArray): Binary feature matrix of shape (num_samples, d).
            y (NDArray): Target vector of shape (num_samples,).

        Returns:
            BayesFMRegression: The fitted regression model.
        """
        n_samples, d = X.shape
        k = self.n_factors

        # Zero-center y.
        y_mean = np.mean(y)
        y_centered = y - y_mean

        # Initialize parameters if not warm-starting or not yet fitted.
        if not self.warm_start or not self.is_fitted:
            self.intercept_ = 0.0
            self.w_ = np.zeros(d)
            self.V_ = np.random.normal(loc=0.0, scale=1.0, size=(d, k))
            self.sigma2 = 1.0
            self.mu_w = 0.0
            self.mu_V = np.zeros(k)

        # Use stored parameters if warm_start and already fitted.
        b_init: float = self.intercept_
        w_init: NDArray = self.w_.copy()
        V_init: NDArray = self.V_.copy()
        mu_w_init: float = self.mu_w
        mu_V_init: NDArray = self.mu_V.copy()

        # Warm up the sampler.
        if self.n_warmup > 0:
            b_samples, w_samples, V_samples, _, mu_w, mu_V = _sample(
                self.n_warmup,
                X,
                y_centered,
                b_init,
                w_init,
                V_init,
                mu_w_init,
                mu_V_init,
                self.mu_b,
                self.sigma2_b,
                self.alpha_sigma,
                self.beta_sigma,
                self.m_0,
                self.tau2_0,
                self.alpha_0,
                self.beta_0,
                show_progress_bar=self.show_progress_bar,
                progress_bar_desc="Warming",
                log_per=self.log_per,
                rng=self.rng,
            )
            b_init = b_samples[-1]
            w_init = w_samples[-1]
            V_init = V_samples[-1]
            mu_w_init = mu_w
            mu_V_init = mu_V

        # Run the sampler.
        if self.n_mcs > 0:
            b_samples, w_samples, V_samples, sigma2_samples, mu_w, mu_V = _sample(
                self.n_mcs,
                X,
                y_centered,
                b_init,
                w_init,
                V_init,
                mu_w_init,
                mu_V_init,
                self.mu_b,
                self.sigma2_b,
                self.alpha_sigma,
                self.beta_sigma,
                self.m_0,
                self.tau2_0,
                self.alpha_0,
                self.beta_0,
                show_progress_bar=self.show_progress_bar,
                progress_bar_desc="Sampling",
                log_per=self.log_per,
                rng=self.rng,
            )

        if self.save_samples:
            self.samples["b"] = b_samples + y_mean
            self.samples["w"] = w_samples
            self.samples["V"] = V_samples
            self.samples["sigma2"] = sigma2_samples

        # Use the final sample as the parameter estimates.
        b_final = b_samples[-1]
        w_final = w_samples[-1]
        V_final = V_samples[-1]

        # Adjust intercept by adding back the y mean.
        self.intercept_ = b_final + y_mean
        self.w_ = w_final
        self.V_ = V_final

        self.is_fitted = True
        return self

    def to_bqm(self) -> dimod.BinaryQuadraticModel:
        """Converts the fitted model to a BinaryQuadraticModel.

        Returns:
            dimod.BinaryQuadraticModel: The model as a BinaryQuadraticModel.
        """
        d = len(self.w_)

        linear = {i: w for i, w in enumerate(self.w_)}

        i_idx, j_idx = np.triu_indices(d, k=1)
        quadratic = {}
        for i, j in zip(i_idx, j_idx):
            quadratic[(i, j)] = self.V_[i, :] @ self.V_[j, :]

        offset = self.intercept_
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, offset, dimod.BINARY)
        return bqm

    def predict(self, x: NDArray) -> NDArray | float:
        """Predicts the target value for the input feature vector.

        Args:
            x (NDArray): Input feature vector of shape (d,).

        Returns:
            NDArray: Predicted target value.
        """
        return fm_fast(x, self.intercept_, self.w_, self.V_)
