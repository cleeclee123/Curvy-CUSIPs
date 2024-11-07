from typing import List, Optional, Union

import numpy as np
from scipy import optimize
from scipy.optimize import minimize


class SmithWilsonCurve:
    def __init__(self, ufr: float, alpha: Optional[float] = None):
        self.ufr = ufr  # Ultimate Forward Rate (annualized/annual compounding)
        self.alpha = alpha  # Convergence speed parameter
        self.zeta = None  # To be determined after fitting the model
        self.t_obs = None  # Observed maturities
        self.rates_obs = None  # Observed zero-coupon rates

    @staticmethod
    def calculate_prices(
        rates: Union[np.ndarray, List[float]], t: Union[np.ndarray, List[float]]
    ) -> np.ndarray:
        rates = np.array(rates)
        t = np.array(t)
        return np.power(1 + rates, -t)

    def ufr_discount_factor(self, t: Union[np.ndarray, List[float]]) -> np.ndarray:
        ufr_log = np.log(1 + self.ufr)
        t = np.array(t)
        return np.exp(-ufr_log * t)

    def wilson_function(
        self, t1: Union[np.ndarray, List[float]], t2: Union[np.ndarray, List[float]]
    ) -> np.ndarray:
        t1 = np.array(t1).reshape(-1, 1)  # Ensure t1 is a column vector
        t2 = np.array(t2).reshape(-1, 1)  # Ensure t2 is a column vector

        m = len(t1)
        n = len(t2)

        t1_Mat = np.repeat(t1, n, axis=1)
        t2_Mat = np.repeat(t2, m, axis=1).T

        min_t = np.minimum(t1_Mat, t2_Mat)
        max_t = np.maximum(t1_Mat, t2_Mat)

        ufr_disc = self.ufr_discount_factor(t=(t1_Mat + t2_Mat))
        W = ufr_disc * (
            self.alpha * min_t
            - 0.5
            * np.exp(-self.alpha * max_t)
            * (np.exp(self.alpha * min_t) - np.exp(-self.alpha * min_t))
        )
        return W

    def fit_parameters(
        self, rates: Union[np.ndarray, List[float]], t: Union[np.ndarray, List[float]]
    ) -> np.ndarray:
        W = self.wilson_function(t1=t, t2=t)
        mu = self.ufr_discount_factor(t=t)
        P = self.calculate_prices(rates=rates, t=t)

        zeta = np.linalg.inv(W) @ (mu - P)
        self.zeta = zeta
        return zeta

    def fit_smithwilson_rates(
        self,
        rates_obs: Union[np.ndarray, List[float]],
        t_obs: Union[np.ndarray, List[float]],
        t_target: Union[np.ndarray, List[float]],
    ) -> np.ndarray:
        rates_obs = np.array(rates_obs).reshape((-1, 1))
        t_obs = np.array(t_obs).reshape((-1, 1))
        t_target = np.array(t_target).reshape((-1, 1))

        if self.alpha is None:
            self.alpha = self.fit_convergence_parameter(
                rates_obs=rates_obs, t_obs=t_obs
            )

        self.fit_parameters(rates=rates_obs, t=t_obs)
        ufr_disc = self.ufr_discount_factor(t=t_target)
        W = self.wilson_function(t1=t_target, t2=t_obs)

        P = ufr_disc - W @ self.zeta
        return np.power(1 / P, 1 / t_target) - 1

    def fit_convergence_parameter(
        self,
        rates_obs: Union[np.ndarray, List[float]],
        t_obs: Union[np.ndarray, List[float]],
    ) -> float:
        llp = np.max(t_obs)
        convergence_t = max(llp + 40, 60)

        def forward_difference(alpha: float):
            # Set alpha and fit the model
            self.alpha = alpha
            rates = self.fit_smithwilson_rates(
                rates_obs=rates_obs,
                t_obs=t_obs,
                t_target=[convergence_t, convergence_t + 1],
            )

            forward_rate = (1 + rates[1]) ** (convergence_t + 1) / (
                1 + rates[0]
            ) ** convergence_t - 1
            return -abs(forward_rate - self.ufr) + 1 / 10_000

        result = optimize.minimize(
            lambda alpha: alpha,
            x0=0.15,
            method="SLSQP",
            bounds=[[0.05, 1.0]],
            constraints=[{"type": "ineq", "fun": forward_difference}],
            options={"ftol": 1e-6, "disp": True},
        )

        return float(result.x)

    def fit(
        self,
        rates_obs: Union[np.ndarray, List[float]],
        t_obs: Union[np.ndarray, List[float]],
    ):
        self.t_obs = t_obs
        self.rates_obs = rates_obs
        if self.alpha is None:
            self.alpha = self.fit_convergence_parameter(
                rates_obs=rates_obs, t_obs=t_obs
            )
        self.fit_parameters(rates=rates_obs, t=t_obs)

    def interpolate(self, t_target: Union[np.ndarray, List[float]]) -> np.ndarray:
        return self.fit_smithwilson_rates(self.rates_obs, self.t_obs, t_target)

    def __call__(self, t_target: Union[np.ndarray, List[float]]) -> np.ndarray:
        return self.interpolate(t_target)


# https://www.sciencedirect.com/science/article/pii/S0167668723000963
def find_ufr_ytm(
    maturities, ytms, alpha_min=0.05, tau=0.0001, max_iter=100, ufr_guess=0.04
):
    def wilson_function_matrix(alpha, t):
        # Compute Wilson function matrix W_alpha for given alpha and maturities t
        W = np.zeros((len(t), len(t)))
        for i in range(len(t)):
            for j in range(len(t)):
                W[i, j] = (
                    alpha * min(t[i], t[j])
                    - 0.5 * np.exp(-alpha * abs(t[i] - t[j]))
                    + 0.5 * np.exp(-alpha * (t[i] + t[j]))
                )
        return W

    def objective(ufr, alpha):
        W_alpha = wilson_function_matrix(alpha, maturities)
        discount_factors = np.exp(-ytms * np.array(maturities))
        implied_prices = np.exp(-ufr * np.array(maturities))

        # Ensure the smoothness and correct the discount curve using W_alpha
        residuals = discount_factors - implied_prices
        penalty = np.sum((residuals**2) + alpha**2 * np.dot(W_alpha, residuals) ** 2)

        # Additional penalty based on the convergence condition with tau
        forward_rate_convergence = np.abs(
            residuals[-1]
        )  # Assume last residual represents forward rate convergence
        if forward_rate_convergence > tau:
            penalty += (
                1e6 * (forward_rate_convergence - tau) ** 2
            )  # Large penalty if it exceeds tau

        return penalty

    def find_optimal_alpha(ufr):
        result = minimize(
            lambda alpha: objective(ufr, alpha),
            x0=0.1,
            bounds=[(alpha_min, 1.0)],
            options={"maxiter": max_iter},
        )
        return result.x[0]

    alpha = find_optimal_alpha(ufr_guess)

    result = minimize(
        lambda ufr: objective(ufr, alpha),
        x0=ufr_guess,
        bounds=[(0, None)],
        options={"maxiter": max_iter},
    )
    ufr_optimal = result.x[0]

    return ufr_optimal
