import numpy as np


class DieboldLiCurve:
    def __init__(
        self, beta0: float, beta1: float, beta2: float, lambda_: float
    ) -> None:
        self.beta0 = beta0
        self.beta1 = beta1
        self.beta2 = beta2
        self.lambda_ = lambda_

    def yield_curve(self, tau: np.ndarray) -> np.ndarray:
        small_value = 1e-6  
        tau = np.maximum(tau, small_value)

        term1 = self.beta0
        term2 = self.beta1 * (1 - np.exp(-self.lambda_ * tau)) / (self.lambda_ * tau)
        term3 = self.beta2 * (
            (1 - np.exp(-self.lambda_ * tau)) / (self.lambda_ * tau)
            - np.exp(-self.lambda_ * tau)
        )
        return term1 + term2 + term3

    def __call__(self, tau: np.ndarray) -> np.ndarray:
        return self.yield_curve(tau)
