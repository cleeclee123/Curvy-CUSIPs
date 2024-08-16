from dataclasses import dataclass
from typing import Union, Tuple, Any
import numpy as np
from scipy.optimize import minimize
from numbers import Real

EPS = np.finfo(float).eps


@dataclass
class BjorkChristensenCurve:
    """Implementation of the Bjork-Christensen interest rate curve model."""

    beta0: float
    beta1: float
    beta2: float
    beta3: float
    tau: float

    def factors(
        self, T: Union[float, np.ndarray]
    ) -> Union[Tuple[float, float, float], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Factor loadings for time(s) T, excluding constant."""
        T = np.maximum(T, EPS)  # Ensure T is not zero or negative to avoid instability

        factor1 = (1 - np.exp(-T / self.tau)) / (T / self.tau)
        factor2 = factor1 - np.exp(-T / self.tau)
        factor3 = (1 - np.exp(-2 * T / self.tau)) / (2 * T / self.tau)

        return factor1, factor2, factor3

    def factor_matrix(self, T: Union[float, np.ndarray]) -> np.ndarray:
        """Factor loadings for time(s) T as matrix columns, including constant column (=1.0)."""
        factor1, factor2, factor3 = self.factors(T)
        constant = np.ones_like(T)
        return np.stack([constant, factor1, factor2, factor3], axis=-1)

    def zero(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Zero rate(s) of this curve at time(s) T."""
        factor1, factor2, factor3 = self.factors(T)
        return (
            self.beta0
            + self.beta1 * factor1
            + self.beta2 * factor2
            + self.beta3 * factor3
        )

    def __call__(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Zero rate(s) of this curve at time(s) T."""
        return self.zero(T)

    def forward(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Instantaneous forward rate(s) of this curve at time(s) T."""
        exp_tt1 = np.exp(-T / self.tau)
        exp_tt2 = np.exp(-2 * T / self.tau)
        return (
            self.beta0
            + self.beta1 * exp_tt1
            + self.beta2 * exp_tt1 * T / self.tau
            + self.beta3 * exp_tt2 * T / (2 * self.tau)
        )
