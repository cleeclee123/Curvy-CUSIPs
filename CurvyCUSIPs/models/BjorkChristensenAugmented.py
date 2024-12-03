import numpy as np


class BjorkChristensenAugmentedCurve:
    def __init__(
        self,
        beta0: float,
        beta1: float,
        beta2: float,
        beta3: float,
        beta4: float,
        tau: float,
    ) -> None:
        self.beta0 = beta0
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.beta4 = beta4
        self.tau = tau

    def _time_decay(self, t: np.ndarray) -> np.ndarray:
        return self.beta1 * (t / (2 * self.tau))

    def _hump(self, t: np.ndarray) -> np.ndarray:
        return self.beta2 * ((1 - np.exp(-t / self.tau)) / (t / self.tau))

    def _second_hump(self, t: np.ndarray) -> np.ndarray:
        return self.beta3 * (
            (1 - np.exp(-t / self.tau)) / (t / self.tau) - np.exp(-t / self.tau)
        )

    def _third_hump(self, t: np.ndarray) -> np.ndarray:
        return self.beta4 * ((1 - np.exp(-2 * t / self.tau)) / (2 * t / self.tau))

    def factor_matrix(self, t: np.ndarray) -> np.ndarray:
        """Create the factor matrix for OLS."""
        factor1 = self._time_decay(t)
        factor2 = self._hump(t)
        factor3 = self._second_hump(t)
        factor4 = self._third_hump(t)
        constant = np.ones(t.shape)
        return np.column_stack([constant, factor1, factor2, factor3, factor4])

    def zero(self, t: np.ndarray) -> np.ndarray:
        """Calculate the zero rate for a given maturity."""
        return (
            self.beta0
            + self._time_decay(t)
            + self._hump(t)
            + self._second_hump(t)
            + self._third_hump(t)
        )

    def __call__(self, t: np.ndarray) -> np.ndarray:
        """Calculate the zero rate when the instance is called directly."""
        return self.zero(t)
