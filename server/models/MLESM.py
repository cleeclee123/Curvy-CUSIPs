# reference:
# https://publications.gc.ca/site/archivee-archived.html?url=https://publications.gc.ca/collections/collection_2014/banque-bank-canada/FB3-2-102-29-eng.pdf


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from dataclasses import dataclass


"""Implementation of the Merrill Lynch Exponential Spline interest rate curve model."""


@dataclass
class MerrillLynchExponentialSplineModel:

    alpha: float
    N: int
    lambda_hat: np.ndarray

    def e_basis(self, k: int, T: float) -> float:
        return np.exp(-self.alpha * k * T)

    def discount(self, T: float) -> float:
        return sum(self.lambda_hat[k] * self.e_basis(k, T) for k in range(self.N))

    def construct_H(self, T: np.ndarray) -> np.ndarray:
        H = np.zeros((len(T), self.N))
        for j in range(len(T)):
            for k in range(self.N):
                H[j, k] = self.e_basis(k, T[j])
        return H

    def gls(self, H: np.ndarray, W: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Generalized Least Squares (GLS) estimate of the MLES basis parameters."""
        return np.linalg.inv(H.T @ W @ H) @ H.T @ W @ p

    def fit(self, maturities: np.ndarray, yields: np.ndarray, W: np.ndarray):
        """Fits the model to observed yields."""
        prices = np.exp(-yields * maturities / 100)
        H = self.construct_H(maturities)
        # Perform GLS estimation
        self.lambda_hat = self.gls(H, W, prices)

    def theoretical_yields(self, maturities: np.ndarray) -> np.ndarray:
        prices = np.array([self.discount(T) for T in maturities])
        return -np.log(prices) / maturities * 100

    def __call__(self, maturities: np.ndarray) -> np.ndarray:
        return self.theoretical_yields(maturities)
