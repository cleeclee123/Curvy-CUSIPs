# adapted from https://github.com/luphord/nelson_siegel_svensson/blob/master/nelson_siegel_svensson/calibrate.py
# -*- coding: utf-8 -*-

"""Calibration methods for Nelson-Siegel(-Svensson) Models.
See `calibrate_ns_ols` and `calibrate_nss_ols` for ordinary least squares
(OLS) based methods.
"""

from typing import Tuple, Any, List

import numpy as np
from numpy.linalg import lstsq
from scipy.optimize import minimize

from models.NelsonSiegel import NelsonSiegelCurve
from models.NelsonSiegelSvensson import NelsonSiegelSvenssonCurve
from models.BjorkChristensen import BjorkChristensenCurve
from models.BjorkChristensenAugmented import BjorkChristensenAugmentedCurve
from models.DieboldLi import DieboldLiCurve


def _assert_same_shape(t: np.ndarray, y: np.ndarray) -> None:
    assert t.shape == y.shape, "Mismatching shapes of time and values"


def betas_ns_ols(
    tau: float, t: np.ndarray, y: np.ndarray
) -> Tuple[NelsonSiegelCurve, Any]:
    """Calculate the best-fitting beta-values given tau
    for time-value pairs t and y and return a corresponding
    Nelson-Siegel curve instance.
    """
    _assert_same_shape(t, y)
    curve = NelsonSiegelCurve(0, 0, 0, tau)
    factors = curve.factor_matrix(t)
    lstsq_res = lstsq(factors, y, rcond=None)
    beta = lstsq_res[0]
    return NelsonSiegelCurve(beta[0], beta[1], beta[2], tau), lstsq_res


def errorfn_ns_ols(tau: float, t: np.ndarray, y: np.ndarray) -> float:
    """Sum of squares error function for a Nelson-Siegel model and
    time-value pairs t and y. All betas are obtained by ordinary
    least squares given tau.
    """
    _assert_same_shape(t, y)
    curve, lstsq_res = betas_ns_ols(tau, t, y)
    return np.sum((curve(t) - y) ** 2)


def calibrate_ns_ols(
    t: np.ndarray, y: np.ndarray, tau0: float = 2.0
) -> Tuple[NelsonSiegelCurve, Any]:
    """Calibrate a Nelson-Siegel curve to time-value pairs
    t and y, by optimizing tau and chosing all betas
    using ordinary least squares.
    """
    _assert_same_shape(t, y)
    opt_res = minimize(errorfn_ns_ols, x0=tau0, args=(t, y))
    curve, lstsq_res = betas_ns_ols(opt_res.x[0], t, y)
    return curve, opt_res


def empirical_factors(
    y_3m: float, y_2y: float, y_10y: float
) -> Tuple[float, float, float]:
    """Calculate the empirical factors according to
    Diebold and Li (2006)."""
    return y_10y, y_10y - y_3m, 2 * y_2y - y_3m - y_10y


def betas_nss_ols(
    tau: Tuple[float, float], t: np.ndarray, y: np.ndarray
) -> Tuple[NelsonSiegelSvenssonCurve, Any]:
    """Calculate the best-fitting beta-values given tau (= array of tau1
    and tau2) for time-value pairs t and y and return a corresponding
    Nelson-Siegel-Svensson curve instance.
    """
    _assert_same_shape(t, y)
    curve = NelsonSiegelSvenssonCurve(0, 0, 0, 0, tau[0], tau[1])
    factors = curve.factor_matrix(t)
    lstsq_res = lstsq(factors, y, rcond=None)
    beta = lstsq_res[0]
    return (
        NelsonSiegelSvenssonCurve(beta[0], beta[1], beta[2], beta[3], tau[0], tau[1]),
        lstsq_res,
    )


def errorfn_nss_ols(tau: Tuple[float, float], t: np.ndarray, y: np.ndarray) -> float:
    """Sum of squares error function for a Nelson-Siegel-Svensson
    model and time-value pairs t and y. All betas are obtained
    by ordinary least squares given tau (= array of tau1
    and tau2).
    """
    _assert_same_shape(t, y)
    curve, lstsq_res = betas_nss_ols(tau, t, y)
    return np.sum((curve(t) - y) ** 2)


def calibrate_nss_ols(
    t: np.ndarray, y: np.ndarray, tau0: Tuple[float, float] = (2.0, 5.0)
) -> Tuple[NelsonSiegelSvenssonCurve, Any]:
    """Calibrate a Nelson-Siegel-Svensson curve to time-value
    pairs t and y, by optimizing tau1 and tau2 and chosing
    all betas using ordinary least squares. This method does
    not work well regarding the recovery of true parameters.
    """
    _assert_same_shape(t, y)
    opt_res = minimize(errorfn_nss_ols, x0=np.array(tau0), args=(t, y))
    curve, lstsq_res = betas_nss_ols(opt_res.x, t, y)
    return curve, opt_res


def errorfn_bc_ols(tau: float, t: np.ndarray, y: np.ndarray) -> float:
    """Error function for the Bjork-Christensen OLS calibration."""
    curve, _ = betas_bc_ols(tau, t, y)
    estimated_yields = curve(t)
    return np.sum((estimated_yields - y) ** 2)


def betas_bc_ols(
    tau: float, t: np.ndarray, y: np.ndarray
) -> Tuple[BjorkChristensenCurve, np.linalg.LinAlgError]:
    """Compute the beta parameters for the Bjork-Christensen model using OLS."""
    curve = BjorkChristensenCurve(0, 0, 0, 0, tau)
    F = curve.factor_matrix(t)
    betas, _, _, _ = np.linalg.lstsq(F, y, rcond=None)
    curve.beta0, curve.beta1, curve.beta2, curve.beta3 = betas
    return curve, betas


def calibrate_bc_ols(
    t: np.ndarray, y: np.ndarray, tau0: float = 1.0
) -> Tuple[BjorkChristensenCurve, Any]:
    """Calibrate a Bjork-Christensen curve to time-value pairs t and y."""
    _assert_same_shape(t, y)
    opt_res = minimize(errorfn_bc_ols, x0=np.array([tau0]), args=(t, y))
    curve, lstsq_res = betas_bc_ols(opt_res.x[0], t, y)
    return curve, opt_res


def calibrate_bc_augmented_ols(
    maturities: List[float], yields: List[float]
) -> Tuple[BjorkChristensenAugmentedCurve, Any]:
    """Fit the Bjork-Christensen Augmented model to the given yields."""

    def objective(params: List[float]) -> float:
        beta0, beta1, beta2, beta3, beta4, tau = params
        curve = BjorkChristensenAugmentedCurve(beta0, beta1, beta2, beta3, beta4, tau)
        model_yields = curve(np.array(maturities))
        return np.sum((model_yields - np.array(yields)) ** 2)

    # Initial guess for the parameters
    initial_params = [0.01, 0.01, 0.01, 0.01, 0.01, 1.0]
    bounds = [
        (0, None),
        (None, None),
        (None, None),
        (None, None),
        (None, None),
        (0, None),
    ]

    result = minimize(objective, initial_params, method="L-BFGS-B", bounds=bounds)
    fitted_params = result.x
    curve = BjorkChristensenAugmentedCurve(*fitted_params)
    return curve, result


def calibrate_diebold_li_ols(
    maturities: List[float], yields: List[float]
) -> Tuple[DieboldLiCurve, Any]:
    """Fit the Diebold-Li model to the given yields."""
    # Initial guesses for beta0, beta1, beta2, and lambda_
    initial_guess = [np.mean(yields), -0.02, 0.02, 0.1]

    # Objective function to minimize (sum of squared errors)
    def objective(params: List[float]) -> float:
        curve = DieboldLiCurve(params[0], params[1], params[2], params[3])
        return np.sum((curve(np.array(maturities)) - np.array(yields)) ** 2)

    # Use scipy's optimization routine to minimize the objective function
    result = minimize(objective, initial_guess, method="BFGS")

    # Extract the optimized parameters and create the calibrated yield curve model
    optimized_params = result.x
    curve = DieboldLiCurve(*optimized_params)
    return curve, result
