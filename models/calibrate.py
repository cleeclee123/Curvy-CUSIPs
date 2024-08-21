# adapted from https://github.com/luphord/nelson_siegel_svensson/blob/master/nelson_siegel_svensson/calibrate.py
# -*- coding: utf-8 -*-

from typing import Any, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from models.BjorkChristensen import BjorkChristensenCurve
from models.BjorkChristensenAugmented import BjorkChristensenAugmentedCurve
from models.DieboldLi import DieboldLiCurve
from models.MLESM import MerrillLynchExponentialSplineModel
from models.NelsonSiegel import NelsonSiegelCurve
from models.NelsonSiegelSvensson import NelsonSiegelSvenssonCurve
from models.PCA import PCACurve
from models.SmithWilson import SmithWilsonCurve, find_ufr_ytm
from numpy.linalg import lstsq
from scipy.optimize import OptimizeResult, minimize


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
    maturities: npt.NDArray[np.float64], yields: npt.NDArray[np.float64]
) -> Tuple[BjorkChristensenAugmentedCurve, Any]:
    """Fit the Bjork-Christensen Augmented model to the given yields."""

    def objective(params: npt.NDArray[np.float64]) -> float:
        beta0, beta1, beta2, beta3, beta4, tau = params
        curve = BjorkChristensenAugmentedCurve(beta0, beta1, beta2, beta3, beta4, tau)
        model_yields = curve(np.array(maturities))
        return np.sum((model_yields - np.array(yields)) ** 2)

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
    maturities: npt.NDArray[np.float64], yields: npt.NDArray[np.float64]
) -> Tuple[DieboldLiCurve, Any]:
    """Fit the Diebold-Li model to the given yields."""
    # Initial guesses for beta0, beta1, beta2, and lambda_
    initial_guess = [np.mean(yields), -0.02, 0.02, 0.1]

    def objective(params: npt.NDArray[np.float64]) -> float:
        curve = DieboldLiCurve(params[0], params[1], params[2], params[3])
        return np.sum((curve(np.array(maturities)) - np.array(yields)) ** 2)

    result = minimize(objective, initial_guess, method="BFGS")

    optimized_params = result.x
    curve = DieboldLiCurve(*optimized_params)
    return curve, result


def calibrate_mles_ols(
    maturities: npt.NDArray[np.float64],
    yields: npt.NDArray[np.float64],
    N: int = 8,
    regularization: float = 1e-4,
    overnight_rate: Optional[float] = None,
) -> Tuple[MerrillLynchExponentialSplineModel, Any]:
    if maturities[0] != 0:
        short_rate = overnight_rate or yields[0]
        maturities = np.insert(maturities, 0, 0)
        yields = np.insert(yields, 0, short_rate)

    """Fit the MLES model to the given yields using OLS."""
    initial_guess = [0.1] + [1.0] * N

    def objective(params: npt.NDArray[np.float64]) -> float:
        alpha = params[0]
        lambda_hat = np.array(params[1:])
        curve = MerrillLynchExponentialSplineModel(alpha, N, lambda_hat)
        curve.fit(np.array(maturities), np.array(yields), np.eye(len(maturities)))

        theoretical_yields = curve.theoretical_yields(np.array(maturities))

        # Regularization to penalize large gradients (especially at the front end)
        regularization_term = regularization * np.sum(np.diff(lambda_hat) ** 2)

        return (
            np.sum((theoretical_yields - np.array(yields)) ** 2) + regularization_term
        )

    # Optimize alpha and lambda_hat
    result = minimize(objective, initial_guess, method="BFGS")

    optimized_params = result.x
    optimized_alpha = optimized_params[0]
    optimized_lambda_hat = np.array(optimized_params[1:])
    curve = MerrillLynchExponentialSplineModel(optimized_alpha, N, optimized_lambda_hat)
    curve.fit(np.array(maturities), np.array(yields), np.eye(len(maturities)))

    return curve, result


def calibrate_smith_wilson_ols(
    maturities: npt.NDArray[np.float64],
    yields: npt.NDArray[np.float64],
    ufr: Optional[float] = None,
    alpha_initial: float = 0.1,
    overnight_rate: Optional[float] = None,
) -> Tuple[MerrillLynchExponentialSplineModel, Any]:
    if maturities[0] != 0:
        short_rate = overnight_rate or yields[0]
        maturities = np.insert(maturities, 0, 1 / 365)
        yields = np.insert(yields, 0, short_rate)

    if not ufr:
        ufr = find_ufr_ytm(maturities=maturities, ytms=yields)

    def objective(alpha: float) -> float:
        curve = SmithWilsonCurve(ufr, alpha)
        curve.fit(yields, maturities)
        fitted_yields = curve(maturities)
        return np.sum((fitted_yields - yields) ** 2)

    result = minimize(
        objective, x0=alpha_initial, bounds=[(0.01, 1.0)], method="L-BFGS-B"
    )

    optimal_alpha = result.x[0]
    calibrated_curve = SmithWilsonCurve(ufr, optimal_alpha)
    calibrated_curve.fit(yields, maturities)

    return calibrated_curve, result


def calibrate_pca_yield_curve(
    historical_df: pd.DataFrame, n_components: int = 3, use_changes: bool = False
) -> Tuple[PCACurve, Any]:
    if use_changes:
        historical_df = historical_df.diff().dropna()
    pca_model = PCACurve(n_components=n_components)
    pca_model.fit(historical_df)
    return pca_model, pca_model.explained_variance
