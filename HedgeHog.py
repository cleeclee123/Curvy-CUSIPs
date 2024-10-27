import warnings
from datetime import datetime, timedelta
from typing import Annotated, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy
import seaborn as sns
import statsmodels.api as sm
import ujson as json
from scipy.odr import ODR, Data, Model, Output, RealData
from scipy.optimize import minimize
from scipy.stats import linregress, tstd, zscore
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from termcolor import colored

from CurveBuilder import calc_ust_impl_spot_n_fwd_curve, calc_ust_metrics
from CurveDataFetcher import CurveDataFetcher
from utils.arbitragelab import JohansenPortfolio, construct_spread, EngleGrangerPortfolio


def dv01_neutral_steepener_hegde_ratio(
    as_of_date: datetime,
    front_leg_bond_row: Dict | pd.Series,
    back_leg_bond_row: Dict | pd.Series,
    curve_data_fetcher: CurveDataFetcher,
    scipy_interp_curve: scipy.interpolate.interpolate,
    repo_rate: float,
    quote_type: Optional[str] = "eod",
    front_leg_par_amount: Optional[int] = None,
    back_leg_par_amount: Optional[int] = None,
    verbose: Optional[bool] = True,
    very_verbose: Optional[bool] = False,
):
    front_leg_info = curve_data_fetcher.ust_data_fetcher.cusip_to_ust_label(cusip=front_leg_bond_row["cusip"])
    back_leg_info = curve_data_fetcher.ust_data_fetcher.cusip_to_ust_label(cusip=back_leg_bond_row["cusip"])

    front_leg_metrics = calc_ust_metrics(
        bond_info=front_leg_info,
        curr_price=front_leg_bond_row[f"{quote_type}_price"],
        curr_ytm=front_leg_bond_row[f"{quote_type}_yield"],
        as_of_date=as_of_date,
        scipy_interp=scipy_interp_curve,
        on_rate=repo_rate,
    )
    back_leg_metrics = calc_ust_metrics(
        bond_info=back_leg_info,
        curr_price=back_leg_bond_row[f"{quote_type}_price"],
        curr_ytm=back_leg_bond_row[f"{quote_type}_yield"],
        as_of_date=as_of_date,
        scipy_interp=scipy_interp_curve,
        on_rate=repo_rate,
    )

    front_leg_ttm: float = (front_leg_info["maturity_date"] - as_of_date).days / 365
    back_leg_ttm: float = (back_leg_info["maturity_date"] - as_of_date).days / 365
    impl_spot_3m_fwds = calc_ust_impl_spot_n_fwd_curve(n=0.25, scipy_interp_curve=scipy_interp_curve, return_scipy=True)
    impl_spot_6m_fwds = calc_ust_impl_spot_n_fwd_curve(n=0.5, scipy_interp_curve=scipy_interp_curve, return_scipy=True)
    impl_spot_12m_fwds = calc_ust_impl_spot_n_fwd_curve(n=1, scipy_interp_curve=scipy_interp_curve, return_scipy=True)

    if very_verbose:
        print("Front Leg Info: ")
        print(front_leg_bond_row)
        print(front_leg_metrics)
        print("Back Leg Info: ")
        print(back_leg_bond_row)
        print(back_leg_metrics)

    if front_leg_bond_row["rank"] == 0 and back_leg_bond_row["rank"] == 0:
        print(f"{front_leg_bond_row["original_security_term"].split("-")[0]}s{back_leg_bond_row["original_security_term"].split("-")[0]}s")
    print(f"{back_leg_bond_row["ust_label"]} - {front_leg_bond_row["ust_label"]}") if verbose else None

    hr = back_leg_metrics["bps"] / front_leg_metrics["bps"]
    print("Hedge Ratio (relative to backleg): ", "\x1b[0;30;47m", hr, "\x1b[0m") if verbose else None

    if front_leg_par_amount and back_leg_par_amount:
        print("Both Leg Par Amounts passed in - using backleg par amount")
        front_leg_par_amount = None

    if not front_leg_par_amount and not back_leg_par_amount:
        back_leg_par_amount = 1_000_000

    if back_leg_par_amount:
        print(
            f"{front_leg_bond_row["ust_label"]} === {front_leg_bond_row["original_security_term"]}, TTM = {front_leg_bond_row["time_to_maturity"]:3f} (Frontleg) Par Amount = {back_leg_par_amount * hr:_}"
            if verbose
            else None
        )
        print(
            f"{back_leg_bond_row["ust_label"]} === {back_leg_bond_row["original_security_term"]}, TTM = {back_leg_bond_row["time_to_maturity"]:3f} (Backleg) Par Amount = {back_leg_par_amount:_}"
            if verbose
            else None
        )
        front_leg_par_amount = back_leg_par_amount * hr

    elif front_leg_par_amount:
        print(
            f"{front_leg_bond_row["ust_label"]} === {front_leg_bond_row["original_security_term"]}, TTM = {front_leg_bond_row["time_to_maturity"]:3f} (Frontleg) Par Amount = {front_leg_par_amount:_}"
            if verbose
            else None
        )
        print(
            f"{back_leg_bond_row["ust_label"]} === {back_leg_bond_row["original_security_term"]}, TTM = {back_leg_bond_row["time_to_maturity"]:3f} (Backleg) Par Amount = {front_leg_par_amount / hr:_}"
            if verbose
            else None
        )
        back_leg_par_amount = front_leg_par_amount / hr

    return {
        "curr_spread": (back_leg_bond_row[f"{quote_type}_yield"] - front_leg_bond_row[f"{quote_type}_yield"]) * 100,
        "rough_3m_impl_fwd_spread": (impl_spot_3m_fwds(back_leg_ttm) - impl_spot_3m_fwds(front_leg_ttm)) * 100,
        "rough_6m_impl_fwd_spread": (impl_spot_6m_fwds(back_leg_ttm) - impl_spot_6m_fwds(front_leg_ttm)) * 100,
        "rough_12m_impl_fwd_spread": (impl_spot_12m_fwds(back_leg_ttm) - impl_spot_12m_fwds(front_leg_ttm)) * 100,
        "front_leg_metrics": front_leg_metrics,
        "back_leg_metrics": back_leg_metrics,
        "hedge_ratio": hr,
        "front_leg_par_amount": front_leg_par_amount,
        "back_leg_par_amount": back_leg_par_amount,
        "spread_dv01": np.abs(back_leg_metrics["bps"] * back_leg_par_amount / 100),
        "rough_3m_carry_roll": (back_leg_metrics["rough_carry"] + back_leg_metrics["rough_3m_rolldown"])
        - hr * (front_leg_metrics["rough_carry"] + front_leg_metrics["rough_3m_rolldown"]),
        "rough_6m_carry_roll": (back_leg_metrics["rough_carry"] + back_leg_metrics["rough_6m_rolldown"])
        - hr * (front_leg_metrics["rough_carry"] + front_leg_metrics["rough_6m_rolldown"]),
        "rough_12m_carry_roll": (back_leg_metrics["rough_carry"] + back_leg_metrics["rough_12m_rolldown"])
        - hr * (front_leg_metrics["rough_carry"] + front_leg_metrics["rough_12m_rolldown"]),
    }


# reference point is buying the belly => fly spread down
def dv01_neutral_butterfly_hegde_ratio(
    as_of_date: datetime,
    front_wing_bond_row: Dict | pd.Series,
    belly_bond_row: Dict | pd.Series,
    back_wing_bond_row: Dict | pd.Series,
    curve_data_fetcher: CurveDataFetcher,
    scipy_interp_curve: scipy.interpolate.interpolate,
    repo_rate: float,
    quote_type: Optional[str] = "eod",
    front_wing_par_amount: Optional[int] = None,
    belly_par_amount: Optional[int] = None,
    back_wing_par_amount: Optional[int] = None,
    verbose: Optional[bool] = True,
    very_verbose: Optional[bool] = False,
):
    front_wing_info = curve_data_fetcher.ust_data_fetcher.cusip_to_ust_label(cusip=front_wing_bond_row["cusip"])
    belly_info = curve_data_fetcher.ust_data_fetcher.cusip_to_ust_label(cusip=belly_bond_row["cusip"])
    back_wing_info = curve_data_fetcher.ust_data_fetcher.cusip_to_ust_label(cusip=back_wing_bond_row["cusip"])

    front_wing_metrics = calc_ust_metrics(
        bond_info=front_wing_info,
        curr_price=front_wing_bond_row[f"{quote_type}_price"],
        curr_ytm=front_wing_bond_row[f"{quote_type}_yield"],
        as_of_date=as_of_date,
        scipy_interp=scipy_interp_curve,
        on_rate=repo_rate,
    )
    belly_metrics = calc_ust_metrics(
        bond_info=belly_info,
        curr_price=belly_bond_row[f"{quote_type}_price"],
        curr_ytm=belly_bond_row[f"{quote_type}_yield"],
        as_of_date=as_of_date,
        scipy_interp=scipy_interp_curve,
        on_rate=repo_rate,
    )
    back_wing_metrics = calc_ust_metrics(
        bond_info=back_wing_info,
        curr_price=back_wing_bond_row[f"{quote_type}_price"],
        curr_ytm=back_wing_bond_row[f"{quote_type}_yield"],
        as_of_date=as_of_date,
        scipy_interp=scipy_interp_curve,
        on_rate=repo_rate,
    )

    front_wing_ttm: float = (front_wing_info["maturity_date"] - as_of_date).days / 365
    belly_ttm: float = (belly_info["maturity_date"] - as_of_date).days / 365
    back_wing_ttm: float = (back_wing_info["maturity_date"] - as_of_date).days / 365
    impl_spot_3m_fwds = calc_ust_impl_spot_n_fwd_curve(n=0.25, scipy_interp_curve=scipy_interp_curve, return_scipy=True)
    impl_spot_6m_fwds = calc_ust_impl_spot_n_fwd_curve(n=0.5, scipy_interp_curve=scipy_interp_curve, return_scipy=True)
    impl_spot_12m_fwds = calc_ust_impl_spot_n_fwd_curve(n=1, scipy_interp_curve=scipy_interp_curve, return_scipy=True)

    if very_verbose:
        print("Front Wing Info: ")
        print(front_wing_info)
        print(front_wing_metrics)

        print("Belly Info: ")
        print(belly_info)
        print(belly_metrics)

        print("Back Wing Info: ")
        print(back_wing_info)
        print(back_wing_metrics)

    hedge_ratios = {
        "front_wing_hr": belly_metrics["bps"] / front_wing_metrics["bps"] / 2,
        "belly_hr": 1,
        "back_wing_hr": belly_metrics["bps"] / back_wing_metrics["bps"] / 2,
    }

    if verbose:
        if front_wing_bond_row["rank"] == 0 and belly_bond_row["rank"] == 0 and back_wing_bond_row["rank"] == 0:
            print(
                f"{front_wing_bond_row["original_security_term"].split("-")[0]}s{belly_bond_row["original_security_term"].split("-")[0]}s{back_wing_bond_row["original_security_term"].split("-")[0]}s"
            )

        print(f"{front_wing_bond_row["ust_label"]} - {belly_bond_row["ust_label"]} - {back_wing_bond_row["ust_label"]} Fly")
        print("\x1b[0;30;47m", "Normalized Belly Fly Hedge Ratio:", "\x1b[0m")
        print(json.dumps(hedge_ratios, indent=4))

        if belly_par_amount:
            front_wing_par_amount = (belly_metrics["bps"] / front_wing_metrics["bps"] / 2) * belly_par_amount
            belly_par_amount = belly_par_amount
            back_wing_par_amount = (belly_metrics["bps"] / back_wing_metrics["bps"] / 2) * belly_par_amount
        elif front_wing_par_amount:
            front_wing_par_amount = front_wing_par_amount
            belly_par_amount = (2 * front_wing_metrics["bps"] / belly_metrics["bps"]) * front_wing_par_amount
            back_wing_par_amount = (
                (belly_metrics["bps"] / back_wing_metrics["bps"] / 2) * (2 * front_wing_metrics["bps"] / belly_metrics["bps"]) * front_wing_par_amount
            )
        elif back_wing_par_amount:
            front_wing_par_amount = (
                (belly_metrics["bps"] / front_wing_metrics["bps"] / 2) * (2 * back_wing_metrics["bps"] / belly_metrics["bps"]) * back_wing_par_amount
            )
            belly_par_amount = (2 * back_wing_metrics["bps"] / belly_metrics["bps"]) * back_wing_par_amount
            back_wing_par_amount = back_wing_par_amount

        print(f"Front Wing Par Amount = {front_wing_par_amount:_}")
        print(f"Belly Par Amount = {belly_par_amount:_}")
        print(f"Back Wing Par Amount = {back_wing_par_amount:_}")

    return {
        "curr_spread": (
            (belly_bond_row[f"{quote_type}_yield"] - front_wing_bond_row[f"{quote_type}_yield"])
            - (back_wing_bond_row[f"{quote_type}_yield"] - belly_bond_row[f"{quote_type}_yield"])
        )
        * 100,
        "rough_3m_impl_fwd_spread": (
            (impl_spot_3m_fwds(belly_ttm) - impl_spot_3m_fwds(front_wing_ttm)) - (impl_spot_3m_fwds(back_wing_ttm) - impl_spot_3m_fwds(belly_ttm))
        )
        * 100,
        "rough_6m_impl_fwd_spread": (
            (impl_spot_6m_fwds(belly_ttm) - impl_spot_6m_fwds(front_wing_ttm)) - (impl_spot_6m_fwds(back_wing_ttm) - impl_spot_6m_fwds(belly_ttm))
        )
        * 100,
        "rough_12m_impl_fwd_spread": (
            (impl_spot_12m_fwds(belly_ttm) - impl_spot_12m_fwds(front_wing_ttm)) - (impl_spot_12m_fwds(back_wing_ttm) - impl_spot_12m_fwds(belly_ttm))
        )
        * 100,
        "front_wing_metrics": front_wing_metrics,
        "belly_metrics": belly_metrics,
        "back_wing_metrics": back_wing_metrics,
        "hedge_ratio": hedge_ratios,
        "front_wing_par_amount": front_wing_par_amount,
        "belly_par_amount": belly_par_amount,
        "back_leg_par_amount": back_wing_par_amount,
        "spread_dv01": np.abs(belly_metrics["bps"] * belly_par_amount / 100),
        "rough_3m_carry_roll": (belly_metrics["rough_carry"] + belly_metrics["rough_3m_rolldown"])
        - (hedge_ratios["front_wing_hr"] * (front_wing_metrics["rough_carry"] + front_wing_metrics["rough_3m_rolldown"]))
        - (hedge_ratios["back_wing_hr"] * (back_wing_metrics["rough_carry"] + back_wing_metrics["rough_3m_rolldown"])),
        "rough_6m_carry_roll": (belly_metrics["rough_carry"] + belly_metrics["rough_6m_rolldown"])
        - (hedge_ratios["front_wing_hr"] * (front_wing_metrics["rough_carry"] + front_wing_metrics["rough_6m_rolldown"]))
        - (hedge_ratios["back_wing_hr"] * (back_wing_metrics["rough_carry"] + back_wing_metrics["rough_6m_rolldown"])),
        "rough_12m_carry_roll": (belly_metrics["rough_carry"] + belly_metrics["rough_12m_rolldown"])
        - (hedge_ratios["front_wing_hr"] * (front_wing_metrics["rough_carry"] + front_wing_metrics["rough_12m_rolldown"]))
        - (hedge_ratios["back_wing_hr"] * (back_wing_metrics["rough_carry"] + back_wing_metrics["rough_12m_rolldown"])),
    }


def _run_odr(df: pd.DataFrame, x_cols: List[str], y_col: str, x_errs: Optional[npt.ArrayLike] = None, y_errs: Optional[npt.ArrayLike] = None):
    def orthoregress(
        x: pd.Series | npt.ArrayLike, y: pd.Series | npt.ArrayLike, x_errs: Optional[npt.ArrayLike] = None, y_errs: Optional[npt.ArrayLike] = None
    ):
        # calc weights (inverse variances)
        wd = None
        we = None
        if x_errs is not None:
            wd = 1.0 / np.square(x_errs)
        if y_errs is not None:
            we = 1.0 / np.square(y_errs)

        def f(p, x):
            return (p[0] * x) + p[1]

        od = ODR(Data(x, y, wd=wd, we=we), Model(f), beta0=linregress(x, y)[0:2])
        out = od.run()
        return out

    def orthoregress_multilinear(
        X: pd.DataFrame | pd.Series | npt.ArrayLike,
        y: pd.Series | npt.ArrayLike,
        x_errs: Optional[npt.ArrayLike] = None,
        y_errs: Optional[npt.ArrayLike] = None,
    ):
        # calc weights (inverse variances)
        wd = None
        we = None
        if x_errs is not None:
            x_errs = np.asarray(x_errs)
            wd = 1.0 / np.square(x_errs.T)  # transpose to match ODR shape
        if y_errs is not None:
            we = 1.0 / np.square(y_errs)

        def multilinear_f(p, x):
            return np.dot(p[:-1], x) + p[-1]

        X = np.asarray(X)
        y = np.asarray(y)
        X_odr = X.T
        y_flat = y.flatten()
        X_with_intercept = np.column_stack((X, np.ones(X.shape[0])))
        beta_init, _, _, _ = np.linalg.lstsq(X_with_intercept, y_flat, rcond=None)
        beta0 = np.append(beta_init[:-1], beta_init[-1])
        model = Model(multilinear_f)
        data = Data(X_odr, y, wd=wd, we=we)
        odr_instance = ODR(data, model, beta0=beta0)
        output = odr_instance.run()
        return output

    if len(x_cols) > 1:
        out = orthoregress_multilinear(df[x_cols], df[y_col], x_errs, y_errs)
    else:
        out = orthoregress(df[x_cols[0]], df[y_col], x_errs, y_errs)
    out.beta = np.roll(out.beta, 1)
    return out


def calc_pca_loadings_matrix(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    run_on_level_changes: Optional[bool] = False,
    run_pca_on_corr_mat: Optional[bool] = False,
    scale_loadings: Optional[bool] = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if cols:
        df = df[cols].copy()
    else:
        df = df.copy()

    if run_on_level_changes:
        df = df.diff().dropna()

    if run_pca_on_corr_mat:
        scaler = StandardScaler()
        values_to_fit = scaler.fit_transform(df)
    else:
        values_to_fit = df.values

    pca = PCA()
    pc_scores_signs_not_flipped = pca.fit_transform(values_to_fit)
    pc_scores_df = pd.DataFrame(
        {
            "Date": df.index,
            "PC1": pc_scores_signs_not_flipped[:, 0],
            # https://stackoverflow.com/questions/44765682/in-sklearn-decomposition-pca-why-are-components-negative
            "PC2": -1 * pc_scores_signs_not_flipped[:, 1],
            "PC3": -1 * pc_scores_signs_not_flipped[:, 2] if len(df.columns) > 3 else None,
        }
    )

    if scale_loadings:
        # the sensitivity of each original tenor to changes in each principal component
        # the loadings reflect how much variance (in original units) is explained by each PC b/c scaled by sqrt(eigenvalues)
        loadings_df = pd.DataFrame(
            pca.components_.T * np.sqrt(pca.explained_variance_), index=df.columns, columns=[f"PC{i+1}" for i in range(len(df.columns))]
        )
    else:
        loadings_df = pd.DataFrame(pca.components_.T, index=df.columns, columns=[f"PC{i+1}" for i in range(len(df.columns))])

    return {"pca": pca, "loading_matrix": loadings_df, "pc_scores_matrix": pc_scores_df}


# Adapted from https://github.com/hudson-and-thames/arbitragelab/blob/master/arbitragelab/hedge_ratios/box_tiao.py
def get_box_tiao_hedge_ratio(price_data: pd.DataFrame, dependent_variable: str) -> Tuple[dict, pd.DataFrame, None, pd.Series]:
    """
    Perform Box-Tiao canonical decomposition on the assets dataframe.

    The resulting ratios are the weightings of each asset in the portfolio. There are N decompositions for N assets,
    where each column vector corresponds to one portfolio. The order of the weightings corresponds to the
    descending order of the eigenvalues.

    :param price_data: (pd.DataFrame) DataFrame with security prices.
    :param dependent_variable: (str) Column name which represents the dependent variable (y).
    :return: (Tuple) Hedge ratios, X, and fit residuals.
    """

    def _least_square_VAR_fit(demeaned_price_data: pd.DataFrame) -> np.array:
        """
        Calculate the least square estimate of the VAR(1) matrix.

        :param demeaned_price_data: (pd.DataFrame) Demeaned price data.
        :return: (np.array) Least square estimate of VAR(1) matrix.
        """

        # Fit VAR(1) model
        var_model = sm.tsa.VAR(demeaned_price_data)

        # The statsmodels package will give the least square estimate
        least_sq_est = np.squeeze(var_model.fit(1).coefs, axis=0)

        return least_sq_est, var_model

    X = price_data.copy()
    X = X[[dependent_variable] + [x for x in X.columns if x != dependent_variable]]

    demeaned = X - X.mean()  # Subtract mean columns

    # Calculate the least square estimate of the price with VAR(1) model
    least_sq_est, var_model = _least_square_VAR_fit(demeaned)

    # Construct the matrix from which the eigenvectors need to be computed
    covar = demeaned.cov()
    box_tiao_matrix = np.linalg.inv(covar) @ least_sq_est @ covar @ least_sq_est.T

    # Calculate the eigenvectors and sort by eigenvalue
    eigvals, eigvecs = np.linalg.eig(box_tiao_matrix)

    # Sort the eigenvectors by eigenvalues by descending order
    bt_eigvecs = eigvecs[:, np.argsort(eigvals)[::-1]]
    hedge_ratios = dict(zip(X.columns, bt_eigvecs[:, -1]))

    beta_weights = []
    # Convert to a format expected by `construct_spread` function and normalize such that dependent has a hedge ratio 1
    for ticker, h in hedge_ratios.items():
        if ticker != dependent_variable:
            beta = -h / hedge_ratios[dependent_variable]
            hedge_ratios[ticker] = beta
            beta_weights.append(beta)
    hedge_ratios[dependent_variable] = 1.0

    residuals = construct_spread(price_data, hedge_ratios=hedge_ratios, dependent_variable=dependent_variable)
    # return hedge_ratios, X, residuals
    return {
        "beta_weights": beta_weights,
        "hedge_ratios_dict": hedge_ratios,
        "X": X,
        "residuals": residuals,
        "results": var_model,
    }


def get_johansen_hedge_ratio(price_data: pd.DataFrame, dependent_variable: str) -> Tuple[dict, pd.DataFrame, pd.Series, pd.Series]:
    """
    Get hedge ratio from Johansen test eigenvector.

    :param price_data: (pd.DataFrame) DataFrame with security prices.
    :param dependent_variable: (str) Column name which represents the dependent variable (y).
    :return: (Tuple) Hedge ratios, X, and y and OLS fit residuals.
    """

    # Construct a Johansen portfolio
    port = JohansenPortfolio()
    port.fit(price_data, dependent_variable)

    X = price_data.copy()
    X.drop(columns=dependent_variable, axis=1, inplace=True)

    y = price_data[dependent_variable].copy()

    # Convert to a format expected by `construct_spread` function and normalize such that dependent has a hedge ratio 1.
    hedge_ratios = port.hedge_ratios.iloc[0].to_dict()
    residuals = construct_spread(price_data, hedge_ratios=hedge_ratios, dependent_variable=dependent_variable)

    hedge_ratios_copy = hedge_ratios.copy()
    del hedge_ratios_copy[dependent_variable]

    # Normalize Johansen cointegration vectors such that dependent variable has a hedge ratio of 1.
    return {
        "beta_weights": list(hedge_ratios_copy.values()),
        "hedge_ratios_dict": hedge_ratios,
        "X": X,
        "y": y,
        "residuals": residuals,
        "results": port,
    }


def get_minimum_hl_hedge_ratio(price_data: pd.DataFrame, dependent_variable: str) -> Tuple[dict, pd.DataFrame, pd.Series, pd.Series, object]:
    """
    Get hedge ratio by minimizing spread half-life of mean reversion.

    :param price_data: (pd.DataFrame) DataFrame with security prices.
    :param dependent_variable: (str) Column name which represents the dependent variable (y).
    :return: (Tuple) Hedge ratios, X, and y, OLS fit residuals and optimization object.
    """

    def get_half_life_of_mean_reversion(data: pd.Series) -> float:
        """
        Get half-life of mean-reversion under the assumption that data follows the Ornstein-Uhlenbeck process.

        :param data: (np.array) Data points.
        :return: (float) Half-life of mean reversion.
        """

        reg = LinearRegression(fit_intercept=True)

        training_data = data.shift(1).dropna().values.reshape(-1, 1)
        target_values = data.diff().dropna()
        reg.fit(X=training_data, y=target_values)

        half_life = -np.log(2) / reg.coef_[0]

        return half_life

    def _min_hl_function(beta: np.array, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Fitness function to minimize in Minimum Half-Life Hedge Ratio algorithm.

        :param beta: (np.array) Array of hedge ratios.
        :param X: (pd.DataFrame) DataFrame of dependent variables. We hold `beta` units of X assets.
        :param y: (pd.Series) Series of target variable. For this asset we hold 1 unit.
        :return: (float) Half-life of mean-reversion.
        """

        spread = y - (beta * X).sum(axis=1)
        return abs(get_half_life_of_mean_reversion(spread))

    X = price_data.copy()
    X.drop(columns=dependent_variable, axis=1, inplace=True)

    y = price_data[dependent_variable].copy()
    initial_guess = (y[0] / X).mean().values
    result = minimize(_min_hl_function, x0=initial_guess, method="BFGS", tol=1e-5, args=(X, y))
    residuals = y - (result.x * X).sum(axis=1)

    hedge_ratios = result.x
    hedge_ratios_dict = dict(zip([dependent_variable] + X.columns.tolist(), np.insert(hedge_ratios, 0, 1.0)))
    if result.status != 0:
        warnings.warn("Minimum Half Life Optimization failed to converge. Please check output hedge ratio! The result can be unstable!")
    return {
        "beta_weights": list(hedge_ratios),
        "hedge_ratios_dict": hedge_ratios_dict,
        "X": X,
        "y": y,
        "residuals": residuals,
        "results": result,
    }


def get_adf_optimal_hedge_ratio(price_data: pd.DataFrame, dependent_variable: str) -> Tuple[dict, pd.DataFrame, pd.Series, pd.Series, object]:
    """
    Get hedge ratio by minimizing ADF test statistic.

    :param price_data: (pd.DataFrame) DataFrame with security prices.
    :param dependent_variable: (str) Column name which represents the dependent variable (y).
    :return: (Tuple) Hedge ratios, X, and y, OLS fit residuals and optimization object.
    """

    def _min_adf_stat(beta: np.array, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Fitness function to minimize in ADF test statistic algorithm.

        :param beta: (np.array) Array of hedge ratios.
        :param X: (pd.DataFrame) DataFrame of dependent variables. We hold `beta` units of X assets.
        :param y: (pd.Series) Series of target variable. For this asset we hold 1 unit.
        :return: (float) Half-life of mean-reversion.
        """

        # Performing Engle-Granger test on spread
        portfolio = EngleGrangerPortfolio()
        spread = y - (beta * X).sum(axis=1)
        portfolio.perform_eg_test(spread)

        return portfolio.adf_statistics.loc["statistic_value"].iloc[0]

    X = price_data.copy()
    X.drop(columns=dependent_variable, axis=1, inplace=True)

    y = price_data[dependent_variable].copy()
    initial_guess = (y[0] / X).mean().values
    result = minimize(_min_adf_stat, x0=initial_guess, method="BFGS", tol=1e-5, args=(X, y))
    residuals = y - (result.x * X).sum(axis=1)

    hedge_ratios = result.x
    hedge_ratios_dict = dict(zip([dependent_variable] + X.columns.tolist(), np.insert(hedge_ratios, 0, 1.0)))
    if result.status != 0:
        warnings.warn("ADF Optimization failed to converge. Please check output hedge ratio! The result can be unstable!")

    return {
        "beta_weights": list(hedge_ratios),
        "hedge_ratios_dict": hedge_ratios_dict,
        "X": X,
        "y": y,
        "residuals": residuals,
        "results": result,
    }


def beta_estimates(
    df: pd.DataFrame,
    x_cols: List[str],
    y_col: str,
    run_on_level_changes: Optional[bool] = False,
    x_errs: Optional[npt.ArrayLike] = None,
    y_errs: Optional[npt.ArrayLike] = None,
    pc_scores_df: Optional[pd.DataFrame] = None,
    loadings_df: Optional[pd.DataFrame] = None,
) -> Dict[str, sm.regression.linear_model.RegressionResults | Output | pd.DataFrame]:

    df = df[["Date"] + x_cols + [y_col]].copy()
    df_level = df.copy()

    if len(x_cols) == 1:
        df["spread"] = df[y_col] - df[x_cols[0]]
    elif len(x_cols) == 2:
        df["spread"] = (df[y_col] - df[x_cols[0]]) - (df[x_cols[1]] - df[y_col])
    else:
        raise ValueError("Too many x_cols")

    if run_on_level_changes:
        date_col = df["Date"]
        df = df[x_cols + [y_col] + ["spread"]].diff()
        df["Date"] = date_col
    df = df.dropna()

    pc1_beta = None
    if loadings_df is not None:
        ep_x0_pc1 = loadings_df.loc[x_cols[0], "PC1"]
        ep_x0_pc2 = loadings_df.loc[x_cols[0], "PC2"]
        ep_x0_pc3 = loadings_df.loc[x_cols[0], "PC3"]

        ep_y_pc1 = loadings_df.loc[y_col, "PC1"]
        ep_y_pc2 = loadings_df.loc[y_col, "PC2"]
        ep_y_pc3 = loadings_df.loc[y_col, "PC3"]

        if len(x_cols) > 1:
            ep_x1_pc1 = loadings_df.loc[x_cols[1], "PC1"]
            ep_x1_pc2 = loadings_df.loc[x_cols[1], "PC2"]
            ep_x1_pc3 = loadings_df.loc[x_cols[1], "PC3"]

            # see  Doug Huggins, Christian Schaller Fixed Income Relative Value Analysis ed2 page 76 APPROPRIATE HEDGING
            r"""
                Hedge ratios against more factors are best calculated via matrix inversion. 
                For example, the hedge ratio for a 2Y-5Y-10Y butterfly which is neutral to changes in the first and 
                second factor can be calculated for a given notional $n_5$ for 5Y by:
                    $$
                        \begin{pmatrix}
                            n_2 \\
                            n_{10}
                        \end{pmatrix}
                        = 
                        \begin{pmatrix}
                            BPV_2 \cdot e_{12} & BPV_2 \cdot e_{22} \\
                            BPV_{10} \cdot e_{12} & BPV_{10} \cdot e_{22}
                            \end{pmatrix}^{-1} 
                            \begin{pmatrix}
                            - n_5 \cdot BPV_5 \cdot e_{15} \\
                            - n_5 \cdot BPV_5 \cdot e_{25}
                        \end{pmatrix}
                    $$
            """
            pc1_beta = list(np.dot(np.linalg.inv(np.array([[ep_x0_pc1, ep_x1_pc1], [ep_x0_pc2, ep_x1_pc2]])), np.array([ep_y_pc1, ep_y_pc2])))
        else:
            pc1_beta = ep_y_pc1 / ep_x0_pc1

    regression_results = {
        "ols": sm.OLS(df[y_col], sm.add_constant(df[x_cols])).fit(),
        "tls": _run_odr(df=df, x_cols=x_cols, y_col=y_col, x_errs=None, y_errs=None),
        # ODR becomes TLS if errors not specified
        "odr": _run_odr(df=df, x_cols=x_cols, y_col=y_col, x_errs=x_errs, y_errs=y_errs) if x_errs or y_errs else None,
        "box_tiao": get_box_tiao_hedge_ratio(df_level[["Date"] + x_cols + [y_col]].set_index("Date"), dependent_variable=y_col),
        "johansen": get_johansen_hedge_ratio(df_level[["Date"] + x_cols + [y_col]].set_index("Date"), dependent_variable=y_col),
        "minimum_half_life": get_minimum_hl_hedge_ratio(df_level[["Date"] + x_cols + [y_col]].set_index("Date"), dependent_variable=y_col),
        "adf_optimal": get_adf_optimal_hedge_ratio(df_level[["Date"] + x_cols + [y_col]].set_index("Date"), dependent_variable=y_col),
        "pcr_pc1": sm.OLS(df["spread"].to_numpy(), sm.add_constant(pc_scores_df["PC1"].to_numpy())).fit() if pc_scores_df is not None else None,
        "pcr_pc2": sm.OLS(df["spread"].to_numpy(), sm.add_constant(pc_scores_df["PC2"].to_numpy())).fit() if pc_scores_df is not None else None,
        "pcr_pc3": sm.OLS(df["spread"].to_numpy(), sm.add_constant(pc_scores_df["PC3"].to_numpy())).fit() if pc_scores_df is not None else None,
    }

    beta_estimates = {
        "ols": (
            regression_results["ols"].params[1] if len(x_cols) == 1 else [regression_results["ols"].params[1], regression_results["ols"].params[2]]
        ),
        "tls": regression_results["tls"].beta[1] if len(x_cols) == 1 else [regression_results["tls"].beta[1], regression_results["tls"].beta[2]],
        "odr": (
            regression_results["odr"].beta[1]
            if (x_errs or y_errs) and len(x_cols) == 1
            else [regression_results["odr"].beta[1], regression_results["odr"].beta[2]] if x_errs or y_errs else None
        ),
        "pc1_beta": pc1_beta,
        "box_tiao": regression_results["box_tiao"]["beta_weights"],
        "johansen": regression_results["johansen"]["beta_weights"],
        "minimum_half_life": regression_results["minimum_half_life"]["beta_weights"],
        "adf_optimal": regression_results["adf_optimal"]["beta_weights"],
    }

    pcs_exposures = {
        "pcr_pc1_exposure": regression_results["pcr_pc1"].params[1] if pc_scores_df is not None else None,
        "pcr_pc2_exposure": regression_results["pcr_pc2"].params[1] if pc_scores_df is not None else None,
        "pcr_pc3_exposure": regression_results["pcr_pc3"].params[1] if pc_scores_df is not None else None,
        # checking 50-50 duration - if non-zero => exposures exists
        "epsilon_pc1_loadings_exposure": (
            ep_y_pc1 - (ep_x0_pc1 + ep_x1_pc1) / 2.0 if len(x_cols) > 1 else ep_x0_pc1 - ep_y_pc1 if loadings_df is not None else None
        ),
        "epsilon_pc2_loadings_exposure": (
            ep_y_pc2 - (ep_x0_pc2 + ep_x1_pc2) / 2.0 if len(x_cols) > 1 else ep_x0_pc2 - ep_y_pc2 if loadings_df is not None else None
        ),
        "epsilon_pc3_loadings_exposure": (
            ep_y_pc3 - (ep_x0_pc3 + ep_x1_pc3) / 2.0 if len(x_cols) > 1 else ep_x0_pc3 - ep_y_pc3 if loadings_df is not None else None
        ),
    }

    return {"betas": beta_estimates, "regression_results": regression_results, "pc_exposures": pcs_exposures}
