import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import ujson as json
from scipy.optimize import minimize
from termcolor import colored

from CurveBuilder import calc_ust_impl_spot_n_fwd_curve, calc_ust_metrics
from CurveDataFetcher import CurveDataFetcher
from utils.regression_utils import run_odr
from utils.arbitragelab import JohansenPortfolio, construct_spread, EngleGrangerPortfolio


def dv01_neutral_curve_hegde_ratio(
    as_of_date: datetime,
    front_leg_bond_row: Dict | pd.Series,
    back_leg_bond_row: Dict | pd.Series,
    curve_data_fetcher: CurveDataFetcher,
    scipy_interp_curve: scipy.interpolate.interpolate,
    repo_rate: float,
    quote_type: Optional[str] = "eod",
    front_leg_par_amount: Optional[int] = None,
    back_leg_par_amount: Optional[int] = None,
    total_trade_par_amount: Optional[int] = None,
    yvx_beta_adjustment: Optional[int] = None,
    verbose: Optional[bool] = True,
    very_verbose: Optional[bool] = False,
):
    if isinstance(front_leg_bond_row, pd.Series) or isinstance(front_leg_bond_row, pd.DataFrame):
        front_leg_bond_row = front_leg_bond_row.to_dict("records")[0]
    if isinstance(back_leg_bond_row, pd.Series) or isinstance(back_leg_bond_row, pd.DataFrame):
        back_leg_bond_row = back_leg_bond_row.to_dict("records")[0]

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
    print(f"{front_leg_bond_row["ust_label"]} / {back_leg_bond_row["ust_label"]}") if verbose else None

    hr = back_leg_metrics["bps"] / front_leg_metrics["bps"]
    print(colored(f"BPV Neutral Hedge Ratio: {hr}", "light_blue")) if verbose else None
    if yvx_beta_adjustment:
        (print(colored(f"Beta Weighted Hedge Ratio: {hr * yvx_beta_adjustment:3f}", "light_magenta")) if verbose else None)
        hr = hr * yvx_beta_adjustment

    if total_trade_par_amount is not None:
        if front_leg_par_amount is not None or back_leg_par_amount is not None:
            raise ValueError("Cannot provide total_trade_par_amount along with individual leg par amounts.")

        if hr <= 0:
            raise ValueError("Hedge ratio must be positive to calculate par amounts.")

        front_leg_par_amount = total_trade_par_amount / (hr + 1)
        back_leg_par_amount = hr * front_leg_par_amount

    else:
        if front_leg_par_amount and back_leg_par_amount:
            raise ValueError("'front_leg_par_amount' and 'back_leg_par_amount' are both defined!")
        if not front_leg_par_amount and not back_leg_par_amount:
            back_leg_par_amount = 1_000_000
        if back_leg_par_amount:
            front_leg_par_amount = back_leg_par_amount * hr
        elif front_leg_par_amount:
            back_leg_par_amount = front_leg_par_amount / hr

    if verbose:
        print(
            f"Front Leg: {front_leg_bond_row["ust_label"]} (OST {front_leg_bond_row["original_security_term"]}, TTM = {front_leg_bond_row["time_to_maturity"]:3f}) Par Amount = {front_leg_par_amount :_}"
        )
        print(
            f"Back Leg: {back_leg_bond_row["ust_label"]} (OST {back_leg_bond_row["original_security_term"]}, TTM = {back_leg_bond_row["time_to_maturity"]:3f}) Par Amount = {back_leg_par_amount:_}"
        )
        print(f"Total Trade Par Amount: {front_leg_par_amount + back_leg_par_amount:_}")
        risk_weight = (front_leg_par_amount * front_leg_metrics["bps"] / 100) / (back_leg_par_amount * back_leg_metrics["bps"] / 100)
        print(f"Risk Weights: {risk_weight:3f} : 100")

    return {
        "current_spread": (back_leg_bond_row[f"{quote_type}_yield"] - front_leg_bond_row[f"{quote_type}_yield"]) * 100,
        "current_bpv_neutral_spread": (
            back_leg_bond_row[f"{quote_type}_yield"]
            - (front_leg_bond_row[f"{quote_type}_yield"] * (back_leg_metrics["bps"] / front_leg_metrics["bps"]))
        )
        * 100,
        "current_beta_weighted_spread": (
            (back_leg_bond_row[f"{quote_type}_yield"] - (front_leg_bond_row[f"{quote_type}_yield"] * hr)) * 100 if yvx_beta_adjustment else None
        ),
        "rough_3m_impl_fwd_spread": (impl_spot_3m_fwds(back_leg_ttm) - impl_spot_3m_fwds(front_leg_ttm)) * 100,
        "rough_6m_impl_fwd_spread": (impl_spot_6m_fwds(back_leg_ttm) - impl_spot_6m_fwds(front_leg_ttm)) * 100,
        "rough_12m_impl_fwd_spread": (impl_spot_12m_fwds(back_leg_ttm) - impl_spot_12m_fwds(front_leg_ttm)) * 100,
        "front_leg_metrics": front_leg_metrics,
        "back_leg_metrics": back_leg_metrics,
        "bpv_hedge_ratio": back_leg_metrics["bps"] / front_leg_metrics["bps"],
        "beta_weighted_hedge_ratio": ((back_leg_metrics["bps"] / front_leg_metrics["bps"]) * yvx_beta_adjustment if yvx_beta_adjustment else None),
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
    total_trade_par_amount: Optional[int] = None,
    yvx_front_wing_beta_adjustment: Optional[int] = None,
    yvx_back_wing_beta_adjustment: Optional[int] = None,
    verbose: Optional[bool] = True,
    very_verbose: Optional[bool] = False,
):
    if isinstance(front_wing_bond_row, pd.Series) or isinstance(front_wing_bond_row, pd.DataFrame):
        front_wing_bond_row = front_wing_bond_row.to_dict("records")[0]
    if isinstance(belly_bond_row, pd.Series) or isinstance(belly_bond_row, pd.DataFrame):
        belly_bond_row = belly_bond_row.to_dict("records")[0]
    if isinstance(back_wing_bond_row, pd.Series) or isinstance(back_wing_bond_row, pd.DataFrame):
        back_wing_bond_row = back_wing_bond_row.to_dict("records")[0]

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

        (print(f"{front_wing_bond_row["ust_label"]} - {belly_bond_row["ust_label"]} - {back_wing_bond_row["ust_label"]} Fly") if verbose else None)
        print(colored(f"BPV Neutral Hedge Ratio:", "light_blue")) if verbose else None
        print(json.dumps(hedge_ratios, indent=4)) if verbose else None

        if yvx_front_wing_beta_adjustment and yvx_back_wing_beta_adjustment:
            print(colored(f"Beta Weighted Hedge Ratio:", "light_magenta")) if verbose else None
            hedge_ratios = {
                "front_wing_hr": (belly_metrics["bps"] / front_wing_metrics["bps"] / 2) * yvx_front_wing_beta_adjustment,
                "belly_hr": 1,
                "back_wing_hr": (belly_metrics["bps"] / back_wing_metrics["bps"] / 2) * yvx_back_wing_beta_adjustment,
            }
            print(json.dumps(hedge_ratios, indent=4)) if verbose else None

        if total_trade_par_amount is not None:
            if front_wing_par_amount is not None or belly_par_amount is not None or back_wing_par_amount is not None:
                raise ValueError("Cannot provide total_trade_par_amount along with individual leg par amounts.")

            total_hr_abs = abs(hedge_ratios["front_wing_hr"]) + abs(hedge_ratios["belly_hr"]) + abs(hedge_ratios["back_wing_hr"])
            belly_par_amount = total_trade_par_amount / total_hr_abs
            front_wing_par_amount = hedge_ratios["front_wing_hr"] * belly_par_amount
            back_wing_par_amount = hedge_ratios["back_wing_hr"] * belly_par_amount

        else:
            if belly_par_amount:
                front_wing_par_amount = hedge_ratios["front_wing_hr"] * belly_par_amount
                belly_par_amount = belly_par_amount
                back_wing_par_amount = hedge_ratios["back_wing_hr"] * belly_par_amount
            elif front_wing_par_amount:
                front_wing_par_amount = front_wing_par_amount
                belly_par_amount = front_wing_par_amount / hedge_ratios["front_wing_hr"]
                back_wing_par_amount = hedge_ratios["back_wing_hr"] * (front_wing_par_amount / hedge_ratios["front_wing_hr"])
            elif back_wing_par_amount:
                front_wing_par_amount = hedge_ratios["front_wing_hr"] * (back_wing_par_amount / hedge_ratios["back_wing_hr"])
                belly_par_amount = back_wing_par_amount / hedge_ratios["back_wing_hr"]
                back_wing_par_amount = back_wing_par_amount

        print(
            f"Front Wing: {front_wing_bond_row["ust_label"]} (OST {front_wing_bond_row["original_security_term"]}, TTM = {front_wing_bond_row["time_to_maturity"]:3f}) Par Amount = {front_wing_par_amount:_}"
        )
        print(
            f"Belly: {belly_bond_row["ust_label"]} (OST {belly_bond_row["original_security_term"]}, TTM = {belly_bond_row["time_to_maturity"]:3f}) Par Amount = {belly_par_amount:_}"
        )
        print(
            f"Back Wing: {back_wing_bond_row["ust_label"]} (OST {back_wing_bond_row["original_security_term"]}, TTM = {back_wing_bond_row["time_to_maturity"]:3f}) Par Amount = {back_wing_par_amount:_}"
        )
        print(f"Total Trade Par Amount: {front_wing_par_amount + belly_par_amount + back_wing_par_amount:_}")
        (
            print(
                f"Risk Weights - Front Wing: {yvx_front_wing_beta_adjustment:.3%}, Back Wing: {yvx_back_wing_beta_adjustment:.3%}, Sum: {yvx_front_wing_beta_adjustment + yvx_back_wing_beta_adjustment:.3%}"
            )
            if yvx_front_wing_beta_adjustment and yvx_back_wing_beta_adjustment
            else None
        )

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
        "bpv_neutral_hedge_ratio": {
            "front_wing_hr": belly_metrics["bps"] / front_wing_metrics["bps"] / 2,
            "belly_hr": 1,
            "back_wing_hr": belly_metrics["bps"] / back_wing_metrics["bps"] / 2,
        },
        "beta_weighted_hedge_ratio": (hedge_ratios if yvx_front_wing_beta_adjustment and yvx_back_wing_beta_adjustment else None),
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


# https://github.com/hudson-and-thames/arbitragelab/blob/master/arbitragelab/hedge_ratios/box_tiao.py
def get_box_tiao_hedge_ratio(price_data: pd.DataFrame, dependent_variable: str) -> Tuple[dict, pd.DataFrame, None, pd.Series]:
    """
    Perform Box-Tiao canonical decomposition on the assets dataframe.
    The resulting ratios are the weightings of each asset in the portfolio. There are N decompositions for N assets,
    where each column vector corresponds to one portfolio. The order of the weightings corresponds to the
    descending order of the eigenvalues.
    """

    def _least_square_VAR_fit(demeaned_price_data: pd.DataFrame) -> np.array:
        var_model = sm.tsa.VAR(demeaned_price_data)
        least_sq_est = np.squeeze(var_model.fit(1).coefs, axis=0)
        return least_sq_est, var_model

    X = price_data.copy()
    X = X[[dependent_variable] + [x for x in X.columns if x != dependent_variable]]

    demeaned = X - X.mean()
    least_sq_est, var_model = _least_square_VAR_fit(demeaned)
    covar = demeaned.cov()
    box_tiao_matrix = np.linalg.inv(covar) @ least_sq_est @ covar @ least_sq_est.T
    eigvals, eigvecs = np.linalg.eig(box_tiao_matrix)
    bt_eigvecs = eigvecs[:, np.argsort(eigvals)[::-1]]
    hedge_ratios = dict(zip(X.columns, bt_eigvecs[:, -1]))

    beta_weights = []
    for ticker, h in hedge_ratios.items():
        if ticker != dependent_variable:
            beta = -h / hedge_ratios[dependent_variable]
            hedge_ratios[ticker] = beta
            beta_weights.append(beta)
    hedge_ratios[dependent_variable] = 1.0
    residuals = construct_spread(price_data, hedge_ratios=hedge_ratios, dependent_variable=dependent_variable)

    return {
        "beta_weights": beta_weights,
        "hedge_ratios_dict": hedge_ratios,
        "X": X,
        "residuals": residuals,
        "results": var_model,
    }


# https://github.com/hudson-and-thames/arbitragelab/blob/master/arbitragelab/hedge_ratios/johansen.py
def get_johansen_hedge_ratio(price_data: pd.DataFrame, dependent_variable: str) -> Tuple[dict, pd.DataFrame, pd.Series, pd.Series]:
    """
    Get hedge ratio from Johansen test eigenvector
    https://en.wikipedia.org/wiki/Johansen_test
    """

    port = JohansenPortfolio()
    port.fit(price_data, dependent_variable)

    X = price_data.copy()
    X.drop(columns=dependent_variable, axis=1, inplace=True)

    y = price_data[dependent_variable].copy()

    hedge_ratios = port.hedge_ratios.iloc[0].to_dict()
    residuals = construct_spread(price_data, hedge_ratios=hedge_ratios, dependent_variable=dependent_variable)

    hedge_ratios_copy = hedge_ratios.copy()
    del hedge_ratios_copy[dependent_variable]

    return {
        "beta_weights": list(hedge_ratios_copy.values()),
        "hedge_ratios_dict": hedge_ratios,
        "X": X,
        "y": y,
        "residuals": residuals,
        "results": port,
    }


# https://github.com/hudson-and-thames/arbitragelab/blob/master/arbitragelab/hedge_ratios/half_life.py
def get_minimum_hl_hedge_ratio(price_data: pd.DataFrame, dependent_variable: str) -> Tuple[dict, pd.DataFrame, pd.Series, pd.Series, object]:
    """
    Get hedge ratio by minimizing spread half-life of mean reversion.
    https://quant.stackexchange.com/questions/77953/interpretation-and-intuition-behind-half-life-of-a-mean-reverting-process
    """

    def get_half_life_of_mean_reversion_ou_process(data: pd.Series) -> float:
        reg = LinearRegression(fit_intercept=True)
        training_data = data.shift(1).dropna().values.reshape(-1, 1)
        target_values = data.diff().dropna()
        reg.fit(X=training_data, y=target_values)
        half_life = -np.log(2) / reg.coef_[0]
        return half_life

    def _min_hl_function(beta: np.array, X: pd.DataFrame, y: pd.Series) -> float:
        spread = y - (beta * X).sum(axis=1)
        return abs(get_half_life_of_mean_reversion_ou_process(spread))

    X = price_data.copy()
    X.drop(columns=dependent_variable, axis=1, inplace=True)

    y = price_data[dependent_variable].copy()
    initial_guess = (y[0] / X).mean().values
    result = minimize(_min_hl_function, x0=initial_guess, method="BFGS", tol=1e-5, args=(X, y))

    if result.status != 0:
        warnings.warn("Minimum Half Life Optimization failed to converge. Please check output hedge ratio! The result can be unstable!")

    residuals = y - (result.x * X).sum(axis=1)

    hedge_ratios = result.x
    hedge_ratios_dict = dict(zip([dependent_variable] + X.columns.tolist(), np.insert(hedge_ratios, 0, 1.0)))

    return {
        "beta_weights": list(hedge_ratios),
        "hedge_ratios_dict": hedge_ratios_dict,
        "X": X,
        "y": y,
        "residuals": residuals,
        "results": result,
    }


# https://github.com/hudson-and-thames/arbitragelab/blob/master/arbitragelab/hedge_ratios/adf_optimal.py
def get_adf_optimal_hedge_ratio(price_data: pd.DataFrame, dependent_variable: str) -> Tuple[dict, pd.DataFrame, pd.Series, pd.Series, object]:
    """
    Get hedge ratio by minimizing ADF test statistic.
    https://www.statisticshowto.com/adf-augmented-dickey-fuller-test/
    """

    def _min_adf_stat(beta: np.array, X: pd.DataFrame, y: pd.Series) -> float:
        portfolio = EngleGrangerPortfolio()
        spread = y - (beta * X).sum(axis=1)
        portfolio.perform_eg_test(spread)
        return portfolio.adf_statistics.loc["statistic_value"].iloc[0]

    X = price_data.copy()
    X.drop(columns=dependent_variable, axis=1, inplace=True)
    y = price_data[dependent_variable].copy()
    initial_guess = (y[0] / X).mean().values
    result = minimize(_min_adf_stat, x0=initial_guess, method="BFGS", tol=1e-5, args=(X, y))

    if result.status != 0:
        warnings.warn("ADF Optimization failed to converge. Please check output hedge ratio! The result can be unstable!")

    residuals = y - (result.x * X).sum(axis=1)

    hedge_ratios = result.x
    hedge_ratios_dict = dict(zip([dependent_variable] + X.columns.tolist(), np.insert(hedge_ratios, 0, 1.0)))

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
) -> Dict:
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
        ep_x0_pc3 = loadings_df.loc[x_cols[0], "PC3"] if len(x_cols) > 1 else None

        ep_y_pc1 = loadings_df.loc[y_col, "PC1"]
        ep_y_pc2 = loadings_df.loc[y_col, "PC2"]
        ep_y_pc3 = loadings_df.loc[y_col, "PC3"] if len(x_cols) > 1 else None

        if len(x_cols) > 1:
            ep_x1_pc1 = loadings_df.loc[x_cols[1], "PC1"]
            ep_x1_pc2 = loadings_df.loc[x_cols[1], "PC2"]
            ep_x1_pc3 = loadings_df.loc[x_cols[1], "PC3"] if len(x_cols) > 1 else None

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
            pc1_beta = list(
                np.dot(
                    np.linalg.inv(np.array([[ep_x0_pc1, ep_x1_pc1], [ep_x0_pc2, ep_x1_pc2]])),
                    np.array([ep_y_pc1, ep_y_pc2]),
                )
            )
        else:
            pc1_beta = ep_y_pc1 / ep_x0_pc1

    # avoiding divide by zero errors
    small_value = 1e-8
    if x_errs is not None:
        x_errs[x_errs == 0] = small_value
    if y_errs is not None:
        y_errs[y_errs == 0] = small_value

    regression_results = {
        "ols": sm.OLS(df[y_col], sm.add_constant(df[x_cols])).fit(),
        "tls": run_odr(df=df, x_cols=x_cols, y_col=y_col, x_errs=None, y_errs=None),
        # ODR becomes TLS if errors not specified
        "odr": (run_odr(df=df, x_cols=x_cols, y_col=y_col, x_errs=x_errs, y_errs=y_errs) if x_errs is not None or y_errs is not None else None),
        "box_tiao": get_box_tiao_hedge_ratio(df_level[["Date"] + x_cols + [y_col]].set_index("Date"), dependent_variable=y_col),
        "johansen": get_johansen_hedge_ratio(df_level[["Date"] + x_cols + [y_col]].set_index("Date"), dependent_variable=y_col),
        "minimum_half_life": get_minimum_hl_hedge_ratio(df_level[["Date"] + x_cols + [y_col]].set_index("Date"), dependent_variable=y_col),
        "adf_optimal": get_adf_optimal_hedge_ratio(df_level[["Date"] + x_cols + [y_col]].set_index("Date"), dependent_variable=y_col),
        "pcr_pc1": (sm.OLS(df["spread"].to_numpy(), sm.add_constant(pc_scores_df["PC1"].to_numpy())).fit() if pc_scores_df is not None else None),
        "pcr_pc2": (sm.OLS(df["spread"].to_numpy(), sm.add_constant(pc_scores_df["PC2"].to_numpy())).fit() if pc_scores_df is not None else None),
        "pcr_pc3": (sm.OLS(df["spread"].to_numpy(), sm.add_constant(pc_scores_df["PC3"].to_numpy())).fit() if pc_scores_df is not None else None),
    }

    beta_estimates = {
        "ols": (
            regression_results["ols"].params[1] if len(x_cols) == 1 else [regression_results["ols"].params[1], regression_results["ols"].params[2]]
        ),
        "tls": (regression_results["tls"].beta[1] if len(x_cols) == 1 else [regression_results["tls"].beta[1], regression_results["tls"].beta[2]]),
        "odr": (
            regression_results["odr"].beta[1]
            if (x_errs is not None or y_errs is not None) and len(x_cols) == 1
            else [regression_results["odr"].beta[1], regression_results["odr"].beta[2]] if x_errs or y_errs else None
        ),
        "pc1": pc1_beta,
        "box_tiao": regression_results["box_tiao"]["beta_weights"],
        "johansen": regression_results["johansen"]["beta_weights"],
        "minimum_half_life": regression_results["minimum_half_life"]["beta_weights"],
        "adf_optimal": regression_results["adf_optimal"]["beta_weights"],
    }

    pcs_exposures = {
        "pcr_pc1_exposure": regression_results["pcr_pc1"].params[1] if pc_scores_df is not None else None,
        "pcr_pc2_exposure": regression_results["pcr_pc2"].params[1] if pc_scores_df is not None else None,
        "pcr_pc3_exposure": (regression_results["pcr_pc3"].params[1] if pc_scores_df is not None and len(x_cols) > 1 else None),
        # checking 50-50 duration - if non-zero => exposures exists
        "epsilon_pc1_loadings_exposure": (
            ep_y_pc1 - (ep_x0_pc1 + ep_x1_pc1) / 2.0 if len(x_cols) > 1 else ep_x0_pc1 - ep_y_pc1 if loadings_df is not None else None
        ),
        "epsilon_pc2_loadings_exposure": (
            ep_y_pc2 - (ep_x0_pc2 + ep_x1_pc2) / 2.0 if len(x_cols) > 1 else ep_x0_pc2 - ep_y_pc2 if loadings_df is not None else None
        ),
        "epsilon_pc3_loadings_exposure": (
            (ep_y_pc3 - (ep_x0_pc3 + ep_x1_pc3) / 2.0 if len(x_cols) > 1 else ep_x0_pc3 - ep_y_pc3 if loadings_df is not None else None)
            if len(x_cols) > 1
            else None
        ),
    }

    return {"betas": beta_estimates, "regression_results": regression_results, "pc_exposures": pcs_exposures}


# TODO
def rolling_beta_estimates():
    pass
