import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
import QuantLib as ql
import rateslib as rl
import scipy.interpolate
from pandas.tseries.offsets import BDay

from CurvyCUSIPs.CurveDataFetcher import CurveDataFetcher
from CurvyCUSIPs.utils.ust_utils import pydatetime_to_quantlib_date, quantlib_date_to_pydatetime, to_quantlib_fixed_rate_bond_obj

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)


def _calc_spot_rates_on_tenors(
    yield_curve: ql.DiscountCurve | ql.ZeroCurve,
    on_rate: float,
    day_count: ql.ActualActual = ql.ActualActual(ql.ActualActual.ISDA),
    price_col: Optional[str] = None,
    custom_price_col: Optional[str] = None,
    continuous_compounded_zero: Optional[bool] = False,
):
    spots = []
    tenors = []
    maturity_dates = []
    ref_date = yield_curve.referenceDate()

    dates = yield_curve.dates()
    for i, d in enumerate(dates):
        yrs = day_count.yearFraction(ref_date, d)
        if i == 0:
            tenors.append(1 / 360)
            spots.append(on_rate)
            t_plus_1_sr: pd.Timestamp = quantlib_date_to_pydatetime(d) - BDay(1)
            t_plus_1_sr = t_plus_1_sr.to_pydatetime()
            t_plus_1_sr = t_plus_1_sr.replace(hour=0, minute=0, second=0, microsecond=0)
            maturity_dates.append(t_plus_1_sr)
            continue

        if continuous_compounded_zero:
            zero_rate = yield_curve.zeroRate(yrs, ql.Continuous, True)
            eq_rate = zero_rate.equivalentRate(day_count, ql.Continuous, ql.NoFrequency, ref_date, d).rate()
        else:
            compounding = ql.Compounded
            freq = ql.Semiannual
            zero_rate = yield_curve.zeroRate(yrs, compounding, freq, True)
            eq_rate = zero_rate.equivalentRate(day_count, compounding, freq, ref_date, d).rate()

        tenors.append(yrs)
        spots.append(100 * eq_rate)
        maturity_dates.append(quantlib_date_to_pydatetime(d))

    price_col_type = price_col.split("_")[0] if price_col else None
    spot_col_name = f"{price_col_type}_spot_rate" if price_col else "spot_rate"
    if custom_price_col:
        spot_col_name = custom_price_col
    return pd.DataFrame(
        {
            "maturity_date": maturity_dates,
            "time_to_maturity": tenors,
            spot_col_name: spots,
        }
    )


def _calc_spot_rates_intep_months(
    yield_curve: ql.DiscountCurve | ql.ZeroCurve,
    on_rate: float,
    months: Optional[int] = 361,
    month_freq: Optional[float] = 1,
    custom_tenors: Optional[List[int]] = None,
    day_count=ql.ActualActual(ql.ActualActual.ISDA),
    calendar=ql.UnitedStates(m=ql.UnitedStates.GovernmentBond),
    price_col: Optional[str] = None,
    custom_price_col: Optional[str] = None,
    continuous_compounded_zero: Optional[bool] = False,
):
    spots = []
    tenors = []
    maturity_dates = []
    ref_date = yield_curve.referenceDate()
    to_iterate = custom_tenors if custom_tenors else range(0, months, month_freq)
    for month in to_iterate:
        d = calendar.advance(ref_date, ql.Period(month, ql.Months))
        yrs = month / 12.0
        if yrs == 0:
            tenors.append(1 / 360)
            spots.append(on_rate)
            t_plus_1_sr: pd.Timestamp = quantlib_date_to_pydatetime(d) - BDay(1)
            t_plus_1_sr = t_plus_1_sr.to_pydatetime()
            t_plus_1_sr = t_plus_1_sr.replace(hour=0, minute=0, second=0, microsecond=0)
            maturity_dates.append(t_plus_1_sr)
            continue

        if continuous_compounded_zero:
            zero_rate = yield_curve.zeroRate(yrs, ql.Continuous, True)
            eq_rate = zero_rate.equivalentRate(day_count, ql.Continuous, ql.NoFrequency, ref_date, d).rate()
        else:
            compounding = ql.Compounded
            freq = ql.Semiannual
            zero_rate = yield_curve.zeroRate(yrs, compounding, freq, True)
            eq_rate = zero_rate.equivalentRate(day_count, compounding, freq, ref_date, d).rate()

        tenors.append(yrs)
        spots.append(100 * eq_rate)
        maturity_dates.append(quantlib_date_to_pydatetime(d))

    price_col_type = price_col.split("_")[0] if price_col else None
    spot_col_name = f"{price_col_type}_spot_rate" if price_col else "spot_rate"
    if custom_price_col:
        spot_col_name = custom_price_col
    return pd.DataFrame(
        {
            "maturity_date": maturity_dates,
            "time_to_maturity": tenors,
            spot_col_name: spots,
        }
    )


def _calc_spot_rates_intep_days(
    yield_curve: ql.DiscountCurve | ql.ZeroCurve,
    on_rate: float,
    days: Optional[int] = 361 * 30,
    custom_tenors: Optional[List[int]] = None,
    day_count=ql.ActualActual(ql.ActualActual.ISDA),
    calendar=ql.UnitedStates(m=ql.UnitedStates.GovernmentBond),
    price_col: Optional[str] = None,
    custom_price_col: Optional[str] = None,
    continuous_compounded_zero: Optional[bool] = False,
):
    spots = []
    tenors = []
    maturity_dates = []
    ref_date = yield_curve.referenceDate()
    to_iterate = custom_tenors if custom_tenors else range(0, days, 1)
    for day in to_iterate:
        d = calendar.advance(ref_date, ql.Period(day, ql.Days))
        yrs = day / 365.0
        if yrs == 0:
            tenors.append(1 / 360)
            spots.append(on_rate)
            t_plus_1_sr: pd.Timestamp = quantlib_date_to_pydatetime(d) - BDay(1)
            t_plus_1_sr = t_plus_1_sr.to_pydatetime()
            t_plus_1_sr = t_plus_1_sr.replace(hour=0, minute=0, second=0, microsecond=0)
            maturity_dates.append(t_plus_1_sr)
            continue

        if continuous_compounded_zero:
            zero_rate = yield_curve.zeroRate(yrs, ql.Continuous, True)
            eq_rate = zero_rate.equivalentRate(day_count, ql.Continuous, ql.NoFrequency, ref_date, d).rate()
        else:
            compounding = ql.Compounded
            freq = ql.Semiannual
            zero_rate = yield_curve.zeroRate(yrs, compounding, freq, True)
            eq_rate = zero_rate.equivalentRate(day_count, compounding, freq, ref_date, d).rate()

        tenors.append(yrs)
        spots.append(100 * eq_rate)
        maturity_dates.append(quantlib_date_to_pydatetime(d))

    price_col_type = price_col.split("_")[0] if price_col else None
    spot_col_name = f"{price_col_type}_spot_rate" if price_col else "spot_rate"
    if custom_price_col:
        spot_col_name = custom_price_col
    return pd.DataFrame(
        {
            "maturity_date": maturity_dates,
            "time_to_maturity": tenors,
            spot_col_name: spots,
        }
    )


"""
Using QuantLib's Piecewise yield term structure for bootstrapping market observed prices to zeros rates at the respective ttms
- small differences between methods
- flag to take the averages of all Piecewise methods or pass in a specifc method
- passing in multiple ql_bootstrap_methods will take the average of the spot rates calculated from the different methods 
"""


def ql_piecewise_method_pretty(bs_method):
    ql_piecewise_methods_pretty_dict = {
        "ql_plld": "Piecewise Log Linear Discount",
        "ql_lcd": "Piecewise Log Cubic Discount",
        "ql_lz": "Piecewise Linear Zero",
        "ql_cz": "Piecewise Cubic Zero",
        "ql_lf": "Piecewise Linear Forward",
        "ql_spd": "Piecewise Spline Cubic Discount",
        "ql_kz": "Piecewise Kruger Zero",
        "ql_kld": "Piecewise Kruger Log Discount",
        "ql_mcf": "Piecewise Convex Monotone Forward",
        "ql_mcz": "Piecewise Convex Monotone Zero",
        "ql_ncz": "Piecewise Natural Cubic Zero",
        "ql_nlcd": "Piecewise Natural Log Cubic Discount",
        "ql_lmlcd": "Piecewise Log Mixed Linear Cubic Discount",
        "ql_pcz": "Piecewise Parabolic Cubic Zero",
        "ql_mpcz": "Piecewise Monotonic Parabolic Cubic Zero",
        "ql_lpcd": "Piecewise Log Parabolic Cubic Discount",
        "ql_mlpcd": "Piecewise Monotonic Log Parabolic Cubic Discount",
        "ql_f_ns": "Nelson-Siegel Fitting",
        "ql_f_nss": "Svensson Fitting",
        "ql_f_np": "Simple Polynomial Fitting",
        "ql_f_es": "Exponential Splines Fitting",
        "ql_f_cbs": "Cubic B-Splines Fitting",
    }
    return ql_piecewise_methods_pretty_dict[bs_method]


def get_spot_rates_bootstrapper(
    curve_set_df: pd.DataFrame,
    as_of_date: datetime,
    on_rate: float,
    ql_bootstrap_interp_methods: Optional[
        List[
            Literal[
                "ql_plld",
                "ql_lcd",
                "ql_lz",
                "ql_cz",
                "ql_lf",
                "ql_spd",
                "ql_kz",
                "ql_kld",
                "ql_mcf",
                "ql_mcz",
                "ql_ncz",
                "ql_nlcd",
                "ql_lmlcd",
                "ql_pcz",
                "ql_mpcz",
                "ql_lpcd",
                "ql_mlpcd",
            ]
        ]
    ] = ["ql_plld"],
    return_ql_zero_curve: Optional[bool] = False,
    interpolated_months_num: Optional[int] = None,
    interpolated_curve_yearly_freq: Optional[int] = 1,
    custom_yearly_tenors: Optional[List[int]] = None,
    # return_rel_val_df: Optional[bool] = False,
    daily_interpolation: Optional[bool] = False,
    return_scipy_interp_func: Optional[bool] = False,
    continuous_compounded_zero: Optional[bool] = False,
) -> Dict[str, pd.DataFrame | ql.DiscountCurve | ql.ZeroCurve]:
    ql_piecewise_methods: Dict[str, ql.DiscountCurve | ql.ZeroCurve] = {
        "ql_plld": ql.PiecewiseLogLinearDiscount,
        "ql_lcd": ql.PiecewiseLogCubicDiscount,
        "ql_lz": ql.PiecewiseLinearZero,
        "ql_cz": ql.PiecewiseCubicZero,
        "ql_lf": ql.PiecewiseLinearForward,
        "ql_spd": ql.PiecewiseSplineCubicDiscount,
        "ql_kz": ql.PiecewiseKrugerZero,
        "ql_kld": ql.PiecewiseKrugerLogDiscount,
        "ql_mcf": ql.PiecewiseConvexMonotoneForward,
        "ql_mcz": ql.PiecewiseConvexMonotoneZero,
        "ql_ncz": ql.PiecewiseNaturalCubicZero,
        "ql_nlcd": ql.PiecewiseNaturalLogCubicDiscount,
        "ql_lmlcd": ql.PiecewiseLogMixedLinearCubicDiscount,
        "ql_pcz": ql.PiecewiseParabolicCubicZero,
        "ql_mpcz": ql.PiecewiseMonotonicParabolicCubicZero,
        "ql_lpcd": ql.PiecewiseLogParabolicCubicDiscount,
        "ql_mlpcd": ql.PiecewiseMonotonicLogParabolicCubicDiscount,
        "ql_f_ns": ql.NelsonSiegelFitting,
        "ql_f_nss": ql.SvenssonFitting,
        "ql_f_np": ql.SimplePolynomialFitting,
        "ql_f_es": ql.ExponentialSplinesFitting,
        "ql_f_cbs": ql.CubicBSplinesFitting,
    }

    price_cols = ["bid_price", "offer_price", "mid_price", "eod_price"]
    required_cols = ["issue_date", "maturity_date", "int_rate"]
    price_col_exists = any(item in curve_set_df.columns for item in price_cols)
    missing_required_cols = [item for item in required_cols if item not in curve_set_df.columns]

    if not price_col_exists:
        raise ValueError(f"Build Spot Curve - Couldn't find a valid price col in your curve set df - one of {price_cols}")
    if missing_required_cols:
        raise ValueError(f"Build Spot Curve - Missing required curve set cols: {missing_required_cols}")

    price_col = next((item for item in price_cols if item in curve_set_df.columns), None)
    calendar = ql.UnitedStates(m=ql.UnitedStates.GovernmentBond)
    today = calendar.adjust(pydatetime_to_quantlib_date(py_datetime=as_of_date))
    ql.Settings.instance().evaluationDate = today

    t_plus = 1
    bond_settlement_date = calendar.advance(today, ql.Period(t_plus, ql.Days))
    frequency = ql.Semiannual
    day_count = ql.ActualActual(ql.ActualActual.ISDA)
    par = 100.0

    bond_helpers = []
    for _, row in curve_set_df.iterrows():
        maturity = pydatetime_to_quantlib_date(row["maturity_date"])
        if np.isnan(row["int_rate"]):
            quote = ql.QuoteHandle(ql.SimpleQuote(row[price_col]))
            tbill = ql.ZeroCouponBond(
                t_plus,
                calendar,
                par,
                maturity,
                ql.ModifiedFollowing,
                100.0,
                bond_settlement_date,
            )
            helper = ql.BondHelper(quote, tbill)
        else:
            schedule = ql.Schedule(
                bond_settlement_date,
                maturity,
                ql.Period(frequency),
                calendar,
                ql.ModifiedFollowing,
                ql.ModifiedFollowing,
                ql.DateGeneration.Backward,
                False,
            )
            helper = ql.FixedRateBondHelper(
                ql.QuoteHandle(ql.SimpleQuote(row[price_col])),
                t_plus,
                100.0,
                schedule,
                [row["int_rate"] / 100],
                day_count,
                ql.ModifiedFollowing,
                par,
            )

        bond_helpers.append(helper)

    spot_dfs: List[pd.DataFrame] = []
    ql_curves: Dict[str, ql.DiscountCurve | ql.ZeroCurve] = {}

    for bs_method in ql_bootstrap_interp_methods:
        if bs_method.split("_")[1] == "f":
            ql_fit_method = ql_piecewise_methods[bs_method]
            curr_curve = ql.FittedBondDiscountCurve(bond_settlement_date, bond_helpers, day_count, ql_fit_method())
            curr_curve.enableExtrapolation()
            ql_curves[bs_method] = curr_curve
        else:
            curr_curve = ql_piecewise_methods[bs_method](bond_settlement_date, bond_helpers, day_count)
            curr_curve.enableExtrapolation()
            ql_curves[bs_method] = curr_curve
        if interpolated_months_num or custom_yearly_tenors:
            if daily_interpolation:
                curr_spot_df = _calc_spot_rates_intep_days(
                    yield_curve=curr_curve,
                    on_rate=on_rate,
                    days=interpolated_months_num * 31,
                    custom_price_col=f"{bs_method}_spot_rate",
                    continuous_compounded_zero=continuous_compounded_zero,
                )
            else:
                curr_spot_df = _calc_spot_rates_intep_months(
                    yield_curve=curr_curve,
                    on_rate=on_rate,
                    months=interpolated_months_num,
                    month_freq=interpolated_curve_yearly_freq,
                    custom_tenors=([i * 12 for i in custom_yearly_tenors] if custom_yearly_tenors else None),
                    custom_price_col=f"{bs_method}_spot_rate",
                    continuous_compounded_zero=continuous_compounded_zero,
                )
        else:
            curr_spot_df = _calc_spot_rates_on_tenors(
                yield_curve=curr_curve,
                on_rate=on_rate,
                custom_price_col=f"{bs_method}_spot_rate",
                continuous_compounded_zero=continuous_compounded_zero,
            )

        spot_dfs.append(curr_spot_df)

    if len(spot_dfs) == 1:
        zero_rates_df = spot_dfs[0]
    else:
        maturity_dates = spot_dfs[0]["maturity_date"].to_list()
        tenors = spot_dfs[0]["time_to_maturity"].to_list()
        merged_df = pd.concat(
            [df[[col for col in df.columns if "spot_rate" in col]] for df in spot_dfs],
            axis=1,
        )
        avg_spot_rate_col = merged_df.mean(axis=1).to_list()

        merged_df.insert(0, "maturity_date", maturity_dates)
        merged_df.insert(1, "time_to_maturity", tenors)
        merged_df["avg_spot_rate"] = avg_spot_rate_col
        zero_rates_df = merged_df

    to_return_dict = {
        "ql_zero_curve_obj": None,
        "spot_rate_df": None,
        "scipy_interp_funcs": None,
    }

    if return_ql_zero_curve:
        if len(ql_bootstrap_interp_methods) > 1:
            print("Get Spot Rates - multiple bs methods passed - returning ql zero curve based on first bs method")
        bs_method = ql_bootstrap_interp_methods[0]
        if bs_method.split("_")[1] == "f":
            ql_fit_method = ql_piecewise_methods[bs_method]
            zero_curve = ql.FittedBondDiscountCurve(bond_settlement_date, bond_helpers, day_count, ql_fit_method())
        else:
            zero_curve = ql_piecewise_methods[bs_method](bond_settlement_date, bond_helpers, day_count)
        zero_curve.enableExtrapolation()
        to_return_dict["ql_zero_curve_obj"] = zero_curve

    to_return_dict["spot_rate_df"] = zero_rates_df

    if return_scipy_interp_func:
        scipy_interp_funcs = {}
        for bs_method in ql_bootstrap_interp_methods:
            scipy_interp_funcs[bs_method] = scipy.interpolate.interp1d(
                to_return_dict["spot_rate_df"]["time_to_maturity"],
                to_return_dict["spot_rate_df"][f"{bs_method}_spot_rate"],
                axis=0,
                kind="linear",
                # bounds_error=False,
                # fill_value="extrapolate",
            )
        to_return_dict["scipy_interp_funcs"] = scipy_interp_funcs

    return to_return_dict


def get_par_rates(
    spot_rates: List[float],
    tenors: List[int],
    select_every_nth_spot_rate: Optional[int] = None,
) -> pd.DataFrame:
    if select_every_nth_spot_rate:
        spot_rates = spot_rates[::select_every_nth_spot_rate]
    par_rates = []
    for tenor in tenors:
        periods = np.arange(0, tenor + 0.5, 0.5)
        curr_spot_rates = spot_rates[: len(periods)].copy()
        discount_factors = [1 / (1 + (s / 100) / 2) ** (2 * t) for s, t in zip(curr_spot_rates, periods)]
        sum_of_dfs = sum(discount_factors[:-1])
        par_rate = (1 - discount_factors[-1]) / sum_of_dfs * 2
        par_rates.append(par_rate * 100)

    return pd.DataFrame(
        {
            "tenor": tenors,
            "par_rate": par_rates,
        }
    )


# TODO match ql interp method with scipy interp func
def get_spot_rates_fitter(
    curve_set_df: pd.DataFrame,
    as_of_date: datetime,
    on_rate: float,
    ql_fitting_methods: Optional[
        List[
            Literal[
                "ql_f_ns",
                "ql_f_nss",
                "ql_f_sp",
                "ql_f_es",
                "ql_f_cbs",
            ]
        ]
    ] = ["ql_f_nss"],
    ql_zero_curve_interp_method: Optional[
        Literal[
            "ql_z_interp_log_lin",
            "ql_z_interp_cubic",
            "ql_z_interp_nat_cubic",
            "ql_z_interp_log_cubic",
            "ql_z_interp_monot_cubic",
        ]
    ] = None,
    daily_interpolation: Optional[bool] = False,
    simple_poly: Optional[int] = None,
    knots: Optional[List[float]] = None,
):
    ql_fitting_methods_dict: Dict[str, ql.DiscountCurve | ql.ZeroCurve] = {
        "ql_f_ns": ql.NelsonSiegelFitting,
        "ql_f_nss": ql.SvenssonFitting,
        "ql_f_sp": ql.SimplePolynomialFitting,
        "ql_f_es": ql.ExponentialSplinesFitting,
        "ql_f_cbs": ql.CubicBSplinesFitting,
    }

    ql_zero_curve_interp_methods_dict: Dict[str, ql.ZeroCurve] = {
        "ql_z_interp_log_lin": ql.LogLinearZeroCurve,
        "ql_z_interp_cubic": ql.CubicZeroCurve,
        "ql_z_interp_nat_cubic": ql.NaturalCubicZeroCurve,
        "ql_z_interp_log_cubic": ql.LogCubicZeroCurve,
        "ql_z_interp_monot_cubic": ql.MonotonicCubicZeroCurve,
    }

    price_cols = ["bid_price", "offer_price", "mid_price", "eod_price"]
    required_cols = ["issue_date", "maturity_date", "int_rate"]
    price_col_exists = any(item in curve_set_df.columns for item in price_cols)
    missing_required_cols = [item for item in required_cols if item not in curve_set_df.columns]

    if not price_col_exists:
        raise ValueError(f"Build Spot Curve - Couldn't find a valid price col in your curve set df - one of {price_cols}")
    if missing_required_cols:
        raise ValueError(f"Build Spot Curve - Missing required curve set cols: {missing_required_cols}")

    price_col = next((item for item in price_cols if item in curve_set_df.columns), None)
    calendar = ql.UnitedStates(m=ql.UnitedStates.GovernmentBond)
    today = calendar.adjust(pydatetime_to_quantlib_date(py_datetime=as_of_date))
    ql.Settings.instance().evaluationDate = today

    t_plus = 1
    bond_settlement_date = calendar.advance(today, ql.Period(t_plus, ql.Days))
    frequency = ql.Semiannual
    day_count = ql.ActualActual(ql.ActualActual.ISDA)
    par = 100.0

    bond_helpers = []
    for _, row in curve_set_df.iterrows():
        maturity = pydatetime_to_quantlib_date(row["maturity_date"])
        if np.isnan(row["int_rate"]):
            quote = ql.QuoteHandle(ql.SimpleQuote(row[price_col]))
            tbill = ql.ZeroCouponBond(
                t_plus,
                calendar,
                par,
                maturity,
                ql.ModifiedFollowing,
                100.0,
                bond_settlement_date,
            )
            helper = ql.BondHelper(quote, tbill)
        else:
            schedule = ql.Schedule(
                bond_settlement_date,
                maturity,
                ql.Period(frequency),
                calendar,
                ql.ModifiedFollowing,
                ql.ModifiedFollowing,
                ql.DateGeneration.Backward,
                False,
            )
            helper = ql.FixedRateBondHelper(
                ql.QuoteHandle(ql.SimpleQuote(row[price_col])),
                t_plus,
                100.0,
                schedule,
                [row["int_rate"] / 100],
                day_count,
                ql.ModifiedFollowing,
                par,
            )

        bond_helpers.append(helper)

    ql_curves: Dict[str, Dict[str, ql.DiscountCurve | ql.ZeroCurve | pd.DataFrame]] = {}
    for fit_method in ql_fitting_methods:
        if fit_method == "ql_f_sp" and not simple_poly:
            continue
        if fit_method == "ql_f_cbs" and not knots:
            continue

        ql_fit_method = ql_fitting_methods_dict[fit_method]

        if fit_method == "ql_f_sp":
            called_ql_fit_method = ql_fit_method(simple_poly)
        elif fit_method == "ql_f_cbs":
            called_ql_fit_method = ql_fit_method(knots)
        else:
            called_ql_fit_method = ql_fit_method()

        curr_curve = ql.FittedBondDiscountCurve(bond_settlement_date, bond_helpers, day_count, called_ql_fit_method)
        curr_curve.enableExtrapolation()
        if fit_method not in ql_curves:
            ql_curves[fit_method] = {
                "ql_fitted_curve": None,
                "ql_zero_curve": None,
                "zero_interp_func": None,
                "df_interp_func": None,
                "comparison_df": None,
            }

        ql_curves[fit_method]["ql_curve"] = curr_curve

        if daily_interpolation:
            dates = [bond_settlement_date + ql.Period(i, ql.Days) for i in range(0, 12 * 30 * 30, 1)]
        else:
            dates = [bond_settlement_date + ql.Period(i, ql.Months) for i in range(0, 12 * 30, 1)]

        discount_factors = [curr_curve.discount(d) for d in dates]
        ttm = [(ql.Date.to_date(d) - ql.Date.to_date(bond_settlement_date)).days / 365 for d in dates]

        eq_zero_rates = []
        eq_zero_rates_dec = []
        for d in dates:
            yrs = (ql.Date.to_date(d) - ql.Date.to_date(bond_settlement_date)).days / 365.0
            zero_rate = curr_curve.zeroRate(yrs, ql.Continuous, True)
            eq_rate = zero_rate.equivalentRate(day_count, ql.Continuous, ql.NoFrequency, bond_settlement_date, d).rate()
            eq_zero_rates.append(eq_rate * 100)
            eq_zero_rates_dec.append(eq_rate)
        eq_zero_rates[0] = on_rate
        eq_zero_rates_dec[0] = on_rate / 100

        ql_curves[fit_method]["zero_interp_func"] = scipy.interpolate.interp1d(ttm, eq_zero_rates, axis=0, kind="linear", fill_value="extrapolate")
        ql_curves[fit_method]["df_interp_func"] = scipy.interpolate.interp1d(ttm, discount_factors, axis=0, kind="linear", fill_value="extrapolate")
        zero_curve = (
            ql.ZeroCurve(dates, eq_zero_rates_dec, day_count)
            if not ql_zero_curve_interp_method
            else ql_zero_curve_interp_methods_dict[ql_zero_curve_interp_method](dates, eq_zero_rates_dec, day_count)
        )
        zero_curve.enableExtrapolation()
        ql_curves[fit_method]["ql_zero_curve"] = zero_curve

    return ql_curves


def reprice_bonds_single_zero_curve(as_of_date: datetime, ql_zero_curve: ql.ZeroCurve, curve_set_df: pd.DataFrame) -> pd.DataFrame:
    yield_curve_handle = ql.YieldTermStructureHandle(ql_zero_curve)
    engine = ql.DiscountingBondEngine(yield_curve_handle)

    price_cols = ["bid_price", "offer_price", "mid_price", "eod_price"]
    required_cols = ["issue_date", "maturity_date", "int_rate"]
    price_col_exists = any(item in curve_set_df.columns for item in price_cols)
    missing_required_cols = [item for item in required_cols if item not in curve_set_df.columns]

    if not price_col_exists:
        raise ValueError(f"Build Spot Curve - Couldn't find a valid price col in your curve set df - one of {price_cols}")
    if missing_required_cols:
        raise ValueError(f"Build Spot Curve - Missing required curve set cols: {missing_required_cols}")

    price_col = next((item for item in price_cols if item in curve_set_df.columns), None)
    quote_type = price_col.split("_")[0]
    yield_col = f"{quote_type}_yield"

    if yield_col not in curve_set_df.columns:
        raise ValueError(f"Build Spot Curve - Missing required curve set cols: {yield_col}")

    calendar = ql.UnitedStates(m=ql.UnitedStates.GovernmentBond)
    today = calendar.adjust(pydatetime_to_quantlib_date(py_datetime=as_of_date))
    ql.Settings.instance().evaluationDate = today

    t_plus = 1
    bond_settlement_date = calendar.advance(today, ql.Period(t_plus, ql.Days))
    frequency = ql.Semiannual
    day_count = ql.ActualActual(ql.ActualActual.ISDA)
    par = 100.0

    bonds: List[Dict[str, ql.FixedRateBond | ql.ZeroCouponBond]] = []
    for _, row in curve_set_df.iterrows():
        try:
            maturity = pydatetime_to_quantlib_date(row["maturity_date"])
            if np.isnan(row["int_rate"]):
                bond_ql = ql.ZeroCouponBond(
                    t_plus,
                    calendar,
                    par,
                    maturity,
                    ql.ModifiedFollowing,
                    100.0,
                    bond_settlement_date,
                )
                bond_rl: rl.FixedRateBond = rl.Bill(
                    termination=row["maturity_date"],
                    effective=row["issue_date"],
                    calendar="nyc",
                    modifier="NONE",
                    currency="usd",
                    convention="Act360",
                    settle=1,
                    curves="bill_curve",
                    calc_mode="us_gbb",
                )
            else:
                schedule = ql.Schedule(
                    bond_settlement_date,
                    maturity,
                    ql.Period(frequency),
                    calendar,
                    ql.ModifiedFollowing,
                    ql.ModifiedFollowing,
                    ql.DateGeneration.Backward,
                    False,
                )
                bond_ql = ql.FixedRateBond(
                    t_plus,
                    100.0,
                    schedule,
                    [row["int_rate"] / 100],
                    day_count,
                    ql.ModifiedFollowing,
                )
                bond_rl: rl.FixedRateBond = rl.FixedRateBond(
                    effective=row["issue_date"],
                    termination=row["maturity_date"],
                    fixed_rate=row["int_rate"],
                    spec="ust",
                    calc_mode="ust_31bii",
                )

            curr_accrued_amount = bond_rl.accrued(quantlib_date_to_pydatetime(bond_settlement_date))
            bond_ql.setPricingEngine(engine)
            curr_calced_npv = bond_ql.NPV()
            curr_calced_ytm = (
                bond_ql.bondYield(
                    curr_calced_npv,
                    day_count,
                    ql.Compounded,
                    frequency,
                    bond_settlement_date,
                )
                * 100
            )
            bonds.append(
                {
                    "cusip": row["cusip"],
                    "label": row["label"],
                    "issue_date": row["issue_date"],
                    "maturity_date": row["maturity_date"],
                    "time_to_maturity": row["time_to_maturity"],
                    "high_investment_rate": row["high_investment_rate"],
                    "int_rate": row["int_rate"],
                    "rank": row["rank"] if "rank" in curve_set_df.columns else None,
                    "outstanding": (row["outstanding_amt"] if "outstanding_amt" in curve_set_df.columns else None),
                    "soma_holdings": (row["parValue"] if "parValue" in curve_set_df.columns else None),
                    "stripping_amount": (row["portion_stripped_amt"] if "portion_stripped_amt" in curve_set_df.columns else None),
                    "free_float": (row["free_float"] if "free_float" in curve_set_df.columns else None),
                    yield_col: row[yield_col],
                    price_col: row[price_col],
                    "accured": curr_accrued_amount,
                    "repriced_npv": curr_calced_npv,
                    "repriced_ytm": curr_calced_ytm,
                    "price_spread": row[price_col] + curr_accrued_amount - curr_calced_npv,
                    "ytm_spread": (row[yield_col] - curr_calced_ytm) * 100,
                }
            )
        except Exception as e:
            print(row["cusip"], e)

    return pd.DataFrame(bonds)


def reprice_bonds(
    as_of_date: datetime,
    ql_zero_curves: Dict[str, ql.ZeroCurve],
    curve_set_df: pd.DataFrame,
):
    price_cols = ["bid_price", "offer_price", "mid_price", "eod_price"]
    required_cols = ["issue_date", "maturity_date", "int_rate"]
    price_col_exists = any(item in curve_set_df.columns for item in price_cols)
    missing_required_cols = [item for item in required_cols if item not in curve_set_df.columns]

    if not price_col_exists:
        raise ValueError(f"Build Spot Curve - Couldn't find a valid price col in your curve set df - one of {price_cols}")
    if missing_required_cols:
        raise ValueError(f"Build Spot Curve - Missing required curve set cols: {missing_required_cols}")

    price_col = next((item for item in price_cols if item in curve_set_df.columns), None)
    quote_type = price_col.split("_")[0]
    yield_col = f"{quote_type}_yield"

    if yield_col not in curve_set_df.columns:
        raise ValueError(f"Build Spot Curve - Missing required curve set cols: {yield_col}")

    calendar = ql.UnitedStates(m=ql.UnitedStates.GovernmentBond)
    today = calendar.adjust(pydatetime_to_quantlib_date(py_datetime=as_of_date))
    ql.Settings.instance().evaluationDate = today

    t_plus = 1
    bond_settlement_date = calendar.advance(today, ql.Period(t_plus, ql.Days))
    frequency = ql.Semiannual
    day_count = ql.ActualActual(ql.ActualActual.ISDA)
    par = 100.0

    bonds: List[Dict[str, ql.FixedRateBond | ql.ZeroCouponBond]] = []
    for _, row in curve_set_df.iterrows():
        try:
            maturity = pydatetime_to_quantlib_date(row["maturity_date"])
            if np.isnan(row["int_rate"]):
                bond_ql = ql.ZeroCouponBond(
                    t_plus,
                    calendar,
                    par,
                    maturity,
                    ql.ModifiedFollowing,
                    100.0,
                    bond_settlement_date,
                )
                bond_rl: rl.FixedRateBond = rl.Bill(
                    termination=row["maturity_date"],
                    effective=row["issue_date"],
                    calendar="nyc",
                    modifier="NONE",
                    currency="usd",
                    convention="Act360",
                    settle=1,
                    curves="bill_curve",
                    calc_mode="us_gbb",
                )
            else:
                schedule = ql.Schedule(
                    bond_settlement_date,
                    maturity,
                    ql.Period(frequency),
                    calendar,
                    ql.ModifiedFollowing,
                    ql.ModifiedFollowing,
                    ql.DateGeneration.Backward,
                    False,
                )
                bond_ql = ql.FixedRateBond(
                    t_plus,
                    100.0,
                    schedule,
                    [row["int_rate"] / 100],
                    day_count,
                    ql.ModifiedFollowing,
                )
                bond_rl: rl.FixedRateBond = rl.FixedRateBond(
                    effective=row["issue_date"],
                    termination=row["maturity_date"],
                    fixed_rate=row["int_rate"],
                    spec="ust",
                    calc_mode="ust_31bii",
                )

            curr_accrued_amount = bond_rl.accrued(quantlib_date_to_pydatetime(bond_settlement_date))
            curr_row = {
                "cusip": row["cusip"],
                "label": row["label"],
                "issue_date": row["issue_date"],
                "maturity_date": row["maturity_date"],
                "time_to_maturity": row["time_to_maturity"],
                "high_investment_rate": row["high_investment_rate"],
                "int_rate": row["int_rate"],
                "rank": row["rank"] if "rank" in curve_set_df.columns else None,
                "outstanding": (row["outstanding_amt"] if "outstanding_amt" in curve_set_df.columns else None),
                "soma_holdings": (row["parValue"] if "parValue" in curve_set_df.columns else None),
                "stripping_amount": (row["portion_stripped_amt"] if "portion_stripped_amt" in curve_set_df.columns else None),
                "free_float": (row["free_float"] if "free_float" in curve_set_df.columns else None),
                yield_col: row[yield_col],
                price_col: row[price_col],
                "accured": curr_accrued_amount,
            }

            for label, ql_zero_curve in ql_zero_curves.items():
                yield_curve_handle = ql.YieldTermStructureHandle(ql_zero_curve)
                engine = ql.DiscountingBondEngine(yield_curve_handle)
                bond_ql.setPricingEngine(engine)
                curr_calced_npv = bond_ql.NPV()
                curr_calced_ytm = (
                    bond_ql.bondYield(
                        curr_calced_npv,
                        day_count,
                        ql.Compounded,
                        frequency,
                        bond_settlement_date,
                    )
                    * 100
                )

                curr_price_spread = row[price_col] + curr_accrued_amount - curr_calced_npv
                curr_ytm_spread = (row[yield_col] - curr_calced_ytm) * 100

                curr_row[f"{label}_repriced_npv"] = curr_calced_npv
                curr_row[f"{label}_repriced_ytm"] = curr_calced_ytm
                curr_row[f"{label}_price_spread"] = curr_price_spread
                curr_row[f"{label}_ytm_spread"] = curr_ytm_spread

            bonds.append(curr_row)

        except Exception as e:
            print(row["cusip"], e)

    return pd.DataFrame(bonds)


# def reprice_bonds_single_scipy_interpolated_curve(
#     as_of_date: datetime,
#     scipy_interp_curve: scipy.interpolate.interp1d,
#     curve_set_df: pd.DataFrame,
#     yield_col: str,
#     price_col: str,
#     coupon_col: str,
#     mat_col: str,
# ) -> pd.DataFrame:
#     # price_cols = ["bid_price", "offer_price", "mid_price", "eod_price"]
#     # required_cols = ["issue_date", "maturity_date", "int_rate"]
#     # price_col_exists = any(item in curve_set_df.columns for item in price_cols)
#     # missing_required_cols = [
#     #     item for item in required_cols if item not in curve_set_df.columns
#     # ]

#     # if not price_col_exists:
#     #     raise ValueError(
#     #         f"Build Spot Curve - Couldn't find a valid price col in your curve set df - one of {price_cols}"
#     #     )
#     # if missing_required_cols:
#     #     raise ValueError(
#     #         f"Build Spot Curve - Missing required curve set cols: {missing_required_cols}"
#     #     )

#     # price_col = next(
#     #     (item for item in price_cols if item in curve_set_df.columns), None
#     # )
#     # quote_type = price_col.split("_")[0]
#     # yield_col = f"{quote_type}_yield"

#     # if yield_col not in curve_set_df.columns:
#     #     raise ValueError(
#     #         f"Build Spot Curve - Missing required curve set cols: {yield_col}"
#     #     )

#     as_of_date = np.datetime64(as_of_date)
#     t_plus = 1
#     bond_settlement_date = as_of_date + BDay(t_plus)
#     frequency = 2
#     par = 100.0

#     bonds: List[Dict[str, float]] = []
#     for _, row in curve_set_df.iterrows():
#         try:
#             maturity_date = np.datetime64(row[mat_col])
#             time_to_maturity = (maturity_date - bond_settlement_date).days / 365
#             discount_factor = np.exp(
#                 -scipy_interp_curve(time_to_maturity) * time_to_maturity
#             )

#             if np.isnan(row[coupon_col]):
#                 curr_calced_npv = par * discount_factor
#             else:
#                 num_periods = int(frequency * time_to_maturity)
#                 cash_flows = [
#                     (par * (row[coupon_col] / 100) / frequency) * discount_factor**i
#                     for i in range(1, num_periods + 1)
#                 ]
#                 cash_flows.append(par * discount_factor)
#                 curr_calced_npv = sum(cash_flows)

#             curr_calced_ytm = -np.log(curr_calced_npv / par) / time_to_maturity * 100
#             bonds.append(
#                 {
#                     "label": row["label"],
#                     yield_col: row[yield_col],
#                     price_col: row[price_col],
#                     "repriced_npv": curr_calced_npv,
#                     "repriced_ytm": curr_calced_ytm,
#                     "price_spread": row[price_col] - curr_calced_npv,
#                     "ytm_spread": (row[yield_col] - curr_calced_ytm) * 100,
#                 }
#             )
#         except Exception as e:
#             print(row["label"], e)

#     return pd.DataFrame(bonds)


def calc_ust_metrics(
    bond_info: str, 
    curr_price: float, 
    curr_ytm: float, 
    on_rate: float,
    as_of_date: datetime, 
    scipy_interp: scipy.interpolate.interpolate, 
    print_bond_info=False, 
):
    calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    day_count = ql.ActualActual(ql.ActualActual.ISDA)
    times = np.arange(0, 30.5, 0.5)

    dates = []
    zero_rates = []
    today = calendar.adjust(pydatetime_to_quantlib_date(py_datetime=as_of_date))
    ql.Settings.instance().evaluationDate = today
    t_plus = 2
    bond_settlement_date = calendar.advance(today, ql.Period(t_plus, ql.Days))

    for t in times:
        if t == 0:
            dates.append(today)
            zero_rate = scipy_interp(0)
            zero_rates.append(float(zero_rate) / 100)
        else:
            maturity_date = calendar.advance(pydatetime_to_quantlib_date(as_of_date), ql.Period(int(round(t * 365)), ql.Days))
            dates.append(maturity_date)
            zero_rate = scipy_interp(t)
            zero_rates.append(float(zero_rate) / 100)

    zero_curve = ql.ZeroCurve(dates, zero_rates, day_count, calendar)
    yield_curve_handle = ql.YieldTermStructureHandle(zero_curve)
    engine = ql.DiscountingBondEngine(yield_curve_handle)

    ql_fixed_rate_bond_obj = to_quantlib_fixed_rate_bond_obj(bond_info=bond_info, as_of_date=as_of_date, print_bond_info=print_bond_info)
    ql_fixed_rate_bond_obj.setPricingEngine(engine)

    try:
        zspread = ql.BondFunctions.zSpread(
            ql_fixed_rate_bond_obj,
            curr_price,
            yield_curve_handle.currentLink(),
            day_count,
            ql.Compounded,
            ql.Semiannual,
            pydatetime_to_quantlib_date(as_of_date),
            1.0e-16,
            1000000,
            0.0,
        )
        spread1 = ql.SimpleQuote(zspread)
        spread_handle1 = ql.QuoteHandle(spread1)
        ts_spreaded1 = ql.ZeroSpreadedTermStructure(yield_curve_handle, spread_handle1, ql.Compounded, ql.Semiannual)
        ts_spreaded_handle1 = ql.YieldTermStructureHandle(ts_spreaded1)
        ycsin = ts_spreaded_handle1
        bond_engine = ql.DiscountingBondEngine(ycsin)
        ql_fixed_rate_bond_obj.setPricingEngine(bond_engine)
        zspread_impl_clean_price = ql_fixed_rate_bond_obj.cleanPrice()
        zspread = zspread * 10000
    except:
        zspread = None
        zspread_impl_clean_price = None

    rate = ql.InterestRate(curr_ytm / 100, day_count, ql.Compounded, ql.Semiannual)
    bps_value = ql.BondFunctions.basisPointValue(ql_fixed_rate_bond_obj, rate, bond_settlement_date)
    dv01_1mm = bps_value * 1_000_000 / 100
    impl_spot_3m_fwds = calc_ust_impl_spot_n_fwd_curve(n=0.25, scipy_interp_curve=scipy_interp, return_scipy=True)
    impl_spot_6m_fwds = calc_ust_impl_spot_n_fwd_curve(n=0.5, scipy_interp_curve=scipy_interp, return_scipy=True)
    impl_spot_12m_fwds = calc_ust_impl_spot_n_fwd_curve(n=1, scipy_interp_curve=scipy_interp, return_scipy=True)
    bond_ttm: timedelta = (bond_info["maturity_date"] - as_of_date)

    metrics = {
        "Date": as_of_date,
        "zspread": zspread,
        "zspread_impl_clean_price": zspread_impl_clean_price,
        "clean_price": ql.BondFunctions.cleanPrice(ql_fixed_rate_bond_obj, rate),
        "dirty_price": ql.BondFunctions.dirtyPrice(ql_fixed_rate_bond_obj, yield_curve_handle.currentLink(), bond_settlement_date),
        "accrued_amount": ql.BondFunctions.accruedAmount(ql_fixed_rate_bond_obj, bond_settlement_date),
        "bps": bps_value,
        "dv01_1mm": dv01_1mm,
        "mac_duration": ql.BondFunctions.duration(ql_fixed_rate_bond_obj, rate, ql.Duration.Macaulay),
        "mod_duration": ql.BondFunctions.duration(ql_fixed_rate_bond_obj, rate, ql.Duration.Modified),
        "convexity": ql.BondFunctions.convexity(ql_fixed_rate_bond_obj, rate, bond_settlement_date),
        "basis_point_value": ql.BondFunctions.basisPointValue(ql_fixed_rate_bond_obj, rate, bond_settlement_date),
        "yield_value_basis_point": ql.BondFunctions.yieldValueBasisPoint(ql_fixed_rate_bond_obj, rate, bond_settlement_date),
        "rough_carry": curr_ytm - on_rate, 
        "rough_3m_rolldown": (impl_spot_3m_fwds(float(bond_ttm.days / 365)) - curr_ytm) * 100,
        "rough_6m_rolldown": (impl_spot_6m_fwds(float(bond_ttm.days / 365)) - curr_ytm) * 100,
        "rough_12m_rolldown": (impl_spot_12m_fwds(float(bond_ttm.days / 365)) - curr_ytm) * 100
    }

    return metrics


def calc_ust_impl_spot_n_fwd_curve(n: float | int, scipy_interp_curve: scipy.interpolate.interpolate, return_scipy=False) -> Dict[float, float]:
    cfs = np.arange(0.5, 30 + 1, 0.5)
    implied_spot_rates = []
    first_n_cfs = 0
    for t in cfs:
        if t > n:
            Z_t_temp = scipy_interp_curve(t)
            Z_n = scipy_interp_curve(n)
            Z_n_t = (Z_t_temp * t - Z_n * n) / (t - n)
            implied_spot_rates.append(Z_n_t)
        else:
            if return_scipy:
                # implied_spot_rates.append(0)
                first_n_cfs += 1
            else:
                implied_spot_rates.append(np.nan)
    
    if return_scipy:
        return scipy.interpolate.CubicSpline(x=cfs[first_n_cfs:], y=implied_spot_rates)
    return dict(zip(cfs, implied_spot_rates))        


def calc_ust_metrics_parallel(
    bonds: List[Dict],
    curve_data_fetcher: CurveDataFetcher,
    inter_func_key: str,
    max_workers=16,
) -> List[Dict]:

    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_args = {
            executor.submit(
                calc_ust_metrics,
                curve_data_fetcher.ust_data_fetcher.cusip_to_ust_label(cusip=bond["cusip"]),
                bond["eod_price"],
                bond["eod_yield"],
                bond["Date"],
                bond[inter_func_key],
            ): (curve_data_fetcher.ust_data_fetcher.cusip_to_ust_label(cusip=bond["cusip"]), bond["Date"])
            for bond in bonds
        }

        for future in as_completed(future_to_args):
            bond_info, as_of_date = future_to_args[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"Bond {bond_info} for date {as_of_date} generated an exception: {exc}")

    return results
