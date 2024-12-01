from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import QuantLib as ql
import rateslib as rl
import scipy
import scipy.interpolate
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import BDay, BMonthEnd
from termcolor import colored

UPI_MIGRATE_DATE = datetime(2024, 1, 29)
SCHEMA_CHANGE_2022 = datetime(2022, 12, 3)

DEFAULT_SWAP_TENORS = [
    "1D",
    "1W",
    "2W",
    "3W",
    "1M",
    "2M",
    "3M",
    "4M",
    "5M",
    "6M",
    "9M",
    "12M",
    "18M",
    "2Y",
    "3Y",
    "4Y",
    "5Y",
    "6Y",
    "7Y",
    "8Y",
    "9Y",
    "10Y",
    "12Y",
    "15Y",
    "20Y",
    "25Y",
    "30Y",
    "40Y",
    "50Y",
]


def datetime_to_ql_date(dt):
    day = dt.day
    month = dt.month
    year = dt.year

    ql_month = {
        1: ql.January,
        2: ql.February,
        3: ql.March,
        4: ql.April,
        5: ql.May,
        6: ql.June,
        7: ql.July,
        8: ql.August,
        9: ql.September,
        10: ql.October,
        11: ql.November,
        12: ql.December,
    }[month]

    return ql.Date(day, ql_month, year)


def tenor_to_ql_period(tenor):
    unit = tenor[-1]
    value = int(tenor[:-1])

    if unit == "D":
        return ql.Period(value, ql.Days)
    elif unit == "W":
        return ql.Period(value, ql.Weeks)
    elif unit == "M":
        return ql.Period(value, ql.Months)
    elif unit == "Y":
        return ql.Period(value, ql.Years)
    else:
        raise ValueError("Invalid tenor unit. Must be one of 'D', 'W', 'M', 'Y'.")


def parse_frequency(frequency):
    number = int("".join(filter(str.isdigit, frequency)))
    unit = "".join(filter(str.isalpha, frequency))

    if unit == "Y":
        return number, ql.Years
    elif unit == "M":
        return number, ql.Months
    elif unit == "D":
        return number, ql.Days
    elif unit == "W":
        return number, ql.Weeks
    else:
        raise ValueError("Invalid period string format")


def tenor_to_years(tenor):
    num = float(tenor[:-1])
    unit = tenor[-1].upper()
    if unit == "D":
        return num / 360
    if unit == "W":
        return num / 52
    elif unit == "M":
        return num / 12
    elif unit == "Y":
        return num
    else:
        raise ValueError(f"Unknown tenor unit: {tenor}")


def expiry_to_tenor(expiry: datetime, as_of_date: datetime):
    delta_days = (expiry - as_of_date).days
    if delta_days < 0:
        raise ValueError("Expiry date must be after the as_of_date.")
    if delta_days == 1:
        return "1D"
    elif delta_days <= 7:
        return "1W"
    elif delta_days <= 14:
        return "2W"
    elif delta_days <= 21:
        return "3W"
    elif delta_days <= 31:
        return "1M"
    elif delta_days <= 61:
        return "2M"
    elif delta_days <= 92:
        return "3M"
    elif delta_days <= 122:
        return "4M"
    elif delta_days <= 152:
        return "5M"
    elif delta_days <= 183:
        return "6M"
    elif delta_days <= 274:
        return "9M"
    elif delta_days <= 365:
        return "12M"
    elif delta_days <= 547:
        return "18M"
    elif delta_days <= 730:
        return "2Y"
    elif delta_days <= 1095:
        return "3Y"
    elif delta_days <= 1460:
        return "4Y"
    elif delta_days <= 1825:
        return "5Y"
    elif delta_days <= 2190:
        return "6Y"
    elif delta_days <= 2555:
        return "7Y"
    elif delta_days <= 2920:
        return "8Y"
    elif delta_days <= 3285:
        return "9Y"
    elif delta_days <= 3650:
        return "10Y"
    elif delta_days <= 4380:
        return "12Y"
    elif delta_days <= 5475:
        return "15Y"
    elif delta_days <= 7300:
        return "20Y"
    elif delta_days <= 9125:
        return "25Y"
    elif delta_days <= 10950:
        return "30Y"
    elif delta_days <= 14600:
        return "40Y"
    else:
        return "50Y"


def get_sofr_ois(
    forward_term_structure,
    fixed_coupon_rate,
    notional,
    start_date,
    maturity_date,
    fixed_payment_frequency,
    pay_rec=ql.Swap.Payer,
):
    calendar = ql.UnitedStates(ql.UnitedStates.FederalReserve)

    if not isinstance(start_date, ql.Date):
        start_date = datetime_to_ql_date(start_date)
    if not isinstance(maturity_date, ql.Date):
        maturity_date = datetime_to_ql_date(maturity_date)

    sofr_index = ql.Sofr(forward_term_structure)
    fixed_leg_tenor = ql.Period(*parse_frequency(fixed_payment_frequency))
    fixed_schedule = ql.Schedule(
        start_date, maturity_date, fixed_leg_tenor, calendar, ql.ModifiedFollowing, ql.ModifiedFollowing, ql.DateGeneration.Forward, False
    )
    ois_swap = ql.OvernightIndexedSwap(pay_rec, notional, fixed_schedule, fixed_coupon_rate, ql.Actual360(), sofr_index)

    return ois_swap


def build_ql_piecewise_curves(
    df: pd.DataFrame,
    as_of_date: datetime,
    is_ois: bool,
    ql_index,
    tenor_col: str = "Tenor",
    fixed_rate_col: str = "Fixed Rate",
    settlement_t_plus: int = 2,
    payment_lag: int = 2,
    fixed_leg_frequency=None,
    fixed_leg_daycount=None,
    fixed_leg_convention=None,
    fixed_leg_calendar=None,
    logLinearDiscount: bool = False,
    logCubicDiscount: bool = False,
    linearZero: bool = False,
    cubicZero: bool = False,
    linearForward: bool = False,
    splineCubicDiscount: bool = False,
) -> Dict[
    str,
    ql.PiecewiseLogLinearDiscount
    | ql.PiecewiseLogCubicDiscount
    | ql.PiecewiseLinearZero
    | ql.PiecewiseCubicZero
    | ql.PiecewiseLinearForward
    | ql.PiecewiseSplineCubicDiscount,
]:
    if not isinstance(as_of_date, ql.Date):
        evaluation_date = datetime_to_ql_date(as_of_date)

    try:
        ql.Settings.instance().evaluationDate = evaluation_date
    except Exception as e:
        if "degenerate" in str(e):
            plus_one_bday_as_of_date = as_of_date + BDay(1)
            ql.Settings.instance().evaluationDate = datetime_to_ql_date(plus_one_bday_as_of_date)
        else:
            raise e

    # examples:
    # https://quant.stackexchange.com/questions/64342/rfr-boostrapping-using-rfr-ois-is-convexity-adjustment-technically-necessary?rq=1
    # https://quant.stackexchange.com/questions/74766/bootstrapping-sofr-curve-and-swap-payment-lag
    # https://www.jpmorgan.com/content/dam/jpm/global/disclosures/IN/usd-inr-irs.pdf
    # https://www.tradeweb.com/49b49f/globalassets/our-businesses/market-regulation/sef-2023/tw---mat-submission.5.22.23.pdf

    # can just ignore payment pay b/c convexity adjustment is negligible for bootstrapping purposes
    # cubic/spline methods are relatively ill conditioned to the piecewie linear methods -

    if is_ois:
        settlement_date = as_of_date + BDay(2)
        is_end_of_bmonth = BMonthEnd().rollforward(settlement_date) == settlement_date

        try:
            helpers = [
                ql.OISRateHelper(
                    settlement_t_plus,
                    tenor_to_ql_period(row[tenor_col]),
                    ql.QuoteHandle(ql.SimpleQuote(row[fixed_rate_col])),
                    ql_index,
                    paymentConvention=fixed_leg_convention,
                    paymentCalendar=fixed_leg_calendar,
                    paymentFrequency=fixed_leg_frequency if tenor_to_ql_period(row[tenor_col]) > ql.Period("18M") else ql.Once,
                    pillar=ql.Pillar.MaturityDate,
                    telescopicValueDates=False,
                    endOfMonth=is_end_of_bmonth,
                )
                for _, row in df.iterrows()
            ]
        except Exception as e:
            # https://github.com/lballabio/QuantLib/issues/1405
            if "degenerate" in str(e):
                plus_one_bday_as_of_date = as_of_date + BDay(1)
                ql.Settings.instance().evaluationDate = datetime_to_ql_date(plus_one_bday_as_of_date)
                helpers = [
                    ql.OISRateHelper(
                        settlement_t_plus,
                        tenor_to_ql_period(row[tenor_col]),
                        ql.QuoteHandle(ql.SimpleQuote(row[fixed_rate_col])),
                        ql_index,
                        paymentConvention=fixed_leg_convention,
                        paymentCalendar=fixed_leg_calendar,
                        paymentFrequency=fixed_leg_frequency if tenor_to_ql_period(row[tenor_col]) > ql.Period("18M") else ql.Once,
                        pillar=ql.Pillar.MaturityDate,
                        telescopicValueDates=False,
                        endOfMonth=is_end_of_bmonth,
                    )
                    for _, row in df.iterrows()
                ]
            else:
                raise e

        try:
            # https://github.com/lballabio/QuantLib/issues/700
            with_payment_lag_helpers = [
                ql.OISRateHelper(
                    settlement_t_plus,
                    tenor_to_ql_period(row[tenor_col]),
                    ql.QuoteHandle(ql.SimpleQuote(row[fixed_rate_col])),
                    ql_index,
                    paymentLag=payment_lag,
                    paymentConvention=fixed_leg_convention,
                    paymentCalendar=fixed_leg_calendar,
                    paymentFrequency=fixed_leg_frequency if tenor_to_ql_period(row[tenor_col]) > ql.Period("18M") else ql.Once,
                    pillar=ql.Pillar.MaturityDate,
                    telescopicValueDates=False,
                    endOfMonth=is_end_of_bmonth,
                )
                for _, row in df.iterrows()
            ]
        except Exception as e:
            # https://github.com/lballabio/QuantLib/issues/1405
            if "degenerate" in str(e):
                plus_one_bday_as_of_date = as_of_date + BDay(1)
                ql.Settings.instance().evaluationDate = datetime_to_ql_date(plus_one_bday_as_of_date)
                # https://github.com/lballabio/QuantLib/issues/700
                with_payment_lag_helpers = [
                    ql.OISRateHelper(
                        settlement_t_plus,
                        tenor_to_ql_period(row[tenor_col]),
                        ql.QuoteHandle(ql.SimpleQuote(row[fixed_rate_col])),
                        ql_index,
                        paymentLag=payment_lag,
                        paymentConvention=fixed_leg_convention,
                        paymentCalendar=fixed_leg_calendar,
                        paymentFrequency=fixed_leg_frequency if tenor_to_ql_period(row[tenor_col]) > ql.Period("18M") else ql.Once,
                        pillar=ql.Pillar.MaturityDate,
                        telescopicValueDates=False,
                        endOfMonth=is_end_of_bmonth,
                    )
                    for _, row in df.iterrows()
                ]
            else:
                raise e

        piecewise_with_payment_lag_params = [evaluation_date, with_payment_lag_helpers, fixed_leg_daycount]

    else:
        helpers = [
            ql.SwapRateHelper(
                ql.QuoteHandle(ql.SimpleQuote(row[fixed_rate_col])),
                tenor_to_ql_period(row[tenor_col]),
                fixed_leg_calendar,
                fixed_leg_frequency,
                fixed_leg_convention,
                fixed_leg_daycount,
                ql_index,
            )
            for _, row in df.iterrows()
        ]

    piecewise_params = [evaluation_date, helpers, fixed_leg_daycount]
    piecewise_curves = {}

    try:
        if is_ois:
            piecewise_curves["logLinearDiscount"] = (
                ql.PiecewiseLogLinearDiscount(*piecewise_with_payment_lag_params, ql.IterativeBootstrap(accuracy=1e-3, maxEvaluations=150))
                if logLinearDiscount
                else None
            )
        else:
            piecewise_curves["logLinearDiscount"] = (
                ql.PiecewiseLogLinearDiscount(*piecewise_params, ql.IterativeBootstrap(accuracy=1e-3, maxEvaluations=150))
                if logLinearDiscount
                else None
            )
    except Exception as e:
        piecewise_curves["logLinearDiscount"] = None
        print(colored(f"Quantlib Curve Build Error --- {as_of_date} - logLinearDiscount --- {str(e)}"))
    try:
        piecewise_curves["logCubicDiscount"] = (
            ql.PiecewiseLogCubicDiscount(*piecewise_params, ql.IterativeBootstrap(accuracy=1e-3, maxEvaluations=150)) if logCubicDiscount else None
        )
    except Exception as e:
        piecewise_curves["logCubicDiscount"] = None
        print(colored(f"Quantlib Curve Build Error --- {as_of_date} - logCubicDiscount --- {str(e)}"))
    try:
        if is_ois:
            piecewise_curves["linearZero"] = (
                ql.PiecewiseLinearZero(*piecewise_with_payment_lag_params, ql.IterativeBootstrap(accuracy=1e-3, maxEvaluations=150))
                if linearZero
                else None
            )
        else:
            piecewise_curves["linearZero"] = (
                ql.PiecewiseLinearZero(*piecewise_params, ql.IterativeBootstrap(accuracy=1e-3, maxEvaluations=150)) if linearZero else None
            )
    except Exception as e:
        piecewise_curves["linearZero"] = None
        print(colored(f"Quantlib Curve Build Error --- {as_of_date} - linearZero --- {str(e)}"))
    try:
        piecewise_curves["cubicZero"] = (
            ql.PiecewiseCubicZero(*piecewise_params, ql.IterativeBootstrap(accuracy=1e-3, maxEvaluations=150)) if cubicZero else None
        )
    except Exception as e:
        piecewise_curves["cubicZero"] = None
        print(colored(f"Quantlib Curve Build Error --- {as_of_date} - cubicZero --- {str(e)}"))
    try:
        if is_ois:
            piecewise_curves["linearForward"] = (
                ql.PiecewiseLinearForward(*piecewise_with_payment_lag_params, ql.IterativeBootstrap(accuracy=1e-3, maxEvaluations=150))
                if linearForward
                else None
            )
        else:
            piecewise_curves["linearForward"] = (
                ql.PiecewiseLinearForward(*piecewise_params, ql.IterativeBootstrap(accuracy=1e-3, maxEvaluations=150)) if linearForward else None
            )
    except Exception as e:
        piecewise_curves["linearForward"] = None
        print(colored(f"Quantlib Curve Build Error --- {as_of_date} - linearForward --- {str(e)}"))
    try:
        piecewise_curves["splineCubicDiscount"] = (
            ql.PiecewiseSplineCubicDiscount(*piecewise_params, ql.IterativeBootstrap(accuracy=1e-3, maxEvaluations=150))
            if splineCubicDiscount
            else None
        )
    except Exception as e:
        piecewise_curves["splineCubicDiscount"] = None
        print(colored(f"Quantlib Curve Build Error --- {as_of_date} - splineCubicDiscount --- {str(e)}"))

    return piecewise_curves

def format_swap_time_and_sales(df: pd.DataFrame, as_of_date: datetime, tenors_to_interpolate: Optional[List[str] | bool] = None, verbose=False):
    swap_columns = [
        "Event timestamp",
        "Execution Timestamp",
        "Effective Date",
        "Expiration Date",
        # "Platform identifier",
        "Tenor",
        "Fwd",
        "Fixed Rate",
        "Direction",
        "Notional Amount",
        "Unique Product Identifier",
        "UPI FISN",
        "UPI Underlier Name",
    ]

    if as_of_date < SCHEMA_CHANGE_2022:
        df = df.rename(
            columns={
                "Event Timestamp": "Event timestamp",
                "Leg 1 - Floating Rate Index": "Underlier ID-Leg 1",
                "Leg 2 - Floating Rate Index": "Underlier ID-Leg 2",
                "Fixed Rate 1": "Fixed rate-Leg 1",
                "Fixed Rate 2": "Fixed rate-Leg 2",
                "Notional Amount 1": "Notional amount-Leg 1",
                "Notional Amount 2": "Notional amount-Leg 2",
            }
        )

    if as_of_date < UPI_MIGRATE_DATE or as_of_date < SCHEMA_CHANGE_2022:
        df["Unique Product Identifier"] = "DNE"
        df["UPI FISN"] = "DNE"
        df["UPI Underlier Name"] = df["Underlier ID-Leg 1"].combine_first(df["Underlier ID-Leg 2"])

    date_columns = [
        "Event timestamp",
        "Execution Timestamp",
        "Effective Date",
        "Expiration Date",
    ]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])
        df[col] = df[col].dt.tz_localize(None)

    df = df[df["Event timestamp"].dt.date >= as_of_date.date()]
    df = df[df["Execution Timestamp"].dt.date >= as_of_date.date()]

    year_day_count = 360
    df["Expiry (d)"] = (df["Expiration Date"] - df["Effective Date"]).dt.days
    df["Expiry (w)"] = df["Expiry (d)"] / 7
    df["Expiry (m)"] = df["Expiry (d)"] / 30
    df["Expiry (yr)"] = df["Expiry (d)"] / year_day_count

    settle_date = as_of_date + BDay(2)
    df["Fwd (d)"] = (df["Effective Date"] - settle_date).dt.days
    df["Fwd (w)"] = df["Fwd (d)"] / 7
    df["Fwd (m)"] = df["Fwd (d)"] / 30
    df["Fwd (yr)"] = df["Fwd (d)"] / year_day_count

    def format_tenor(row: pd.Series, col: str):
        if row[f"{col} (d)"] < 7:
            return f"{int(row[f'{col} (d)'])}D"
        if row[f"{col} (w)"] <= 3.75:
            return f"{int(row[f'{col} (w)'])}W"
        if row[f"{col} (m)"] < 23.5:
            return f"{int(row[f'{col} (m)'])}M"
        return f"{int(row[f'{col} (yr)'])}Y"

    df["Tenor"] = df.apply(lambda row: format_tenor(row, col="Expiry"), axis=1)
    df["Fwd"] = df.apply(lambda row: format_tenor(row, col="Fwd"), axis=1)

    df["Notional amount-Leg 1"] = pd.to_numeric(df["Notional amount-Leg 1"].str.replace(",", ""), errors="coerce")
    df["Notional amount-Leg 2"] = pd.to_numeric(df["Notional amount-Leg 2"].str.replace(",", ""), errors="coerce")
    df["Notional Amount"] = df["Notional amount-Leg 1"].combine_first(df["Notional amount-Leg 2"])
    df["Fixed rate-Leg 1"] = pd.to_numeric(df["Fixed rate-Leg 1"], errors="coerce")
    df["Fixed rate-Leg 2"] = pd.to_numeric(df["Fixed rate-Leg 2"], errors="coerce")
    df["Fixed Rate"] = df["Fixed rate-Leg 1"].combine_first(df["Fixed rate-Leg 2"])
    df["Direction"] = df.apply(
        lambda row: "receive" if pd.notna(row["Fixed rate-Leg 1"]) else ("pay" if pd.notna(row["Fixed rate-Leg 2"]) else None), axis=1
    )

    # let ignore negative rates
    df = df[df["Fixed Rate"] > 0]
    df = df[(df["Notional Amount"] > 0) & (df["Notional Amount"].notna())]

    df = df.drop(
        columns=[
            "Fixed rate-Leg 1",
            "Fixed rate-Leg 2",
            "Expiry (d)",
            "Expiry (m)",
            "Expiry (yr)",
        ]
    )

    if tenors_to_interpolate is not None:
        print("Tenors that will be Interpolated: ", tenors_to_interpolate) if verbose else None
        df = df[~df["Tenor"].isin(set(tenors_to_interpolate))]

    return df[swap_columns]


def scipy_linear_interp_func(xx, yy, kind="linear", logspace=False):
    if logspace:
        log_y = np.log(yy)
        interp_func = scipy.interpolate.interp1d(xx, log_y, kind="linear", fill_value="extrapolate", bounds_error=False)

        def log_linear_interp(x_new):
            return np.exp(interp_func(x_new))

        return log_linear_interp

    return scipy.interpolate.interp1d(xx, yy, kind=kind, fill_value="extrapolate", bounds_error=False)

def format_swap_ohlc(
    df: pd.DataFrame,
    as_of_date: datetime,
    overnight_rate: Optional[float] = None,
    minimum_time_and_sales_trades: int = 100,
    dtcc_interp_func: Optional[Callable] = scipy_linear_interp_func,
    default_tenors=DEFAULT_SWAP_TENORS,
    filter_extreme_time_and_sales=False,
    quantile_smoothing_range: Optional[Tuple[float, float]] = None,
    specifc_tenor_quantile_smoothing_range: Optional[Dict[str, Tuple[float, float]]] = None,
    remove_tenors: Optional[List[str]] = None,
    t_plus_another_one=False,
    ny_hours=False,
    # is_intraday=False,
    verbose=False,
):
    filtered_df = df[(df["Fwd"] == "0D")]

    if len(filtered_df) < minimum_time_and_sales_trades:
        t_plus_another_one = True

    if filtered_df.empty or t_plus_another_one:
        (
            print(f'Initial OHLC Filtering Results: {len(filtered_df)} < {minimum_time_and_sales_trades} --- Enabled "t_plus_another_one"')
            if verbose
            else None
        )
        filtered_df = df[(df["Fwd"] == "0D") | (df["Fwd"] == "1D")]

    if filtered_df.empty or len(filtered_df) < minimum_time_and_sales_trades:
        (
            print(f'Initial OHLC Filtering Results: {len(filtered_df)} < {minimum_time_and_sales_trades} --- Enabled "t_plus_another_two"')
            if verbose
            else None
        )
        filtered_df = df[(df["Fwd"] == "0D") | (df["Fwd"] == "1D") | (df["Fwd"] == "2D")]

    if filtered_df.empty or len(filtered_df) < minimum_time_and_sales_trades:
        (
            print(f'Initial OHLC Filtering Results: {len(filtered_df)} < {minimum_time_and_sales_trades} --- Enabled "t_plus_another_three"')
            if verbose
            else None
        )
        filtered_df = df[(df["Fwd"] == "0D") | (df["Fwd"] == "1D") | (df["Fwd"] == "2D") | (df["Fwd"] == "3D")]

    if filtered_df.empty or len(filtered_df) < minimum_time_and_sales_trades:
        (
            print(f'Initial OHLC Filtering Results: {len(filtered_df)} < {minimum_time_and_sales_trades} --- Enabled "t_plus_another_four"')
            if verbose
            else None
        )
        filtered_df = df[(df["Fwd"] == "0D") | (df["Fwd"] == "1D") | (df["Fwd"] == "2D") | (df["Fwd"] == "3D") | (df["Fwd"] == "4D")]

    if filter_extreme_time_and_sales and quantile_smoothing_range:
        filtered_df = filtered_df.groupby(["Tenor", "Fwd"], group_keys=False).apply(
            lambda group: group[
                (group["Fixed Rate"] > group["Fixed Rate"].quantile(quantile_smoothing_range[0]))
                & (group["Fixed Rate"] < group["Fixed Rate"].quantile(quantile_smoothing_range[1]))
            ]
        )

    if specifc_tenor_quantile_smoothing_range:
        for tenor, quantile_range in specifc_tenor_quantile_smoothing_range.items():
            lower_quantile = filtered_df[filtered_df["Tenor"] == tenor]["Fixed Rate"].quantile(quantile_range[0])
            upper_quantile = filtered_df[filtered_df["Tenor"] == tenor]["Fixed Rate"].quantile(quantile_range[1])
            filtered_df = filtered_df[
                ~((filtered_df["Tenor"] == tenor) & ((filtered_df["Fixed Rate"] < lower_quantile) | (filtered_df["Fixed Rate"] > upper_quantile)))
            ]

    if remove_tenors:
        for rm_tenor in remove_tenors:
            filtered_df = filtered_df[filtered_df["Tenor"] != rm_tenor]

    filtered_df = filtered_df.sort_values(by=["Tenor", "Execution Timestamp", "Fixed Rate"], ascending=True)

    if ny_hours:
        filtered_df = filtered_df[filtered_df["Execution Timestamp"].dt.hour >= 12]
        filtered_df = filtered_df[filtered_df["Execution Timestamp"].dt.hour <= 23]

    tenor_df = (
        filtered_df.groupby("Tenor")
        .agg(
            Open=("Fixed Rate", "first"),
            High=("Fixed Rate", "max"),
            Low=("Fixed Rate", "min"),
            Close=("Fixed Rate", "last"),
            VWAP=(
                "Fixed Rate",
                lambda x: (x * filtered_df.loc[x.index, "Notional Amount"]).sum() / filtered_df.loc[x.index, "Notional Amount"].sum(),
            ),
        )
        .reset_index()
    )

    tenor_df["Tenor"] = pd.Categorical(tenor_df["Tenor"], categories=sorted(tenor_df["Tenor"], key=lambda x: (x[-1], int(x[:-1]))))
    tenor_df["Expiry"] = [rl.add_tenor(as_of_date + BDay(2), _, "F", "nyc") for _ in tenor_df["Tenor"]]
    tenor_df = tenor_df.sort_values("Expiry").reset_index(drop=True)
    tenor_df = tenor_df[["Tenor", "Open", "High", "Low", "Close", "VWAP"]]
    tenor_df = tenor_df[tenor_df["Tenor"].isin(default_tenors)].reset_index(drop=True)

    if dtcc_interp_func is not None:
        x = np.array([tenor_to_years(t) for t in tenor_df["Tenor"]])
        x_new = np.array([tenor_to_years(t) for t in default_tenors])
        interp_df = pd.DataFrame({"Tenor": default_tenors})

        tenor_map = {
            tenor_to_years(t): {col: tenor_df.loc[tenor_df["Tenor"] == t, col].values[0] for col in ["Open", "High", "Low", "Close", "VWAP"]}
            for t in tenor_df["Tenor"]
        }

        for col in ["Open", "High", "Low", "Close", "VWAP"]:
            y = tenor_df[col].values

            if overnight_rate is not None:
                x = np.array([1 / 360] + [tenor_to_years(t) for t in tenor_df["Tenor"]])
                x_new = np.array([tenor_to_years(t) for t in default_tenors])
                y = [overnight_rate] + list(y)
                y_new = dtcc_interp_func(x, y)(x_new)
            else:
                y_new = dtcc_interp_func(x, y)(x_new)

            interp_df[col] = [
                tenor_map[tenor_to_years(t)][col] if tenor_to_years(t) in tenor_map else y_val for t, y_val in zip(default_tenors, y_new)
            ]

        interp_df.insert(1, "Expiry", [rl.add_tenor(as_of_date + BDay(2), _, "F", "nyc") for _ in interp_df["Tenor"]])
        interp_df = interp_df.sort_values("Expiry").reset_index(drop=True)

        return interp_df

    return tenor_df


def calculate_tenor_combined(effective_date, expiration_date):
    delta = relativedelta(expiration_date, effective_date)
    tenor_parts = []

    if delta.years == 0:
        if delta.months > 0:
            if delta.days > 0:
                days = delta.months * 30 + delta.days
                if days % 30 <= 5 or days % 30 >= 25:
                    return f"{round(days / 30)}M"
                return f"{days}D"
            return f"{delta.months}M"

        if delta.days == 0:
            return "1D"
        days = delta.months * 30 + delta.days
        if days % 30 <= 5 or days % 30 >= 25 or days >= 25:
            return f"{round(days / 30)}M"
        return f"{delta.days}D"

    has_months = False
    if delta.years > 0:
        tenor_parts.append(f"{delta.years}Y")
    if delta.months > 0:
        tenor_parts.append(f"{delta.months}M")
        has_months = True
    if delta.days > 10 and has_months:
        days = (delta.years * 365) + (delta.months * 30) + delta.days
        if days % 30 < 5 or days % 30 > 25:
            return f"{round(days / 30)}M"
        return f"{days}D"

    if len(tenor_parts) == 1 and delta.years == 1:
        return "12M"

    return "".join(tenor_parts)


def format_vanilla_swaption_time_and_sales(df: pd.DataFrame, as_of_date: datetime):
    swap_columns = [
        # "Action type",
        # "Event type",
        "Event timestamp",
        "Execution Timestamp",
        "Effective Date",
        "Expiration Date",
        "Maturity date of the underlier",
        "Fwd",
        "Option Tenor",
        "Underlying Tenor",
        "Strike Price",
        "Option Premium Amount",
        "Notional Amount",
        "Option Premium per Notional",
        "Direction",
        "Style",
        # "Package indicator",
        "Unique Product Identifier",
        "UPI FISN",
        "UPI Underlier Name",
    ]

    if as_of_date < UPI_MIGRATE_DATE:
        # not consistent among reported - assume none are fwd vol traded
        df["Effective Date"] = as_of_date
        df["Unique Product Identifier"] = "DNE"
        df["UPI FISN"] = "DNE"
        df["UPI Underlier Name"] = df["Underlier ID-Leg 1"].combine_first(df["Underlier ID-Leg 2"])

    date_columns = [
        "Event timestamp",
        "Execution Timestamp",
        "Effective Date",
        "Expiration Date",
        "Maturity date of the underlier",
        "First exercise date",
    ]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        df[col] = df[col].dt.tz_localize(None)

    df = df[df["Event timestamp"].dt.date >= as_of_date.date()]
    df = df[df["Execution Timestamp"].dt.date >= as_of_date.date()]
    # df = df[df["Expiration Date"].dt.date == df["First exercise date"].dt.date]

    df["Notional Amount"] = df["Notional amount-Leg 1"].combine_first(df["Notional amount-Leg 2"])
    df = df[(df["Notional Amount"] > 0) & (df["Notional Amount"].notna())]
    df["Fixed rate-Leg 1"] = pd.to_numeric(df["Fixed rate-Leg 1"], errors="coerce")
    df["Fixed rate-Leg 2"] = pd.to_numeric(df["Fixed rate-Leg 2"], errors="coerce")
    df["Strike Price"] = pd.to_numeric(df["Strike Price"], errors="coerce")
    df = df[df["Strike Price"] > 0]
    df = df[df["Strike Price"] == df["Fixed rate-Leg 1"].combine_first(df["Fixed rate-Leg 2"])]
    df["Direction"] = df.apply(
        lambda row: "buyer" if pd.notna(row["Fixed rate-Leg 1"]) else ("underwritter" if pd.notna(row["Fixed rate-Leg 2"]) else None), axis=1
    )

    if as_of_date < UPI_MIGRATE_DATE:
        df["Style"] = df["Option Type"].str.lower()
    else:
        df["Style"] = df.apply(
            lambda row: ("receiver" if "NA/O P Epn" in row["UPI FISN"] else ("payer" if "NA/O Call Epn" in row["UPI FISN"] else row["UPI FISN"])),
            axis=1,
        )

    df["Option Tenor"] = df.apply(
        lambda row: calculate_tenor_combined(effective_date=row["Effective Date"], expiration_date=row["First exercise date"]), axis=1
    )
    df["Underlying Tenor"] = df.apply(
        lambda row: calculate_tenor_combined(effective_date=row["First exercise date"], expiration_date=row["Maturity date of the underlier"]), axis=1
    )
    df["Fwd"] = df.apply(lambda row: f"{(row["Effective Date"] - as_of_date).days}D", axis=1)
    df["Option Premium per Notional"] = df["Option Premium Amount"] / df["Notional Amount"]
    return df[swap_columns]
