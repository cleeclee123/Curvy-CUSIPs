from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple
from pandas.tseries.offsets import BDay

import numpy as np
import pandas as pd
import pandas.tseries.offsets as offsets
import QuantLib as ql
import rateslib as rl
import scipy
import scipy.interpolate
from termcolor import colored

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


def format_swap_time_and_sales(
    df: pd.DataFrame, as_of_date: datetime, tenors_to_interpolate: Optional[List[str] | bool] = None, minimum_num_trades=0, verbose=False
):
    swap_columns = [
        "Event timestamp",
        "Execution Timestamp",
        "Effective Date",
        "Expiration Date",
        "Platform identifier",
        "Tenor",
        "Fwd",
        "Fixed Rate",
        "Direction",
        "Notional Amount",
        "Unique Product Identifier",
        "UPI FISN",
        "UPI Underlier Name",
    ]

    UPI_MIGRATE_DATE = datetime(2024, 1, 29)
    SCHEMA_CHANGE_2022 = datetime(2023, 1, 1)

    if as_of_date < SCHEMA_CHANGE_2022:
        df = df.rename(
            columns={
                "Event Timestamp": "Event timestamp",
                "Leg 1 - Floating Rate Index": "Underlier ID-Leg 1",
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

    # stuff to interpolate instead of observing due to low liquidity
    if tenors_to_interpolate is not None:
        # for tenor in DEFAULT_SWAP_TENORS:
        #     if len(df[(df["Tenor"] == tenor) & (df["Fwd"] == "0D")]) <= minimum_num_trades:
        #         tenors_to_interpolate.append(tenor)

        print("Tenors that will be Interpolated: ", tenors_to_interpolate) if verbose else None
        df = df[~df["Tenor"].isin(set(tenors_to_interpolate))]

    return df[swap_columns]


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


def default_interp_func(x, y):
    return scipy.interpolate.interp1d(x, y, kind="linear", fill_value="extrapolate", bounds_error=False)


def format_swap_ohlc(
    df: pd.DataFrame,
    as_of_date: datetime,
    dtcc_interp_func: Optional[Callable] = default_interp_func,
    default_tenors=DEFAULT_SWAP_TENORS,
    t_plus_another_one=False,
    ny_hours=False,
):
    filtered_df = df[(df["Fwd"] == "0D")]
    
    if len(filtered_df) < 100:
        t_plus_another_one = True
    
    if filtered_df.empty or t_plus_another_one:
        filtered_df = df[
            (df["Fwd"] == "0D")
            | (df["Fwd"] == "1D")
            # | (df["Fwd"] == "2D")
            # | (df["Fwd"] == "-2D")
            # | (df["Fwd"] == "3D")
            # | (df["Fwd"] == "-3D")
            # | (df["Fwd"] == "4D")
            # | (df["Fwd"] == "-4D")
        ]

    filtered_df = filtered_df.sort_values(by=["Tenor", "Execution Timestamp", "Fixed Rate"])
    filtered_df = filtered_df.sort_values(by=["Execution Timestamp"], ascending=True)

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
    tenor_df["Expiry"] = [rl.add_tenor(as_of_date + offsets.BDay(2), _, "F", "nyc") for _ in tenor_df["Tenor"]]
    tenor_df = tenor_df.sort_values("Expiry").reset_index(drop=True)
    tenor_df = tenor_df[["Tenor", "Open", "High", "Low", "Close", "VWAP"]]
    tenor_df = tenor_df[tenor_df["Tenor"].isin(default_tenors)].reset_index(drop=True)

    if dtcc_interp_func is not None:
        x = np.array([tenor_to_years(t) for t in tenor_df["Tenor"]])
        x_new = np.array([tenor_to_years(t) for t in default_tenors])
        interp_df = pd.DataFrame({"Tenor": default_tenors})
        for col in ["Open", "High", "Low", "Close", "VWAP"]:
            y = tenor_df[col].values
            y_new = dtcc_interp_func(x, y)(x_new)
            interp_df[col] = y_new

        interp_df.insert(1, "Expiry", [rl.add_tenor(as_of_date + offsets.BDay(2), _, "F", "nyc") for _ in interp_df["Tenor"]])
        interp_df = interp_df.sort_values("Expiry").reset_index(drop=True)

        return interp_df

    return tenor_df


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

    ql.Settings.instance().evaluationDate = evaluation_date

    # settlementDays,
    # Period tenor,
    # QuoteHandle rate,
    # ext::shared_ptr< OvernightIndex > const & index,
    # YieldTermStructureHandle discountingCurve={},
    # bool telescopicValueDates=False,
    # Integer paymentLag=0,
    # BusinessDayConvention paymentConvention=Following,
    # Frequency paymentFrequency=Annual,
    # Calendar paymentCalendar=Calendar(),
    # Period forwardStart=0*Days,
    # Spread const overnightSpread=0.0,
    # Pillar::Choice pillar=LastRelevantDate,
    # Date customPillarDate=Date(),
    # RateAveraging::Type averagingMethod=Compound,
    # ext::optional< bool > endOfMonth=ext::nullopt,
    # ext::optional< Frequency > fixedPaymentFrequency=ext::nullopt,
    # Calendar fixedCalendar=Calendar(),
    # Natural lookbackDays=Null< Natural >(),
    # Natural lockoutDays=0,
    # bool applyObservationShift=False,
    # ext::shared_ptr< FloatingRateCouponPricer > const & pricer={})

    # examples:
    # https://quant.stackexchange.com/questions/64342/rfr-boostrapping-using-rfr-ois-is-convexity-adjustment-technically-necessary?rq=1
    # https://quant.stackexchange.com/questions/74766/bootstrapping-sofr-curve-and-swap-payment-lag
    # https://www.jpmorgan.com/content/dam/jpm/global/disclosures/IN/usd-inr-irs.pdf

    # can just ignore payment pay b/c convexity adjustment is negligible for bootstrapping purposes
    # cubic/spline methods are relatively ill conditioned to the piecewie linear methods -

    if is_ois:
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
                endOfMonth=False,
            )
            for _, row in df.iterrows()
        ]

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
                endOfMonth=False,
            )
            for _, row in df.iterrows()
        ]
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


def build_term_structure_df(
    df: pd.DataFrame,
    as_of_date: datetime,
    curve: (
        ql.PiecewiseLogLinearDiscount
        | ql.PiecewiseLogCubicDiscount
        | ql.PiecewiseLinearZero
        | ql.PiecewiseCubicZero
        | ql.PiecewiseLinearForward
        | ql.PiecewiseSplineCubicDiscount
    ),
    interp_func: Callable = default_interp_func,
    tenor_col: str = "Tenor",
    fixed_rate_col: str = "Fixed Rate",
    fwd_rate_tenors: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame | Dict[str, scipy.interpolate.interp1d]]:
    if not isinstance(as_of_date, ql.Date):
        evaluation_date = datetime_to_ql_date(as_of_date)

    calendar = ql.UnitedStates(ql.UnitedStates.FederalReserve)
    settlement_date: ql.Date = calendar.advance(evaluation_date, ql.Period(2, ql.Days))
    tenors: List[ql.Period] = []
    rates: List[float] = []
    py_maturity_dates: List[datetime] = []
    ql_maturity_dates: List[ql.Date] = []
    zero_rates: List[float] = []
    discount_factors: List[float] = []
    for tenor, rate in dict(zip(df[tenor_col], df[fixed_rate_col])).items():
        tenor = tenor_to_ql_period(tenor)
        tenors.append(tenor)
        rates.append(rate * 100)
        maturity_date: ql.Date = calendar.advance(settlement_date, tenor, ql.ModifiedFollowing, True)
        ql_maturity_dates.append(maturity_date)
        py_maturity_dates.append(maturity_date.to_date())
        discount_factor: float = curve.discount(maturity_date)
        discount_factors.append(discount_factor)
        zero_rate: float = -100.0 * np.log(discount_factor) * 365.0 / (maturity_date - evaluation_date)
        zero_rates.append(zero_rate)

    x_interp = np.array([tenor_to_years(t) for t in df[tenor_col]])
    scipy_interp_dict = {
        "fixed_rate": interp_func(x_interp, rates),
        "zero_rate": interp_func(x_interp, zero_rates),
        "discount": interp_func(x_interp, discount_factors),
    }

    dict_for_df = {"Tenor": tenors, "Expiry": py_maturity_dates, "Fixed Rate": rates, "Zero Rate": zero_rates, "Discount": discount_factors}
    for fwd_rate_tenor in fwd_rate_tenors:
        dict_for_df[f"{fwd_rate_tenor} Fwd"] = [
            (curve.forwardRate(date, date + tenor_to_ql_period(fwd_rate_tenor), ql.Actual360(), ql.Simple).rate()) * 100 for date in ql_maturity_dates
        ]
        scipy_interp_dict[f"{fwd_rate_tenor}_fwd"] = interp_func(x_interp, dict_for_df[f"{fwd_rate_tenor} Fwd"])

    return pd.DataFrame(dict_for_df), scipy_interp_dict


# curve/fly stuff
"""
sofr_ois_swaps_pcakage_df = sdr_df[(sdr_df["Unique Product Identifier"] == underlying_dict["Identifier_UPI"]) & (sdr_df["Package indicator"] == True)]
df = sofr_ois_swaps_pcakage_df 
df['Effective Date'] = pd.to_datetime(df['Effective Date'])
df['Expiration Date'] = pd.to_datetime(df['Expiration Date'])

# Calculate tenor in years for each trade
df['Tenor'] = ((df['Expiration Date'] - df['Effective Date']).dt.days / 360).round()

# Group trades by Event timestamp to identify related legs
grouped = df.groupby('Event timestamp')

curve_structures = []

for timestamp, group in grouped:
    if len(group) > 1:  # Multiple trades with same timestamp are likely part of a curve
        tenors = sorted(group['Tenor'].unique())
        
        # Create curve trade description
        if len(tenors) == 2:
            curve_structures.append({
                'timestamp': timestamp,
                'structure': f"{int(tenors[0])}s{int(tenors[-1])}s",
                'tenors': tenors,
                'rates': group['Fixed rate-Leg 1'].tolist(),
                'notionals': group['Notional amount-Leg 1'].tolist()
            })
        elif len(tenors) == 3:
            curve_structures.append({
                'timestamp': timestamp,
                'structure': f"{int(tenors[0])}s{int(tenors[1])}s{int(tenors[-1])}s",
                'tenors': tenors,
                'rates': group['Fixed rate-Leg 1'].tolist(),
                'notionals': group['Notional amount-Leg 1'].tolist()
            })

pd.DataFrame(curve_structures)

"""
