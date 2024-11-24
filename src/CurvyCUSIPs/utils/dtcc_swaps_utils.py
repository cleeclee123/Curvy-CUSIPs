from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple
from pandas.tseries.offsets import BDay, BMonthEnd

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
    # interp_func: Callable = default_interp_func,
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
    # scipy_interp_dict = {
    #     "fixed_rate": interp_func(x_interp, rates),
    #     "zero_rate": interp_func(x_interp, zero_rates),
    #     "discount": interp_func(x_interp, discount_factors),
    # }

    dict_for_df = {"Tenor": tenors, "Expiry": py_maturity_dates, "Fixed Rate": rates, "Zero Rate": zero_rates, "Discount": discount_factors}
    for fwd_rate_tenor in fwd_rate_tenors:
        dict_for_df[f"{fwd_rate_tenor} Fwd"] = [
            (curve.forwardRate(date, date + tenor_to_ql_period(fwd_rate_tenor), ql.Actual360(), ql.Simple).rate()) * 100 for date in ql_maturity_dates
        ]
        # scipy_interp_dict[f"{fwd_rate_tenor}_fwd"] = interp_func(x_interp, dict_for_df[f"{fwd_rate_tenor} Fwd"])

    return pd.DataFrame(dict_for_df)


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
