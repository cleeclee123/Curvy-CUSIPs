from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
import QuantLib as ql
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



def scipy_linear_interp_func(xx, yy, kind="linear", logspace=False):
    if logspace:
        log_y = np.log(yy)
        interp_func = scipy.interpolate.interp1d(xx, log_y, kind="linear", fill_value="extrapolate", bounds_error=False)

        def log_linear_interp(x_new):
            return np.exp(interp_func(x_new))

        return log_linear_interp

    return scipy.interpolate.interp1d(xx, yy, kind=kind, fill_value="extrapolate", bounds_error=False)




def calculate_tenor_combined(effective_date, expiration_date):
    delta = relativedelta(expiration_date, effective_date)
    tenor_parts = []

    if delta.years == 0:
        if delta.months > 0:
            if delta.days > 5:
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

