import asyncio
import math
import os
import shutil
import ujson as json
from datetime import datetime, timedelta
from itertools import product
from typing import Dict, List, Literal, Optional, TypeAlias, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import aiohttp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import QuantLib as ql
import requests
from scipy.optimize import minimize

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None


def auction_df_filterer(historical_auctions_df: pd.DataFrame):
    historical_auctions_df = historical_auctions_df.copy()
    historical_auctions_df["issue_date"] = pd.to_datetime(
        historical_auctions_df["issue_date"]
        # , errors="coerce"
    )
    historical_auctions_df["maturity_date"] = pd.to_datetime(
        historical_auctions_df["maturity_date"]
        # , errors="coerce"
    )
    historical_auctions_df["auction_date"] = pd.to_datetime(
        historical_auctions_df["auction_date"]
        # , errors="coerce"
    )
    historical_auctions_df.loc[
        historical_auctions_df["original_security_term"].str.contains("29-Year", case=False, na=False),
        "original_security_term",
    ] = "30-Year"
    historical_auctions_df.loc[
        historical_auctions_df["original_security_term"].str.contains("30-", case=False, na=False),
        "original_security_term",
    ] = "30-Year"
    historical_auctions_df = historical_auctions_df[
        (historical_auctions_df["security_type"] == "Bill")
        | (historical_auctions_df["security_type"] == "Note")
        | (historical_auctions_df["security_type"] == "Bond")
    ]
    return historical_auctions_df


def build_treasurydirect_header(
    host_str: Optional[str] = "api.fiscaldata.treasury.gov",
    cookie_str: Optional[str] = None,
    origin_str: Optional[str] = None,
    referer_str: Optional[str] = None,
):
    return {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7,application/json",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Cookie": cookie_str or "",
        "DNT": "1",
        "Host": host_str or "",
        "Origin": origin_str or "",
        "Referer": referer_str or "",
        "Pragma": "no-cache",
        "Sec-CH-UA": '"Not)A;Brand";v="99", "Google Chrome";v="127", "Chromium";v="127"',
        "Sec-CH-UA-Mobile": "?0",
        "Sec-CH-UA-Platform": '"Windows"',
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    }


# n == 0 => On-the-runs
def get_last_n_off_the_run_cusips(
    auction_json: Optional[JSON] = None,
    auctions_df: Optional[pd.DataFrame] = None,
    n=0,
    filtered=False,
    as_of_date=datetime.today(),
    use_issue_date=False,
) -> List[Dict[str, str]] | pd.DataFrame:
    if not auction_json and auctions_df is None:
        return pd.DataFrame(columns=historical_auction_cols())

    if auction_json and auctions_df is None:
        auctions_df = pd.DataFrame(auction_json)

    auctions_df = auctions_df[
        (auctions_df["security_type"] != "TIPS")
        & (auctions_df["security_type"] != "TIPS Note")
        & (auctions_df["security_type"] != "TIPS Bond")
        & (auctions_df["security_type"] != "FRN")
        & (auctions_df["security_type"] != "FRN Note")
        & (auctions_df["security_type"] != "FRN Bond")
        & (auctions_df["security_type"] != "CMB")
    ]
    # auctions_df = auctions_df.drop(
    #     auctions_df[
    #         (auctions_df["security_type"] == "Bill")
    #         & (
    #             auctions_df["original_security_term"]
    #             != auctions_df["security_term_week_year"]
    #         )
    #     ].index
    # )
    auctions_df["auction_date"] = pd.to_datetime(auctions_df["auction_date"])
    auctions_df["issue_date"] = pd.to_datetime(auctions_df["issue_date"])
    current_date = as_of_date
    auctions_df = auctions_df[auctions_df["auction_date" if not use_issue_date else "issue_date"].dt.date <= current_date.date()]
    auctions_df = auctions_df.sort_values("auction_date" if not use_issue_date else "issue_date", ascending=False)

    mapping = {
        "4-Week": 0.077,
        "8-Week": 0.15,
        "13-Week": 0.25,
        "17-Week": 0.33,
        "26-Week": 0.5,
        "52-Week": 1,
        "2-Year": 2,
        "3-Year": 3,
        "5-Year": 5,
        "7-Year": 7,
        "10-Year": 10,
        "20-Year": 20,
        "30-Year": 30,
    }

    on_the_run = auctions_df.groupby("original_security_term").first().reset_index()
    on_the_run = on_the_run[(on_the_run["security_type"] == "Note") | (on_the_run["security_type"] == "Bond")]
    on_the_run_result = on_the_run[
        [
            "original_security_term",
            "security_type",
            "cusip",
            "auction_date",
            "issue_date",
        ]
    ]

    on_the_run_bills = auctions_df.groupby("security_term").first().reset_index()
    on_the_run_bills = on_the_run_bills[on_the_run_bills["security_type"] == "Bill"]
    on_the_run_result_bills = on_the_run_bills[
        [
            "original_security_term",
            "security_type",
            "cusip",
            "auction_date",
            "issue_date",
        ]
    ]

    on_the_run = pd.concat([on_the_run_result_bills, on_the_run_result])

    if n == 0:
        return on_the_run

    off_the_run = auctions_df[~auctions_df.index.isin(on_the_run.index)]
    off_the_run_result = off_the_run.groupby("original_security_term").nth(list(range(1, n + 1))).reset_index()

    combined_result = pd.concat([on_the_run_result, off_the_run_result], ignore_index=True)
    combined_result = combined_result.sort_values(by=["original_security_term", "issue_date"], ascending=[True, False])

    combined_result["target_tenor"] = combined_result["original_security_term"].replace(mapping)
    mask = combined_result["original_security_term"].isin(mapping.keys())
    mapped_and_filtered_df = combined_result[mask]
    grouped = mapped_and_filtered_df.groupby("original_security_term")
    max_size = grouped.size().max()
    wrapper = []
    for i in range(max_size):
        sublist = []
        for _, group in grouped:
            if i < len(group):
                sublist.append(group.iloc[i].to_dict())
        sublist = sorted(sublist, key=lambda d: d["target_tenor"])
        if filtered:
            wrapper.append({auctioned_dict["target_tenor"]: auctioned_dict["cusip"] for auctioned_dict in sublist})
        else:
            wrapper.append(sublist)

    return wrapper


def get_historical_on_the_run_cusips(
    auctions_df: pd.DataFrame,
    as_of_date=datetime.today(),
    use_issue_date=False,
) -> pd.DataFrame:

    current_date = as_of_date
    auctions_df = auctions_df[auctions_df["auction_date" if not use_issue_date else "issue_date"].dt.date <= current_date.date()]
    auctions_df = auctions_df[auctions_df["maturity_date"].dt.date >= current_date.date()]
    auctions_df = auctions_df.sort_values("auction_date" if not use_issue_date else "issue_date", ascending=False)

    mapping = {
        "17-Week": 0.25,
        "26-Week": 0.5,
        "52-Week": 1,
        "2-Year": 2,
        "3-Year": 3,
        "5-Year": 5,
        "7-Year": 7,
        "10-Year": 10,
        "20-Year": 20,
        "30-Year": 30,
    }

    on_the_run_df = auctions_df.groupby("original_security_term").first().reset_index()
    on_the_run_filtered_df = on_the_run_df[
        [
            "original_security_term",
            "security_type",
            "cusip",
            "auction_date",
            "issue_date",
        ]
    ]
    on_the_run_filtered_df["target_tenor"] = on_the_run_filtered_df["original_security_term"].replace(mapping)

    return on_the_run_filtered_df


def get_active_cusips(
    auction_json: Optional[JSON] = None,
    historical_auctions_df: Optional[pd.DataFrame] = None,
    as_of_date=datetime.today(),
    use_issue_date=False,
) -> pd.DataFrame:
    if not auction_json and historical_auctions_df is None:
        return pd.DataFrame(columns=historical_auction_cols())

    if auction_json and historical_auctions_df is None:
        historical_auctions_df = pd.DataFrame(auction_json)

    historical_auctions_df = auction_df_filterer(historical_auctions_df)
    historical_auctions_df = historical_auctions_df[
        historical_auctions_df["auction_date" if not use_issue_date else "issue_date"].dt.date <= as_of_date.date()
    ]
    historical_auctions_df = historical_auctions_df[historical_auctions_df["maturity_date"] >= as_of_date]
    historical_auctions_df = historical_auctions_df.drop_duplicates(subset=["cusip"], keep="first")
    historical_auctions_df["int_rate"] = pd.to_numeric(historical_auctions_df["int_rate"], errors="coerce")
    historical_auctions_df["time_to_maturity"] = (historical_auctions_df["maturity_date"] - as_of_date).dt.days / 365
    return historical_auctions_df


def last_day_n_months_ago(given_date: datetime, n: int = 1, return_all: bool = False) -> datetime | List[datetime]:
    if return_all:
        given_date = pd.Timestamp(given_date)
        return [(given_date - pd.offsets.MonthEnd(i)).to_pydatetime() for i in range(1, n + 1)]

    given_date = pd.Timestamp(given_date)
    last_day = given_date - pd.offsets.MonthEnd(n)
    return last_day.to_pydatetime()


def cookie_string_to_dict(cookie_string):
    cookie_pairs = cookie_string.split("; ")
    cookie_dict = {pair.split("=")[0]: pair.split("=")[1] for pair in cookie_pairs if "=" in pair}
    return cookie_dict


def is_valid_ust_cusip(potential_ust_cusip: str):
    return len(potential_ust_cusip) == 9 and "912" in potential_ust_cusip


def historical_auction_cols():
    return [
        "cusip",
        "security_type",
        "auction_date",
        "issue_date",
        "maturity_date",
        "price_per100",
        "allocation_pctage",
        "avg_med_yield",
        "bid_to_cover_ratio",
        "comp_accepted",
        "comp_tendered",
        "corpus_cusip",
        "tint_cusip_1",
        "currently_outstanding",
        "direct_bidder_accepted",
        "direct_bidder_tendered",
        "est_pub_held_mat_by_type_amt",
        "fima_included",
        "fima_noncomp_accepted",
        "fima_noncomp_tendered",
        "high_discnt_rate",
        "high_investment_rate",
        "high_price",
        "high_yield",
        "indirect_bidder_accepted",
        "indirect_bidder_tendered",
        "int_rate",
        "low_investment_rate",
        "low_price",
        "low_discnt_margin",
        "low_yield",
        "max_comp_award",
        "max_noncomp_award",
        "noncomp_accepted",
        "noncomp_tenders_accepted",
        "offering_amt",
        "security_term",
        "original_security_term",
        "security_term_week_year",
        "primary_dealer_accepted",
        "primary_dealer_tendered",
        "reopening",
        "total_accepted",
        "total_tendered",
        "treas_retail_accepted",
        "treas_retail_tenders_accepted",
    ]


def ust_labeler(row: pd.Series):
    mat_date = row["maturity_date"]
    tenor = row["original_security_term"]
    if np.isnan(row["int_rate"]):
        return str(row["high_investment_rate"])[:5] + "s , " + mat_date.strftime("%b %y") + "s" + ", " + tenor
    return str(row["int_rate"]) + "s, " + mat_date.strftime("%b %y") + "s, " + tenor


def ust_sorter(term: str):
    if " " in term:
        term = term.split(" ")[0]
    num, unit = term.split("-")
    num = int(num)
    unit_multiplier = {"Year": 365, "Month": 30, "Week": 7, "Day": 1}
    return num * unit_multiplier[unit]


def get_otr_cusips_by_date(historical_auctions_df: pd.DataFrame, dates: list, use_issue_date: bool = True):
    historical_auctions_df = auction_df_filterer(historical_auctions_df)
    date_column = "issue_date" if use_issue_date else "auction_date"
    historical_auctions_df = historical_auctions_df.sort_values(by=[date_column], ascending=False)
    historical_auctions_df = historical_auctions_df.drop_duplicates(subset=["cusip"], keep="last")
    grouped = historical_auctions_df.groupby("original_security_term")
    otr_cusips_by_date = {date: [] for date in dates}
    for _, group in grouped:
        group = group.reset_index(drop=True)
        for date in dates:
            filtered_group = group[(group[date_column] <= date) & (group["maturity_date"] > date)]
            if not filtered_group.empty:
                otr_cusip = filtered_group.iloc[0]["cusip"]
                otr_cusips_by_date[date].append(otr_cusip)

    return otr_cusips_by_date


def process_cusip_otr_daterange(cusip, historical_auctions_df, date_column):
    try:
        tenor = historical_auctions_df[historical_auctions_df["cusip"] == cusip]["original_security_term"].iloc[0]
        tenor_df: pd.DataFrame = historical_auctions_df[historical_auctions_df["original_security_term"] == tenor].reset_index()
        otr_df = tenor_df[tenor_df["cusip"] == cusip]
        otr_index = otr_df.index[0]
        start_date: pd.Timestamp = otr_df[date_column].iloc[0]
        start_date = start_date.to_pydatetime()

        if otr_index == 0:
            return cusip, (start_date, datetime.today().date())

        if otr_index < len(tenor_df) - 1:
            end_date: pd.Timestamp = tenor_df[date_column].iloc[otr_index - 1]
            end_date = end_date.to_pydatetime()
        else:
            end_date = datetime.today().date()

        return cusip, {"start_date": start_date, "end_date": end_date}
    except Exception as e:
        # print(f"Something went wrong for {cusip}: {e}")
        return cusip, {"start_date": None, "end_date": None}


def get_otr_date_ranges(historical_auctions_df: pd.DataFrame, cusips: List[str], use_issue_date: bool = True) -> Dict[str, Tuple[datetime, datetime]]:

    historical_auctions_df = auction_df_filterer(historical_auctions_df)
    date_column = "issue_date" if use_issue_date else "auction_date"
    historical_auctions_df = historical_auctions_df.sort_values(by=[date_column], ascending=False)
    historical_auctions_df = historical_auctions_df[historical_auctions_df["issue_date"].dt.date < datetime.today().date()]
    historical_auctions_df = historical_auctions_df.drop_duplicates(subset=["cusip"], keep="last")

    cusip_daterange_map = {}
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_cusip_otr_daterange, cusip, historical_auctions_df, date_column): cusip for cusip in cusips}

        for future in as_completed(futures):
            cusip, date_range = future.result()
            cusip_daterange_map[cusip] = date_range

    return cusip_daterange_map


def pydatetime_to_quantlib_date(py_datetime: datetime) -> ql.Date:
    return ql.Date(py_datetime.day, py_datetime.month, py_datetime.year)


def quantlib_date_to_pydatetime(ql_date: ql.Date):
    return datetime(ql_date.year(), ql_date.month(), ql_date.dayOfMonth())


def get_isin_from_cusip(cusip_str, country_code: str = "US"):
    """
    >>> get_isin_from_cusip('037833100', 'US')
    'US0378331005'
    """
    isin_to_digest = country_code + cusip_str.upper()

    get_numerical_code = lambda c: str(ord(c) - 55)
    encode_letters = lambda c: c if c.isdigit() else get_numerical_code(c)
    to_digest = "".join(map(encode_letters, isin_to_digest))

    ints = [int(s) for s in to_digest[::-1]]
    every_second_doubled = [x * 2 for x in ints[::2]] + ints[1::2]

    sum_digits = lambda i: sum(divmod(i, 10))
    digit_sum = sum([sum_digits(i) for i in every_second_doubled])

    check_digit = (10 - digit_sum % 10) % 10
    return isin_to_digest + str(check_digit)


def get_cstrips_cusips(
    historical_auctions_df: pd.DataFrame,
    as_of_date: Optional[datetime] = None,
):
    historical_auctions_df = auction_df_filterer(historical_auctions_df)
    active_df = historical_auctions_df[historical_auctions_df["maturity_date"] > as_of_date]
    tint_cusip = "tint_cusip_1"
    active_df[tint_cusip] = active_df[tint_cusip].replace("null", np.nan)
    active_df = active_df[active_df[tint_cusip].notna()]
    active_df = active_df.sort_values(by=["maturity_date"]).reset_index(drop=True)
    return active_df[["maturity_date", tint_cusip]]


def original_security_term_to_ql_period(original_security_term: str):
    ost_ql_period = {
        "52-Week": ql.Period("1Y"),
        "2-Year": ql.Period("2Y"),
        "3-Year": ql.Period("3Y"),
        "5-Year": ql.Period("5Y"),
        "7-Year": ql.Period("7Y"),
        "10-Year": ql.Period("10Y"),
        "20-Year": ql.Period("20Y"),
        "30-Year": ql.Period("30Y"),
    }
    return ost_ql_period[original_security_term]


def enhanced_plotly_blue_scale():
    enhanced_blue_scale = [
        [0.0, "#0508b8"],  # Deepest blue
        [0.02, "#0508b8"],  # Extending deepest blue
        [0.04, "#0610b9"],  # Very subtle shift
        [0.06, "#0712ba"],  # Slightly lighter
        [0.08, "#0814bb"],  # Incrementally lighter
        [0.10, "#0916bc"],  # Another slight lightening
        [0.12, "#0a18bd"],  # Continuing the trend
        [0.14, "#0b1abd"],  # Gradual lightening
        [0.16, "#0c1cbe"],  # More noticeable change
        [0.18, "#0d1ebe"],  # Further lightening
        [0.20, "#0e20bf"],  # Approaching mid blues
        [0.25, "#1116e8"],  # Jumping to a lighter blue for contrast
        [0.3, "#1910d8"],  # Original Plotly3 blue
    ]

    # Transitioning back to the original Plotly3 scale, starting from a point after the enhanced blue
    transition_scale = [
        [0.35, "#3c19f0"],
        [0.40, "#6b1cfb"],
        [0.45, "#981cfd"],
        [0.50, "#bf1cfd"],
        [0.55, "#dd2bfd"],
        [0.60, "#f246fe"],
        [0.65, "#fc67fd"],
        [0.70, "#fe88fc"],
        [0.75, "#fea5fd"],
        [0.80, "#febefe"],
        [0.85, "#fec3fe"],
    ]

    # Combine the more granular blue scale with the rest of the Plotly3 colors
    combined_scale = enhanced_blue_scale + transition_scale

    return combined_scale


def to_quantlib_fixed_rate_bond_obj(bond_info: Dict[str, str], as_of_date: datetime, print_bond_info=False):
    if print_bond_info:
        print(json.dumps(bond_info, indent=4, default=str))
    maturity_date: pd.Timestamp = bond_info["maturity_date"]
    calendar = ql.UnitedStates(m=ql.UnitedStates.GovernmentBond)
    today = calendar.adjust(pydatetime_to_quantlib_date(py_datetime=as_of_date))
    ql.Settings.instance().evaluationDate = today
    t_plus = 1
    bond_settlement_date = calendar.advance(today, ql.Period(t_plus, ql.Days))

    schedule = ql.Schedule(
        bond_settlement_date,
        pydatetime_to_quantlib_date(maturity_date),
        ql.Period(ql.Semiannual),
        calendar,
        ql.ModifiedFollowing,
        ql.ModifiedFollowing,
        ql.DateGeneration.Backward,
        False,
    )
    return ql.FixedRateBond(
        t_plus,
        100.0,
        schedule,
        [bond_info["int_rate"] / 100],
        ql.ActualActual(ql.ActualActual.ISDA),
    )

class NoneReturningSpline:
    def __init__(self, *args, **kwargs):
        # Accept the same parameters as a usual spline, but do nothing with them
        pass

    def __call__(self, x):
        # Always return None for any input x
        if isinstance(x, (np.ndarray, list)):
            return [None] * len(x)  # Return a list of None for multiple inputs
        else:
            return None