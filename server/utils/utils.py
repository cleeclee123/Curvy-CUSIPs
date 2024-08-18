import pandas as pd
import aiohttp
import asyncio
import os
import requests
import math
import shutil
import QuantLib as ql
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime
from itertools import product
from typing import TypeAlias, Optional, List, Dict, Literal

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None





# n == 0 => On-the-runs
def get_last_n_off_the_run_cusips(
    auction_json: Optional[JSON] = None,
    auctions_df: Optional[pd.DataFrame] = None,
    n=0,
    filtered=False,
    as_of_date=datetime.today(),
    use_issue_date=False,
) -> List[Dict[str, str]]:
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
    auctions_df = auctions_df.drop(
        auctions_df[
            (auctions_df["security_type"] == "Bill")
            & (
                auctions_df["original_security_term"]
                != auctions_df["security_term_week_year"]
            )
        ].index
    )
    auctions_df["auction_date"] = pd.to_datetime(auctions_df["auction_date"])
    auctions_df["issue_date"] = pd.to_datetime(auctions_df["issue_date"])
    current_date = as_of_date
    auctions_df = auctions_df[
        auctions_df["auction_date" if not use_issue_date else "issue_date"].dt.date
        <= current_date.date()
    ]
    auctions_df = auctions_df.sort_values(
        "auction_date" if not use_issue_date else "issue_date", ascending=False
    )

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

    on_the_run = auctions_df.groupby("original_security_term").first().reset_index()
    on_the_run_result = on_the_run[
        [
            "original_security_term",
            "security_type",
            "cusip",
            "auction_date",
            "issue_date",
        ]
    ]

    off_the_run = auctions_df[~auctions_df.index.isin(on_the_run.index)]
    off_the_run_result = (
        off_the_run.groupby("original_security_term")
        .nth(list(range(1, n + 1)))
        .reset_index()
    )

    combined_result = pd.concat(
        [on_the_run_result, off_the_run_result], ignore_index=True
    )
    combined_result = combined_result.sort_values(
        by=["original_security_term", "issue_date"], ascending=[True, False]
    )

    combined_result["target_tenor"] = combined_result["original_security_term"].replace(
        mapping
    )
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
            wrapper.append(
                {
                    auctioned_dict["target_tenor"]: auctioned_dict["cusip"]
                    for auctioned_dict in sublist
                }
            )
        else:
            wrapper.append(sublist)

    return wrapper


def get_historical_on_the_run_cusips(
    auctions_df: pd.DataFrame,
    as_of_date=datetime.today(),
    use_issue_date=False,
) -> pd.DataFrame:

    current_date = as_of_date
    auctions_df = auctions_df[
        auctions_df["auction_date" if not use_issue_date else "issue_date"].dt.date
        <= current_date.date()
    ]
    auctions_df = auctions_df.sort_values(
        "auction_date" if not use_issue_date else "issue_date", ascending=False
    )

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
    on_the_run_filtered_df["target_tenor"] = on_the_run_filtered_df[
        "original_security_term"
    ].replace(mapping)

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

    historical_auctions_df["issue_date"] = pd.to_datetime(
        historical_auctions_df["issue_date"]
    )
    historical_auctions_df["maturity_date"] = pd.to_datetime(
        historical_auctions_df["maturity_date"]
    )
    historical_auctions_df["auction_date"] = pd.to_datetime(
        historical_auctions_df["auction_date"]
    )

    historical_auctions_df.loc[
        historical_auctions_df["original_security_term"].str.contains(
            "29-Year", case=False, na=False
        ),
        "original_security_term",
    ] = "30-Year"
    historical_auctions_df.loc[
        historical_auctions_df["original_security_term"].str.contains(
            "30-", case=False, na=False
        ),
        "original_security_term",
    ] = "30-Year"

    historical_auctions_df = historical_auctions_df[
        (historical_auctions_df["security_type"] == "Bill")
        | (historical_auctions_df["security_type"] == "Note")
        | (historical_auctions_df["security_type"] == "Bond")
    ]
    historical_auctions_df = historical_auctions_df.drop(
        historical_auctions_df[
            (historical_auctions_df["security_type"] == "Bill")
            & (
                historical_auctions_df["original_security_term"]
                != historical_auctions_df["security_term"]
            )
        ].index
    )
    historical_auctions_df = historical_auctions_df[
        historical_auctions_df[
            "auction_date" if not use_issue_date else "issue_date"
        ].dt.date
        <= as_of_date.date()
    ]
    historical_auctions_df = historical_auctions_df[
        historical_auctions_df["maturity_date"] >= as_of_date
    ]
    historical_auctions_df = historical_auctions_df.drop_duplicates(
        subset=["cusip"], keep="first"
    )
    historical_auctions_df["int_rate"] = pd.to_numeric(
        historical_auctions_df["int_rate"], errors="coerce"
    )
    return historical_auctions_df


def last_day_n_months_ago(
    given_date: datetime, n: int = 1, return_all: bool = False
) -> datetime | List[datetime]:
    if return_all:
        given_date = pd.Timestamp(given_date)
        return [
            (given_date - pd.offsets.MonthEnd(i)).to_pydatetime()
            for i in range(1, n + 1)
        ]

    given_date = pd.Timestamp(given_date)
    last_day = given_date - pd.offsets.MonthEnd(n)
    return last_day.to_pydatetime()


def cookie_string_to_dict(cookie_string):
    cookie_pairs = cookie_string.split("; ")
    cookie_dict = {
        pair.split("=")[0]: pair.split("=")[1] for pair in cookie_pairs if "=" in pair
    }
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


def ust_labeler(mat_date: datetime | pd.Timestamp):
    return mat_date.strftime("%b %y") + "s"


def ust_sorter(term: str):
    if " " in term:
        term = term.split(" ")[0]
    num, unit = term.split("-")
    num = int(num)
    unit_multiplier = {"Year": 365, "Month": 30, "Week": 7, "Day": 1}
    return num * unit_multiplier[unit]
