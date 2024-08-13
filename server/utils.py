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


def build_treasurydirect_header(
    host_str: str = "api.fiscaldata.treasury.gov",
    cookie_str: Optional[str] = None,
):
    return {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Cookie": cookie_str or "",
        "DNT": "1",
        "Host": host_str or "",
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


def multi_download_year_treasury_par_yield_curve_rate(
    years: List[int],
    raw_path: str,
    download=False,
    real_par_yields=False,
    run_all=False,
    verbose=False,
) -> pd.DataFrame:
    async def fetch_from_treasurygov(
        session: aiohttp.ClientSession, url: str, curr_year: int
    ) -> pd.DataFrame:
        try:
            headers = build_treasurydirect_header()
            treasurygov_data_type = "".join(url.split("?type=")[1].split("&field")[0])
            full_file_path = os.path.join(
                raw_path, "temp", f"{treasurygov_data_type}.csv"
            )
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    with open(full_file_path, "wb") as f:
                        chunk_size = 8192
                        while True:
                            chunk = await response.content.read(chunk_size)
                            if not chunk:
                                break
                            f.write(chunk)
                    return {
                        treasurygov_data_type: await convert_csv_to_excel(
                            full_file_path
                        )
                    }
                else:
                    raise Exception(f"Bad Status: {response.status}")
        except Exception as e:
            print(e) if verbose else None
            return {treasurygov_data_type: pd.DataFrame()}

    async def convert_csv_to_excel(full_file_path: str | None) -> str:
        if not full_file_path:
            return

        copy = full_file_path
        rdir_path = copy.split("\\")
        rdir_path.remove("temp")
        renamed = str.join("\\", rdir_path)
        renamed = f"{renamed.split('.')[0]}.xlsx"

        df_temp = pd.read_csv(full_file_path)
        df_temp["Date"] = pd.to_datetime(df_temp["Date"])
        df_temp["Date"] = df_temp["Date"].dt.strftime("%Y-%m-%d")
        if download:
            df_temp.to_excel(f"{renamed.split('.')[0]}.xlsx", index=False)
        os.remove(full_file_path)
        return df_temp

    async def get_promises(session: aiohttp.ClientSession):
        tasks = []
        for year in years:
            daily_par_yield_curve_url = f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv"
            daily_par_real_yield_curve_url = f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{year}/all?type=daily_treasury_real_yield_curve&field_tdr_date_value={year}&amp;page&amp;_format=csv"
            daily_treasury_bill_rates_url = f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{year}/all?type=daily_treasury_bill_rates&field_tdr_date_value={year}&page&_format=csv"
            daily_treaury_long_term_rates_extrapolation_factors_url = f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{year}/all?type=daily_treasury_long_term_rate&field_tdr_date_value={year}&page&_format=csv"
            daily_treasury_real_long_term_rates_averages = f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{year}/all?type=daily_treasury_real_long_term&field_tdr_date_value={year}&page&_format=csv"
            if run_all:
                tasks.extend(
                    [
                        fetch_from_treasurygov(
                            session, daily_par_yield_curve_url, year
                        ),
                        fetch_from_treasurygov(
                            session, daily_par_real_yield_curve_url, year
                        ),
                        fetch_from_treasurygov(
                            session, daily_treasury_bill_rates_url, year
                        ),
                        fetch_from_treasurygov(
                            session,
                            daily_treaury_long_term_rates_extrapolation_factors_url,
                            year,
                        ),
                        fetch_from_treasurygov(
                            session, daily_treasury_real_long_term_rates_averages, year
                        ),
                    ]
                )
            else:
                curr_url = (
                    daily_par_yield_curve_url
                    if not real_par_yields
                    else daily_par_real_yield_curve_url
                )
                task = fetch_from_treasurygov(session, curr_url, year)
                tasks.append(task)

        return await asyncio.gather(*tasks)

    async def run_fetch_all() -> List[pd.DataFrame]:
        async with aiohttp.ClientSession() as session:
            all_data = await get_promises(session)
            return all_data

    os.mkdir(f"{raw_path}/temp")
    dfs: List[Dict[str, pd.DataFrame]] = asyncio.run(run_fetch_all())
    shutil.rmtree(f"{raw_path}/temp")

    if not run_all:
        dfs = [next(iter(dictionary.values())) for dictionary in dfs]
        yield_df = pd.concat(dfs, ignore_index=True)
        return yield_df

    organized_by_ust_type_dict: Dict[str, List[pd.DataFrame]] = {}
    for dictionary in dfs:
        ust_data_type, df = next(iter(dictionary)), next(iter(dictionary.values()))
        if not ust_data_type or df is None or df.empty:
            continue
        if ust_data_type not in organized_by_ust_type_dict:
            organized_by_ust_type_dict[ust_data_type] = []
        organized_by_ust_type_dict[ust_data_type].append(df)

    organized_by_ust_type_df_dict_concated: Dict[str, pd.DataFrame] = {}
    for ust_data_type in organized_by_ust_type_dict.keys():
        dfs = organized_by_ust_type_dict[ust_data_type]
        concated_df = pd.concat(dfs, ignore_index=True)
        organized_by_ust_type_df_dict_concated[ust_data_type] = concated_df

    return organized_by_ust_type_df_dict_concated


# n == 0 => On-the-runs
def get_last_n_off_the_run_cusips(
    auctions_json: JSON, n=0, filtered=False, as_of_date=datetime.today()
) -> List[Dict[str, str]]:
    auctions_df = pd.DataFrame(auctions_json)
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
    current_date = as_of_date
    auctions_df = auctions_df[auctions_df["auction_date"] <= current_date]

    auctions_df["issue_date"] = pd.to_datetime(auctions_df["issue_date"])
    auctions_df = auctions_df.sort_values("issue_date", ascending=False)

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


def get_active_cusips(auction_json: JSON, as_of_date=datetime.today()) -> pd.DataFrame:
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
    historical_auctions_df = historical_auctions_df[
        (historical_auctions_df["security_type"] == "Bill")
        | (historical_auctions_df["security_type"] == "Note")
        | (historical_auctions_df["security_type"] == "Bond")
    ]
    historical_auctions_df = historical_auctions_df.drop(
        historical_auctions_df[
            (historical_auctions_df["security_type"] == "Bill")
            & (historical_auctions_df["original_security_term"] != historical_auctions_df["security_term"])
        ].index
    )
    historical_auctions_df = historical_auctions_df[
        historical_auctions_df["auction_date"] <= as_of_date
    ]
    historical_auctions_df = historical_auctions_df[
        historical_auctions_df["maturity_date"] >= as_of_date
    ]
    historical_auctions_df = historical_auctions_df.drop_duplicates(subset=["cusip"], keep="first")
    return historical_auctions_df 