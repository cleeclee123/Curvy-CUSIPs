import asyncio
import multiprocessing as mp
import shelve
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Dict

import httpx
import numpy as np
import pandas as pd
import ujson as json
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import BDay, CustomBusinessDay

sys.path.insert(0, "../")
from CurvyCUSIPs.CurveDataFetcher import CurveDataFetcher
from CurvyCUSIPs.utils.QL_BondPricer import QL_BondPricer
from CurvyCUSIPs.utils.ShelveDBWrapper import ShelveDBWrapper


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def runner(dates, curve_data_fetcher: CurveDataFetcher):
    async def build_tasks(client: httpx.AsyncClient, dates):
        tasks = await curve_data_fetcher.fedinvest_data_fetcher._build_fetch_tasks_cusip_prices_fedinvest(
            client=client,
            dates=dates,
            max_concurrent_tasks=5,
        )
        return await asyncio.gather(*tasks)

    async def run_fetch_all(dates):
        limits = httpx.Limits(max_connections=20, max_keepalive_connections=10)
        async with httpx.AsyncClient(limits=limits) as client:
            all_data = await build_tasks(client=client, dates=dates)
            return all_data

    results = asyncio.run(run_fetch_all(dates=dates))
    return dict(results)


def get_business_days_groups(start_date: datetime, end_date: datetime, group_size=3):
    date_range = pd.date_range(start=start_date, end=end_date, freq=CustomBusinessDay(calendar=USFederalHolidayCalendar()))
    business_day_groups = [[bday.to_pydatetime() for bday in date_range[i : i + group_size].tolist()] for i in range(0, len(date_range), group_size)]
    return business_day_groups


def ust_labeler(mat_date: datetime | pd.Timestamp):
    return mat_date.strftime("%b %y") + "s"


def process_dataframe(key: datetime, df: pd.DataFrame, raw_auctions_df: pd.DataFrame):
    max_retries = 3
    counter = 0
    raw_auctions_df = raw_auctions_df.copy()
    try:
        while counter < max_retries:
            try:
                raw_auctions_df = raw_auctions_df.sort_values(by=["issue_date"], ascending=False)
                raw_auctions_df = raw_auctions_df[raw_auctions_df["issue_date"] < key]
                raw_auctions_df = raw_auctions_df.drop_duplicates(subset=["cusip"], keep="first")

                cusip_ref_df = raw_auctions_df[raw_auctions_df["cusip"].isin(df["cusip"].to_list())][
                    [
                        "cusip",
                        "security_type",
                        "security_term",
                        "original_security_term",
                        "auction_date",
                        "issue_date",
                        "maturity_date",
                        "int_rate",
                        "high_investment_rate",
                    ]
                ]

                merged_df = pd.merge(left=df, right=cusip_ref_df, on=["cusip"])
                merged_df = merged_df.replace("null", np.nan)
                merged_df["eod_price"] = merged_df["eod_price"].replace(0, np.nan)
                merged_df["bid_price"] = merged_df["bid_price"].replace(0, np.nan)
                merged_df["offer_price"] = merged_df["offer_price"].replace(0, np.nan)

                merged_df["eod_yield"] = merged_df.apply(
                    lambda row: QL_BondPricer.bond_price_to_ytm(
                        type=row["security_type"],
                        issue_date=row["issue_date"],
                        maturity_date=row["maturity_date"],
                        as_of=key,
                        coupon=float(row["int_rate"]) / 100,
                        price=row["eod_price"],
                    ),
                    axis=1,
                )
                merged_df["bid_yield"] = merged_df.apply(
                    lambda row: QL_BondPricer.bond_price_to_ytm(
                        type=row["security_type"],
                        issue_date=row["issue_date"],
                        maturity_date=row["maturity_date"],
                        as_of=key,
                        coupon=float(row["int_rate"]) / 100,
                        price=row["bid_price"],
                    ),
                    axis=1,
                )
                merged_df["offer_yield"] = merged_df.apply(
                    lambda row: QL_BondPricer.bond_price_to_ytm(
                        type=row["security_type"],
                        issue_date=row["issue_date"],
                        maturity_date=row["maturity_date"],
                        as_of=key,
                        coupon=float(row["int_rate"]) / 100,
                        price=row["offer_price"],
                    ),
                    axis=1,
                )

                merged_df["mid_price"] = (merged_df["offer_price"] + merged_df["bid_price"]) / 2
                merged_df["mid_yield"] = (merged_df["offer_yield"] + merged_df["bid_yield"]) / 2

                merged_df = merged_df[
                    [
                        "cusip",
                        "security_type",
                        "security_term",
                        "original_security_term",
                        "auction_date",
                        "issue_date",
                        "maturity_date",
                        "int_rate",
                        "high_investment_rate",
                        "bid_price",
                        "offer_price",
                        "mid_price",
                        "eod_price",
                        "bid_yield",
                        "offer_yield",
                        "mid_yield",
                        "eod_yield",
                    ]
                ]
                merged_df = merged_df.replace({np.nan: None})
                records = merged_df.to_dict(orient="records")

                if len(records) == 0:
                    raise ValueError("records is empty after processing")

                json_structure = {"data": records}
                return key, json_structure

            except Exception as e:
                counter += 1
                print(bcolors.FAIL + f"FAILED DF PROCESSING {key} - retry attempts remaining: {max_retries - counter} - {str(e)}" + bcolors.ENDC)

        raise ValueError("failed cusip set df processing")

    except Exception as e:
        return key, {"data": str(e)}


def parallel_process(dict_df, raw_auctions_df):
    result_dict = {}

    with ProcessPoolExecutor(max_workers=mp.cpu_count() - 3) as executor:
        futures = {executor.submit(process_dataframe, key, df, raw_auctions_df): key for key, df in dict_df.items()}
        for future in as_completed(futures):
            key, json_structure = future.result()
            result_dict[key] = json_structure

    return result_dict


if __name__ == "__main__":

    DB_PATH_CUSIP_CURVE_SET = "../db/ust_cusip_set"
    DB_PATH_CUSIP_TIMESERIES = "../db/ust_cusip_timeseries"

    t1_parent = time.time()
    t1 = time.time()

    if len(sys.argv) >= 2 and sys.argv[1] == "init":
        start_date = datetime(2018, 1, 1)
        ybday: pd.Timestamp = datetime.today() - BDay(1)
        end_date = ybday.to_pydatetime()
    else:
        cusip_curve_set_db = ShelveDBWrapper(DB_PATH_CUSIP_CURVE_SET)
        cusip_curve_set_db.open()
        most_recent_db_dt = datetime.strptime(max(cusip_curve_set_db.keys()), "%Y-%m-%d")
        bday_offset = ((datetime.today() - BDay(1)) - most_recent_db_dt).days
        cusip_curve_set_db.close()

        if bday_offset == 0 and len(sys.argv) == 1:
            print(bcolors.OKBLUE + "DB is up to date - exiting..." + bcolors.ENDC)
            sys.exit()

        y2bday = (datetime.today() - BDay(bday_offset)).to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0)
        ybday = (datetime.today() - BDay(1)).to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = y2bday
        end_date = ybday

    bdates = pd.date_range(
        start=start_date,
        end=end_date,
        freq=CustomBusinessDay(calendar=USFederalHolidayCalendar()),
    )

    ##################

    print(bcolors.OKBLUE + f"Fetching UST Prices for {start_date} and {end_date}" + bcolors.ENDC)
    weeks = get_business_days_groups(start_date, end_date, group_size=60)

    curve_data_fetcher = CurveDataFetcher(use_ust_issue_date=True, error_verbose=True)
    raw_auctions_df = curve_data_fetcher.ust_data_fetcher._historical_auctions_df.copy()
    raw_auctions_df["issue_date"] = pd.to_datetime(raw_auctions_df["issue_date"], errors="coerce")
    raw_auctions_df["maturity_date"] = pd.to_datetime(raw_auctions_df["maturity_date"], errors="coerce")
    raw_auctions_df["auction_date"] = pd.to_datetime(raw_auctions_df["auction_date"], errors="coerce")
    raw_auctions_df.loc[
        raw_auctions_df["original_security_term"].str.contains("29-Year", case=False, na=False),
        "original_security_term",
    ] = "30-Year"
    raw_auctions_df.loc[
        raw_auctions_df["original_security_term"].str.contains("30-", case=False, na=False),
        "original_security_term",
    ] = "30-Year"

    raw_auctions_df = raw_auctions_df[
        (raw_auctions_df["security_type"] == "Bill") | (raw_auctions_df["security_type"] == "Note") | (raw_auctions_df["security_type"] == "Bond")
    ]

    cusip_curve_set_db = ShelveDBWrapper(DB_PATH_CUSIP_CURVE_SET, create=len(sys.argv) >= 2 and sys.argv[1] == "init")
    cusip_curve_set_db.open()

    for week in weeks:
        dict_df: Dict[datetime, pd.DataFrame] = runner(dates=week, curve_data_fetcher=curve_data_fetcher)
        t_p = time.time()
        to_write = parallel_process(dict_df, raw_auctions_df)
        print(bcolors.OKBLUE + f"CUSIP set processing took: {time.time() - t_p} seconds" + bcolors.ENDC)
        for key, json_structure in to_write.items():
            try:
                if len(json_structure["data"]) == 0:
                    raise ValueError("CUSIP SET IS EMPTY")

                date_str = key.strftime("%Y-%m-%d")
                cusip_curve_set_db.set(date_str, json.dumps(json_structure, default=str))

                print(bcolors.OKGREEN + f"WROTE {key} to DB" + bcolors.ENDC)

            except Exception as e:
                print(bcolors.FAIL + f"FAILED DB WRITE {key} - {str(e)}" + bcolors.ENDC)

    print(f"FedInvest Scraper Script took: {time.time() - t1} seconds")

    ##################

    print(bcolors.OKBLUE + "STARTING TIMESERIES SCRIPT" + bcolors.ENDC)
    time.sleep(5)
    t1 = time.time()

    cusip_timeseries = defaultdict(list)
    keys_to_include = [
        "Date",
        "cusip",
        "bid_price",
        "offer_price",
        "mid_price",
        "eod_price",
        "bid_yield",
        "offer_yield",
        "mid_yield",
        "eod_yield",
    ]

    cusip_timeseries_db = ShelveDBWrapper(DB_PATH_CUSIP_TIMESERIES, create=len(sys.argv) >= 2 and sys.argv[1] == "init")
    cusip_timeseries_db.open()

    for curr_date in bdates:
        try:
            date_str = curr_date.to_pydatetime().strftime("%Y-%m-%d")
            for entry in json.loads(cusip_curve_set_db.get(date_str))["data"]:
                cusip = entry["cusip"]
                to_write = {
                    "Date": date_str,
                    "bid_price": entry["bid_price"],
                    "offer_price": entry["offer_price"],
                    "mid_price": entry["mid_price"],
                    "eod_price": entry["eod_price"],
                    "bid_yield": entry["bid_yield"],
                    "offer_yield": entry["offer_yield"],
                    "mid_yield": entry["mid_yield"],
                    "eod_yield": entry["eod_yield"],
                }
                cusip_timeseries[cusip].append(to_write)
            print(bcolors.OKBLUE + f"Saw {date_str}" + bcolors.ENDC)
        except Exception as e:
            print(bcolors.FAIL + f"FAILED {date_str} - {str(e)}" + bcolors.ENDC)

    for cusip, timeseries in cusip_timeseries.items():
        try:
            if cusip_timeseries_db.exists(cusip):
                existing_cusip_timeseries = json.loads(cusip_timeseries_db.get(cusip))
                existing_cusip_timeseries += timeseries
                cusip_timeseries_db.set(cusip, json.dumps(existing_cusip_timeseries))
            else:
                cusip_timeseries_db.set(cusip, json.dumps(timeseries))
            print(bcolors.OKGREEN + f"Wrote time series for CUSIP {cusip} to DB" + bcolors.ENDC)
        except Exception as e:
            print(bcolors.FAIL + f"FAILED to Write {cusip} to DB: {e}" + bcolors.ENDC)


    print(f"Timeseries Script took: {time.time() - t1} seconds")

    ##################

    print(bcolors.OKBLUE + "STARTING CT YIELDS SCRIPT" + bcolors.ENDC)
    time.sleep(5)
    t1 = time.time()

    results_dict = {}

    DB_PATH_CT_EOD_YIELDS = "../db/ust_eod_ct_yields"
    DB_PATH_CT_BID_YIELDS = "../db/ust_bid_ct_yields"
    DB_PATH_CT_MID_YIELDS = "../db/ust_mid_ct_yields"
    DB_PATH_CT_OFFER_YIELDS = "../db/ust_offer_ct_yields"

    ct_db_mapper = {
        "eod_yield": shelve.open(DB_PATH_CT_EOD_YIELDS),
        "bid_yield": shelve.open(DB_PATH_CT_BID_YIELDS),
        "mid_yield": shelve.open(DB_PATH_CT_MID_YIELDS),
        "offer_yield": shelve.open(DB_PATH_CT_OFFER_YIELDS),
    }

    for curr_date in bdates:
        try:
            df = pd.DataFrame(json.loads(cusip_curve_set_db.get(curr_date.to_pydatetime().strftime("%Y-%m-%d")))["data"])
            date_cols = ["auction_date", "issue_date", "maturity_date"]
            for date_col in date_cols:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

            df_coups = df.sort_values(by=["issue_date"], ascending=False)
            df_coups = df_coups[(df_coups["security_type"] == "Note") | (df_coups["security_type"] == "Bond")]
            df_coups = df_coups.groupby("original_security_term").first().reset_index()

            df_bills = df.sort_values(by=["maturity_date"], ascending=True)
            df_bills = df_bills[df_bills["security_type"] == "Bill"]
            df_bills = df_bills.groupby("security_term").last().reset_index()

            df_bills.loc[df_bills["security_term"] == "4-Week", "original_security_term"] = "4-Week"
            df_bills.loc[df_bills["security_term"] == "8-Week", "original_security_term"] = "8-Week"
            df_bills.loc[df_bills["security_term"] == "13-Week", "original_security_term"] = "13-Week"
            df_bills.loc[df_bills["security_term"] == "17-Week", "original_security_term"] = "17-Week"
            df_bills.loc[df_bills["security_term"] == "26-Week", "original_security_term"] = "26-Week"
            df_bills.loc[df_bills["security_term"] == "52-Week", "original_security_term"] = "52-Week"

            df = pd.concat([df_bills, df_coups])
            cusip_to_term_dict = dict(zip(df["cusip"], df["original_security_term"]))
            df["cusip"] = df["cusip"].replace(cusip_to_term_dict)
            df = df.set_index("cusip")

            otr_df = df.groupby(by=["original_security_term"]).first()

            for ytm_type in ["eod_yield", "bid_yield", "mid_yield", "offer_yield"]:
                ct_db_mapper[ytm_type][curr_date.strftime("%Y-%m-%d")] = dict(zip(otr_df.index, otr_df[ytm_type]))

            print(bcolors.OKGREEN + f"Wrote {curr_date.date()} CT yields to DB" + bcolors.ENDC)

        except Exception as e:
            print(bcolors.FAIL + f"Error occured formatting CT Yields at {curr_date.date()}: {str(e)}" + bcolors.ENDC)

    print(f"CT Yields Script took: {time.time() - t1} seconds")
    print(f"UST Update Script took: {time.time() - t1_parent} seconds")
