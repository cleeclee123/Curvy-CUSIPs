import asyncio
import math
import os
import warnings
from datetime import datetime
from typing import Dict, List, Optional

import time
import httpx
import numpy as np
import pandas as pd
import requests

from DataFetcher.base import DataFetcherBase
from utils.fetch_ust_par_yields import (
    multi_download_year_treasury_par_yield_curve_rate,
)
from utils.utils import (
    JSON,
    build_treasurydirect_header,
    get_active_cusips,
    historical_auction_cols,
    last_day_n_months_ago,
)

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

import sys

if sys.platform == "win32":
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)


class USTreasuryDataFetcher(DataFetcherBase):
    _use_ust_issue_date: bool = False
    _historical_auctions_df: pd.DataFrame = (None,)
    _valid_tenors_strings = [
        "CMT1M",
        "CMT2M",
        "CMT3M",
        "CMT4M",
        "CMT6M",
        "CMT1",
        "CMT2",
        "CMT3",
        "CMT5",
        "CMT7",
        "CMT10",
        "CMT20",
        "CMT30",
    ]

    def __init__(
        self,
        use_ust_issue_date: Optional[bool] = False,
        global_timeout: int = 10,
        proxies: Optional[Dict[str, str]] = None,
        debug_verbose: Optional[bool] = False,
        info_verbose: Optional[bool] = False,
        error_verbose: Optional[bool] = False,
    ):
        super().__init__(
            global_timeout=global_timeout,
            proxies=proxies,
            debug_verbose=debug_verbose,
            info_verbose=info_verbose,
            error_verbose=error_verbose,
        )

        self._use_ust_issue_date = use_ust_issue_date
        time.sleep(0.01)
        self._historical_auctions_df = self.get_auctions_df()
        self._historical_auctions_df["auction_date"] = pd.to_datetime(self._historical_auctions_df["auction_date"])
        self._historical_auctions_df["issue_date"] = pd.to_datetime(self._historical_auctions_df["issue_date"])
        self._historical_auctions_df["maturity_date"] = pd.to_datetime(self._historical_auctions_df["maturity_date"])
        self._historical_auctions_df["int_rate"] = pd.to_numeric(self._historical_auctions_df["int_rate"], errors="coerce")
        self._historical_auctions_df["high_investment_rate"] = pd.to_numeric(self._historical_auctions_df["high_investment_rate"], errors="coerce")
        self._historical_auctions_df["ust_label"] = self._historical_auctions_df.apply(
            lambda row: (
                f"{row['high_investment_rate']}% {row['maturity_date'].strftime('%b-%y')}"
                if pd.isna(row["int_rate"])
                else f"{row['int_rate']}% {row['maturity_date'].strftime('%b-%y')}"
            ),
            axis=1,
        )

    # ust label: f"{coupon}s {datetime.strftime("%Y-%m-%d")}"
    def cme_ust_label_to_cusip(self, ust_label: str):
        try:
            coupon = float(ust_label.split("s")[0])
            maturity = ust_label.split(" ")[1]
            ust_row = self._historical_auctions_df[
                (self._historical_auctions_df["int_rate"] == coupon) & (self._historical_auctions_df["maturity_date"] == maturity)
            ]
            return ust_row.to_dict("records")[0]
        except:
            raise Exception("LABEL NOT FOUND")

    def cusip_to_cme_ust_label(self, cusip: str):
        ust_row = self._historical_auctions_df[self._historical_auctions_df["cusip"] == cusip].to_dict("records")[0]
        return f"{ust_row["int_rate"]}s {ust_row["maturity_date"]}"

    # ust_label = f"{row['int_rate']:.3f}% {row['maturity_date'].strftime('%b-%y')}"
    def ust_label_to_cusip(self, ust_label: str):
        try:
            ust_row = self._historical_auctions_df[self._historical_auctions_df["ust_label"] == ust_label]
            return ust_row.to_dict("records")[0]
        except:
            raise Exception("LABEL NOT FOUND")

    def cusip_to_ust_label(self, cusip: str):
        ust_row = self._historical_auctions_df[self._historical_auctions_df["cusip"] == cusip].to_dict("records")[0]
        return ust_row["ust_label"]

    async def _build_fetch_tasks_historical_treasury_auctions(
        self,
        client: httpx.AsyncClient,
        assume_data_size=True,
        uid: Optional[str | int] = None,
        return_df: Optional[bool] = False,
        as_of_date: Optional[datetime] = None,  # active cusips as of
        max_retries: Optional[int] = 5,
        backoff_factor: Optional[int] = 1,
    ):
        MAX_TREASURY_GOV_API_CONTENT_SIZE = 10000
        NUM_REQS_NEEDED_TREASURY_GOV_API = 2

        def get_treasury_query_sizing() -> List[str]:
            base_url = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query?page[number]=1&page[size]=1"
            res = requests.get(base_url, headers=build_treasurydirect_header(), proxies=self._proxies)
            if res.ok:
                meta = res.json()["meta"]
                size = meta["total-count"]
                number_requests = math.ceil(size / MAX_TREASURY_GOV_API_CONTENT_SIZE)
                return [
                    f"https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query?page[number]={i+1}&page[size]={MAX_TREASURY_GOV_API_CONTENT_SIZE}"
                    for i in range(0, number_requests)
                ]
            else:
                raise ValueError(f"UST Auctions - Query Sizing Bad Status: ", {res.status_code})

        links = (
            get_treasury_query_sizing()
            if not assume_data_size
            else [
                f"https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query?page[number]={i+1}&page[size]={MAX_TREASURY_GOV_API_CONTENT_SIZE}"
                for i in range(0, NUM_REQS_NEEDED_TREASURY_GOV_API)
            ]
        )
        self._logger.debug(f"UST Auctions - Number of Links to Fetch: {len(links)}")
        self._logger.debug(f"UST Auctions - Links: {links}")

        async def fetch(
            client: httpx.AsyncClient,
            url,
            as_of_date: Optional[datetime] = None,
            return_df: Optional[bool] = False,
            uid: Optional[str | int] = None,
            max_retries: Optional[int] = 5,
            backoff_factor: Optional[int] = 1,
        ):
            retries = 0
            try:
                while retries < max_retries:
                    try:
                        response = await client.get(
                            url,
                            headers=build_treasurydirect_header(),
                        )
                        response.raise_for_status()
                        json_data: JSON = response.json()
                        if as_of_date:
                            df = get_active_cusips(
                                auction_json=json_data["data"],
                                as_of_date=as_of_date,
                                use_issue_date=self._use_ust_issue_date,
                            )
                            if uid:
                                return df[historical_auction_cols()], uid
                            return df[historical_auction_cols()]

                        if return_df and not as_of_date:
                            if uid:
                                return (
                                    pd.DataFrame(json_data["data"])[historical_auction_cols()],
                                    uid,
                                )
                            return pd.DataFrame(json_data["data"])[historical_auction_cols()]
                        if uid:
                            return json_data["data"], uid
                        return json_data["data"]

                    except httpx.HTTPStatusError as e:
                        self._logger.error(f"UST AUCTIONS - Bad Status: {response.status_code}")

                        # UST's website is very weird sometimes - maybe getting throttled?
                        # if response.status_code == 404:
                        #     if uid:
                        #         return pd.DataFrame(columns=historical_auction_cols()), uid
                        #     return pd.DataFrame(columns=historical_auction_cols())

                        retries += 1
                        wait_time = backoff_factor * (2 ** (retries - 1))
                        self._logger.debug(f"UST AUCTIONS - Throttled. Waiting for {wait_time} seconds before retrying...")
                        await asyncio.sleep(wait_time)

                    except Exception as e:
                        self._logger.error(f"UST AUCTIONS - Error: {e}")
                        retries += 1
                        wait_time = backoff_factor * (2 ** (retries - 1))
                        self._logger.debug(f"UST STRIPPING Activity - Throttled. Waiting for {wait_time} seconds before retrying...")
                        await asyncio.sleep(wait_time)

                raise ValueError(f"UST AUCTIONS Activity - Max retries exceeded")

            except Exception as e:
                self._logger.error(e)
                raise e
                # if uid:
                #     return pd.DataFrame(columns=historical_auction_cols()), uid
                # return pd.DataFrame(columns=historical_auction_cols())

        tasks = [
            fetch(client=client, url=url, as_of_date=as_of_date, return_df=return_df, uid=uid, max_retries=max_retries, backoff_factor=backoff_factor)
            for url in links
        ]
        return tasks

    def get_auctions_df(self, as_of_date: Optional[datetime] = None) -> pd.DataFrame:
        async def build_tasks(client: httpx.AsyncClient, as_of_date: datetime):
            tasks = await self._build_fetch_tasks_historical_treasury_auctions(client=client, as_of_date=as_of_date, return_df=True)
            return await asyncio.gather(*tasks)

        async def run_fetch_all(as_of_date: datetime):
            async with httpx.AsyncClient(proxy=self._proxies["https"]) as client:
                all_data = await build_tasks(client=client, as_of_date=as_of_date)
                return all_data

        dfs = asyncio.run(run_fetch_all(as_of_date=as_of_date))
        auctions_df: pd.DataFrame = pd.concat(dfs)
        auctions_df = auctions_df.sort_values(by=["auction_date"], ascending=False)
        return auctions_df

    def get_historical_cmt_yields(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        treasury_data_dir: Optional[str] = None,
        download_treasury_par_yields: Optional[bool] = False,
        apply_long_term_extrapolation_factor: Optional[bool] = False,
        tenors: Optional[List[str]] = None,
    ):
        print("Fetching from treasury.gov...")
        dir_path = treasury_data_dir or os.getcwd()
        start_year = start_date.year if start_date else 1990
        end_year = end_date.year if end_date else 2024
        if start_year == end_year:
            start_year = start_year - 1
        years = [str(x) for x in range(end_year, start_year, -1)]
        ust_daily_data = multi_download_year_treasury_par_yield_curve_rate(
            years,
            dir_path,
            run_all=True,
            download=download_treasury_par_yields,
            proxy=self._proxies["https"],
        )
        if "daily_treasury_yield_curve" not in ust_daily_data:
            raise ValueError("CMT Yield - Fetch Failed - Fetching from treasury.gov")
        df_par_rates: pd.DataFrame = ust_daily_data["daily_treasury_yield_curve"]
        df_par_rates["Date"] = pd.to_datetime(df_par_rates["Date"])
        df_par_rates = df_par_rates.sort_values(by=["Date"], ascending=True).reset_index(drop=True)

        if start_date:
            df_par_rates = df_par_rates[df_par_rates["Date"] >= start_date]
        if end_date:
            df_par_rates = df_par_rates[df_par_rates["Date"] <= end_date]

        if apply_long_term_extrapolation_factor:
            try:
                df_lt_avg_rate = ust_daily_data["daily_treasury_long_term_rate"]
                df_lt_avg_rate["Date"] = pd.to_datetime(df_lt_avg_rate["Date"])
                df_par_rates_lt_adj = pd.merge(df_par_rates, df_lt_avg_rate, on="Date", how="left")
                df_par_rates_lt_adj["20 Yr"] = df_par_rates_lt_adj["20 Yr"].fillna(df_par_rates_lt_adj["TREASURY 20-Yr CMT"])
                df_par_rates_lt_adj["30 Yr"] = np.where(
                    df_par_rates_lt_adj["30 Yr"].isna(),
                    df_par_rates_lt_adj["20 Yr"] + df_par_rates_lt_adj["Extrapolation Factor"],
                    df_par_rates_lt_adj["30 Yr"],
                )
                df_par_rates_lt_adj = df_par_rates_lt_adj.drop(
                    columns=[
                        "LT COMPOSITE (>10 Yrs)",
                        "TREASURY 20-Yr CMT",
                        "Extrapolation Factor",
                    ]
                )
                df_par_rates_lt_adj.columns = ["Date"] + self._valid_tenors_strings
                if tenors:
                    tenors = ["Date"] + tenors
                    df_par_rates_lt_adj[tenors]
                if download_treasury_par_yields:
                    df_par_rates_lt_adj.to_excel(
                        os.path.join(
                            dir_path,
                            "daily_treasury_par_yields_with_lt_extrap.xlsx",
                        )
                    )
                return df_par_rates_lt_adj
            except Exception as e:
                self._logger.error(f"UST CMT Yields - LT Extra Failed: {str(e)}")
                msg = "Applying Long-Term Extrapolation Factor Failed"
                is_to_recent = start_date > datetime(2006, 2, 9)
                if is_to_recent:
                    msg += f" - {start_date} is too recent - pick a starting date older than February 9, 2006"
                print(msg)

        df_par_rates.columns = ["Date"] + self._valid_tenors_strings

        if tenors:
            tenors = ["Date"] + tenors
            return df_par_rates[tenors]

        return df_par_rates

    async def _fetch_mspd_table_5(
        self,
        client: httpx.AsyncClient,
        date: datetime,
        uid: Optional[str | int] = None,
        max_retries: Optional[int] = 3,
        backoff_factor: Optional[int] = 1,
    ):
        cols_to_return = [
            "cusip",
            "corpus_cusip",
            "outstanding_amt",  # not using this col since not all USTs have stripping activity - using MSPD table 3 for "outstanding_amt"
            "portion_unstripped_amt",
            "portion_stripped_amt",
            "reconstituted_amt",
        ]
        retries = 0
        try:
            while retries < max_retries:
                try:
                    last_n_months_last_business_days: List[datetime] = last_day_n_months_ago(date, n=2, return_all=True)
                    self._logger.debug(f"STRIPping - BDays: {last_n_months_last_business_days}")
                    dates_str_query = ",".join([date.strftime("%Y-%m-%d") for date in last_n_months_last_business_days])
                    url = f"https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/debt/mspd/mspd_table_5?filter=record_date:in:({dates_str_query})&page[number]=1&page[size]=10000"
                    self._logger.debug(f"STRIPping - {date} url: {url}")
                    response = await client.get(
                        url,
                        headers=build_treasurydirect_header(host_str="api.fiscaldata.treasury.gov"),
                    )
                    response.raise_for_status()
                    curr_stripping_activity_json = response.json()
                    curr_stripping_activity_df = pd.DataFrame(curr_stripping_activity_json["data"])
                    curr_stripping_activity_df = curr_stripping_activity_df[
                        curr_stripping_activity_df["security_class1_desc"] != "Treasury Inflation-Protected Securities"
                    ]
                    curr_stripping_activity_df["record_date"] = pd.to_datetime(curr_stripping_activity_df["record_date"], errors="coerce")
                    latest_date = curr_stripping_activity_df["record_date"].max()
                    curr_stripping_activity_df = curr_stripping_activity_df[curr_stripping_activity_df["record_date"] == latest_date]
                    curr_stripping_activity_df["outstanding_amt"] = pd.to_numeric(curr_stripping_activity_df["outstanding_amt"], errors="coerce")
                    curr_stripping_activity_df["portion_unstripped_amt"] = pd.to_numeric(
                        curr_stripping_activity_df["portion_unstripped_amt"],
                        errors="coerce",
                    )
                    curr_stripping_activity_df["portion_stripped_amt"] = pd.to_numeric(
                        curr_stripping_activity_df["portion_stripped_amt"], errors="coerce"
                    )
                    curr_stripping_activity_df["reconstituted_amt"] = pd.to_numeric(curr_stripping_activity_df["reconstituted_amt"], errors="coerce")
                    col1 = "cusip"
                    col2 = "security_class2_desc"
                    curr_stripping_activity_df.columns = [
                        col2 if col == col1 else col1 if col == col2 else col for col in curr_stripping_activity_df.columns
                    ]
                    curr_stripping_activity_df = curr_stripping_activity_df.rename(columns={"security_class2_desc": "corpus_cusip"})

                    if uid:
                        return date, curr_stripping_activity_df[cols_to_return], uid

                    return date, curr_stripping_activity_df[cols_to_return]

                except httpx.HTTPStatusError as e:
                    self._logger.error(f"STRIPping - Bad Status: {response.status_code}")
                    if response.status_code == 404:
                        if uid:
                            return date, pd.DataFrame(columns=cols_to_return), uid
                        return date, pd.DataFrame(columns=cols_to_return)

                    retries += 1
                    wait_time = backoff_factor * (2 ** (retries - 1))
                    self._logger.debug(f"UST STRIPPING Activity - Throttled. Waiting for {wait_time} seconds before retrying...")
                    await asyncio.sleep(wait_time)

                except Exception as e:
                    self._logger.error(f"UST STRIPPING Activity - Error for {date}: {e}")
                    retries += 1
                    wait_time = backoff_factor * (2 ** (retries - 1))
                    self._logger.debug(f"UST STRIPPING Activity - Throttled. Waiting for {wait_time} seconds before retrying...")
                    await asyncio.sleep(wait_time)

            raise ValueError(f"UST STRIPPING Activity - Max retries exceeded for {date}")

        except Exception as e:
            self._logger.error(e)
            if uid:
                return date, pd.DataFrame(columns=cols_to_return), uid
            return date, pd.DataFrame(columns=cols_to_return)

    async def _fetch_mspd_table_5_with_semaphore(self, semaphore, *args, **kwargs):
        async with semaphore:
            return await self._fetch_mspd_table_5(*args, **kwargs)

    async def _build_fetch_tasks_historical_stripping_activity(
        self,
        client: httpx.AsyncClient,
        dates: List[datetime],
        uid: Optional[str | int] = None,
        minimize_api_calls=False,
        max_retries: Optional[int] = 3,
        backoff_factor: Optional[int] = 1,
        max_concurrent_tasks: int = 64,
        my_semaphore: Optional[asyncio.Semaphore] = None,
    ):

        semaphore = my_semaphore or asyncio.Semaphore(max_concurrent_tasks)

        if minimize_api_calls:
            seen_month_years = set()
            filtered_list = []
            for date in dates:
                month_year = (date.year, date.month)
                if month_year not in seen_month_years:
                    seen_month_years.add(month_year)
                    filtered_list.append(date)

            tasks = [
                self._fetch_mspd_table_5_with_semaphore(
                    client=client, date=date, semaphore=semaphore, uid=uid, max_retries=max_retries, backoff_factor=backoff_factor
                )
                for date in filtered_list
            ]
            return tasks

        else:
            tasks = [
                self._fetch_mspd_table_5_with_semaphore(
                    client=client, date=date, semaphore=semaphore, uid=uid, max_retries=max_retries, backoff_factor=backoff_factor
                )
                for date in dates
            ]
            return tasks
