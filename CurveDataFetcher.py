import asyncio
import logging
import math
import multiprocessing as mp
import time
import warnings
from collections import defaultdict
from datetime import datetime
from functools import partial, reduce
from typing import Dict, List, Optional, Tuple, Literal
import os
import aiohttp
import httpx
import numpy as np
import pandas as pd
import requests
import ujson as json

from utils.QL_BondPricer import QL_BondPricer
from utils.RL_BondPricer import RL_BondPricer
from utils.utils import (
    JSON,
    build_treasurydirect_header,
    cookie_string_to_dict,
    get_active_cusips,
    get_last_n_off_the_run_cusips,
    historical_auction_cols,
    is_valid_ust_cusip,
    last_day_n_months_ago,
    ust_labeler,
    ust_sorter,
    get_isin_from_cusip,
)
from utils.fred import Fred
from utils.fetch_ust_par_yields import (
    multi_download_year_treasury_par_yield_curve_rate,
)

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

import sys

Valid_String_Tenors = [
    "4-Week",
    "8-Week",
    "13-Week",
    "17-Week",
    "26-Week",
    "52-Week",
    "2-Year",
    "3-Year",
    "5-Year",
    "7-Year",
    "10-Year",
    "20-Year",
    "30-Year",
]

if sys.platform == "win32":
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)


def calculate_yields(row, as_of_date, use_quantlib=False):
    if use_quantlib:
        offer_yield = QL_BondPricer.bond_price_to_ytm(
            type=row["security_type"],
            issue_date=row["issue_date"],
            maturity_date=row["maturity_date"],
            as_of=as_of_date,
            coupon=row["int_rate"] / 100,
            price=row["offer_price"],
        )
        bid_yield = QL_BondPricer.bond_price_to_ytm(
            type=row["security_type"],
            issue_date=row["issue_date"],
            maturity_date=row["maturity_date"],
            as_of=as_of_date,
            coupon=row["int_rate"] / 100,
            price=row["bid_price"],
        )
        eod_yield = QL_BondPricer.bond_price_to_ytm(
            type=row["security_type"],
            issue_date=row["issue_date"],
            maturity_date=row["maturity_date"],
            as_of=as_of_date,
            coupon=row["int_rate"] / 100,
            price=row["eod_price"],
        )
    else:
        offer_yield = RL_BondPricer.bond_price_to_ytm(
            type=row["security_type"],
            issue_date=row["issue_date"],
            maturity_date=row["maturity_date"],
            as_of=as_of_date,
            coupon=row["int_rate"] / 100,
            price=row["offer_price"],
        )
        bid_yield = RL_BondPricer.bond_price_to_ytm(
            type=row["security_type"],
            issue_date=row["issue_date"],
            maturity_date=row["maturity_date"],
            as_of=as_of_date,
            coupon=row["int_rate"] / 100,
            price=row["bid_price"],
        )
        eod_yield = RL_BondPricer.bond_price_to_ytm(
            type=row["security_type"],
            issue_date=row["issue_date"],
            maturity_date=row["maturity_date"],
            as_of=as_of_date,
            coupon=row["int_rate"] / 100,
            price=row["eod_price"],
        )

    return offer_yield, bid_yield, eod_yield


# TODO
# - find additional source for ust outstanding amount
# - public.com scrape
class CurveDataFetcher:
    _use_ust_issue_date: bool = False
    _global_timeout: int = 10
    _historical_auctions_df: pd.DataFrame = (None,)
    _fred: Fred = None
    _proxies: Dict[str, str] = {"http": None, "https": None}
    _httpx_proxies: Dict[str, str] = {"http://": None, "https://": None}
    _public_dotcom_jwt: str = None

    _logger = logging.getLogger(__name__)
    _debug_verbose: bool = False
    _error_verbose: bool = False
    _info_verbose: bool = False  # performance benchmarking mainly
    _no_logs_plz: bool = False

    def __init__(
        self,
        use_ust_issue_date: Optional[bool] = False,
        global_timeout: int = 10,
        fred_api_key: Optional[str] = None,
        proxies: Optional[Dict[str, str]] = None,
        debug_verbose: Optional[bool] = False,
        info_verbose: Optional[bool] = False,
        error_verbose: Optional[bool] = False,
        no_logs_plz: Optional[bool] = False,
    ):
        self._use_ust_issue_date = use_ust_issue_date
        self._global_timeout = global_timeout

        self._historical_auctions_df = self.get_auctions_df()
        self._proxies = proxies if proxies else {"http": None, "https": None}
        self._httpx_proxies["http://"] = self._proxies["http"]
        self._httpx_proxies["https://"] = self._proxies["https"]

        if fred_api_key:
            self._fred = Fred(api_key=fred_api_key, proxies=self._proxies)

        self._public_dotcom_jwt = self._fetch_public_dotcome_jwt()

        self._debug_verbose = debug_verbose
        self._error_verbose = error_verbose
        self._info_verbose = info_verbose
        self._no_logs_plz = no_logs_plz

        if not self._logger.hasHandlers():
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            self._logger.addHandler(handler)

        if self._debug_verbose:
            self._logger.setLevel(logging.DEBUG)
        elif self._info_verbose:
            self._logger.setLevel(logging.INFO)
        elif self._error_verbose:
            self._logger.setLevel(logging.ERROR)
        else:
            self._logger.setLevel(logging.WARNING)

        if self._no_logs_plz:
            self._logger.disabled = True
            self._logger.propagate = False

        if self._debug_verbose or self._info_verbose or self._error_verbose:
            self._logger.setLevel(logging.DEBUG)

    async def _build_fetch_tasks_historical_treasury_auctions(
        self,
        client: httpx.AsyncClient,
        assume_data_size=True,
        uid: Optional[str | int] = None,
        return_df: Optional[bool] = False,
        as_of_date: Optional[datetime] = None,  # active cusips as of
    ):
        MAX_TREASURY_GOV_API_CONTENT_SIZE = 10000
        NUM_REQS_NEEDED_TREASURY_GOV_API = 2

        def get_treasury_query_sizing() -> List[str]:
            base_url = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query?page[number]=1&page[size]=1"
            res = requests.get(
                base_url, headers=build_treasurydirect_header(), proxies=self._proxies
            )
            if res.ok:
                meta = res.json()["meta"]
                size = meta["total-count"]
                number_requests = math.ceil(size / MAX_TREASURY_GOV_API_CONTENT_SIZE)
                return [
                    f"https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query?page[number]={i+1}&page[size]={MAX_TREASURY_GOV_API_CONTENT_SIZE}"
                    for i in range(0, number_requests)
                ]
            else:
                raise ValueError(
                    f"UST Auctions - Query Sizing Bad Status: ", {res.status_code}
                )

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
        ):
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
                self._logger.debug(f"UST Prices - Bad Status: {response.status_code}")
                if uid:
                    return pd.DataFrame(columns=historical_auction_cols()), uid
                return pd.DataFrame(columns=historical_auction_cols())
            except Exception as e:
                self._logger.debug(f"UST Prices - Error: {e}")
                if uid:
                    return pd.DataFrame(columns=historical_auction_cols()), uid
                return pd.DataFrame(columns=historical_auction_cols())

        tasks = [
            fetch(
                client=client,
                url=url,
                as_of_date=as_of_date,
                return_df=return_df,
                uid=uid,
            )
            for url in links
        ]
        return tasks

    def get_auctions_df(self, as_of_date: Optional[datetime] = None) -> pd.DataFrame:
        async def build_tasks(client: httpx.AsyncClient, as_of_date: datetime):
            tasks = await self._build_fetch_tasks_historical_treasury_auctions(
                client=client, as_of_date=as_of_date, return_df=True
            )
            return await asyncio.gather(*tasks)

        async def run_fetch_all(as_of_date: datetime):
            async with httpx.AsyncClient(proxy=self._proxies["https"]) as client:
                all_data = await build_tasks(client=client, as_of_date=as_of_date)
                return all_data

        dfs = asyncio.run(run_fetch_all(as_of_date=as_of_date))
        auctions_df: pd.DataFrame = pd.concat(dfs)
        auctions_df = auctions_df.sort_values(by=["auction_date"], ascending=False)
        return auctions_df

    async def _fetch_prices_from_treasury_date_search(
        self,
        client: httpx.AsyncClient,
        date: datetime,
        cusips: List[str],
        uid: Optional[int | str],
        max_retries: Optional[int] = 3,
        backoff_factor: Optional[int] = 1,
    ):
        payload = {
            "priceDate.month": date.month,
            "priceDate.day": date.day,
            "priceDate.year": date.year,
            "submit": "Show Prices",
        }
        headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "max-age=0",
            "Connection": "keep-alive",
            # "Content-Length": "100",
            "Content-Type": "application/x-www-form-urlencoded",
            "Dnt": "1",
            "Host": "savingsbonds.gov",
            "Origin": "https://savingsbonds.gov",
            "Referer": "https://savingsbonds.gov/GA-FI/FedInvest/selectSecurityPriceDate",
            "Sec-Ch-Ua": '"Google Chrome";v="125", "Chromium";v="125", "Not.A/Brand";v="24"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        }
        self._logger.debug(f"UST Prices - {date} Payload: {payload}")
        cols_to_return = ["cusip", "offer_price", "bid_price", "eod_price"]
        retries = 0
        try:
            while retries < max_retries:
                try:
                    url = "https://savingsbonds.gov/GA-FI/FedInvest/selectSecurityPriceDate"
                    response = await client.post(
                        url,
                        data=payload,
                        headers=headers,
                        follow_redirects=False,
                        timeout=self._global_timeout,
                    )
                    if response.is_redirect:
                        redirect_url = response.headers.get("Location")
                        self._logger.debug(
                            f"UST Prices - {date} Redirecting to {redirect_url}"
                        )
                        response = await client.get(redirect_url, headers=headers)

                    response.raise_for_status()
                    tables = pd.read_html(response.content, header=0)
                    df = tables[0]
                    if cusips:
                        missing_cusips = [
                            cusip for cusip in cusips if cusip not in df["CUSIP"].values
                        ]
                        if missing_cusips:
                            self._logger.warning(
                                f"UST Prices Warning - The following CUSIPs are not found in the DataFrame: {missing_cusips}"
                            )
                    df = df[df["CUSIP"].isin(cusips)] if cusips else df
                    df.columns = df.columns.str.lower()
                    df = df.query("`security type` not in ['TIPS', 'MARKET BASED FRN']")
                    df = df.rename(
                        columns={
                            "buy": "offer_price",
                            "sell": "bid_price",
                            "end of day": "eod_price",
                        }
                    )
                    if uid:
                        return date, df[cols_to_return], uid
                    return date, df[cols_to_return]

                except httpx.HTTPStatusError as e:
                    self._logger.error(
                        f"UST Prices - Bad Status for {date}: {response.status_code}"
                    )
                    if response.status_code == 404:
                        if uid:
                            return date, df[cols_to_return], uid
                        return date, df[cols_to_return]
                    retries += 1
                    wait_time = backoff_factor * (2 ** (retries - 1))
                    self._logger.debug(
                        f"UST Prices - Throttled. Waiting for {wait_time} seconds before retrying..."
                    )
                    await asyncio.sleep(wait_time)

                except Exception as e:
                    self._logger.error(f"UST Prices - Error for {date}: {e}")
                    retries += 1
                    wait_time = backoff_factor * (2 ** (retries - 1))
                    self._logger.debug(
                        f"UST Prices - Throttled. Waiting for {wait_time} seconds before retrying..."
                    )
                    await asyncio.sleep(wait_time)

            raise ValueError(f"UST Prices - Max retries exceeded for {date}")
        except Exception as e:
            self._logger.error(e)
            if uid:
                return date, pd.DataFrame(columns=cols_to_return), uid
            return date, pd.DataFrame(columns=cols_to_return)

    async def _fetch_prices_with_semaphore(self, semaphore, *args, **kwargs):
        async with semaphore:
            return await self._fetch_prices_from_treasury_date_search(*args, **kwargs)

    async def _build_fetch_tasks_historical_cusip_prices(
        self,
        client: httpx.AsyncClient,
        dates: List[datetime],
        cusips: Optional[List[str]] = None,
        uid: Optional[str | int] = None,
        max_concurrent_tasks: int = 64,
    ):
        semaphore = asyncio.Semaphore(max_concurrent_tasks)
        tasks = [
            self._fetch_prices_with_semaphore(
                semaphore,
                client=client,
                date=date,
                cusips=cusips,
                uid=uid,
            )
            for date in dates
        ]
        return tasks

    def github_headers(self, path: str):
        return {
            "authority": "raw.githubusercontent.com",
            "method": "GET",
            "path": path,
            "scheme": "https",
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "en-US,en;q=0.9",
            "cache-control": "no-cache",
            "dnt": "1",
            "pragma": "no-cache",
            "priority": "u=0, i",
            "sec-ch-ua": '"Not)A;Brand";v="99", "Google Chrome";v="127", "Chromium";v="127"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "document",
            "sec-fetch-mode": "navigate",
            "sec-fetch-site": "none",
            "sec-fetch-user": "?1",
            "upgrade-insecure-requests": "1",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
        }

    async def _fetch_ust_prices_from_github(
        self,
        client: httpx.AsyncClient,
        date: datetime,
        cusips: Optional[List[str]] = None,
        uid: Optional[str | int] = None,
        max_retries: Optional[int] = 3,
        backoff_factor: Optional[int] = 1,
        cols_to_return: Optional[List[str]] = [
            "cusip",
            "offer_yield",
            "bid_yield",
            "eod_yield",
        ],
        cusip_ref_replacement_dict: Optional[Dict[str, str]] = None,
        return_transpose_df: Optional[bool] = False,
        assume_otrs: Optional[bool] = False,
        set_cusips_as_index: Optional[bool] = False,
    ):
        date_str = date.strftime("%Y-%m-%d")
        headers = self.github_headers(
            path=f"/cleeclee123/CUSIP-Set/main/{date_str}.json"
        )
        url = f"https://raw.githubusercontent.com/cleeclee123/CUSIP-Set/main/{date_str}.json"
        retries = 0
        cols_to_return_copy = cols_to_return.copy()
        try:
            while retries < max_retries:
                try:
                    response = await client.get(url, headers=headers)
                    response.raise_for_status()
                    res_json = response.json()
                    df = pd.DataFrame(res_json["data"])
                    if df.empty:
                        self._logger.error(
                            f"UST Prices GitHub - Data is Empty for {date}"
                        )
                        if uid:
                            return date, None, uid
                        return date, None

                    df["issue_date"] = pd.to_datetime(df["issue_date"])
                    df["maturity_date"] = pd.to_datetime(df["maturity_date"])

                    if cusips:
                        missing_cusips = [
                            cusip for cusip in cusips if cusip not in df["cusip"].values
                        ]
                        if missing_cusips:
                            self._logger.warning(
                                f"UST Prices Warning - The following CUSIPs are not found in the DataFrame: {missing_cusips}"
                            )
                    df = df[df["cusip"].isin(cusips)] if cusips else df

                    if cusip_ref_replacement_dict:
                        df["cusip"] = df["cusip"].replace(cusip_ref_replacement_dict)

                    if assume_otrs and "original_security_term" in df.columns:
                        df = df.sort_values(by=["issue_date"], ascending=False)
                        df = df.groupby("original_security_term").first().reset_index()
                        cusip_to_term_dict = dict(
                            zip(df["cusip"], df["original_security_term"])
                        )
                        df["cusip"] = df["cusip"].replace(cusip_to_term_dict)

                    if set_cusips_as_index:
                        df = df.set_index("cusip")
                        cols_to_return_copy.remove("cusip")

                    if return_transpose_df:
                        df = df[cols_to_return_copy].T
                    else:
                        df = df[cols_to_return_copy]

                    if uid:
                        return date, df, uid
                    return date, df

                except httpx.HTTPStatusError as e:
                    self._logger.error(f"UST Prices GitHub - Error for {date}: {e}")
                    if response.status_code == 404:
                        if uid:
                            return date, None, uid
                        return date, None
                    retries += 1
                    wait_time = backoff_factor * (2 ** (retries - 1))
                    self._logger.debug(
                        f"UST Prices GitHub - Throttled for {date}. Waiting for {wait_time} seconds before retrying..."
                    )
                    await asyncio.sleep(wait_time)
                except Exception as e:
                    self._logger.error(f"UST Prices GitHub - Error for {date}: {e}")
                    retries += 1
                    wait_time = backoff_factor * (2 ** (retries - 1))
                    self._logger.debug(
                        f"UST Prices GitHub - Throttled for {date}. Waiting for {wait_time} seconds before retrying..."
                    )
                    await asyncio.sleep(wait_time)

            raise ValueError(f"UST Prices GitHub - Max retries exceeded for {date}")

        except Exception as e:
            self._logger.error(e)
            if uid:
                return date, None, uid
            return date, None

    async def _fetch_ust_prices_from_github_with_semaphore(
        self, semaphore, *args, **kwargs
    ):
        async with semaphore:
            return await self._fetch_ust_prices_from_github(*args, **kwargs)

    async def _build_fetch_tasks_historical_cusip_prices_github(
        self,
        client: httpx.AsyncClient,
        dates: List[datetime],
        cusips: Optional[List[str]] = None,
        uid: Optional[str | int] = None,
        max_concurrent_tasks: int = 64,
        cols_to_return: Optional[List[str]] = [
            "cusip",
            "offer_yield",
            "bid_yield",
            "eod_yield",
        ],
    ):
        semaphore = asyncio.Semaphore(max_concurrent_tasks)
        tasks = [
            self._fetch_ust_prices_from_github_with_semaphore(
                semaphore,
                client=client,
                date=date,
                cusips=cusips,
                uid=uid,
                cols_to_return=cols_to_return,
            )
            for date in dates
        ]
        return tasks

    def get_historical_ct_yields(
        self,
        start_date: datetime,
        end_date: datetime,
        tenors: Optional[List[str]] = None,
        use_bid_side: Optional[bool] = False,
        use_offer_side: Optional[bool] = False,
        use_mid_side: Optional[bool] = False,
    ):
        side = "eod"
        if use_bid_side:
            side = "bid"
        elif use_offer_side:
            side = "offer"
        elif use_mid_side:
            side = "mid"

        url = f"https://raw.githubusercontent.com/cleeclee123/CUSIP-Timeseries/main/historical_ct_yields_{side}_side.json"
        try:
            res = requests.get(
                url,
                headers=self.github_headers(
                    path=f"/cleeclee123/CUSIP-Timeseries/main/historical_ct_yields_{side}_side.json"
                ),
                proxies=self._proxies,
            )
            res.raise_for_status()
            df = pd.DataFrame(res.json())
            df["Date"] = pd.to_datetime(df["Date"])
            df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
            df = df.reset_index(drop=True)
            if tenors:
                tenors = ["Date"] + tenors
                return df[tenors]
            return df
        except Exception as e:
            self._logger.error(f"Historical CT Yields GitHub - {str(e)}")

    def get_historical_cmt_yields(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        use_treasury_par: Optional[bool] = False,
        treasury_data_dir: Optional[str] = None,
        download_treasury_par_yields: Optional[bool] = False,
        apply_long_term_extrapolation_factor: Optional[bool] = False,
        tenors: Optional[List[str]] = None,
    ):
        if self._fred is not None and not use_treasury_par:
            print("Fetching from FRED...")
            df = self._fred.get_multiple_series(
                series_ids=[
                    "DTB3",
                    "DTB6",
                    "DGS1",
                    "DGS2",
                    "DGS3",
                    "DGS5",
                    "DGS7",
                    "DGS10",
                    "DGS20",
                    "DGS30",
                ],
                one_df=True,
                observation_start=start_date,
                observation_end=end_date,
            )
            df.columns = [
                "13-Week",
                "26-Week",
                "52-Week",
                "2-Year",
                "3-Year",
                "5-Year",
                "7-Year",
                "10-Year",
                "20-Year",
                "30-Year",
            ]
            if tenors:
                tenors = ["Date"] + tenors
                return df[tenors]
            df = df.dropna()
            df = df.rename_axis("Date").reset_index()
            return df

        if use_treasury_par:
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
                raise ValueError(
                    "CMT Yield - Fetch Failed - Fetching from treasury.gov"
                )
            df_par_rates: pd.DataFrame = ust_daily_data["daily_treasury_yield_curve"]
            df_par_rates["Date"] = pd.to_datetime(df_par_rates["Date"])
            df_par_rates = df_par_rates.sort_values(
                by=["Date"], ascending=True
            ).reset_index(drop=True)

            if start_date:
                df_par_rates = df_par_rates[df_par_rates["Date"] >= start_date]
            if end_date:
                df_par_rates = df_par_rates[df_par_rates["Date"] <= end_date]

            if apply_long_term_extrapolation_factor:
                try:
                    df_lt_avg_rate = ust_daily_data["daily_treasury_long_term_rate"]
                    df_lt_avg_rate["Date"] = pd.to_datetime(df_lt_avg_rate["Date"])
                    df_par_rates_lt_adj = pd.merge(
                        df_par_rates, df_lt_avg_rate, on="Date", how="left"
                    )
                    df_par_rates_lt_adj["20 Yr"] = df_par_rates_lt_adj["20 Yr"].fillna(
                        df_par_rates_lt_adj["TREASURY 20-Yr CMT"]
                    )
                    df_par_rates_lt_adj["30 Yr"] = np.where(
                        df_par_rates_lt_adj["30 Yr"].isna(),
                        df_par_rates_lt_adj["20 Yr"]
                        + df_par_rates_lt_adj["Extrapolation Factor"],
                        df_par_rates_lt_adj["30 Yr"],
                    )
                    df_par_rates_lt_adj = df_par_rates_lt_adj.drop(
                        columns=[
                            "LT COMPOSITE (>10 Yrs)",
                            "TREASURY 20-Yr CMT",
                            "Extrapolation Factor",
                        ]
                    )
                    df_par_rates_lt_adj.columns = ["Date"] + Valid_String_Tenors
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

            df_par_rates.columns = ["Date"] + Valid_String_Tenors
            if tenors:
                tenors = ["Date"] + tenors
                return df_par_rates[tenors]
            return df_par_rates

        print("Plz put ur Fred API key or enable 'use_treasury_par' ")

    async def _fetch_single_ust_timeseries_github(
        self,
        client: httpx.AsyncClient,
        cusip: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        uid: Optional[str | int] = None,
        max_retries: Optional[int] = 3,
        backoff_factor: Optional[int] = 1,
    ):
        retries = 0
        try:
            while retries < max_retries:
                url = f"https://raw.githubusercontent.com/cleeclee123/CUSIP-Timeseries/main/{cusip}.json"
                headers = self.github_headers(
                    path=f"/cleeclee123/CUSIP-Timeseries/main/{cusip}.json"
                )
                try:
                    response = await client.get(url, headers=headers)
                    response.raise_for_status()
                    response_json = response.json()
                    df = pd.DataFrame(response_json)
                    df["Date"] = pd.to_datetime(df["Date"])
                    df = df.sort_values(by="Date")
                    if start_date:
                        df = df[df["Date"].dt.date >= start_date.date()]
                    if end_date:
                        df = df[df["Date"].dt.date <= end_date.date()]
                    if uid:
                        return cusip, df, uid
                    return cusip, df

                except httpx.HTTPStatusError as e:
                    self._logger.error(
                        f"UST Timeseries GitHub - Error for {cusip}: {e}"
                    )
                    if response.status_code == 404:
                        if uid:
                            return cusip, None, uid
                        return cusip, None
                    retries += 1
                    wait_time = backoff_factor * (2 ** (retries - 1))
                    self._logger.debug(
                        f"UST Timeseries GitHub - Throttled for {cusip}. Waiting for {wait_time} seconds before retrying..."
                    )
                    await asyncio.sleep(wait_time)
                except Exception as e:
                    self._logger.error(
                        f"UST Timeseries GitHub - Error for {cusip}: {e}"
                    )
                    retries += 1
                    wait_time = backoff_factor * (2 ** (retries - 1))
                    self._logger.debug(
                        f"UST Timeseries GitHub - Throttled for {cusip}. Waiting for {wait_time} seconds before retrying..."
                    )
                    await asyncio.sleep(wait_time)

            raise ValueError(
                f"UST Timeseries GitHub - Max retries exceeded for {cusip}"
            )

        except Exception as e:
            self._logger.error(e)
            if uid:
                return cusip, None, uid
            return cusip, None

    async def _fetch_ust_timeseries_github_with_semaphore(
        self, semaphore, *args, **kwargs
    ):
        async with semaphore:
            return await self._fetch_single_ust_timeseries_github(*args, **kwargs)

    def cusips_timeseries(
        self,
        cusips: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_concurrent_tasks: int = 64,
        max_keepalive_connections: int = 5,
    ):
        async def build_tasks(
            client: httpx.AsyncClient,
            cusips: List[str],
        ):
            tasks = []
            semaphore = asyncio.Semaphore(max_concurrent_tasks)
            for cusip in cusips:
                task = self._fetch_ust_timeseries_github_with_semaphore(
                    semaphore=semaphore,
                    client=client,
                    start_date=start_date,
                    end_date=end_date,
                    cusip=cusip,
                )
                tasks.append(task)

            return await asyncio.gather(*tasks)

        async def run_fetch_all(cusips: List[str]):
            limits = httpx.Limits(
                max_connections=max_concurrent_tasks,
                max_keepalive_connections=max_keepalive_connections,
            )
            async with httpx.AsyncClient(limits=limits) as client:
                all_data = await build_tasks(client=client, cusips=cusips)
                return all_data

        results: List[Tuple[str, pd.DataFrame]] = asyncio.run(
            run_fetch_all(cusips=cusips)
        )
        results_dict: Dict[str, pd.DataFrame] = {
            dt: df for dt, df in results if dt is not None and df is not None
        }
        return results_dict

    async def _build_fetch_tasks_historical_soma_holdings(
        self,
        client: httpx.AsyncClient,
        dates: List[datetime],
        uid: Optional[str | int] = None,
    ):
        valid_soma_holding_dates_reponse = requests.get(
            "https://markets.newyorkfed.org/api/soma/asofdates/list.json",
            headers=build_treasurydirect_header(host_str="markets.newyorkfed.org"),
            proxies=self._proxies,
        )
        if valid_soma_holding_dates_reponse.ok:
            valid_soma_holding_dates_json = valid_soma_holding_dates_reponse.json()
            valid_soma_dates_dt = [
                datetime.strptime(dt_string, "%Y-%m-%d")
                for dt_string in valid_soma_holding_dates_json["soma"]["asOfDates"]
            ]
        else:
            raise ValueError(
                f"SOMA Holdings - Status Code: {valid_soma_holding_dates_reponse.status_code}"
            )

        valid_soma_dates_from_input = {}
        for dt in dates:
            valid_closest_date = min(
                (valid_date for valid_date in valid_soma_dates_dt if valid_date <= dt),
                key=lambda valid_date: abs(dt - valid_date),
            )
            valid_soma_dates_from_input[dt] = valid_closest_date
        self._logger.debug(
            f"SOMA Holdings - Valid SOMA Holding Dates: {valid_soma_dates_from_input}"
        )

        async def fetch_single_soma_holding_day(
            client: httpx.AsyncClient,
            date: datetime,
            uid: Optional[str | int] = None,
        ):
            cols_to_return = [
                "cusip",
                "asOfDate",
                "parValue",
                "percentOutstanding",
                "est_outstanding_amt",
                # "changeFromPriorWeek",
                # "changeFromPriorYear",
            ]
            try:
                date_str = valid_soma_dates_from_input[date].strftime("%Y-%m-%d")
                print(f"Using SOMA Holdings Data As of {date_str}")
                url = f"https://markets.newyorkfed.org/api/soma/tsy/get/asof/{date_str}.json"
                response = await client.get(
                    url,
                    headers=build_treasurydirect_header(
                        host_str="markets.newyorkfed.org"
                    ),
                )
                response.raise_for_status()
                curr_soma_holdings_json = response.json()
                curr_soma_holdings_df = pd.DataFrame(
                    curr_soma_holdings_json["soma"]["holdings"]
                )
                curr_soma_holdings_df = curr_soma_holdings_df.fillna("")
                curr_soma_holdings_df["asOfDate"] = pd.to_datetime(
                    curr_soma_holdings_df["asOfDate"], errors="coerce"
                )
                curr_soma_holdings_df["parValue"] = pd.to_numeric(
                    curr_soma_holdings_df["parValue"], errors="coerce"
                )
                curr_soma_holdings_df["percentOutstanding"] = pd.to_numeric(
                    curr_soma_holdings_df["percentOutstanding"], errors="coerce"
                )
                curr_soma_holdings_df["est_outstanding_amt"] = (
                    curr_soma_holdings_df["parValue"]
                    / curr_soma_holdings_df["percentOutstanding"]
                )
                curr_soma_holdings_df = curr_soma_holdings_df[
                    (curr_soma_holdings_df["securityType"] != "TIPS")
                    & (curr_soma_holdings_df["securityType"] != "FRNs")
                ]
                if uid:
                    return date, curr_soma_holdings_df[cols_to_return], uid
                return date, curr_soma_holdings_df[cols_to_return]

            except httpx.HTTPStatusError as e:
                self._logger.error(f"SOMA Holding - Bad Status: {response.status_code}")
                if uid:
                    return date, pd.DataFrame(columns=cols_to_return), uid
                return date, pd.DataFrame(columns=cols_to_return)

            except Exception as e:
                self._logger.error(f"SOMA Holding - Error: {str(e)}")
                if uid:
                    return date, pd.DataFrame(columns=cols_to_return), uid
                return date, pd.DataFrame(columns=cols_to_return)

        tasks = [
            fetch_single_soma_holding_day(client=client, date=date, uid=uid)
            for date in dates
        ]
        return tasks

    async def _build_fetch_tasks_historical_amount_outstanding(
        self,
        client: httpx.AsyncClient,
        dates: List[datetime],
        only_marketable: Optional[bool] = False,
        uid: Optional[str | int] = None,
    ):
        async def fetch_mspd_table_3_market(
            client: httpx.AsyncClient,
            date: datetime,
            uid: Optional[str | int] = None,
        ):
            cols_to_return = [
                "cusip",
                "issued_amt",
                "redeemed_amt",
                "outstanding_amt",
            ]
            try:
                last_n_months_last_business_days: List[datetime] = (
                    last_day_n_months_ago(date, n=2, return_all=True)
                )
                self._logger.debug(
                    f"UST Outstanding - BDays: {last_n_months_last_business_days}"
                )
                dates_str_query = ",".join(
                    [
                        date.strftime("%Y-%m-%d")
                        for date in last_n_months_last_business_days
                    ]
                )
                if only_marketable:
                    url = f"https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/debt/mspd/mspd_table_3_market?filter=record_date:in:({dates_str_query})&page[number]=1&page[size]=10000"
                else:
                    url = f"https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/debt/mspd/mspd_table_3?filter=record_date:in:({dates_str_query})&page[number]=1&page[size]=10000"

                self._logger.debug(f"UST Outstanding - {date} url: {url}")
                response = await client.get(
                    url,
                    headers=build_treasurydirect_header(
                        host_str="api.fiscaldata.treasury.gov"
                    ),
                )
                response.raise_for_status()
                ust_outstanding_json = response.json()
                ust_outstanding_df = pd.DataFrame(ust_outstanding_json["data"])

                col1 = "cusip"
                col2 = "security_class2_desc"
                ust_outstanding_df.columns = [
                    col2 if col == col1 else col1 if col == col2 else col
                    for col in ust_outstanding_df.columns
                ]

                ust_outstanding_df["record_date"] = pd.to_datetime(
                    ust_outstanding_df["record_date"], errors="coerce"
                )
                latest_date = ust_outstanding_df["record_date"].max()
                print(f"Using Outstanding Amount Data As of {latest_date}")
                ust_outstanding_df = ust_outstanding_df[
                    ust_outstanding_df["record_date"] == latest_date
                ]

                for col in cols_to_return[1:]:
                    ust_outstanding_df[col] = pd.to_numeric(
                        ust_outstanding_df[col], errors="coerce"
                    )

                ust_outstanding_df = ust_outstanding_df[
                    ust_outstanding_df["cusip"].apply(is_valid_ust_cusip)
                ]

                if uid:
                    return date, ust_outstanding_df[cols_to_return], uid
                return date, ust_outstanding_df[cols_to_return]

            except httpx.HTTPStatusError as e:
                self._logger.error(
                    f"UST Outstanding - Bad Status: {response.status_code}"
                )
                if uid:
                    return date, pd.DataFrame(columns=cols_to_return), uid
                return date, pd.DataFrame(columns=cols_to_return)

            except Exception as e:
                self._logger.error(f"UST Outstanding - Error: {str(e)}")
                if uid:
                    return date, pd.DataFrame(columns=cols_to_return), uid
                return date, pd.DataFrame(columns=cols_to_return)

        tasks = [
            fetch_mspd_table_3_market(client=client, date=date, uid=uid)
            for date in dates
        ]
        return tasks

    async def _build_fetch_tasks_historical_stripping_activity(
        self,
        client: httpx.AsyncClient,
        dates: List[datetime],
        uid: Optional[str | int] = None,
    ):
        async def fetch_mspd_table_5(
            client: httpx.AsyncClient,
            date: datetime,
            uid: Optional[str | int] = None,
        ):
            cols_to_return = [
                "cusip",
                "corpus_cusip",
                "outstanding_amt",  # not using this col since not all USTs have stripping activity - using MSPD table 3 for "outstanding_amt"
                "portion_unstripped_amt",
                "portion_stripped_amt",
                "reconstituted_amt",
            ]
            try:
                last_n_months_last_business_days: List[datetime] = (
                    last_day_n_months_ago(date, n=2, return_all=True)
                )
                self._logger.debug(
                    f"STRIPping - BDays: {last_n_months_last_business_days}"
                )
                dates_str_query = ",".join(
                    [
                        date.strftime("%Y-%m-%d")
                        for date in last_n_months_last_business_days
                    ]
                )
                url = f"https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/debt/mspd/mspd_table_5?filter=record_date:in:({dates_str_query})&page[number]=1&page[size]=10000"
                self._logger.debug(f"STRIPping - {date} url: {url}")
                response = await client.get(
                    url,
                    headers=build_treasurydirect_header(
                        host_str="api.fiscaldata.treasury.gov"
                    ),
                )
                response.raise_for_status()
                curr_stripping_activity_json = response.json()
                curr_stripping_activity_df = pd.DataFrame(
                    curr_stripping_activity_json["data"]
                )
                curr_stripping_activity_df = curr_stripping_activity_df[
                    curr_stripping_activity_df["security_class1_desc"]
                    != "Treasury Inflation-Protected Securities"
                ]
                curr_stripping_activity_df["record_date"] = pd.to_datetime(
                    curr_stripping_activity_df["record_date"], errors="coerce"
                )
                latest_date = curr_stripping_activity_df["record_date"].max()
                print(f"Using STRIPping Data As of {latest_date}")
                curr_stripping_activity_df = curr_stripping_activity_df[
                    curr_stripping_activity_df["record_date"] == latest_date
                ]
                curr_stripping_activity_df["outstanding_amt"] = pd.to_numeric(
                    curr_stripping_activity_df["outstanding_amt"], errors="coerce"
                )
                curr_stripping_activity_df["portion_unstripped_amt"] = pd.to_numeric(
                    curr_stripping_activity_df["portion_unstripped_amt"],
                    errors="coerce",
                )
                curr_stripping_activity_df["portion_stripped_amt"] = pd.to_numeric(
                    curr_stripping_activity_df["portion_stripped_amt"], errors="coerce"
                )
                curr_stripping_activity_df["reconstituted_amt"] = pd.to_numeric(
                    curr_stripping_activity_df["reconstituted_amt"], errors="coerce"
                )
                col1 = "cusip"
                col2 = "security_class2_desc"
                curr_stripping_activity_df.columns = [
                    col2 if col == col1 else col1 if col == col2 else col
                    for col in curr_stripping_activity_df.columns
                ]
                curr_stripping_activity_df = curr_stripping_activity_df.rename(
                    columns={"security_class2_desc": "corpus_cusip"}
                )

                if uid:
                    return date, curr_stripping_activity_df[cols_to_return], uid
                return date, curr_stripping_activity_df[cols_to_return]

            except httpx.HTTPStatusError as e:
                self._logger.error(f"STRIPping - Bad Status: {response.status_code}")
                if uid:
                    return date, pd.DataFrame(columns=cols_to_return), uid
                return date, pd.DataFrame(columns=cols_to_return)

            except Exception as e:
                self._logger.error(f"STRIPping - Error: {str(e)}")
                if uid:
                    return date, pd.DataFrame(columns=cols_to_return), uid
                return date, pd.DataFrame(columns=cols_to_return)

        tasks = [
            fetch_mspd_table_5(client=client, date=date, uid=uid) for date in dates
        ]
        return tasks

    def fetch_historcal_trace_trade_history_by_cusip(
        self,
        cusips: List[str],
        start_date: datetime,
        end_date: datetime,
        xlsx_path: Optional[str] = None,
        session_timeout_minutes: Optional[int] = 5,
    ):
        total_t1 = time.time()

        async def build_fetch_tasks_historical_trace_data(
            session: aiohttp.ClientSession,
            cusips: List[str],
            start_date: datetime,
            end_date: datetime,
            uid: Optional[str | int] = None,
        ):
            finra_cookie_headers = {
                "authority": "services-dynarep.ddwa.finra.org",
                "method": "OPTIONS",
                "path": "/public/reporting/v2/data/group/FixedIncomeMarket/name/TreasuryTradeHistory",
                "scheme": "https",
                "accept": "*/*",
                "accept-encoding": "gzip, deflate, br, zstd",
                "accept-language": "en-US,en;q=0.9",
                "access-control-request-headers": "content-type,x-xsrf-token",
                "access-control-request-method": "POST",
                "cache-control": "no-cache",
                "origin": "https://www.finra.org",
                "pragma": "no-cache",
                "priority": "u=1, i",
                "referer": "https://www.finra.org/",
                "sec-fetch-dest": "empty",
                "sec-fetch-mode": "cors",
                "sec-fetch-site": "same-site",
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
            }

            finra_cookie_t1 = time.time()
            finra_cookie_url = "https://services-dynarep.ddwa.finra.org/public/reporting/v2/group/Firm/name/ActiveIndividual/dynamiclookup/examCode"
            finra_cookie_response = requests.get(
                finra_cookie_url, headers=finra_cookie_headers, proxies=self._proxies
            )
            if not finra_cookie_response.ok:
                raise ValueError(
                    f"TRACE - FINRA Cookies Request Bad Status: {finra_cookie_response.status_code}"
                )
            finra_cookie_str = dict(finra_cookie_response.headers)["set-cookie"]
            finra_cookie_dict = cookie_string_to_dict(cookie_string=finra_cookie_str)
            self._logger.info(
                f"TRACE - FINRA Cookie Fetch Took: {time.time() - finra_cookie_t1} seconds"
            )

            def build_finra_trade_history_headers(
                cookie_str: str, x_xsrf_token_str: str
            ):
                return {
                    "authority": "services-dynarep.ddwa.finra.org",
                    "method": "POST",
                    "path": "/public/reporting/v2/data/group/FixedIncomeMarket/name/TreasuryTradeHistory",
                    "scheme": "https",
                    "accept": "application/json, text/plain, */*",
                    "accept-encoding": "gzip, deflate, br, zstd",
                    "accept-language": "en-US,en;q=0.9",
                    "cache-control": "no-cache",
                    "content-type": "application/json",
                    "dnt": "1",
                    "origin": "https://www.finra.org",
                    "pragma": "no-cache",
                    "priority": "u=1, i",
                    "referer": "https://www.finra.org/",
                    "sec-ch-ua": '"Not)A;Brand";v="99", "Google Chrome";v="127", "Chromium";v="127"',
                    "sec-ch-ua-mobile": "?0",
                    "sec-ch-ua-platform": '"Windows"',
                    "sec-fetch-dest": "empty",
                    "sec-fetch-mode": "cors",
                    "sec-fetch-site": "same-site",
                    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
                    "x-xsrf-token": x_xsrf_token_str,
                    "cookie": cookie_str,
                }

            # maps size of trade history records between given start and end dates of said cusip
            def build_finra_trade_history_payload(
                cusip: str,
                start_date: datetime,
                end_date: datetime,
                limit: int,
                offset: int,
            ) -> Dict[str, int]:
                return {
                    "fields": [
                        "issueSymbolIdentifier",
                        "cusip",
                        "tradeDate",
                        "tradeTime",
                        "reportedTradeVolume",
                        "priceType",
                        "lastSalePrice",
                        "lastSaleYield",
                        "reportingSideCode",
                        "contraPartyTypeCode",
                    ],
                    "dateRangeFilters": [
                        {
                            "fieldName": "tradeDate",
                            "startDate": start_date.strftime("%Y-%m-%d"),
                            "endDate": end_date.strftime("%Y-%m-%d"),
                        },
                    ],
                    "compareFilters": [
                        {
                            "fieldName": "cusip",
                            "fieldValue": cusip,
                            "compareType": "EQUAL",
                        },
                    ],
                    "limit": limit,  # 5000 is Max Limit
                    "offset": offset,
                }

            def get_cusips_finra_pagination_configs(
                cusips: List[str], start_date: datetime, end_date: datetime
            ):
                async def fetch_finra_cusip_pagination_config(
                    config_session: aiohttp.ClientSession,
                    cusip: str,
                    start_date: datetime,
                    end_date: datetime,
                ):
                    try:
                        url = "https://services-dynarep.ddwa.finra.org/public/reporting/v2/data/group/FixedIncomeMarket/name/TreasuryTradeHistory"
                        config_response = await config_session.post(
                            url,
                            headers=build_finra_trade_history_headers(
                                cookie_str=finra_cookie_str,
                                x_xsrf_token_str=finra_cookie_dict["XSRF-TOKEN"],
                            ),
                            json=build_finra_trade_history_payload(
                                cusip=cusip,
                                start_date=start_date,
                                end_date=end_date,
                                limit=1,
                                offset=1,
                            ),
                            proxy=self._proxies["https"],
                        )
                        config_response.raise_for_status()
                        record_total_json = await config_response.json()
                        record_total_str = record_total_json["returnBody"]["headers"][
                            "Record-Total"
                        ][0]
                        return cusip, record_total_str
                    except aiohttp.ClientResponseError:
                        self._logger.error(
                            f"TRACE - CONFIGs Bad Status: {config_response.status}"
                        )
                        return cusip, -1

                    except Exception as e:
                        self._logger.error(f"TRACE - CONFIGs Error : {str(e)}")
                        return cusip, -1

                async def build_finra_config_tasks(
                    config_session: aiohttp.ClientSession,
                    cusips: List[str],
                    start_date: datetime,
                    end_date: datetime,
                ):
                    tasks = [
                        fetch_finra_cusip_pagination_config(
                            config_session=config_session,
                            cusip=cusip,
                            start_date=start_date,
                            end_date=end_date,
                        )
                        for cusip in cusips
                    ]
                    return await asyncio.gather(*tasks)

                async def run_fetch_all(
                    cusips: List[str], start_date: datetime, end_date: datetime
                ) -> List[pd.DataFrame]:
                    async with aiohttp.ClientSession(
                        proxy=self._proxies["https"]
                    ) as config_session:
                        all_data = await build_finra_config_tasks(
                            config_session=config_session,
                            cusips=cusips,
                            start_date=start_date,
                            end_date=end_date,
                        )
                        return all_data

                cusip_finra_api_payload_configs = dict(
                    asyncio.run(
                        run_fetch_all(
                            cusips=cusips, start_date=start_date, end_date=end_date
                        )
                    )
                )
                return cusip_finra_api_payload_configs

            cusip_finra_api_payload_configs_t1 = time.time()
            cusip_finra_api_payload_configs = get_cusips_finra_pagination_configs(
                cusips=cusips, start_date=start_date, end_date=end_date
            )
            self._logger.info(
                f"TRACE - FINRA CUSIP API Payload Configs Took: {time.time() - cusip_finra_api_payload_configs_t1} seconds"
            )
            self._logger.debug(
                f"TRACE - CUSIP API Payload Configs: {cusip_finra_api_payload_configs}"
            )

            async def fetch_finra_cusip_trade_history(
                session: aiohttp.ClientSession,
                cusip: str,
                start_date: datetime,
                end_date: datetime,
                offset: int,
            ):
                try:
                    url = "https://services-dynarep.ddwa.finra.org/public/reporting/v2/data/group/FixedIncomeMarket/name/TreasuryTradeHistory"
                    response = await session.post(
                        url,
                        headers=build_finra_trade_history_headers(
                            cookie_str=finra_cookie_str,
                            x_xsrf_token_str=finra_cookie_dict["XSRF-TOKEN"],
                        ),
                        json=build_finra_trade_history_payload(
                            cusip=cusip,
                            start_date=start_date,
                            end_date=end_date,
                            limit=5000,
                            offset=offset,
                        ),
                        proxy=self._proxies["https"],
                    )
                    response.raise_for_status()
                    trade_history_json = await response.json()
                    trade_data_json = json.loads(
                        trade_history_json["returnBody"]["data"]
                    )
                    df = pd.DataFrame(trade_data_json)
                    if uid:
                        return cusip, df, uid
                    return cusip, df
                except aiohttp.ClientResponseError:
                    self._logger.error(
                        f"TRACE - Trade History Bad Status: {response.status}"
                    )
                    if uid:
                        return cusip, None, uid
                    return cusip, None

                except Exception as e:
                    self._logger.error(f"TRACE - Trade History Error : {str(e)}")
                    if uid:
                        return cusip, None, uid
                    return cusip, None

            tasks = []
            for cusip in cusips:
                max_record_size = int(cusip_finra_api_payload_configs[cusip])
                if max_record_size == -1:
                    self._logger.debug(
                        f"TRACE - {cusip} had -1 Max Record Size - Does it Exist?"
                    )
                    continue
                num_reqs = math.ceil(max_record_size / 5000)
                self._logger.debug(f"TRACE - {cusip} Reqs: {num_reqs}")
                for i in range(1, num_reqs + 1):
                    curr_offset = i * 5000
                    if curr_offset > max_record_size:
                        break
                    tasks.append(
                        fetch_finra_cusip_trade_history(
                            session=session,
                            cusip=cusip,
                            start_date=start_date,
                            end_date=end_date,
                            offset=curr_offset,
                        )
                    )

            return tasks

        async def build_tasks(
            session: aiohttp.ClientSession,
            start_date: datetime,
            end_date: datetime,
            cusips: List[str],
        ):
            tasks = await build_fetch_tasks_historical_trace_data(
                session=session, cusips=cusips, start_date=start_date, end_date=end_date
            )
            return await asyncio.gather(*tasks)

        async def run_fetch_all(
            start_date: datetime, end_date: datetime, cusips: List[str]
        ):
            session_timeout = aiohttp.ClientTimeout(
                total=None,
                sock_connect=session_timeout_minutes * 60,
                sock_read=session_timeout_minutes * 60,
            )
            async with aiohttp.ClientSession(
                timeout=session_timeout, proxy=self._proxies["https"]
            ) as session:
                all_data = await build_tasks(
                    session=session,
                    cusips=cusips,
                    start_date=start_date,
                    end_date=end_date,
                )
                return all_data

        fetch_all_t1 = time.time()
        results: List[Tuple[str, pd.DataFrame]] = asyncio.run(
            run_fetch_all(start_date=start_date, end_date=end_date, cusips=cusips)
        )
        self._logger.info(
            f"TRACE - Fetch All Took: {time.time() - fetch_all_t1} seconds"
        )
        dfs_by_key = defaultdict(list)
        for key, df in results:
            if df is None:
                continue
            dfs_by_key[key].append(df)

        df_concatation_t1 = time.time()
        concatenated_dfs = {
            key: pd.concat(dfs)
            .sort_values(by=["tradeDate", "tradeTime"])
            .reset_index(drop=True)
            for key, dfs in dfs_by_key.items()
        }
        self._logger.info(
            f"TRACE - DF Concation Took: {time.time() - df_concatation_t1} seconds"
        )

        if xlsx_path:
            xlsx_write_t1 = time.time()
            with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
                for key, df in concatenated_dfs.items():
                    df.to_excel(writer, sheet_name=key, index=False)

            self._logger.info(
                f"TRACE - XLSX Write Took: {time.time() - xlsx_write_t1} seconds"
            )

        self._logger.info(
            f"TRACE - Total Time Elapsed: {time.time() - total_t1} seconds"
        )

        return concatenated_dfs

    def build_curve_set(
        self,
        as_of_date: datetime,
        calc_ytms: Optional[bool] = True,
        use_quantlib: Optional[bool] = False,  # default is rateslib
        include_auction_results: Optional[bool] = False,
        include_soma_holdings: Optional[bool] = False,
        include_stripping_activity: Optional[bool] = False,
        # include_outstanding_amt: Optional[bool] = False,
        # exclude_nonmarketable_outstanding_amt: Optional[bool] = False,
        # auctions_df: Optional[pd.DataFrame] = None,
        sorted: Optional[bool] = False,
        use_github: Optional[bool] = False,
        use_public_dotcom: Optional[bool] = False,
        include_off_the_run_number: Optional[bool] = False,
        market_cols_to_return: List[str] = None,
        calc_free_float: Optional[bool] = False,
        calc_mod_duration: Optional[bool] = False,
    ):
        if as_of_date.date() > datetime.today().date():
            print(
                f"crystal ball feature not implemented, yet - {as_of_date} is in the future"
            )
            return

        if use_github or use_public_dotcom:
            calc_ytms = False

        if market_cols_to_return:
            if "cusip" not in market_cols_to_return:
                market_cols_to_return.insert(0, "cusip")

        quote_type = (
            market_cols_to_return[1].split("_")[0] if market_cols_to_return else "eod"
        )
        filtered_free_float_df_col = False
        if calc_free_float:
            if not include_soma_holdings and not include_auction_results:
                filtered_free_float_df_col = True
                include_auction_results = True
                include_soma_holdings = True
                include_stripping_activity = True
                # include_outstanding_amt = True

        async def gather_tasks(client: httpx.AsyncClient, as_of_date: datetime):
            if use_github:
                ust_historical_prices_tasks = (
                    await self._build_fetch_tasks_historical_cusip_prices_github(
                        client=client,
                        dates=[as_of_date],
                        uid="ust_prices",
                        cols_to_return=(
                            [
                                "cusip",
                                "offer_price",
                                "offer_yield",
                                "bid_price",
                                "bid_yield",
                                "mid_price",
                                "mid_yield",
                                "eod_price",
                                "eod_yield",
                            ]
                            if not market_cols_to_return
                            else market_cols_to_return
                        ),
                    )
                )
            elif use_public_dotcom:
                cusips_to_fetch = get_active_cusips(
                    historical_auctions_df=self._historical_auctions_df,
                    as_of_date=as_of_date,
                    use_issue_date=True,
                )["cusip"].to_list()
                ust_historical_prices_tasks = (
                    await self._build_fetch_tasks_cusip_timeseries_public_dotcome(
                        client=client,
                        cusips=cusips_to_fetch,
                        start_date=as_of_date,
                        end_date=as_of_date,
                        uid="ust_prices_public_dotcom",
                    )
                )
            else:
                ust_historical_prices_tasks = (
                    await self._build_fetch_tasks_historical_cusip_prices(
                        client=client, dates=[as_of_date], uid="ust_prices"
                    )
                )

            tasks = ust_historical_prices_tasks

            if include_soma_holdings:
                tasks += await self._build_fetch_tasks_historical_soma_holdings(
                    client=client, dates=[as_of_date], uid="soma_holdings"
                )
            if include_stripping_activity:
                tasks += await self._build_fetch_tasks_historical_stripping_activity(
                    client=client, dates=[as_of_date], uid="ust_stripping"
                )
            # if include_outstanding_amt:
            #     tasks += await self._build_fetch_tasks_historical_amount_outstanding(
            #         client=client,
            #         dates=[as_of_date],
            #         uid="ust_outstanding_amt",
            #         only_marketable=exclude_nonmarketable_outstanding_amt,
            #     )

            return await asyncio.gather(*tasks)

        async def run_fetch_all(as_of_date: datetime):
            limits = httpx.Limits(max_connections=10)
            async with httpx.AsyncClient(
                limits=limits,
            ) as client:
                all_data = await gather_tasks(client=client, as_of_date=as_of_date)
                return all_data

        results = asyncio.run(run_fetch_all(as_of_date=as_of_date))
        auctions_dfs = []
        dfs = []
        public_dotcom_dicts: List[Dict[str, str]] = []
        for tup in results:
            uid = tup[-1]
            if uid == "ust_auctions":
                auctions_dfs.append(tup[0])
            elif uid == "ust_prices_public_dotcom" and use_public_dotcom:
                if not isinstance(tup[1], pd.DataFrame):
                    continue
                if not tup[1].empty:
                    public_dotcom_dicts.append(
                        {
                            "cusip": tup[0],
                            f"{quote_type}_price": tup[1].iloc[-1]["Price"],
                            f"{quote_type}_ytm": tup[1].iloc[-1]["YTM"],
                        }
                    )
            elif (
                uid == "ust_prices"
                or uid == "soma_holdings"
                or uid == "ust_stripping"
                or uid == "ust_outstanding_amt"
            ):
                dfs.append(tup[1])
            else:
                self._logger.warning(f"CURVE SET - unknown UID, Current Tuple: {tup}")

        auctions_df = get_active_cusips(
            historical_auctions_df=self._historical_auctions_df,
            as_of_date=as_of_date,
            use_issue_date=True,
        )
        otr_cusips_df: pd.DataFrame = get_last_n_off_the_run_cusips(
            auctions_df=auctions_df,
            n=0,
            filtered=True,
            as_of_date=as_of_date,
            use_issue_date=self._use_ust_issue_date,
        )
        auctions_df["is_on_the_run"] = auctions_df["cusip"].isin(
            otr_cusips_df["cusip"].to_list()
        )
        auctions_df["label"] = auctions_df.apply(lambda row: ust_labeler(row), axis=1)
        auctions_df["time_to_maturity"] = (
            auctions_df["maturity_date"] - as_of_date
        ).dt.days / 365

        if not include_auction_results or filtered_free_float_df_col:
            auctions_df = auctions_df[
                [
                    "cusip",
                    "security_type",
                    "auction_date",
                    "issue_date",
                    "maturity_date",
                    "time_to_maturity",
                    "int_rate",
                    "high_investment_rate",
                    "is_on_the_run",
                    "label",
                    "security_term",
                    "original_security_term",
                    "corpus_cusip",
                ]
            ]
        if not use_public_dotcom:
            merged_df = reduce(
                lambda left, right: pd.merge(left, right, on="cusip", how="outer"), dfs
            )
            merged_df = pd.merge(left=auctions_df, right=merged_df, on="cusip", how="outer")
        else:
            merged_df = pd.merge(left=auctions_df, right=pd.DataFrame(public_dotcom_dicts), on="cusip", how="outer")
        
        merged_df = merged_df[merged_df["cusip"].apply(is_valid_ust_cusip)]
            
        if calc_free_float:
            merged_df["parValue"] = pd.to_numeric(
                merged_df["parValue"], errors="coerce"
            ).fillna(0)
            merged_df["portion_stripped_amt"] = (
                pd.to_numeric(
                    merged_df["portion_stripped_amt"], errors="coerce"
                ).fillna(0)
                * 1000
            )
            merged_df["outstanding_amt"] = (
                pd.to_numeric(merged_df["outstanding_amt"], errors="coerce").fillna(0)
                * 1000
            )
            merged_df["free_float"] = (
                merged_df["outstanding_amt"]
                - merged_df["parValue"]
                - merged_df["portion_stripped_amt"]
            ) / 1_000_000

        if calc_mod_duration:
            merged_df["mod_dur"] = merged_df.apply(
                lambda row: RL_BondPricer._bond_mod_duration(
                    issue_date=row["issue_date"],
                    maturity_date=row["maturity_date"],
                    as_of=as_of_date,
                    coupon=row["int_rate"] / 100,
                    ytm=(
                        row[
                            next(
                                (
                                    item
                                    for item in market_cols_to_return
                                    if "yield" in item
                                ),
                                None,
                            )
                        ]
                        if market_cols_to_return
                        else row["eod_yield"]
                    ),
                ),
                axis=1,
            )

        if not market_cols_to_return:
            if not use_github:
                merged_df["mid_price"] = (
                    merged_df["offer_price"] + merged_df["bid_price"]
                ) / 2
            else:
                merged_df["mid_yield"] = (
                    merged_df["offer_yield"] + merged_df["bid_yield"]
                ) / 2

        if calc_ytms:
            calculate_yields_partial = partial(
                calculate_yields, as_of_date=as_of_date, use_quantlib=use_quantlib
            )
            with mp.Pool(mp.cpu_count()) as pool:
                results = pool.map(
                    calculate_yields_partial, [row for _, row in merged_df.iterrows()]
                )
            offer_yields, bid_yields, eod_yields = zip(*results)
            merged_df["offer_yield"] = offer_yields
            merged_df["bid_yield"] = bid_yields
            merged_df["eod_yield"] = eod_yields
            merged_df["mid_yield"] = (
                merged_df["offer_yield"] + merged_df["bid_yield"]
            ) / 2

        merged_df = merged_df.replace("null", np.nan)
        merged_df = merged_df[merged_df["original_security_term"].notna()]
        if sorted:
            merged_df["sort_key"] = merged_df["original_security_term"].apply(
                ust_sorter
            )
            merged_df = (
                merged_df.sort_values(by=["sort_key", "time_to_maturity"])
                .drop(columns="sort_key")
                .reset_index(drop=True)
            )

        if include_off_the_run_number:
            merged_df["rank"] = (
                merged_df.groupby("original_security_term")["time_to_maturity"].rank(
                    ascending=False, method="first"
                )
                - 1
            )

        return merged_df

    def _fetch_public_dotcome_jwt(self) -> str:
        try:
            jwt_headers = {
                "authority": "prod-api.154310543964.hellopublic.com",
                "method": "GET",
                "path": "/static/anonymoususer/credentials.json",
                "scheme": "https",
                "accept": "*/*",
                "accept-encoding": "gzip, deflate, br, zstd",
                "accept-language": "en-US,en;q=0.9",
                "cache-control": "no-cache",
                "content-type": "application/json",
                "dnt": "1",
                "origin": "https://public.com",
                "pragma": "no-cache",
                "priority": "u=1, i",
                "sec-ch-ua": '"Chromium";v="128", "Not;A=Brand";v="24", "Google Chrome";v="128"',
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": '"Windows"',
                "sec-fetch-dest": "empty",
                "sec-fetch-mode": "cors",
                "sec-fetch-site": "cross-site",
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
                "x-app-version": "web-1.0.9",
            }
            jwt_url = "https://prod-api.154310543964.hellopublic.com/static/anonymoususer/credentials.json"
            jwt_res = requests.get(jwt_url, headers=jwt_headers)
            jwt_str = jwt_res.json()["jwt"]
            return jwt_str
        except Exception as e:
            self._logger.error(f"Public.com JWT Request Failed: {e}")
            return None

    async def _fetch_cusip_timeseries_public_dotcom(
        self,
        client: httpx.AsyncClient,
        cusip: str,
        jwt_str: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_retries: Optional[int] = 3,
        backoff_factor: Optional[int] = 1,
        uid: Optional[str | int] = None,
    ):
        cols_to_return = ["Date", "Price", "YTM"]  # YTW is same as YTM for cash USTs
        retries = 0
        try:
            if pd.isna(cusip) or not cusip:
                raise ValueError(f"Public.com - invalid CUSIP passed")

            while retries < max_retries:
                try:
                    span = "MAX"
                    data_headers = {
                        "authority": "prod-api.154310543964.hellopublic.com",
                        "method": "GET",
                        "path": f"/fixedincomegateway/v1/graph/data?cusip={cusip}&span={span}",
                        "scheme": "https",
                        "accept": "*/*",
                        "accept-encoding": "gzip, deflate, br, zstd",
                        "accept-language": "en-US,en;q=0.9",
                        "cache-control": "no-cache",
                        "content-type": "application/json",
                        "dnt": "1",
                        "origin": "https://public.com",
                        "pragma": "no-cache",
                        "priority": "u=1, i",
                        "sec-ch-ua": '"Chromium";v="128", "Not;A=Brand";v="24", "Google Chrome";v="128"',
                        "sec-ch-ua-mobile": "?0",
                        "sec-ch-ua-platform": '"Windows"',
                        "sec-fetch-dest": "empty",
                        "sec-fetch-mode": "cors",
                        "sec-fetch-site": "cross-site",
                        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
                        "x-app-version": "web-1.0.9",
                        "authorization": jwt_str,
                    }

                    data_url = f"https://prod-api.154310543964.hellopublic.com/fixedincomegateway/v1/graph/data?cusip={cusip}&span={span}"
                    response = await client.get(data_url, headers=data_headers)
                    response.raise_for_status()
                    df = pd.DataFrame(response.json()["data"])
                    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                    df["unitPrice"] = pd.to_numeric(df["unitPrice"]) * 100
                    df["yieldToWorst"] = pd.to_numeric(df["yieldToWorst"]) * 100
                    df.columns = cols_to_return
                    if start_date:
                        df = df[df["Date"].dt.date >= start_date.date()]
                    if end_date:
                        df = df[df["Date"].dt.date <= end_date.date()]
                    if uid:
                        return cusip, df, uid
                    return cusip, df

                except httpx.HTTPStatusError as e:
                    self._logger.error(
                        f"Public.com - Bad Status for {cusip}: {response.status_code}"
                    )
                    if (
                        response.status_code == 404 or response.status_code == 400
                    ):  # public.com endpoint doesnt throw a 404 specifically
                        if uid:
                            return cusip, pd.DataFrame(columns=cols_to_return), uid
                        return cusip, pd.DataFrame(columns=cols_to_return)

                    retries += 1
                    wait_time = backoff_factor * (2 ** (retries - 1))
                    self._logger.debug(
                        f"Public.com - Throttled for {cusip}. Waiting for {wait_time} seconds before retrying..."
                    )
                    await asyncio.sleep(wait_time)

                except Exception as e:
                    self._logger.error(f"Public.com - Error: {str(e)}")
                    retries += 1
                    wait_time = backoff_factor * (2 ** (retries - 1))
                    self._logger.debug(
                        f"Public.com - Throttled for {cusip}. Waiting for {wait_time} seconds before retrying..."
                    )
                    await asyncio.sleep(wait_time)

            raise ValueError(f"Public.com - Max retries exceeded for {cusip}")

        except Exception as e:
            self._logger.error(e)
            if uid:
                return cusip, pd.DataFrame(columns=cols_to_return), uid
            return cusip, pd.DataFrame(columns=cols_to_return)

    async def _fetch_cusip_timeseries_public_dotcome_with_semaphore(
        self, semaphore, *args, **kwargs
    ):
        async with semaphore:
            return await self._fetch_cusip_timeseries_public_dotcom(*args, **kwargs)

    async def _build_fetch_tasks_cusip_timeseries_public_dotcome(
        self,
        client: httpx.AsyncClient,
        cusips: List[str],
        start_date: datetime,
        end_date: datetime,
        uid: Optional[str | int] = None,
        max_concurrent_tasks: int = 64,
        refresh_jwt: Optional[bool] = False,
    ):
        if refresh_jwt or not self._public_dotcom_jwt:
            self._public_dotcom_jwt = self._fetch_public_dotcome_jwt()
            if not self._public_dotcom_jwt:
                raise ValueError("Public.com JWT Request Failed")

        semaphore = asyncio.Semaphore(max_concurrent_tasks)
        tasks = [
            self._fetch_cusip_timeseries_public_dotcome_with_semaphore(
                semaphore=semaphore,
                client=client,
                cusip=cusip,
                start_date=start_date,
                end_date=end_date,
                uid=uid,
                jwt_str=self._public_dotcom_jwt,
                max_retries=1,
            )
            for cusip in cusips
        ]
        return tasks

    def public_dotcom_timeseries_api(
        self,
        cusips: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        refresh_jwt: Optional[bool] = False,
        max_concurrent_tasks: int = 64,
    ):
        if refresh_jwt or not self._public_dotcom_jwt:
            self._public_dotcom_jwt = self._fetch_public_dotcome_jwt()
            if not self._public_dotcom_jwt:
                raise ValueError("Public.com JWT Request Failed")

        async def build_tasks(
            client: httpx.AsyncClient,
            cusips: List[str],
            start_date: datetime,
            end_date: datetime,
            jwt_str: str,
        ):
            semaphore = asyncio.Semaphore(max_concurrent_tasks)
            tasks = [
                self._fetch_cusip_timeseries_public_dotcome_with_semaphore(
                    semaphore=semaphore,
                    client=client,
                    cusip=cusip,
                    start_date=start_date,
                    end_date=end_date,
                    jwt_str=jwt_str,
                    max_retries=1,
                )
                for cusip in cusips
            ]
            return await asyncio.gather(*tasks)

        async def run_fetch_all(
            cusips: List[str], start_date: datetime, end_date: datetime, jwt_str: str
        ):
            async with httpx.AsyncClient(proxy=self._proxies["https"]) as client:
                all_data = await build_tasks(
                    client=client,
                    cusips=cusips,
                    start_date=start_date,
                    end_date=end_date,
                    jwt_str=jwt_str,
                )
                return all_data

        dfs: List[Tuple[str, pd.DataFrame]] = asyncio.run(
            run_fetch_all(
                cusips=cusips,
                start_date=start_date,
                end_date=end_date,
                jwt_str=self._public_dotcom_jwt,
            )
        )
        return dict(dfs)

    async def _fetch_cusip_timeseries_bondsupermart(
        self,
        client: httpx.AsyncClient,
        cusip: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_retries: Optional[int] = 3,
        backoff_factor: Optional[int] = 1,
        uid: Optional[str | int] = None,
    ):
        isin = get_isin_from_cusip(cusip)
        headers = {
            "authority": "www.bondsupermart.com",
            "method": "GET",
            "path": f"/main/ws/v3/bond-info/bond-factsheet-chart/US{isin}",
            "scheme": "https",
            "accept": "application/json, text/plain, */*",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "en-US,en;q=0.9",
            "cache-control": "no-cache",
            "dnt": "1",
            "pragma": "no-cache",
            "priority": "u=1, i",
            "referer": f"https://www.bondsupermart.com/bsm/bond-factsheet/US{isin}",
            "sec-ch-ua": '"Chromium";v="128", "Not;A=Brand";v="24", "Google Chrome";v="128"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        }
        cols_to_return = ["Date", "bid_price", "bid_yield", "ask_price", "ask_yield"]
        retries = 0
        try:
            while retries < max_retries:
                try:
                    url = f"https://www.bondsupermart.com/main/ws/v3/bond-info/bond-factsheet-chart/{isin}"
                    response = await client.get(url, headers=headers)
                    response.raise_for_status()
                    response_json = response.json()

                    if (
                        not response_json["yieldChartMap"]
                        or not response_json["priceChartMap"]
                    ):
                        raise ValueError("Data is None")

                    bid_yield_df = pd.DataFrame(
                        [
                            {"Date": ts_yield_list[0], "bid_yield": ts_yield_list[1]}
                            for ts_yield_list in response_json["yieldChartMap"][
                                "SINCE_INCEPTION"
                            ][0]["data"]
                        ]
                    )
                    ask_yield_df = pd.DataFrame(
                        [
                            {"Date": ts_yield_list[0], "ask_yield": ts_yield_list[1]}
                            for ts_yield_list in response_json["yieldChartMap"][
                                "SINCE_INCEPTION"
                            ][1]["data"]
                        ]
                    )
                    bid_price_df = pd.DataFrame(
                        [
                            {"Date": ts_price_list[0], "bid_price": ts_price_list[1]}
                            for ts_price_list in response_json["priceChartMap"][
                                "SINCE_INCEPTION"
                            ][0]["data"]
                        ]
                    )
                    ask_price_df = pd.DataFrame(
                        [
                            {"Date": ts_price_list[0], "ask_price": ts_price_list[1]}
                            for ts_price_list in response_json["priceChartMap"][
                                "SINCE_INCEPTION"
                            ][1]["data"]
                        ]
                    )

                    merged_df = reduce(
                        lambda left, right: pd.merge(
                            left, right, on="Date", how="outer"
                        ),
                        [bid_price_df, bid_yield_df, ask_price_df, ask_yield_df],
                    )
                    merged_df["Date"] = pd.to_datetime(
                        merged_df["Date"], unit="ms", errors="coerce"
                    )
                    merged_df["bid_yield"] = pd.to_numeric(
                        merged_df["bid_yield"], errors="coerce"
                    )
                    merged_df["ask_yield"] = pd.to_numeric(
                        merged_df["ask_yield"], errors="coerce"
                    )
                    merged_df["mid_yield"] = (
                        merged_df["bid_yield"] + merged_df["ask_yield"]
                    ) / 2

                    merged_df["bid_price"] = pd.to_numeric(
                        merged_df["bid_price"], errors="coerce"
                    )
                    merged_df["ask_price"] = pd.to_numeric(
                        merged_df["ask_price"], errors="coerce"
                    )
                    merged_df["mid_price"] = (
                        merged_df["bid_price"] + merged_df["ask_price"]
                    ) / 2

                    if start_date:
                        merged_df = merged_df[
                            merged_df["Date"].dt.date >= start_date.date()
                        ]
                    if end_date:
                        merged_df = merged_df[
                            merged_df["Date"].dt.date <= end_date.date()
                        ]

                    if uid:
                        return cusip, merged_df, uid
                    return cusip, merged_df

                except httpx.HTTPStatusError as e:
                    self._logger.error(
                        f"Bondsupermart.com - Bad Status: {response.status_code}"
                    )
                    if response.status_code == 404:
                        if uid:
                            return cusip, pd.DataFrame(columns=cols_to_return), uid
                        return cusip, pd.DataFrame(columns=cols_to_return)

                    retries += 1
                    wait_time = backoff_factor * (2 ** (retries - 1))
                    self._logger.debug(
                        f"UST Timeseries GitHub - Throttled for {cusip}. Waiting for {wait_time} seconds before retrying..."
                    )
                    await asyncio.sleep(wait_time)

                except Exception as e:
                    self._logger.error(f"Bondsupermart.com - Error: {str(e)}")
                    retries += 1
                    wait_time = backoff_factor * (2 ** (retries - 1))
                    self._logger.debug(
                        f"Bondsupermart.com - Throttled for {cusip}. Waiting for {wait_time} seconds before retrying..."
                    )
                    await asyncio.sleep(wait_time)

            raise ValueError(f"Bondsupermart  - Max retries exceeded for {cusip}")

        except Exception as e:
            self._logger.error(e)
            if uid:
                return cusip, pd.DataFrame(columns=cols_to_return), uid
            return cusip, pd.DataFrame(columns=cols_to_return)

    async def _fetch_cusip_timeseries_bondsupermart_with_semaphore(
        self, semaphore, *args, **kwargs
    ):
        async with semaphore:
            return await self._fetch_cusip_timeseries_bondsupermart(*args, **kwargs)

    def bondsupermart_timeseries_api(
        self,
        cusips: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_concurrent_tasks: int = 64,
    ):
        async def build_tasks(
            client: httpx.AsyncClient,
            cusips: List[str],
            start_date: datetime,
            end_date: datetime,
        ):
            semaphore = asyncio.Semaphore(max_concurrent_tasks)
            tasks = [
                self._fetch_cusip_timeseries_bondsupermart_with_semaphore(
                    semaphore=semaphore,
                    client=client,
                    cusip=cusip,
                    start_date=start_date,
                    end_date=end_date,
                )
                for cusip in cusips
            ]
            return await asyncio.gather(*tasks)

        async def run_fetch_all(
            cusips: List[str],
            start_date: datetime,
            end_date: datetime,
        ):
            async with httpx.AsyncClient(proxy=self._proxies["https"]) as client:
                all_data = await build_tasks(
                    client=client,
                    cusips=cusips,
                    start_date=start_date,
                    end_date=end_date,
                )
                return all_data

        dfs: List[Tuple[str, pd.DataFrame]] = asyncio.run(
            run_fetch_all(
                cusips=cusips,
                start_date=start_date,
                end_date=end_date,
            )
        )
        return dict(dfs)

    # async def _fetch_cusip_timeseries_tradingview(
    #     self,
    #     client: httpx.AsyncClient,
    #     cusip: str,
    #     start_date: Optional[datetime] = None,
    #     end_date: Optional[datetime] = None,
    #     max_retries: Optional[int] = 3,
    #     backoff_factor: Optional[int] = 1,
    #     uid: Optional[str | int] = None,
    #     exchange: Literal["FWB", "BER", "DUS", "MUN", "EUROTLX"] = "FWB",
    # ):
    #     if exchange == "EUROTLX":
    #         cusip = get_isin_from_cusip(cusip_str=cusip)
