import asyncio
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import httpx
import pandas as pd
import requests

from DataFetcher.base import DataFetcherBase

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

import sys

if sys.platform == "win32":
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)


class FedInvestDataFetcher(DataFetcherBase):
    def __init__(
        self,
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

    # scrapes prices directly from FedInvest
    async def _fetch_cusip_prices_fedinvest(
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

    async def _fetch_cusip_prices_fedinvest_with_semaphore(
        self, semaphore, *args, **kwargs
    ):
        async with semaphore:
            return await self._fetch_cusip_prices_fedinvest(*args, **kwargs)

    async def _build_fetch_tasks_cusip_prices_fedinvest(
        self,
        client: httpx.AsyncClient,
        dates: List[datetime],
        cusips: Optional[List[str]] = None,
        uid: Optional[str | int] = None,
        max_concurrent_tasks: int = 64,
    ):
        semaphore = asyncio.Semaphore(max_concurrent_tasks)
        tasks = [
            self._fetch_cusip_prices_fedinvest_with_semaphore(
                semaphore,
                client=client,
                date=date,
                cusips=cusips,
                uid=uid,
            )
            for date in dates
        ]
        return tasks

    # we store FedInvest prices in github b/c it speeds up fetching by a ton!
    def _github_headers(self, path: str):
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

    async def _fetch_cusip_prices_from_github(
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
        headers = self._github_headers(
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

    async def _fetch_cusip_prices_from_github_with_semaphore(
        self, semaphore, *args, **kwargs
    ):
        async with semaphore:
            return await self._fetch_cusip_prices_from_github(*args, **kwargs)

    async def _build_fetch_tasks_cusip_prices_from_github(
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
            self._fetch_cusip_prices_from_github_with_semaphore(
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

    # ct yields are yields of OTR USTs at respective tenors timeseres
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
                headers=self._github_headers(
                    path=f"/cleeclee123/CUSIP-Timeseries/main/historical_ct_yields_{side}_side.json"
                ),
                proxies=self._proxies,
            )
            res.raise_for_status()
            df = pd.DataFrame(res.json())
            df["Date"] = pd.to_datetime(df["Date"])
            df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
            df = df.reset_index(drop=True)
            df.columns = ["Date", "CT2M", "CT3M", "CT6M", "CT1", "CT2", "CT3", "CT5", "CT7", "CT10", "CT20", "CT30"]
            
            if tenors:
                tenors = ["Date"] + tenors
                return df[tenors]
            return df
        except Exception as e:
            self._logger.error(f"Historical CT Yields GitHub - {str(e)}")

    async def _fetch_historical_ct_yields(
        self,
        client: httpx.AsyncClient,
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
            res = await client.get(
                url,
                headers=self._github_headers(
                    path=f"/cleeclee123/CUSIP-Timeseries/main/historical_ct_yields_{side}_side.json"
                ),
            )
            res.raise_for_status()
            df = pd.DataFrame(res.json())
            df["Date"] = pd.to_datetime(df["Date"])
            df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
            df = df.reset_index(drop=True)
            df.columns = ["Date", "CT2M", "CT3M", "CT6M", "CT1", "CT2", "CT3", "CT5", "CT7", "CT10", "CT20", "CT30"]

            if tenors:
                tenors = ["Date"] + tenors
                return df[tenors]
            return df
        except httpx.HTTPStatusError as e:
            self._logger.error(f"HTTP Error: {str(e)}")
        except Exception as e:
            self._logger.error(f"Historical CT Yields GitHub - {str(e)}")

    # We store cusip timeseries data on github
    async def _fetch_cusip_timeseries_github(
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
                headers = self._github_headers(
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

    async def _fetch_cusip_timeseries_github_with_semaphore(
        self, semaphore, *args, **kwargs
    ):
        async with semaphore:
            return await self._fetch_cusip_timeseries_github(*args, **kwargs)

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
                task = self._fetch_cusip_timeseries_github_with_semaphore(
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
