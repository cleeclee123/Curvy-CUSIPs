import asyncio
import warnings
from datetime import datetime
from functools import reduce
from typing import Dict, List, Optional, Tuple

import httpx
import pandas as pd

from DataFetcher.base import DataFetcherBase

from utils.ust_utils import get_isin_from_cusip

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

import sys

if sys.platform == "win32":
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)


class BondSupermartDataFetcher(DataFetcherBase):
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
