import asyncio
import re
import warnings
from datetime import datetime
from typing import Callable, Dict, List, Literal, Optional, Tuple

import httpx
import numpy as np
import pandas as pd
import requests
import scipy
import scipy.interpolate

from CurvyCUSIPs.DataFetcher.base import DataFetcherBase

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

import sys

if sys.platform == "win32":
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)


class BBGSEF_DataFetcher(DataFetcherBase):
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
        self.bbg_sef_url = "https://data.bloombergsef.com/bas/blotdatasvc"
        self.bbg_sef_headers = {
            "Accept": "text/plain, */*; q=0.01",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "application/json",
            "Cookie": "session=eyJmbGFzaCI6e319; session.sig=H0ndcv_UJa_BjmoKTVIBrrCbVpQ",
            "DNT": "1",
            "Host": "data.bloombergsef.com",
            "Origin": "https://data.bloombergsef.com",
            "Pragma": "no-cache",
            "Referer": "https://data.bloombergsef.com/",
            "Sec-CH-UA": '"Chromium";v="130", "Google Chrome";v="130", "Not?A_Brand";v="99"',
            "Sec-CH-UA-Mobile": "?0",
            "Sec-CH-UA-Platform": '"Windows"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
            "X-Requested-With": "XMLHttpRequest",
        }

    def _create_bbg_sef_payload(
        self,
        start_date: datetime,
        end_date: datetime,
        swap_type: Optional[Literal["getBsefEodIrsDataRequest", "getBsefEodFxDataRequest", "getBsefEodCdsDataRequest"]] = "getBsefEodIrsDataRequest",
    ):
        return {
            "Request": {
                swap_type: {
                    "tradeDays": {"startDay": start_date.strftime("%Y-%m-%dT%H:%M:%S.%fZ"), "endDay": end_date.strftime("%Y-%m-%dT%H:%M:%S.%fZ")}
                }
            }
        }

    async def _fetch_bbg_sef_data(
        self,
        client: httpx.AsyncClient,
        start_date: datetime,
        end_date: datetime,
        swap_type: Optional[Literal["getBsefEodIrsDataRequest", "getBsefEodFxDataRequest", "getBsefEodCdsDataRequest"]] = "getBsefEodIrsDataRequest",
        benchmark_rate_filter: Optional[str] = None,
        uid: Optional[int | str] = None,
        max_retries: Optional[int] = 3,
        backoff_factor: Optional[int] = 1,
    ):
        retries = 0
        try:
            while retries < max_retries:
                try:
                    response = await client.post(
                        self.bbg_sef_url,
                        json=self._create_bbg_sef_payload(start_date=start_date, end_date=end_date, swap_type=swap_type),
                        headers=self.bbg_sef_headers,
                        follow_redirects=False,
                        timeout=self._global_timeout,
                    )
                    response.raise_for_status()
                    json_data = response.json()
                    df = pd.DataFrame(json_data["response"]["getBsefEodIrsDataResponse"]["BsefEodData"])

                    if benchmark_rate_filter:
                        df = df[df["security"].str.contains(benchmark_rate_filter)]

                    if uid:
                        return df, uid
                    return df

                except httpx.HTTPStatusError as e:
                    self._logger.error(f"BBG SEF - Bad Status for {swap_type}-{start_date}-{end_date}: {response.status_code}")
                    if response.status_code == 404:
                        if uid:
                            return pd.DataFrame(), uid
                        return pd.DataFrame()

                    retries += 1
                    wait_time = backoff_factor * (2 ** (retries - 1))
                    self._logger.debug(f"BBG SEF - Throttled. Waiting for {wait_time} seconds before retrying...")
                    await asyncio.sleep(wait_time)

                except Exception as e:
                    self._logger.error(f"BBG SEF - Error for {swap_type}-{start_date}-{end_date}: {e}")
                    print(e)
                    retries += 1
                    wait_time = backoff_factor * (2 ** (retries - 1))
                    self._logger.debug(f"BBG SEF - Throttled. Waiting for {wait_time} seconds before retrying...")
                    await asyncio.sleep(wait_time)

            raise ValueError(f"BBG SEF - Max retries exceeded for {swap_type}-{start_date}-{end_date}")

        except Exception as e:
            self._logger.error(e)
            if uid:
                return pd.DataFrame(), uid
            return pd.DataFrame()

    async def _fetch_bbg_sef_data_semaphore(self, semaphore, *args, **kwargs):
        async with semaphore:
            return await self._fetch_bbg_sef_data(*args, **kwargs)

    def _get_business_days_groups(self, start_date: datetime, end_date: datetime, group_size=3):
        date_range = pd.date_range(start=start_date, end=end_date, freq="B")
        business_day_groups = [
            [bday.to_pydatetime() for bday in date_range[i : i + group_size].tolist()] for i in range(0, len(date_range), group_size)
        ]
        return business_day_groups

    def _swap_tenor_sort(self, column_name):
        parts = column_name.split()
        if len(parts) < 4:
            return (float("inf"), float("inf"))

        numeric_part = int(parts[-1][:-1])
        unit = parts[-1][-1]
        unit_order = {"D": 1, "W": 2, "M": 3, "Y": 4}
        return (unit_order.get(unit, float("inf")), numeric_part)

    def _liquid_tenors(self):
        return [
            "USD SWAP VS SOFR 1M",
            "USD SWAP VS SOFR 2M",
            "USD SWAP VS SOFR 3M",
            "USD SWAP VS SOFR 4M",
            "USD SWAP VS SOFR 6M",
            "USD SWAP VS SOFR 9M",
            "USD SWAP VS SOFR 1Y",
            "USD SWAP VS SOFR 18M",
            "USD SWAP VS SOFR 2Y",
            "USD SWAP VS SOFR 3Y",
            "USD SWAP VS SOFR 4Y",
            "USD SWAP VS SOFR 5Y",
            "USD SWAP VS SOFR 6Y",
            "USD SWAP VS SOFR 7Y",
            "USD SWAP VS SOFR 8Y",
            "USD SWAP VS SOFR 9Y",
            "USD SWAP VS SOFR 10Y",
            "USD SWAP VS SOFR 11Y",
            "USD SWAP VS SOFR 12Y",
            # "USD SWAP VS SOFR 13Y",
            # "USD SWAP VS SOFR 14Y",
            "USD SWAP VS SOFR 15Y",
            "USD SWAP VS SOFR 20Y",
            "USD SWAP VS SOFR 25Y",
            "USD SWAP VS SOFR 30Y",
            "USD SWAP VS SOFR 35Y",
            "USD SWAP VS SOFR 40Y",
            "USD SWAP VS SOFR 45Y",
            "USD SWAP VS SOFR 50Y",
        ]

    def _parse_usd_sofr_swap_tenor(self, tenor_str):
        match = re.match(r"USD SWAP VS SOFR (\d+)([MY])", tenor_str)
        if match:
            num, unit = match.groups()
            num = int(num)
            if unit == "M":
                return num / 12
            elif unit == "Y":
                return num
        return np.nan

    def _interpolate_row(self, row, times, interp_func):
        valid_cols = [col for col in row.index if pd.notna(row[col]) and pd.notna(times[col])]
        x = np.array([times[col] for col in valid_cols])
        y = np.array([row[col] for col in valid_cols])

        interp_cols = [col for col in row.index if pd.notna(times[col])]
        x_new = np.array([times[col] for col in interp_cols])

        if len(x) > 1:
            interpolated_rates = interp_func(x, y, x_new)
        else:
            interpolated_rates = np.full_like(x_new, np.nan)

        interpolated_series = pd.Series(interpolated_rates, index=interp_cols)
        for col in row.index:
            if pd.isna(times[col]):
                interpolated_series[col] = np.nan

        interpolated_series = interpolated_series.reindex(row.index)
        return interpolated_series

    def _interpolate_and_extrapolate_missing_rates(self, df, interp_func):
        times = {col: self._parse_usd_sofr_swap_tenor(col) for col in df.columns if col != "Date"}
        df_interpolated = df.apply(lambda row: self._interpolate_row(row, times, interp_func), axis=1)
        return df_interpolated

    def get_usd_sofr_swaps_historical_term_structures(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_concurrent_tasks: int = 64,
        max_keepalive_connections: int = 5,
        ohlc: Optional[Literal["open", "high", "low", "close"]] = None,
        tenors: Optional[List[str]] = None,
        liquid_tenors: Optional[bool] = False,
        interp_extrap_strat: Optional[Callable] = None,
    ) -> pd.DataFrame:
        groups = self._get_business_days_groups(start_date=start_date, end_date=end_date, group_size=30)
        start_end_date_tuples = [(min(group), max(group)) for group in groups]

        async def build_tasks(
            client: httpx.AsyncClient,
            start_end_date_tuples: List[Tuple[datetime, datetime]],
        ):
            tasks = []
            semaphore = asyncio.Semaphore(max_concurrent_tasks)
            for start_date, end_date in start_end_date_tuples:
                task = self._fetch_bbg_sef_data_semaphore(
                    semaphore=semaphore,
                    client=client,
                    start_date=start_date,
                    end_date=end_date,
                    swap_type="getBsefEodIrsDataRequest",
                    benchmark_rate_filter="USD SWAP VS SOFR",
                )
                tasks.append(task)

            return await asyncio.gather(*tasks)

        async def run_fetch_all(start_end_date_tuples: List[Tuple[datetime, datetime]]):
            limits = httpx.Limits(
                max_connections=max_concurrent_tasks,
                max_keepalive_connections=max_keepalive_connections,
            )
            async with httpx.AsyncClient(limits=limits) as client:
                all_data = await build_tasks(client=client, start_end_date_tuples=start_end_date_tuples)
                return all_data

        results: List[Tuple[str, pd.DataFrame]] = asyncio.run(run_fetch_all(start_end_date_tuples=start_end_date_tuples))
        big_df: pd.DataFrame = pd.concat(results)[["tradeDate", "security", f"price{ohlc.title()}" if ohlc else "settlementPrice"]]
        big_df = big_df[big_df["security"].str.len() <= 22]
        big_df["tradeDate"] = pd.to_datetime(big_df["tradeDate"], errors="coerce").dt.tz_localize(None)
        big_df.rename(columns={"tradeDate": "Date"}, inplace=True)
        big_df = big_df.pivot(index="Date", columns="security", values="settlementPrice")
        big_df = big_df[sorted(big_df.columns, key=self._swap_tenor_sort)]
        big_df.reset_index(inplace=True)

        if tenors:
            big_df = big_df[["Date"] + tenors]
        elif liquid_tenors:
            for tenor in self._liquid_tenors():
                if tenor not in big_df.columns:
                    big_df[tenor] = np.nan
            big_df = big_df[["Date"] + self._liquid_tenors()]

        if interp_extrap_strat is not None:
            date_col = big_df["Date"]
            big_df = self._interpolate_and_extrapolate_missing_rates(big_df[big_df.columns[1:]], interp_extrap_strat)
            big_df.insert(0, "Date", date_col)

        big_df.index.name = None
        big_df.columns.name = None

        return big_df

    def calc_sofr_swap_spreads(self, ct_yields_df: pd.DataFrame, sofr_swaps_df: pd.DataFrame) -> pd.DataFrame:
        common_dates = ct_yields_df.index.intersection(sofr_swaps_df.index)
        ct_yields_df = ct_yields_df.loc[common_dates]
        sofr_swaps_df = sofr_swaps_df.loc[common_dates]

        swaps_cols = [
            "USD SWAP VS SOFR 1M",
            "USD SWAP VS SOFR 2M",
            "USD SWAP VS SOFR 3M",
            "USD SWAP VS SOFR 4M",
            "USD SWAP VS SOFR 6M",
            "USD SWAP VS SOFR 1Y",
            "USD SWAP VS SOFR 2Y",
            "USD SWAP VS SOFR 3Y",
            "USD SWAP VS SOFR 5Y",
            "USD SWAP VS SOFR 7Y",
            "USD SWAP VS SOFR 10Y",
            "USD SWAP VS SOFR 20Y",
            "USD SWAP VS SOFR 30Y",
        ]
        ct_cols = ["CT1M", "CT2M", "CT3M", "CT4M", "CT6M", "CT1", "CT2", "CT3", "CT5", "CT7", "CT10", "CT20", "CT30"]

        spreads_df = sofr_swaps_df[swaps_cols].values - ct_yields_df[ct_cols].values
        spreads_df = pd.DataFrame(spreads_df, index=common_dates, columns=[f"{swp} Spread" for swp in swaps_cols])
        return spreads_df

    def get_usd_libor_swaps_historical_term_structures(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_concurrent_tasks: int = 64,
        max_keepalive_connections: int = 5,
        ohlc: Optional[Literal["open", "high", "low", "close"]] = None,
        tenors: Optional[List[str]] = None,
        liquid_tenors: Optional[bool] = False,
        interp_extrap_strat: Optional[Callable] = None,
    ):
        pass
