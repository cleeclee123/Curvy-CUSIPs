import asyncio
import warnings
from datetime import datetime
from typing import Dict, List, Optional

import httpx
import pandas as pd
import requests

from CurvyCUSIPs.DataFetcher.base import DataFetcherBase
from CurvyCUSIPs.utils.ust_utils import build_treasurydirect_header

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

import sys

if sys.platform == "win32":
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)


class NYFRBDataFetcher(DataFetcherBase):
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

    async def _fetch_single_soma_holding_day(
        self,
        client: httpx.AsyncClient,
        date: datetime,
        valid_soma_dates_from_input: Dict[datetime, datetime],
        uid: Optional[str | int] = None,
        minimize_api_calls=False,
        max_retries: Optional[int] = 3,
        backoff_factor: Optional[int] = 1,
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
        retries = 0
        try:
            while retries < max_retries:
                try:
                    if minimize_api_calls:
                        date_str = date.strftime("%Y-%m-%d")
                    else:
                        date_str = valid_soma_dates_from_input[date].strftime("%Y-%m-%d")

                    url = f"https://markets.newyorkfed.org/api/soma/tsy/get/asof/{date_str}.json"
                    response = await client.get(
                        url,
                        headers=build_treasurydirect_header(host_str="markets.newyorkfed.org"),
                    )
                    response.raise_for_status()
                    curr_soma_holdings_json = response.json()
                    curr_soma_holdings_df = pd.DataFrame(curr_soma_holdings_json["soma"]["holdings"])
                    curr_soma_holdings_df = curr_soma_holdings_df.fillna("")
                    curr_soma_holdings_df["asOfDate"] = pd.to_datetime(curr_soma_holdings_df["asOfDate"], errors="coerce")
                    curr_soma_holdings_df["parValue"] = pd.to_numeric(curr_soma_holdings_df["parValue"], errors="coerce")
                    curr_soma_holdings_df["percentOutstanding"] = pd.to_numeric(curr_soma_holdings_df["percentOutstanding"], errors="coerce")
                    curr_soma_holdings_df["est_outstanding_amt"] = curr_soma_holdings_df["parValue"] / curr_soma_holdings_df["percentOutstanding"]
                    curr_soma_holdings_df = curr_soma_holdings_df[
                        (curr_soma_holdings_df["securityType"] != "TIPS") & (curr_soma_holdings_df["securityType"] != "FRNs")
                    ]
                    if uid:
                        return date, curr_soma_holdings_df[cols_to_return], uid

                    return date, curr_soma_holdings_df[cols_to_return]

                except httpx.HTTPStatusError as e:
                    self._logger.error(f"SOMA Holding - Bad Status: {response.status_code}")
                    if response.status_code == 404:
                        if uid:
                            return date, pd.DataFrame(columns=cols_to_return), uid
                        return date, pd.DataFrame(columns=cols_to_return)

                    retries += 1
                    wait_time = backoff_factor * (2 ** (retries - 1))
                    self._logger.debug(f"SOMA Holding - Throttled. Waiting for {wait_time} seconds before retrying...")
                    await asyncio.sleep(wait_time)

                except Exception as e:
                    self._logger.error(f"SOMA Holding - Error: {str(e)}")
                    retries += 1
                    wait_time = backoff_factor * (2 ** (retries - 1))
                    self._logger.debug(f"SOMA Holding - Throttled. Waiting for {wait_time} seconds before retrying...")
                    await asyncio.sleep(wait_time)

            raise ValueError(f"UST STRIPPING Activity - Max retries exceeded for {date}")

        except Exception as e:
            self._logger.error(e)
            if uid:
                return date, pd.DataFrame(columns=cols_to_return), uid
            return date, pd.DataFrame(columns=cols_to_return)

    async def _fetch_single_soma_holding_day_with_semaphore(self, semaphore, *args, **kwargs):
        async with semaphore:
            return await self._fetch_single_soma_holding_day(*args, **kwargs)

    async def _build_fetch_tasks_historical_soma_holdings(
        self,
        client: httpx.AsyncClient,
        dates: List[datetime],
        minimize_api_calls: Optional[bool] = False,
        uid: Optional[str | int] = None,
        max_retries: Optional[int] = 3,
        backoff_factor: Optional[int] = 1,
        max_concurrent_tasks: int = 64,
        my_semaphore: Optional[asyncio.Semaphore] = None,
    ):
        valid_soma_holding_dates_reponse = requests.get(
            "https://markets.newyorkfed.org/api/soma/asofdates/list.json",
            headers=build_treasurydirect_header(host_str="markets.newyorkfed.org"),
            proxies=self._proxies,
        )
        if valid_soma_holding_dates_reponse.ok:
            valid_soma_holding_dates_json = valid_soma_holding_dates_reponse.json()
            valid_soma_dates_dt = [datetime.strptime(dt_string, "%Y-%m-%d") for dt_string in valid_soma_holding_dates_json["soma"]["asOfDates"]]
        else:
            raise ValueError(f"SOMA Holdings - Status Code: {valid_soma_holding_dates_reponse.status_code}")

        valid_soma_dates_from_input = {}
        for dt in dates:
            valid_closest_date = min(
                (valid_date for valid_date in valid_soma_dates_dt if valid_date <= dt),
                key=lambda valid_date: abs(dt - valid_date),
            )
            valid_soma_dates_from_input[dt] = valid_closest_date
        self._logger.debug(f"SOMA Holdings - Valid SOMA Holding Dates: {valid_soma_dates_from_input}")

        semaphore = my_semaphore or asyncio.Semaphore(max_concurrent_tasks)
        if minimize_api_calls:
            tasks = [
                self._fetch_single_soma_holding_day_with_semaphore(
                    semaphore=semaphore,
                    client=client,
                    date=date,
                    valid_soma_dates_from_input=valid_soma_dates_from_input,
                    uid=uid,
                    minimize_api_calls=True,
                    max_retries=max_retries,
                    backoff_factor=backoff_factor,
                )
                for date in list(set(valid_soma_dates_from_input.values()))
            ]
            return tasks
        else:
            tasks = [
                self._fetch_single_soma_holding_day_with_semaphore(
                    semaphore=semaphore,
                    client=client,
                    date=date,
                    valid_soma_dates_from_input=valid_soma_dates_from_input,
                    uid=uid,
                    max_retries=max_retries,
                    backoff_factor=backoff_factor,
                )
                for date in dates
            ]
            return tasks
