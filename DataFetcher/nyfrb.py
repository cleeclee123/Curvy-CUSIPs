import asyncio
import warnings
from datetime import datetime
from typing import Dict, List, Optional 

import httpx
import pandas as pd
import requests

from DataFetcher.base import DataFetcherBase

from utils.utils import build_treasurydirect_header

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
