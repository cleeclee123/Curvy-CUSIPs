import pandas as pd
import aiohttp
import asyncio
import requests
import math
import QuantLib as ql
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from itertools import product
import multiprocessing as mp
from datetime import datetime
from typing import TypeAlias, Optional, List, Dict

from utils import JSON, build_treasurydirect_header, get_active_cusips


class CUSIP_Curve:
    def __init__():
        pass

    async def _build_fetch_tasks_historical_treasury_auctions(
        self,
        session: aiohttp.ClientSession,
        assume_data_size=True,
        uid: Optional[str | int] = None,
    ) -> aiohttp.Coroutine[List[List[JSON]]]:
        MAX_TREASURY_GOV_API_CONTENT_SIZE = 10000
        NUM_REQS_NEEDED_TREASURY_GOV_API = 2

        def get_treasury_query_sizing() -> List[str]:
            base_url = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query?page[number]=1&page[size]=1"
            res = requests.get(base_url, headers=build_treasurydirect_header())
            if res.ok:
                meta = res.json()["meta"]
                size = meta["total-count"]
                number_requests = math.ceil(size / MAX_TREASURY_GOV_API_CONTENT_SIZE)
                return [
                    f"https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query?page[number]={i+1}&page[size]={MAX_TREASURY_GOV_API_CONTENT_SIZE}"
                    for i in range(0, number_requests)
                ]

        links = (
            get_treasury_query_sizing()
            if not assume_data_size
            else [
                f"https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query?page[number]={i+1}&page[size]={MAX_TREASURY_GOV_API_CONTENT_SIZE}"
                for i in range(0, NUM_REQS_NEEDED_TREASURY_GOV_API)
            ]
        )

        async def fetch(session: aiohttp.ClientSession, url):
            async with session.get(
                url, headers=build_treasurydirect_header()
            ) as response:
                json_data = await response.json()
                if uid:
                    return json_data["data"], uid
                return json_data["data"]

        tasks = [fetch(session, url) for url in links]
        return tasks

    async def _build_fetch_tasks_historical_cusip_prices(
        session: aiohttp.ClientSession,
        dates: List[datetime],
        cusips: Optional[List[str]] = None,
        uid: Optional[str | int] = None,
    ) -> aiohttp.Coroutine[Dict[str, str]]:
        url = "https://savingsbonds.gov/GA-FI/FedInvest/selectSecurityPriceDate"

        def build_date_payload(date: datetime):
            return {
                "priceDate.month": date.month,
                "priceDate.day": date.day,
                "priceDate.year": date.year,
                "submit": "Show Prices",
            }

        async def fetch_prices_from_treasury_date_search(
            session: aiohttp.ClientSession,
            date: datetime,
            cusips: Optional[List[str]] = None,
        ) -> Dict:
            payload = build_date_payload(date)
            try:
                response = await session.post(url, data=payload, follow_redirects=True)
                response.raise_for_status()
                tables = pd.read_html(response.content)
                df = tables[0]
                missing_cusips = [
                    cusip for cusip in cusips if cusip not in df["CUSIP"].values
                ]
                if missing_cusips:
                    print(
                        f"The following CUSIPs are not found in the DataFrame: {missing_cusips}"
                    )
                df = df[df["CUSIP"].isin(cusips)] if cusips else df
                if uid:
                    return date, df, uid
                return date, df
            except Exception as e:
                print(f"An error occurred: {e}")
                return date, pd.DataFrame()

        tasks = [
            fetch_prices_from_treasury_date_search(
                session=session, date=date, cusips=cusips
            )
            for date in dates
        ]
        return tasks
