import pandas as pd
import aiohttp
import httpx
import asyncio
import requests
import math
import QuantLib as ql
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from itertools import product
import multiprocessing as mp
import ujson as json
from datetime import datetime
from typing import TypeAlias, Optional, List, Dict

from utils import (
    JSON,
    build_treasurydirect_header,
    get_active_cusips,
    last_day_n_months_ago,
    cookie_string_to_dict,
)


class CUSIP_Curve:
    _verbose: bool = False

    def __init__(self, verbose: Optional[bool] = False):
        self._verbose = verbose

    async def _build_fetch_tasks_historical_treasury_auctions(
        self,
        client: httpx.AsyncClient,
        assume_data_size=True,
        uid: Optional[str | int] = None,
    ):
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

        async def fetch(client: httpx.AsyncClient, url):
            response = await client.get(url, headers=build_treasurydirect_header())
            json_data = response.json()
            if uid:
                return json_data["data"], uid
            return json_data["data"]

        tasks = [fetch(client, url) for url in links]
        return tasks

    async def _build_fetch_tasks_historical_cusip_prices(
        self,
        client: httpx.AsyncClient,
        dates: List[datetime],
        cusips: Optional[List[str]] = None,
        uid: Optional[str | int] = None,
    ):
        url = "https://savingsbonds.gov/GA-FI/FedInvest/selectSecurityPriceDate"

        def build_date_payload(date: datetime):
            return {
                "priceDate.month": date.month,
                "priceDate.day": date.day,
                "priceDate.year": date.year,
                "submit": "Show Prices",
            }

        async def fetch_prices_from_treasury_date_search(
            client: httpx.AsyncClient,
            date: datetime,
            cusips: List[str],
            uid: Optional[int | str],
        ):
            payload = build_date_payload(date)
            try:
                response = await client.post(url, data=payload, follow_redirects=True)
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
            except httpx.HTTPStatusError as e:
                print(f"HTTP error status: {e.response.status_code}")
                if uid:
                    return date, pd.DataFrame(), uid
                return date, pd.DataFrame()
            except Exception as e:
                print(f"An error occurred: {e}")
                if uid:
                    return date, pd.DataFrame(), uid
                return date, pd.DataFrame()

        tasks = [
            fetch_prices_from_treasury_date_search(
                client=client,
                date=date,
                cusips=cusips,
                uid=uid,
            )
            for date in dates
        ]
        return tasks

    async def _build_fetch_tasks_historical_soma_holdings(
        client: httpx.AsyncClient,
        dates: List[datetime],
        uid: Optional[str | int] = None,
    ):
        valid_soma_holding_dates_reponse = requests.get(
            "https://markets.newyorkfed.org/api/soma/asofdates/list.json"
        )
        if valid_soma_holding_dates_reponse.ok:
            valid_soma_holding_dates_json = valid_soma_holding_dates_reponse.json()
            valid_soma_dates_dt = [
                datetime.strptime(dt_string, "%Y-%m-%d")
                for dt_string in valid_soma_holding_dates_json["soma"]["asOfDates"]
            ]
        else:
            raise ValueError(
                f"valid_soma_holding_dates_reponse failed - Status Code: {valid_soma_holding_dates_reponse.status_code}"
            )

        valid_soma_dates_from_input = {}
        for dt in dates:
            valid_closest_date = min(
                (valid_date for valid_date in valid_soma_dates_dt if valid_date <= dt),
                key=lambda valid_date: abs(dt - valid_date),
            )
            valid_soma_dates_from_input[dt] = valid_closest_date

        async def fetch_single_soma_holding_day(
            client: httpx.AsyncClient,
            date: datetime,
            uid: Optional[str | int] = None,
        ):
            try:
                date_str = valid_soma_dates_from_input[date].strftime("%Y-%m-%d")
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
                curr_soma_holdings_df["coupon"] = pd.to_numeric(
                    curr_soma_holdings_df["coupon"], errors="coerce"
                )
                curr_soma_holdings_df["inflationCompensation"] = pd.to_numeric(
                    curr_soma_holdings_df["inflationCompensation"], errors="coerce"
                )

                if uid:
                    return date, curr_soma_holdings_df, uid
                return date, curr_soma_holdings_df

            except Exception as e:
                print(f"An error occurred: {e}")
                if uid:
                    return date, pd.DataFrame(), uid
                return date, pd.DataFrame()

        tasks = [
            fetch_single_soma_holding_day(client=client, date=date, uid=uid)
            for date in dates
        ]
        return tasks

    async def _build_fetch_tasks_historical_stripping_activity(
        client: httpx.AsyncClient,
        dates: List[datetime],
        uid: Optional[str | int] = None,
    ):
        async def fetch_mspd_table_5(
            client: httpx.AsyncClient,
            date: datetime,
            uid: Optional[str | int] = None,
        ) -> Dict:
            try:
                last_n_months_last_business_days: List[datetime] = (
                    last_day_n_months_ago(date, n=2, return_all=True)
                )
                dates_str_query = ",".join(
                    [
                        date.strftime("%Y-%m-%d")
                        for date in last_n_months_last_business_days
                    ]
                )
                url = f"https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/debt/mspd/mspd_table_5?filter=record_date:in:({dates_str_query})"
                response = await client.get(
                    url, headers=build_treasurydirect_header(host_str="api.fiscaldata.treasury.gov")
                )
                response.raise_for_status()
                curr_stripping_activity_json = response.json()
                curr_stripping_activity_df = pd.DataFrame(
                    curr_stripping_activity_json["data"]
                )
                curr_stripping_activity_df["record_date"] = pd.to_datetime(
                    curr_stripping_activity_df["record_date"], errors="coerce"
                )
                latest_date = curr_stripping_activity_df["record_date"].max()
                curr_stripping_activity_df = curr_stripping_activity_df[
                    curr_stripping_activity_df["record_date"] == latest_date
                ]
                curr_stripping_activity_df["maturity_date"] = pd.to_datetime(
                    curr_stripping_activity_df["maturity_date"], errors="coerce"
                )
                curr_stripping_activity_df["interest_rate_pct"] = pd.to_numeric(
                    curr_stripping_activity_df["interest_rate_pct"], errors="coerce"
                )
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

                if uid:
                    return date, curr_stripping_activity_df, uid
                return date, curr_stripping_activity_df

            except Exception as e:
                print(f"An error occurred: {e}")
                if uid:
                    return date, pd.DataFrame(), uid
                return date, pd.DataFrame()

        tasks = [
            fetch_mspd_table_5(client=client, date=date, uid=uid) for date in dates
        ]
        return tasks

    # USING aiohttp!
    async def _build_fetch_tasks_historical_trace_data(
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

        finra_cookie_url = "https://services-dynarep.ddwa.finra.org/public/reporting/v2/group/Firm/name/ActiveIndividual/dynamiclookup/examCode"
        finra_cookie_response = requests.get(
            finra_cookie_url, headers=finra_cookie_headers
        )
        finra_cookie_str = dict(finra_cookie_response.headers)["set-cookie"]
        finra_cookie_dict = cookie_string_to_dict(cookie_string=finra_cookie_str)

        def build_finra_trade_history_headers(cookie_str: str, x_xsrf_token_str: str):
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
                    {"fieldName": "cusip", "fieldValue": cusip, "compareType": "EQUAL"},
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
                    )
                    config_response.raise_for_status()
                    record_total_json = await config_response.json()
                    record_total_str = record_total_json["returnBody"]["headers"][
                        "Record-Total"
                    ][0]
                    return cusip, record_total_str
                except Exception as e:
                    print("Something went wrong: ", e)
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
                async with aiohttp.ClientSession() as config_session:
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

        cusip_finra_api_payload_configs = get_cusips_finra_pagination_configs(
            cusips=cusips, start_date=start_date, end_date=end_date
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
                )
                response.raise_for_status()
                trade_history_json = await response.json()
                trade_data_json = json.loads(trade_history_json["returnBody"]["data"])
                df = pd.DataFrame(trade_data_json)
                if uid:
                    return cusip, df, uid
                return cusip, df
            except Exception as e:
                print(f"Error - CUISP: {cusip} at Offset {offset}")
                print(str(e))

        tasks = []
        for cusip in cusips:
            max_record_size = int(cusip_finra_api_payload_configs[cusip])
            num_reqs = math.ceil(max_record_size / 5000)
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
