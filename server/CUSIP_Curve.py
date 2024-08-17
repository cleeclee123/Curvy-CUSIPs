import asyncio
import logging
import math
import multiprocessing as mp
import time
from collections import defaultdict
from datetime import datetime
from functools import reduce, partial
from typing import Dict, List, Optional, Tuple

import aiohttp
import httpx
import numpy as np
import pandas as pd
import requests
import ujson as json

from server.utils.utils import (
    JSON,
    build_treasurydirect_header,
    cookie_string_to_dict,
    get_active_cusips,
    last_day_n_months_ago,
    is_valid_ust_cusip,
    historical_auction_cols,
    get_last_n_off_the_run_cusips,
    ust_labeler,
    ust_sorter,
)
from server.utils.RL_BondPricer import RL_BondPricer
from server.utils.QL_BondPricer import QL_BondPricer

# TODO
"""
- timeouts and error handling after
- bond price to ytm optimization
- find other cusip historical price source
"""


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


class CUSIP_Curve:
    _logger = logging.getLogger()
    _use_ust_issue_date: bool = False
    _debug_verbose: bool = False
    _info_verbose: bool = False  # performance benchmarking mainly
    _no_logs_plz: bool = False

    def __init__(
        self,
        debug_verbose: Optional[bool] = False,
        info_verbose: Optional[bool] = False,
        no_logs_plz: Optional[bool] = False,  # temp
        use_ust_issue_date: Optional[bool] = False,
    ):
        self._debug_verbose = debug_verbose
        self._info_verbose = info_verbose
        self._no_logs_plz = no_logs_plz
        self._use_ust_issue_date = use_ust_issue_date
        if self._debug_verbose:
            self._logger.setLevel(self._logger.DEBUG)
        if self._info_verbose:
            self._logger.setLevel(self._logger.INFO)
        if self._no_logs_plz:
            self._logger.disabled = True
            self._logger.propagate = False

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
            res = requests.get(base_url, headers=build_treasurydirect_header())
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
                response = await client.get(url, headers=build_treasurydirect_header())
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

                if return_df:
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

    def get_auctions_df(self, as_of_date: datetime) -> pd.DataFrame:
        async def build_tasks(client: httpx.AsyncClient, as_of_date: datetime):
            tasks = await self._build_fetch_tasks_historical_treasury_auctions(
                client=client, as_of_date=as_of_date
            )
            return await asyncio.gather(*tasks)

        async def run_fetch_all(as_of_date: datetime):
            async with httpx.AsyncClient() as client:
                all_data = await build_tasks(client=client, as_of_date=as_of_date)
                return all_data

        dfs = asyncio.run(run_fetch_all(as_of_date=as_of_date))
        return pd.concat(dfs)

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
            self._logger.debug(f"UST Prices - {date} Payload: {payload}")
            cols_to_return = ["cusip", "offer_price", "bid_price", "eod_price"]
            try:
                response = await client.post(url, data=payload, follow_redirects=True)
                response.raise_for_status()
                tables = pd.read_html(response.content)
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
                df = df[
                    (df["security type"] != "TIPS")
                    & (df["security type"] != "MARKET BASED FRN")
                ]
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
                self._logger.debug(f"UST Prices - Bad Status: {response.status_code}")
                if uid:
                    return date, pd.DataFrame(columns=cols_to_return), uid
                return date, pd.DataFrame(columns=cols_to_return)
            except Exception as e:
                self._logger.debug(f"UST Prices - Error: {e}")
                if uid:
                    return date, pd.DataFrame(columns=cols_to_return), uid
                return date, pd.DataFrame(columns=cols_to_return)

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
        self,
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
                "changeFromPriorWeek",
                "changeFromPriorYear",
            ]
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
                curr_soma_holdings_df["changeFromPriorWeek"] = pd.to_numeric(
                    curr_soma_holdings_df["changeFromPriorWeek"], errors="coerce"
                )
                curr_soma_holdings_df["changeFromPriorYear"] = pd.to_numeric(
                    curr_soma_holdings_df["changeFromPriorYear"], errors="coerce"
                )
                curr_soma_holdings_df = curr_soma_holdings_df[
                    (curr_soma_holdings_df["securityType"] != "TIPS")
                    & (curr_soma_holdings_df["securityType"] != "FRNs")
                ]
                if uid:
                    return date, curr_soma_holdings_df[cols_to_return], uid
                return date, curr_soma_holdings_df[cols_to_return]

            except httpx.HTTPStatusError as e:
                self._logger.debug(f"SOMA Holding - Bad Status: {response.status_code}")
                if uid:
                    return date, pd.DataFrame(columns=cols_to_return), uid
                return date, pd.DataFrame(columns=cols_to_return)

            except Exception as e:
                self._logger.debug(f"SOMA Holding - Error: {str(e)}")
                if uid:
                    return date, pd.DataFrame(columns=cols_to_return), uid
                return date, pd.DataFrame(columns=cols_to_return)

        tasks = [
            fetch_single_soma_holding_day(client=client, date=date, uid=uid)
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
                "outstanding_amt",
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

                if uid:
                    return date, curr_stripping_activity_df[cols_to_return], uid
                return date, curr_stripping_activity_df[cols_to_return]

            except httpx.HTTPStatusError as e:
                self._logger.debug(f"STRIPping - Bad Status: {response.status_code}")
                if uid:
                    return date, pd.DataFrame(columns=cols_to_return), uid
                return date, pd.DataFrame(columns=cols_to_return)

            except Exception as e:
                self._logger.debug(f"STRIPping - Error: {str(e)}")
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
                finra_cookie_url, headers=finra_cookie_headers
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
                        )
                        config_response.raise_for_status()
                        record_total_json = await config_response.json()
                        record_total_str = record_total_json["returnBody"]["headers"][
                            "Record-Total"
                        ][0]
                        return cusip, record_total_str
                    except aiohttp.ClientResponseError:
                        self._logger.debug(
                            f"TRACE - CONFIGs Bad Status: {config_response.status}"
                        )
                        return cusip, -1

                    except Exception as e:
                        self._logger.debug(f"TRACE - CONFIGs Error : {str(e)}")
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
                    self._logger.debug(
                        f"TRACE - Trade History Bad Status: {response.status}"
                    )
                    if uid:
                        return cusip, None, uid
                    return cusip, None

                except Exception as e:
                    self._logger.debug(f"TRACE - Trade History Error : {str(e)}")
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
            async with aiohttp.ClientSession(timeout=session_timeout) as session:
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
        auctions_df: Optional[pd.DataFrame] = None,
        sorted: Optional[bool] = False,
    ):
        async def gather_tasks(client: httpx.AsyncClient, as_of_date: datetime):
            ust_historical_prices_tasks = (
                await self._build_fetch_tasks_historical_cusip_prices(
                    client=client, dates=[as_of_date], uid="ust_prices"
                )
            )
            tasks = ust_historical_prices_tasks

            if auctions_df is None:
                tasks += await self._build_fetch_tasks_historical_treasury_auctions(
                    client=client, as_of_date=as_of_date, uid="ust_auctions"
                )
            if include_soma_holdings:
                tasks += await self._build_fetch_tasks_historical_soma_holdings(
                    client=client, dates=[as_of_date], uid="soma_holdings"
                )
            if include_stripping_activity:
                tasks += await self._build_fetch_tasks_historical_stripping_activity(
                    client=client, dates=[as_of_date], uid="ust_stripping"
                )

            return await asyncio.gather(*tasks)

        async def run_fetch_all(as_of_date: datetime):
            async with httpx.AsyncClient() as client:
                all_data = await gather_tasks(client=client, as_of_date=as_of_date)
                return all_data

        results = asyncio.run(run_fetch_all(as_of_date=as_of_date))
        auctions_dfs = []
        dfs = []
        for tup in results:
            uid = tup[-1]
            if uid == "ust_auctions":
                auctions_dfs.append(tup[0])
            elif (
                uid == "ust_prices" or uid == "soma_holdings" or uid == "ust_stripping"
            ):
                dfs.append(tup[1])
            else:
                self._logger.warning(f"CURVE SET - unknown UID, Current Tuple: {tup}")

        auctions_df = pd.concat(auctions_dfs) if auctions_df is None else auctions_df
        otr_cusips_dict = get_last_n_off_the_run_cusips(
            auctions_df=auctions_df,
            n=0,
            filtered=True,
            as_of_date=as_of_date,
            use_issue_date=self._use_ust_issue_date,
        )[0]
        auctions_df["is_on_the_run"] = auctions_df["cusip"].isin(
            list(otr_cusips_dict.values())
        )
        auctions_df["label"] = auctions_df["maturity_date"].apply(ust_labeler)
        auctions_df["time_to_maturity"] = (
            auctions_df["maturity_date"] - as_of_date
        ).dt.days / 365
        if not include_auction_results:
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
                    "original_security_term",
                ]
            ]
        merged_df = reduce(
            lambda left, right: pd.merge(left, right, on="cusip", how="outer"), dfs
        )
        merged_df = pd.merge(left=auctions_df, right=merged_df, on="cusip", how="outer")
        merged_df = merged_df[merged_df["cusip"].apply(is_valid_ust_cusip)]
        merged_df["mid_price"] = (merged_df["offer_price"] + merged_df["bid_price"]) / 2

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
        if sorted:
            merged_df["sort_key"] = merged_df["original_security_term"].apply(
                ust_sorter
            )
            merged_df = (
                merged_df.sort_values(by=["sort_key", "maturity_date"])
                .drop(columns="sort_key")
                .reset_index(drop=True)
            )

        return merged_df
