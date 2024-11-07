import asyncio
import math
import time
import warnings
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Literal

import aiohttp
import pandas as pd
import requests
import ujson as json

from DataFetcher.base import DataFetcherBase

from utils.ust_utils import cookie_string_to_dict

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

import sys

if sys.platform == "win32":
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)


class FinraDataFetcher(DataFetcherBase):
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

    def fetch_historcal_trace_trade_history_by_cusip(
        self,
        start_date: datetime,
        end_date: datetime,
        cusips: Optional[List[str]] = None,
        benchmark_terms: Optional[List[Literal["1Y", "2Y", "3Y", "5Y", "7Y", "10Y" "20Y", "30Y"]]] = None,
        xlsx_path: Optional[str] = None,
        session_timeout_minutes: Optional[int] = 5,
    ):
        total_t1 = time.time()

        async def build_fetch_tasks_historical_trace_data(
            session: aiohttp.ClientSession,
            start_date: datetime,
            end_date: datetime,
            cusips: Optional[List[str]] = None,
            benchmark_terms: Optional[List[Literal["1Y", "2Y", "3Y", "5Y", "7Y", "10Y" "20Y", "30Y"]]] = None,
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
            finra_cookie_str = dict(finra_cookie_response.headers)["Set-Cookie"]
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
                start_date: datetime,
                end_date: datetime,
                limit: int,
                offset: int,
                cusip: Optional[str] = None,
                benchmark_term: Optional[Literal["1Y", "2Y", "3Y", "5Y", "7Y", "10Y" "20Y", "30Y"]] = None,
            ) -> Dict[str, int]:
                if cusip:
                    filter_payload = [{
                        "fieldName": "cusip",
                        "fieldValue": cusip,
                        "compareType": "EQUAL",
                    }]
                elif benchmark_term:
                    filter_payload = [{
                        "fieldName": "benchmarkTermCode",
                        "fieldValue": benchmark_term,
                        "compareType": "EQUAL",
                    }]
                else:
                    raise ValueError("One of 'cusip', 'benchmark_term' must be a parameter - (build_finra_trade_history_payload)")
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
                    "compareFilters": filter_payload, 
                    "limit": limit,  # 5000 is Max Limit
                    "offset": offset,
                }

            def get_cusips_finra_pagination_configs(
                start_date: datetime, 
                end_date: datetime,
                cusips: Optional[List[str]] = None,
                benchmark_terms: Optional[List[Literal["1Y", "2Y", "3Y", "5Y", "7Y", "10Y" "20Y", "30Y"]]] = None,
            ):
                async def fetch_finra_cusip_pagination_config(
                    config_session: aiohttp.ClientSession,
                    start_date: datetime,
                    end_date: datetime,
                    cusip: Optional[str] = None,
                    benchmark_term: Optional[Literal["1Y", "2Y", "3Y", "5Y", "7Y", "10Y" "20Y", "30Y"]] = None,
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
                                start_date=start_date,
                                end_date=end_date,
                                limit=1,
                                offset=1,
                                cusip=cusip,
                                benchmark_term=benchmark_term
                            ),
                            proxy=self._proxies["https"],
                        )
                        config_response.raise_for_status()
                        record_total_json = await config_response.json()
                        record_total_str = record_total_json["returnBody"]["headers"][
                            "Record-Total"
                        ][0]
                        
                        if cusip:
                            return cusip, record_total_str
                        elif benchmark_term: 
                            return benchmark_term, record_total_str
                        else:
                            raise ValueError("One of 'cusip', 'benchmark_term' must be a parameter - (get_cusips_finra_pagination_configs)")
                        
                    except aiohttp.ClientResponseError:
                        self._logger.error(
                            f"TRACE - CONFIGs Bad Status: {config_response.status}"
                        )
                        record_total_str = -1
                        if cusip:
                            return cusip, record_total_str
                        elif benchmark_term: 
                            return benchmark_term, record_total_str
                        else:
                            raise ValueError("One of 'cusip', 'benchmark_term' must be a parameter - (get_cusips_finra_pagination_configs)")

                    except Exception as e:
                        self._logger.error(f"TRACE - CONFIGs Error : {str(e)}")
                        record_total_str = -1
                        if cusip:
                            return cusip, record_total_str
                        elif benchmark_term: 
                            return benchmark_term, record_total_str
                        else:
                            raise ValueError("One of 'cusip', 'benchmark_term' must be a parameter - (get_cusips_finra_pagination_configs)")

                async def build_finra_config_tasks(
                    config_session: aiohttp.ClientSession,
                    start_date: datetime,
                    end_date: datetime,
                    cusips: Optional[List[str]] = None,
                    benchmark_terms: Optional[List[Literal["1Y", "2Y", "3Y", "5Y", "7Y", "10Y" "20Y", "30Y"]]] = None,
                ):
                    if cusips:
                        tasks = [
                            fetch_finra_cusip_pagination_config(
                                config_session=config_session,
                                start_date=start_date,
                                end_date=end_date,
                                cusip=cusip,
                            )
                            for cusip in cusips
                        ]
                    elif benchmark_terms:
                        tasks = [
                            fetch_finra_cusip_pagination_config(
                                config_session=config_session,
                                start_date=start_date,
                                end_date=end_date,
                                benchmark_term=benchmark_term
                            )
                            for benchmark_term in benchmark_terms 
                        ]
                    else:
                        raise ValueError("One of 'cusip', 'benchmark_term' must be a parameter - (build_finra_config_tasks)")
                    
                    return await asyncio.gather(*tasks)

                async def run_fetch_all(
                    start_date: datetime,
                    end_date: datetime,
                    cusips: Optional[List[str]] = None,
                    benchmark_terms: Optional[List[Literal["1Y", "2Y", "3Y", "5Y", "7Y", "10Y" "20Y", "30Y"]]] = None,
                ) -> List[pd.DataFrame]:
                    async with aiohttp.ClientSession() as config_session:
                        all_data = await build_finra_config_tasks(
                            config_session=config_session,
                            start_date=start_date,
                            end_date=end_date,
                            cusips=cusips,
                            benchmark_terms=benchmark_terms
                        )
                        return all_data

                cusip_finra_api_payload_configs = dict(
                    asyncio.run(
                        run_fetch_all(
                            start_date=start_date, end_date=end_date, cusips=cusips, benchmark_terms=benchmark_terms,
                        )
                    )
                )
                return cusip_finra_api_payload_configs

            cusip_finra_api_payload_configs_t1 = time.time()
            cusip_finra_api_payload_configs = get_cusips_finra_pagination_configs(
                start_date=start_date, end_date=end_date, cusips=cusips, benchmark_terms=benchmark_terms
            )
            self._logger.info(
                f"TRACE - FINRA CUSIP API Payload Configs Took: {time.time() - cusip_finra_api_payload_configs_t1} seconds"
            )
            self._logger.debug(
                f"TRACE - CUSIP API Payload Configs: {cusip_finra_api_payload_configs}"
            )

            async def fetch_finra_cusip_trade_history(
                session: aiohttp.ClientSession,
                start_date: datetime,
                end_date: datetime,
                offset: int,
                cusip: Optional[str] = None,
                benchmark_term: Optional[Literal["1Y", "2Y", "3Y", "5Y", "7Y", "10Y" "20Y", "30Y"]] = None,
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
                            start_date=start_date,
                            end_date=end_date,
                            limit=5000,
                            offset=offset,
                            cusip=cusip,
                            benchmark_term=benchmark_term
                        ),
                        proxy=self._proxies["https"],
                    )
                    response.raise_for_status()
                    trade_history_json = await response.json()
                    trade_data_json = json.loads(
                        trade_history_json["returnBody"]["data"]
                    )
                    df = pd.DataFrame(trade_data_json)
                    if cusip:
                        if uid:
                            return cusip, df, uid
                        return cusip, df
                    elif benchmark_term:
                        if uid:
                            return benchmark_term, df, uid
                        return benchmark_term, df
                    else:
                        raise ValueError("One of 'cusip', 'benchmark_term' must be a parameter - (fetch_finra_cusip_trade_history)")
                        
                except aiohttp.ClientResponseError:
                    self._logger.error(
                        f"TRACE - Trade History Bad Status: {response.status}"
                    )
                    if cusip:
                        if uid:
                            return cusip, None, uid
                        return cusip, None
                    elif benchmark_term:
                        if uid:
                            return benchmark_term, None, uid
                        return benchmark_term, None
                    else:
                        raise ValueError("One of 'cusip', 'benchmark_term' must be a parameter - (fetch_finra_cusip_trade_history)")

                except Exception as e:
                    self._logger.error(f"TRACE - Trade History Error : {str(e)}")
                    if cusip:
                        if uid:
                            return cusip, None, uid
                        return cusip, None
                    elif benchmark_term:
                        if uid:
                            return benchmark_term, None, uid
                        return benchmark_term, None
                    else:
                        raise ValueError("One of 'cusip', 'benchmark_term' must be a parameter - (fetch_finra_cusip_trade_history)")

            tasks = []
            if cusips:
                for cusip in cusips:
                    max_record_size = int(cusip_finra_api_payload_configs[cusip])
                    if max_record_size == -1:
                        self._logger.debug(
                            f"TRACE - {cusip} had -1 Max Record Size - Does it Exist?"
                        )
                        continue
                    num_reqs = math.ceil(max_record_size / 5000)
                    self._logger.debug(f"TRACE - {cusip} Reqs: {num_reqs}")
                    for i in range(0, num_reqs + 1):
                        curr_offset = i * 5000
                        offsets.append(curr_offset)
                        # failing one on purpose
                        # if curr_offset > max_record_size:
                        #     curr_offset = max_record_size 
                        tasks.append(
                            fetch_finra_cusip_trade_history(
                                session=session,
                                cusip=cusip,
                                start_date=start_date,
                                end_date=end_date,
                                offset=curr_offset,
                            )
                        )
            elif benchmark_terms:
                for benchmark_term in benchmark_terms:
                    max_record_size = int(cusip_finra_api_payload_configs[benchmark_term])
                    if max_record_size == -1:
                        self._logger.debug(
                            f"TRACE - {benchmark_term} had -1 Max Record Size - Does it Exist?"
                        )
                        continue
                    num_reqs = math.ceil(max_record_size / 5000)
                    self._logger.debug(f"TRACE - {benchmark_term} Reqs: {num_reqs}")
                    offsets = []
                    for i in range(0, num_reqs + 1):
                        curr_offset = i * 5000
                        offsets.append(curr_offset)
                        # failing one on purpose
                        # if curr_offset > max_record_size:
                            # curr_offset = max_record_size 
                        tasks.append(
                            fetch_finra_cusip_trade_history(
                                session=session,
                                benchmark_term=benchmark_term,
                                start_date=start_date,
                                end_date=end_date,
                                offset=curr_offset,
                            )
                        )                
            else:
                raise ValueError("One of 'cusip', 'benchmark_term' must be a parameter - (build_fetch_tasks_historical_trace_data)")
            
            self._logger.debug(f"LEN OF TASKS: {len(tasks)}") 
            self._logger.debug(f"OFFSETS Used: {offsets}") 
            return tasks

        async def build_tasks(
            session: aiohttp.ClientSession,
            start_date: datetime,
            end_date: datetime,
            cusips: Optional[List[str]] = None,
            benchmark_terms: Optional[List[Literal["1Y", "2Y", "3Y", "5Y", "7Y", "10Y" "20Y", "30Y"]]] = None,
        ):
            tasks = await build_fetch_tasks_historical_trace_data(
                session=session, 
                start_date=start_date, 
                end_date=end_date,
                cusips=cusips,
                benchmark_terms=benchmark_terms 
            )
            return await asyncio.gather(*tasks)

        async def run_fetch_all(
            start_date: datetime, 
            end_date: datetime, 
            cusips: Optional[List[str]] = None,
            benchmark_terms: Optional[List[Literal["1Y", "2Y", "3Y", "5Y", "7Y", "10Y" "20Y", "30Y"]]] = None,
        ):
            session_timeout = aiohttp.ClientTimeout(
                total=None,
                sock_connect=session_timeout_minutes * 60,
                sock_read=session_timeout_minutes * 60,
            )
            async with aiohttp.ClientSession(
                timeout=session_timeout 
            ) as session:
                all_data = await build_tasks(
                    session=session,
                    start_date=start_date,
                    end_date=end_date,
                    cusips=cusips,
                    benchmark_terms=benchmark_terms 
                )
                return all_data

        fetch_all_t1 = time.time()
        results: List[Tuple[str, pd.DataFrame]] = asyncio.run(
            run_fetch_all(start_date=start_date, end_date=end_date, cusips=cusips, benchmark_terms=benchmark_terms)
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
