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


class PublicDotcomDataFetcher(DataFetcherBase):
    _public_dotcom_jwt: str = None

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

        self._public_dotcom_jwt = self._fetch_public_dotcome_jwt()

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
