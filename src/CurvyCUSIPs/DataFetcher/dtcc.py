import asyncio
import re
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import reduce, partial
from io import BytesIO
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import warnings
import httpx
import pandas as pd
import scipy
import scipy.interpolate
import tqdm
import tqdm.asyncio
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from pandas.errors import DtypeWarning

from CurvyCUSIPs.DataFetcher.base import DataFetcherBase

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

import sys

if sys.platform == "win32":
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)


class DTCCSDR_DataFetcher(DataFetcherBase):
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

    def _get_dtcc_url_and_header(
        self, agency: Literal["CFTC", "SEC"], asset_class: Literal["COMMODITIES", "CREDITS", "EQUITIES", "FOREX", "RATES"], date: datetime
    ) -> Tuple[str, Dict[str, str]]:
        if (agency == "SEC" and asset_class == "COMMODITIES") or (agency == "SEC" and asset_class == "FOREX"):
            raise ValueError(f"SEC DOES NOT STORE {asset_class} IN THEIR SDR")

        dtcc_url = f"https://kgc0418-tdw-data-0.s3.amazonaws.com/{agency.lower()}/eod/{agency.upper()}_CUMULATIVE_{asset_class.upper()}_{date.strftime("%Y_%m_%d")}.zip"
        dtcc_header = {
            "authority": "pddata.dtcc.com",
            "method": "GET",
            "path": f"/{agency.upper()}_CUMULATIVE_{asset_class.upper()}_{date.strftime("%Y_%m_%d")}",
            "scheme": "https",
            "accept": "application/json, text/plain, */*",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "en-US,en;q=0.9",
            "dnt": "1",
            "priority": "u=1, i",
            "referer": f"https://pddata.dtcc.com/ppd/{agency.lower()}dashboard",
            "sec-ch-ua": '"Chromium";v="130", "Google Chrome";v="130", "Not?A_Brand";v="99"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.3",
        }

        return dtcc_url, dtcc_header

    async def _fetch_dtcc_sdr_data_helper(
        self,
        client: httpx.AsyncClient,
        date: datetime,
        agency: Literal["CFTC", "SEC"],
        asset_class: Literal["COMMODITIES", "CREDITS", "EQUITIES", "FOREX", "RATES"],
        max_retries: Optional[int] = 3,
        backoff_factor: Optional[int] = 1,
    ):
        retries = 0
        dtcc_sdr_url, dtcc_sdr_header = self._get_dtcc_url_and_header(date=date, agency=agency, asset_class=asset_class)
        try:
            while retries < max_retries:
                try:
                    async with client.stream(
                        method="GET",
                        url=dtcc_sdr_url,
                        headers=dtcc_sdr_header,
                        follow_redirects=True,
                        timeout=self._global_timeout,
                    ) as response:
                        response.raise_for_status()
                        zip_buffer = BytesIO()
                        async for chunk in response.aiter_bytes():
                            zip_buffer.write(chunk)
                        zip_buffer.seek(0)

                    return zip_buffer

                except httpx.HTTPStatusError as e:
                    self._logger.error(f"DTCC - Bad Status for {agency}-{asset_class}-{date}: {response.status_code}")
                    if response.status_code == 404:
                        return None

                    retries += 1
                    wait_time = backoff_factor * (2 ** (retries - 1))
                    self._logger.debug(f"DTCC - Throttled. Waiting for {wait_time} seconds before retrying...")
                    await asyncio.sleep(wait_time)

                except Exception as e:
                    self._logger.error(f"DTCC - Error for {agency}-{asset_class}-{date}: {e}")
                    print(e)
                    retries += 1
                    wait_time = backoff_factor * (2 ** (retries - 1))
                    self._logger.debug(f"DTCC - Throttled. Waiting for {wait_time} seconds before retrying...")
                    await asyncio.sleep(wait_time)

            raise ValueError(f"DTCC - Max retries exceeded for {agency}-{asset_class}-{date}")

        except Exception as e:
            self._logger.error(e)
            return None
    
    def _read_file(self, file_data: Tuple[BytesIO, str, bool]) -> Tuple[Union[str, datetime], pd.DataFrame]:
        file_buffer, file_name, convert_key_into_dt = file_data

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DtypeWarning)

            if file_name.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(file_buffer)
            elif file_name.lower().endswith(".csv"):
                df = pd.read_csv(file_buffer, low_memory=False)
            else:
                return None

            if convert_key_into_dt:
                match = re.search(r"_(\d{4})_(\d{2})_(\d{2})$", file_name.split(".")[0])
                if match:
                    year, month, day = map(int, match.groups())
                    key = datetime(year, month, day)
                else:
                    key = file_name
            else:
                key = file_name

            return key, df

    def _parse_dtc_filename_to_datetime(self, filename: str) -> datetime:
        match = re.search(r"_(\d{4})_(\d{2})_(\d{2})$", filename)
        if match:
            year, month, day = map(int, match.groups())
            return datetime(year, month, day)
        else:
            raise ValueError("Filename does not contain a valid date.")

    def _extract_excel_from_zip(self, zip_buffer, convert_key_into_dt=False, parallelize=False, max_extraction_workers=None):
        if not zip_buffer:
            return {}

        with zipfile.ZipFile(zip_buffer) as zip_file:
            allowed_extensions = (".xlsx", ".xls", ".csv")
            matching_files = [
                info.filename for info in zip_file.infolist() if not info.is_dir() and info.filename.lower().endswith(allowed_extensions)
            ]
            if not matching_files:
                raise FileNotFoundError("No Excel or CSV file found in the ZIP archive.")

            if parallelize:
                with ThreadPoolExecutor(max_workers=max_extraction_workers) as executor:
                    file_data = []
                    for file_name in matching_files:
                        with zip_file.open(file_name) as f:
                            # Create a BytesIO object for each file
                            buffer = BytesIO(f.read())
                            file_data.append((buffer, file_name, convert_key_into_dt))

                    results = list(executor.map(self._read_file, file_data))

                return {key: df for result in results if result is not None for key, df in [result]}

            else:
                dataframes = {}
                for file_name in matching_files:
                    with zip_file.open(file_name) as file:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", DtypeWarning)

                            if file_name.lower().endswith((".xlsx", ".xls")):
                                df = pd.read_excel(file)
                            elif file_name.lower().endswith(".csv"):
                                df = pd.read_csv(file)
                            else:
                                self._logger.debug(f"DTCC - Skipping {file_name}")
                                continue

                            if convert_key_into_dt:
                                dataframes[self._parse_dtc_filename_to_datetime(file_name.split(".")[0])] = df
                            else:
                                dataframes[file_name] = df

                return dataframes

    async def _fetch_dtcc_sdr_and_extract_excel(
        self,
        semaphore: asyncio.Semaphore,
        client: httpx.AsyncClient,
        date: datetime,
        agency: Literal["CFTC", "SEC"],
        asset_class: Literal["COMMODITIES", "CREDITS", "EQUITIES", "FOREX", "RATES"],
        parallelize: int = False,
        max_extraction_workers=None,
    ):
        async with semaphore:
            zip_buffer = await self._fetch_dtcc_sdr_data_helper(client, date, agency, asset_class)
        
        if parallelize:
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=max_extraction_workers) as pool:
                df = await loop.run_in_executor(pool, self._extract_excel_from_zip, zip_buffer, True, True, max_extraction_workers)
            return df
        else:
            df = await asyncio.to_thread(self._extract_excel_from_zip, zip_buffer, True, parallelize, max_extraction_workers)
            return df

    def fetch_dtcc_sdr_data_timeseries(
        self,
        start_date: datetime,
        end_date: datetime,
        agency: Literal["CFTC", "SEC"],
        asset_class: Literal["COMMODITIES", "CREDITS", "EQUITIES", "FOREX", "RATES"],
        max_concurrent_tasks: Optional[int] = 64,
        max_keepalive_connections: Optional[int] = 5,
        parallelize: Optional[bool] = False,
        max_extraction_workers: Optional[int] = 3,
    ) -> Dict[datetime, pd.DataFrame]:

        bdates = pd.date_range(start=start_date, end=end_date, freq=CustomBusinessDay(calendar=USFederalHolidayCalendar()))

        async def build_tasks(
            client: httpx.AsyncClient,
            dates: List[datetime],
            agency: Literal["CFTC", "SEC"],
            asset_class: Literal["COMMODITIES", "CREDITS", "EQUITIES", "FOREX", "RATES"],
            parallelize: bool,
            max_extraction_workers: int,
        ):
            tasks = []
            semaphore = asyncio.Semaphore(max_concurrent_tasks)
            for date in dates:
                task = asyncio.create_task(
                    self._fetch_dtcc_sdr_and_extract_excel(
                        semaphore=semaphore,
                        client=client,
                        date=date,
                        agency=agency,
                        asset_class=asset_class,
                        parallelize=parallelize,
                        max_extraction_workers=max_extraction_workers,
                    )
                )
                tasks.append(task)

            # return await asyncio.gather(*tasks)
            return await tqdm.asyncio.tqdm.gather(*tasks, desc="FETCHING DTCC SDR DATASETS...")

        async def run_fetch_all(
            dates: List[datetime],
            agency: Literal["CFTC", "SEC"],
            asset_class: Literal["COMMODITIES", "CREDITS", "EQUITIES", "FOREX", "RATES"],
            parallelize: bool,
            max_extraction_workers: int,
        ):
            limits = httpx.Limits(
                max_connections=max_concurrent_tasks,
                max_keepalive_connections=max_keepalive_connections,
            )
            async with httpx.AsyncClient(limits=limits) as client:
                all_data = await build_tasks(
                    client=client,
                    dates=dates,
                    agency=agency,
                    asset_class=asset_class,
                    parallelize=parallelize,
                    max_extraction_workers=max_extraction_workers,
                )
                return all_data

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DtypeWarning)
            results: List[Tuple[str, pd.DataFrame]] = asyncio.run(
                run_fetch_all(
                    dates=bdates, agency=agency, asset_class=asset_class, parallelize=parallelize, max_extraction_workers=max_extraction_workers
                )
            )
            return reduce(lambda a, b: a | b, results)

