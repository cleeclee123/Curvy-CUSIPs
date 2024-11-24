import asyncio
import re
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import reduce
from io import BytesIO
from typing import Dict, List, Literal, Optional, Tuple, Union, Callable

import httpx
import numpy as np
import pandas as pd
import QuantLib as ql
import rateslib as rl
import scipy.interpolate
import tqdm
import tqdm.asyncio
from pandas.errors import DtypeWarning
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import BDay, CustomBusinessDay
from termcolor import colored

from CurvyCUSIPs.DataFetcher.base import DataFetcherBase
from CurvyCUSIPs.utils.dtcc_swaps_utils import build_ql_piecewise_curves, tenor_to_years, datetime_to_ql_date, DEFAULT_SWAP_TENORS

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

import sys

if sys.platform == "win32":
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)


def scipy_linear_interp_func(xx, yy, kind="linear", logspace=False):
    if logspace:
        log_y = np.log(yy)
        interp_func = scipy.interpolate.interp1d(xx, log_y, kind="linear", fill_value="extrapolate", bounds_error=False)

        def log_linear_interp(x_new):
            return np.exp(interp_func(x_new))

        return log_linear_interp

    return scipy.interpolate.interp1d(xx, yy, kind=kind, fill_value="extrapolate", bounds_error=False)


UPI_MIGRATE_DATE = datetime(2024, 1, 29)
SCHEMA_CHANGE_2022 = datetime(2022, 12, 3)


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

        # TODO make general plz
        self.anna_dsb_swaps_lookup_table_df = pd.read_excel(
            r"C:\Users\chris\Project Bond King\Curvy-CUSIPs\src\CurvyCUSIPs\data\anna_dsb\ff_swaps_loookup.xlsx"
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
        verbose=False,
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
            if results is None or len(results) == 0:
                print('"fetch_dtcc_sdr_data_timeseries" --- empty results') if verbose else None
                return {}

            return reduce(lambda a, b: a | b, results)

    # examples/references:
    # https://www.isda.org/a/z3UgE/2006-FRO-Mapping-to-2021-FROs-Version-3.pdf
    # https://www.rbccm.com/assets/rbccm/docs/legal/doddfrank/Documents/ISDALibrary/2006%20ISDA%20Definitions.pdf
    # https://www.clarusft.com/sofr-swap-nuances/
    # https://www.isda.org/a/xjPgE/Market-Practice-Note-Effective-Date-for-SOFR-transactions-040822.pdf
    # https://www.newyorkfed.org/medialibrary/Microsites/arrc/files/2021/users-guide-to-sofr2021-update.pdf

    def fetch_historical_swaps_term_structure(
        self,
        start_date: datetime,
        end_date: datetime,
        swap_type: Literal["Fixed_Float", "Fixed_Float_OIS", "Fixed_Float_Zero_Coupon"],
        # not an exhaustive list - most common
        reference_floating_rates: List[
            Literal["USD-SOFR-OIS Compound", "USD-SOFR-COMPOUND", "USD-SOFR", "USD-SOFR Compounded Index", "USD-SOFR CME Term"]
        ],
        # not an exhaustive list - most common
        ccy: Literal["USD", "EUR", "JPY", "GBP"],
        reference_floating_rate_term_value: int,
        reference_floating_rate_term_unit: Literal["DAYS", "WEEK", "MNTH", "YEAR"],
        notional_schedule: Literal["Constant", "Accreting", "Amortizing", "Custom"],
        delivery_types: List[Literal["PHYS", "CASH"]],
        settlement_t_plus: int,
        payment_lag: int,
        fixed_leg_frequency,
        fixed_leg_daycount,
        fixed_leg_convention,
        fixed_leg_calendar,
        ql_index,
        flat_forward_curve: Optional[bool] = False,
        overnight_fixings_df: Optional[pd.DataFrame] = None,
        overnight_fixings_date_col: Optional[str] = "effectiveDate",
        overnight_fixings_rate_col: Optional[str] = "percentRate",
        ny_hours_only: Optional[bool] = False,
        is_ois: Optional[bool] = False,
        tenor_col: Optional[str] = "Tenor",
        fixed_rate_col: Optional[str] = "Close",
        logLinearDiscount: Optional[bool] = False,
        logCubicDiscount: Optional[bool] = False,
        linearZero: Optional[bool] = False,
        cubicZero: Optional[bool] = False,
        linearForward: Optional[bool] = False,
        splineCubicDiscount: Optional[bool] = False,
        max_concurrent_tasks: Optional[int] = 64,
        max_keepalive_connections: Optional[int] = 5,
        max_extraction_workers: Optional[int] = 3,
        filter_extreme_time_and_sales: Optional[bool] = False,
        minimum_num_trades_time_and_sales: Optional[int] = 100,
        quantile_smoothing_range: Optional[Tuple[float, float]] = (0.05, 0.95),
        specifc_tenor_quantile_smoothing_range: Optional[Dict[str, Tuple[float, float]]] = None,
        remove_tenors: Optional[List[str]] = None,
        my_scipy_interp_func: Optional[Callable] = scipy_linear_interp_func,
        verbose: Optional[bool] = False,
    ) -> (
        Dict[
            datetime,
            Dict[
                str,
                pd.DataFrame
                | Dict[
                    str,
                    ql.PiecewiseLogLinearDiscount
                    | ql.PiecewiseLogCubicDiscount
                    | ql.PiecewiseLinearZero
                    | ql.PiecewiseCubicZero
                    | ql.PiecewiseLinearForward
                    | ql.PiecewiseSplineCubicDiscount,
                ],
            ],
        ]
        | Dict[datetime, pd.DataFrame]
    ):
        if "OIS" in swap_type or "OIS" in reference_floating_rates:
            is_ois = True

        build_ql_curves = True
        if not (logLinearDiscount | logCubicDiscount | linearZero | cubicZero | linearForward | splineCubicDiscount):
            build_ql_curves = False

        sdr_time_and_sales_dict: Dict[datetime, pd.DataFrame] = self.fetch_dtcc_sdr_data_timeseries(
            start_date=start_date,
            end_date=end_date,
            agency="CFTC",
            asset_class="RATES",
            max_concurrent_tasks=max_concurrent_tasks,
            max_keepalive_connections=max_keepalive_connections,
            max_extraction_workers=max_extraction_workers,
        )

        if sdr_time_and_sales_dict is None or len(sdr_time_and_sales_dict.keys()) == 0:
            self._logger.debug('"fetch_historical_swaps_term_structure" --- SDR Time and Sales Data is Empty')
            print('"fetch_historical_swaps_term_structure" --- SDR Time and Sales Data is Empty') if verbose else None
            return {}

        swaps_term_structures: Dict[
            datetime,
            Tuple[
                pd.DataFrame,
                Dict[
                    str,
                    ql.PiecewiseLogLinearDiscount
                    | ql.PiecewiseLogCubicDiscount
                    | ql.PiecewiseLinearZero
                    | ql.PiecewiseCubicZero
                    | ql.PiecewiseLinearForward
                    | ql.PiecewiseSplineCubicDiscount,
                ],
            ],
        ] = {}

        upi_lookup_df = self.anna_dsb_swaps_lookup_table_df

        UPIS = upi_lookup_df[
            (upi_lookup_df["Header_UseCase"] == swap_type)
            & (upi_lookup_df["Derived_UnderlierName"].isin(reference_floating_rates))
            & ((upi_lookup_df["Attributes_NotionalCurrency"] == ccy))
            & ((upi_lookup_df["Attributes_ReferenceRateTermValue"] == reference_floating_rate_term_value))
            & ((upi_lookup_df["Attributes_ReferenceRateTermUnit"] == reference_floating_rate_term_unit))
            & ((upi_lookup_df["Attributes_NotionalSchedule"] == notional_schedule))
            & ((upi_lookup_df["Attributes_DeliveryType"].isin(delivery_types)))
        ]["Identifier_UPI"].to_numpy()

        legacy_swap_type_mapper = {
            "Fixed_Float": "InterestRate:IRSwap:FixedFloat",
            "Fixed_Float_OIS": "InterestRate:IRSwap:OIS",
            "Fixed_Float_Zero_Coupon": "InterestRate:IRSwap:FixedFloat",
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            for date, time_and_sales_sdr_df in tqdm.tqdm(sdr_time_and_sales_dict.items(), desc="BUILDING SOFR CURVES..."):
                try:
                    if time_and_sales_sdr_df is None or time_and_sales_sdr_df.empty:
                        raise ValueError("swaps_time_and_sales_df is empty")
                    
                    if date < SCHEMA_CHANGE_2022:
                        swaps_time_and_sales_df: pd.DataFrame = format_swap_time_and_sales(
                            time_and_sales_sdr_df[
                                (
                                    (time_and_sales_sdr_df["Product ID"] == legacy_swap_type_mapper[swap_type])
                                    & (
                                        (time_and_sales_sdr_df["Leg 1 - Floating Rate Index"].isin(reference_floating_rates))
                                        | (time_and_sales_sdr_df["Leg 2 - Floating Rate Index"].isin(reference_floating_rates))
                                    )
                                    & ((time_and_sales_sdr_df["Notional Currency 1"] == ccy) | (time_and_sales_sdr_df["Notional Currency 2"] == ccy))
                                    # & (
                                    #     (
                                    #         time_and_sales_sdr_df["Floating Rate Reset Frequency Period 1"]
                                    #         == f"{reference_floating_rate_term_value}{reference_floating_rate_term_unit[0]}"
                                    #     )
                                    #     | (
                                    #         time_and_sales_sdr_df["Floating Rate Reset Frequency Period 2"]
                                    #         == f"{reference_floating_rate_term_value}{reference_floating_rate_term_unit[0]}"
                                    #     )
                                    # )
                                    & (time_and_sales_sdr_df["Action"] == "NEW")
                                    & (time_and_sales_sdr_df["Transaction Type"] == "Trade")
                                    & (
                                        (time_and_sales_sdr_df["Non-Standardized Pricing Indicator"] == "N")
                                        # | (time_and_sales_sdr_df["Non-Standardized Pricing Indicator"].isna())
                                    )
                                )
                            ],
                            as_of_date=date,
                            tenors_to_interpolate=["4Y", "6Y", "8Y"],
                            verbose=verbose,
                        )

                    elif date < UPI_MIGRATE_DATE:
                        swaps_time_and_sales_df: pd.DataFrame = format_swap_time_and_sales(
                            time_and_sales_sdr_df[
                                (
                                    (time_and_sales_sdr_df["Product name"] == legacy_swap_type_mapper[swap_type])
                                    & (
                                        (time_and_sales_sdr_df["Underlier ID-Leg 1"].isin(reference_floating_rates))
                                        | (time_and_sales_sdr_df["Underlier ID-Leg 2"].isin(reference_floating_rates))
                                    )
                                    & (
                                        (time_and_sales_sdr_df["Notional currency-Leg 1"] == ccy)
                                        | (time_and_sales_sdr_df["Notional currency-Leg 2"] == ccy)
                                    )
                                    & (time_and_sales_sdr_df["Action type"] == "NEWT")
                                    & (time_and_sales_sdr_df["Package indicator"] == False)
                                    & (
                                        (time_and_sales_sdr_df["Non-standardized term indicator"] == False)
                                        # | time_and_sales_sdr_df["Non-standardized term indicator"].isna()
                                    )
                                )
                            ],
                            as_of_date=date,
                            tenors_to_interpolate=["4Y", "6Y", "8Y"],
                            verbose=verbose,
                        )

                    else:
                        swaps_time_and_sales_df: pd.DataFrame = format_swap_time_and_sales(
                            time_and_sales_sdr_df[
                                (time_and_sales_sdr_df["Unique Product Identifier"].isin(UPIS))
                                & (time_and_sales_sdr_df["Action type"] == "NEWT")
                                & (time_and_sales_sdr_df["Package indicator"] == False)
                                & (
                                    (time_and_sales_sdr_df["Non-standardized term indicator"] == False)
                                    # | time_and_sales_sdr_df["Non-standardized term indicator"].isna()
                                )
                            ],
                            as_of_date=date,
                            tenors_to_interpolate=["4Y", "6Y", "8Y"],
                            verbose=verbose,
                        )

                    swaps_time_and_sales_df = swaps_time_and_sales_df.sort_values(by=["Execution Timestamp"])
                    swaps_time_and_sales_df = swaps_time_and_sales_df.reset_index(drop=True)
                    filtered_swaps_time_and_sales_df = swaps_time_and_sales_df.copy()
                    swaps_term_structures[date] = {"ohlc": None, "time_and_sales": None, "ql_curves": None}
                    swaps_term_structures[date]["time_and_sales"] = swaps_time_and_sales_df

                    if ny_hours_only:
                        filtered_swaps_time_and_sales_df = filtered_swaps_time_and_sales_df[
                            filtered_swaps_time_and_sales_df["Execution Timestamp"].dt.hour >= 12
                        ]
                        filtered_swaps_time_and_sales_df = filtered_swaps_time_and_sales_df[
                            filtered_swaps_time_and_sales_df["Execution Timestamp"].dt.hour <= 23
                        ]

                    if filtered_swaps_time_and_sales_df.empty or len(filtered_swaps_time_and_sales_df) < minimum_num_trades_time_and_sales:
                        empty_err_message = f"Too little liquidity: {len(filtered_swaps_time_and_sales_df)} < {minimum_num_trades_time_and_sales}"

                        if ny_hours_only:
                            empty_err_message += " - ny_hours_only"
                        if filter_extreme_time_and_sales:
                            empty_err_message += " - filter_extreme_time_and_sales"
                        raise ValueError(empty_err_message)

                    if overnight_fixings_df is not None:
                        on_fixing_rate_df = overnight_fixings_df[overnight_fixings_df[overnight_fixings_date_col].dt.date == date.date()]
                        if on_fixing_rate_df.empty:
                            raise ValueError("Overnight Fixings Rate DF is empty")

                        on_fixing_rate = on_fixing_rate_df[overnight_fixings_rate_col].iloc[-1]
                    else:
                        on_fixing_rate = None

                    try:
                        swaps_ohlc_df = format_swap_ohlc(
                            filtered_swaps_time_and_sales_df,
                            as_of_date=date,
                            overnight_rate=on_fixing_rate,
                            filter_extreme_time_and_sales=filter_extreme_time_and_sales,
                            quantile_smoothing_range=quantile_smoothing_range,
                            specifc_tenor_quantile_smoothing_range=specifc_tenor_quantile_smoothing_range,
                            remove_tenors=remove_tenors,
                            dtcc_interp_func=my_scipy_interp_func,
                            ny_hours=ny_hours_only,
                            verbose=verbose,
                        )
                        swaps_term_structures[date]["ohlc"] = swaps_ohlc_df
                    except Exception as e:
                        self._logger.error(f'"fetch_historical_swaps_term_structure" --- OHLC Error at {date} --- {str(e)}')
                        swaps_term_structures[date]["ohlc"] = None

                    if build_ql_curves:
                        if flat_forward_curve:
                            print("hello")
                            try:
                                ql.Settings.instance().evaluationDate = datetime_to_ql_date(date)
                                flat_curve = ql.FlatForward(
                                    datetime_to_ql_date(date),
                                    ql.QuoteHandle(ql.SimpleQuote(on_fixing_rate)),
                                    ql_index.dayCounter(),
                                    ql.Compounded,
                                    fixed_leg_frequency,
                                )
                                yield_curve_handle = ql.YieldTermStructureHandle(flat_curve)
                                ql_index = ql.OvernightIndex(
                                    ql_index.familyName(),
                                    ql_index.fixingDays(),
                                    ql_index.currency(),
                                    ql_index.fixingCalendar(),
                                    ql_index.dayCounter(),
                                    yield_curve_handle,
                                )
                            except Exception as e:
                                if "degenerate" in str(e):
                                    plus_one_bday_as_of_date = date + BDay(1)
                                    ql.Settings.instance().evaluationDate = datetime_to_ql_date(plus_one_bday_as_of_date)
                                    flat_curve = ql.FlatForward(
                                        datetime_to_ql_date(plus_one_bday_as_of_date),
                                        ql.QuoteHandle(ql.SimpleQuote(on_fixing_rate)),
                                        ql_index.dayCounter(),
                                        ql.Compounded,
                                        fixed_leg_frequency,
                                    )
                                    yield_curve_handle = ql.YieldTermStructureHandle(flat_curve)
                                    ql_index = ql.OvernightIndex(
                                        ql_index.familyName(),
                                        ql_index.fixingDays(),
                                        ql_index.currency(),
                                        ql_index.fixingCalendar(),
                                        ql_index.dayCounter(),
                                        yield_curve_handle,
                                    )
                                else:
                                    raise e

                        try:
                            ql_pw_curves = build_ql_piecewise_curves(
                                df=swaps_ohlc_df,
                                as_of_date=date,
                                settlement_t_plus=settlement_t_plus,
                                payment_lag=payment_lag,
                                is_ois=is_ois,
                                tenor_col=tenor_col,
                                fixed_rate_col=fixed_rate_col,
                                fixed_leg_frequency=fixed_leg_frequency,
                                fixed_leg_daycount=fixed_leg_daycount,
                                fixed_leg_convention=fixed_leg_convention,
                                fixed_leg_calendar=fixed_leg_calendar,
                                ql_index=ql_index,
                                logLinearDiscount=logLinearDiscount,
                                logCubicDiscount=logCubicDiscount,
                                linearZero=linearZero,
                                cubicZero=cubicZero,
                                linearForward=linearForward,
                                splineCubicDiscount=splineCubicDiscount,
                            )

                            swaps_term_structures[date]["ql_curves"] = ql_pw_curves
                        except Exception as e:
                            print(f'"fetch_historical_swaps_term_structure" --- Quantlib Error at {date} --- {str(e)}') if verbose else None
                            self._logger.error(f'"fetch_historical_swaps_term_structure" --- Quantlib Error at {date} --- {str(e)}')
                            swaps_term_structures[date]["ql_curves"] = {
                                "logLinearDiscount": None,
                                "logCubicDiscount": None,
                                "linearZero": None,
                                "cubicZero": None,
                                "linearForward": None,
                                "splineCubicDiscount": None,
                            }

                except Exception as e:
                    print(f'"fetch_historical_swaps_term_structure" --- Curve Building Error at {date} --- {str(e)}') if verbose else None
                    self._logger.error(f"DTCC Swaps Curve Building (fetch_historical_sofr_ois_term_structure) at {date} --- {e}")

            return swaps_term_structures


def format_swap_time_and_sales(df: pd.DataFrame, as_of_date: datetime, tenors_to_interpolate: Optional[List[str] | bool] = None, verbose=False):
    swap_columns = [
        "Event timestamp",
        "Execution Timestamp",
        "Effective Date",
        "Expiration Date",
        # "Platform identifier",
        "Tenor",
        "Fwd",
        "Fixed Rate",
        "Direction",
        "Notional Amount",
        "Unique Product Identifier",
        "UPI FISN",
        "UPI Underlier Name",
    ]

    if as_of_date < SCHEMA_CHANGE_2022:
        df = df.rename(
            columns={
                "Event Timestamp": "Event timestamp",
                "Leg 1 - Floating Rate Index": "Underlier ID-Leg 1",
                "Leg 2 - Floating Rate Index": "Underlier ID-Leg 2",
                "Fixed Rate 1": "Fixed rate-Leg 1",
                "Fixed Rate 2": "Fixed rate-Leg 2",
                "Notional Amount 1": "Notional amount-Leg 1",
                "Notional Amount 2": "Notional amount-Leg 2",
            }
        )

    if as_of_date < UPI_MIGRATE_DATE or as_of_date < SCHEMA_CHANGE_2022:
        df["Unique Product Identifier"] = "DNE"
        df["UPI FISN"] = "DNE"
        df["UPI Underlier Name"] = df["Underlier ID-Leg 1"].combine_first(df["Underlier ID-Leg 2"])

    date_columns = [
        "Event timestamp",
        "Execution Timestamp",
        "Effective Date",
        "Expiration Date",
    ]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])
        df[col] = df[col].dt.tz_localize(None)

    df = df[df["Event timestamp"].dt.date >= as_of_date.date()]
    df = df[df["Execution Timestamp"].dt.date >= as_of_date.date()]

    year_day_count = 360
    df["Expiry (d)"] = (df["Expiration Date"] - df["Effective Date"]).dt.days
    df["Expiry (w)"] = df["Expiry (d)"] / 7
    df["Expiry (m)"] = df["Expiry (d)"] / 30
    df["Expiry (yr)"] = df["Expiry (d)"] / year_day_count

    settle_date = as_of_date + BDay(2)
    df["Fwd (d)"] = (df["Effective Date"] - settle_date).dt.days
    df["Fwd (w)"] = df["Fwd (d)"] / 7
    df["Fwd (m)"] = df["Fwd (d)"] / 30
    df["Fwd (yr)"] = df["Fwd (d)"] / year_day_count

    def format_tenor(row: pd.Series, col: str):
        if row[f"{col} (d)"] < 7:
            return f"{int(row[f'{col} (d)'])}D"
        if row[f"{col} (w)"] <= 3.75:
            return f"{int(row[f'{col} (w)'])}W"
        if row[f"{col} (m)"] < 23.5:
            return f"{int(row[f'{col} (m)'])}M"
        return f"{int(row[f'{col} (yr)'])}Y"

    df["Tenor"] = df.apply(lambda row: format_tenor(row, col="Expiry"), axis=1)
    df["Fwd"] = df.apply(lambda row: format_tenor(row, col="Fwd"), axis=1)

    df["Notional amount-Leg 1"] = pd.to_numeric(df["Notional amount-Leg 1"].str.replace(",", ""), errors="coerce")
    df["Notional amount-Leg 2"] = pd.to_numeric(df["Notional amount-Leg 2"].str.replace(",", ""), errors="coerce")
    df["Notional Amount"] = df["Notional amount-Leg 1"].combine_first(df["Notional amount-Leg 2"])
    df["Fixed rate-Leg 1"] = pd.to_numeric(df["Fixed rate-Leg 1"], errors="coerce")
    df["Fixed rate-Leg 2"] = pd.to_numeric(df["Fixed rate-Leg 2"], errors="coerce")
    df["Fixed Rate"] = df["Fixed rate-Leg 1"].combine_first(df["Fixed rate-Leg 2"])
    df["Direction"] = df.apply(
        lambda row: "receive" if pd.notna(row["Fixed rate-Leg 1"]) else ("pay" if pd.notna(row["Fixed rate-Leg 2"]) else None), axis=1
    )

    # let ignore negative rates
    df = df[df["Fixed Rate"] > 0]
    df = df[(df["Notional Amount"] > 0) & (df["Notional Amount"].notna())]

    df = df.drop(
        columns=[
            "Fixed rate-Leg 1",
            "Fixed rate-Leg 2",
            "Expiry (d)",
            "Expiry (m)",
            "Expiry (yr)",
        ]
    )

    if tenors_to_interpolate is not None:
        print("Tenors that will be Interpolated: ", tenors_to_interpolate) if verbose else None
        df = df[~df["Tenor"].isin(set(tenors_to_interpolate))]

    return df[swap_columns]


def format_swap_ohlc(
    df: pd.DataFrame,
    as_of_date: datetime,
    overnight_rate: Optional[float] = None,
    minimum_time_and_sales_trades: int = 100,
    dtcc_interp_func: Optional[Callable] = scipy_linear_interp_func,
    default_tenors=DEFAULT_SWAP_TENORS,
    filter_extreme_time_and_sales=False,
    quantile_smoothing_range: Optional[Tuple[float, float]] = None,
    specifc_tenor_quantile_smoothing_range: Optional[Dict[str, Tuple[float, float]]] = None,
    remove_tenors: Optional[List[str]] = None,
    t_plus_another_one=False,
    ny_hours=False,
    verbose=False,
):
    filtered_df = df[(df["Fwd"] == "0D")]

    if len(filtered_df) < minimum_time_and_sales_trades:
        t_plus_another_one = True

    if filtered_df.empty or t_plus_another_one:
        (
            print(f'Initial OHLC Filtering Results: {len(filtered_df)} < {minimum_time_and_sales_trades} --- Enabled "t_plus_another_one"')
            if verbose
            else None
        )
        filtered_df = df[(df["Fwd"] == "0D") | (df["Fwd"] == "1D")]

    if filtered_df.empty or len(filtered_df) < minimum_time_and_sales_trades:
        (
            print(f'Initial OHLC Filtering Results: {len(filtered_df)} < {minimum_time_and_sales_trades} --- Enabled "t_plus_another_two"')
            if verbose
            else None
        )
        filtered_df = df[(df["Fwd"] == "0D") | (df["Fwd"] == "1D") | (df["Fwd"] == "2D")]

    if filtered_df.empty or len(filtered_df) < minimum_time_and_sales_trades:
        (
            print(f'Initial OHLC Filtering Results: {len(filtered_df)} < {minimum_time_and_sales_trades} --- Enabled "t_plus_another_three"')
            if verbose
            else None
        )
        filtered_df = df[(df["Fwd"] == "0D") | (df["Fwd"] == "1D") | (df["Fwd"] == "2D") | (df["Fwd"] == "3D")]

    if filtered_df.empty or len(filtered_df) < minimum_time_and_sales_trades:
        (
            print(f'Initial OHLC Filtering Results: {len(filtered_df)} < {minimum_time_and_sales_trades} --- Enabled "t_plus_another_four"')
            if verbose
            else None
        )
        filtered_df = df[(df["Fwd"] == "0D") | (df["Fwd"] == "1D") | (df["Fwd"] == "2D") | (df["Fwd"] == "3D") | (df["Fwd"] == "4D")]

    if filter_extreme_time_and_sales and quantile_smoothing_range:
        filtered_df = filtered_df.groupby(["Tenor", "Fwd"], group_keys=False).apply(
            lambda group: group[
                (group["Fixed Rate"] > group["Fixed Rate"].quantile(quantile_smoothing_range[0]))
                & (group["Fixed Rate"] < group["Fixed Rate"].quantile(quantile_smoothing_range[1]))
            ]
        )

    if specifc_tenor_quantile_smoothing_range:
        for tenor, quantile_range in specifc_tenor_quantile_smoothing_range.items():
            lower_quantile = filtered_df[filtered_df["Tenor"] == tenor]["Fixed Rate"].quantile(quantile_range[0])
            upper_quantile = filtered_df[filtered_df["Tenor"] == tenor]["Fixed Rate"].quantile(quantile_range[1])
            filtered_df = filtered_df[
                ~((filtered_df["Tenor"] == tenor) & ((filtered_df["Fixed Rate"] < lower_quantile) | (filtered_df["Fixed Rate"] > upper_quantile)))
            ]

    if remove_tenors:
        for rm_tenor in remove_tenors:
            filtered_df = filtered_df[filtered_df["Tenor"] != rm_tenor]

    filtered_df = filtered_df.sort_values(by=["Tenor", "Execution Timestamp", "Fixed Rate"], ascending=True)

    if ny_hours:
        filtered_df = filtered_df[filtered_df["Execution Timestamp"].dt.hour >= 12]
        filtered_df = filtered_df[filtered_df["Execution Timestamp"].dt.hour <= 23]

    tenor_df = (
        filtered_df.groupby("Tenor")
        .agg(
            Open=("Fixed Rate", "first"),
            High=("Fixed Rate", "max"),
            Low=("Fixed Rate", "min"),
            Close=("Fixed Rate", "last"),
            VWAP=(
                "Fixed Rate",
                lambda x: (x * filtered_df.loc[x.index, "Notional Amount"]).sum() / filtered_df.loc[x.index, "Notional Amount"].sum(),
            ),
        )
        .reset_index()
    )

    tenor_df["Tenor"] = pd.Categorical(tenor_df["Tenor"], categories=sorted(tenor_df["Tenor"], key=lambda x: (x[-1], int(x[:-1]))))
    tenor_df["Expiry"] = [rl.add_tenor(as_of_date + BDay(2), _, "F", "nyc") for _ in tenor_df["Tenor"]]
    tenor_df = tenor_df.sort_values("Expiry").reset_index(drop=True)
    tenor_df = tenor_df[["Tenor", "Open", "High", "Low", "Close", "VWAP"]]
    tenor_df = tenor_df[tenor_df["Tenor"].isin(default_tenors)].reset_index(drop=True)

    if dtcc_interp_func is not None:
        x = np.array([tenor_to_years(t) for t in tenor_df["Tenor"]])
        x_new = np.array([tenor_to_years(t) for t in default_tenors])
        interp_df = pd.DataFrame({"Tenor": default_tenors})

        tenor_map = {
            tenor_to_years(t): {col: tenor_df.loc[tenor_df["Tenor"] == t, col].values[0] for col in ["Open", "High", "Low", "Close", "VWAP"]}
            for t in tenor_df["Tenor"]
        }

        for col in ["Open", "High", "Low", "Close", "VWAP"]:
            y = tenor_df[col].values

            if overnight_rate is not None:
                x = np.array([1 / 360] + [tenor_to_years(t) for t in tenor_df["Tenor"]])
                x_new = np.array([tenor_to_years(t) for t in default_tenors])
                y = [overnight_rate] + list(y)
                y_new = dtcc_interp_func(x, y)(x_new)
            else:
                y_new = dtcc_interp_func(x, y)(x_new)

            interp_df[col] = [
                tenor_map[tenor_to_years(t)][col] if tenor_to_years(t) in tenor_map else y_val for t, y_val in zip(default_tenors, y_new)
            ]

        interp_df.insert(1, "Expiry", [rl.add_tenor(as_of_date + BDay(2), _, "F", "nyc") for _ in interp_df["Tenor"]])
        interp_df = interp_df.sort_values("Expiry").reset_index(drop=True)

        return interp_df

    return tenor_df
