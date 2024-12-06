import asyncio
import logging
import math
import multiprocessing as mp
from datetime import datetime
from fractions import Fraction
from functools import partial, reduce
from typing import Callable, Dict, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd
import polars as pl
import QuantLib as ql
import tqdm
import tqdm.asyncio
import ujson as json
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import BDay, CustomBusinessDay
from scipy.optimize import newton
from termcolor import colored

from CurvyCUSIPs.CurveDataFetcher import CurveDataFetcher
from CurvyCUSIPs.CurveInterpolator import GeneralCurveInterpolator
from CurvyCUSIPs.utils.ShelveDBWrapper import ShelveDBWrapper
from CurvyCUSIPs.utils.ust_utils import NoneReturningSpline, is_valid_ust_cusip, ust_sorter


class USTs:
    cusip_set_db: ShelveDBWrapper = None
    cusip_timeseries_db: ShelveDBWrapper = None
    ct_yields_db: Dict[str, ShelveDBWrapper] = None
    curve_data_fetcher: CurveDataFetcher = None

    _logger = logging.getLogger(__name__)
    _debug_verbose: bool = False
    _error_verbose: bool = False
    _info_verbose: bool = False
    _no_logs_plz: bool = False

    def __init__(
        self,
        curve_data_fetcher: CurveDataFetcher,
        cusip_set_db_path: str,
        cusip_timeseries_db_path: str,
        ct_eod_db_path: str,
        debug_verbose: Optional[bool] = False,
        info_verbose: Optional[bool] = False,
        error_verbose: Optional[bool] = False,
    ):
        self.curve_data_fetcher = curve_data_fetcher
        self.cusip_set_db = self.setup_db(cusip_set_db_path)
        self.cusip_timeseries_db = self.setup_db(cusip_timeseries_db_path, run_check=False)
        self.ct_yields_db = {
            "eod_yield": self.setup_db(ct_eod_db_path, run_check=False),
            "bid_yield": self.setup_db(ct_eod_db_path.replace("eod", "bid"), run_check=False),
            "offer_yield": self.setup_db(ct_eod_db_path.replace("eod", "offer"), run_check=False),
            "mid_yield": self.setup_db(ct_eod_db_path.replace("eod", "mid"), run_check=False),
        }

        self._historical_auctions_df = self.curve_data_fetcher.ust_data_fetcher._historical_auctions_df

        self._debug_verbose = debug_verbose
        self._error_verbose = error_verbose
        self._info_verbose = info_verbose
        self._no_logs_plz = not debug_verbose and not error_verbose and not info_verbose
        self._setup_logger()

    def _setup_logger(self):
        if not self._logger.hasHandlers():
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            self._logger.addHandler(handler)

        if self._debug_verbose:
            self._logger.setLevel(logging.DEBUG)
        elif self._info_verbose:
            self._logger.setLevel(logging.INFO)
        elif self._error_verbose:
            self._logger.setLevel(logging.ERROR)
        else:
            self._logger.setLevel(logging.WARNING)

        if self._debug_verbose or self._info_verbose or self._error_verbose:
            self._logger.setLevel(logging.DEBUG)

        if self._no_logs_plz:
            self._logger.disabled = True
            self._logger.propagate = False

    def setup_db(self, db_path: str, run_check=True):
        db = ShelveDBWrapper(db_path)
        db.open()

        if run_check:
            most_recent_db_dt = datetime.strptime(max(db.keys()), "%Y-%m-%d")
            self._logger.info(f"Most recent date in db: {most_recent_db_dt}")
            if ((datetime.today() - BDay(1)) - most_recent_db_dt).days >= 1:
                print(
                    colored(
                        f"{db_path} is behind --- cd into 'scripts' and run 'update_ust_cusips_db.py' to update --- most recent date in db: {most_recent_db_dt}",
                        "yellow",
                    )
                )

        return db

    # ust label: f"{coupon (as a fraction)} {datetime.strftime("%m/%d/%Y")}"
    def cme_ust_label_to_cusip(self, ust_label: str):
        try:
            ust_label_split = ust_label.split(" ")
            if len(ust_label_split) == 2:
                coupon = float(ust_label_split[0])
                maturity = datetime.strptime(ust_label.split(" ")[1], "%m/%d/%Y")
            else:
                coupon = float(ust_label_split[0]) + float(Fraction(ust_label.split(" ")[1]))
                maturity = datetime.strptime(ust_label.split(" ")[2], "%m/%d/%Y")
            ust_row = self._historical_auctions_df[
                (self._historical_auctions_df["int_rate"] == coupon) & (self._historical_auctions_df["maturity_date"].dt.date == maturity.date())
            ]
            return ust_row.to_dict("records")[0]
        except:
            raise Exception("LABEL NOT FOUND")

    def cusip_to_cme_ust_label(self, cusip: str):
        ust_row = self._historical_auctions_df[self._historical_auctions_df["cusip"] == cusip].to_dict("records")[0]
        frac, whole = math.modf(ust_row["int_rate"])
        tup = frac.as_integer_ratio()
        return f"{int(whole)} {tup[0]}/{tup[1]} {ust_row["maturity_date"].strftime("%m/%d/%Y")}"

    # ust_label = f"{row['int_rate']:.3f}% {row['maturity_date'].strftime('%b-%y')}"
    def ust_label_to_cusip(self, ust_label: str, return_all_occurrences=False):
        try:
            ust_row = self._historical_auctions_df[self._historical_auctions_df["ust_label"] == ust_label]
            occurences = ust_row.to_dict("records")
            if return_all_occurrences:
                return occurences
            return occurences[-1]  # return the earliest issue date by default
        except Exception as e:
            if ust_row.empty:
                raise Exception(f"LABEL NOT FOUND - {e}")
            else:
                raise Exception(f"SOMETHING WENT WRONG - {e}")

    def cusip_to_ust_label(self, cusip: str, return_all_occurrences=False):
        try:
            ust_row = self._historical_auctions_df[self._historical_auctions_df["cusip"] == cusip]
            occurences = ust_row.to_dict("records")
            if return_all_occurrences:
                return occurences
            return occurences[-1]  # return the earliest issue date by default
        except Exception as e:
            if ust_row.empty:
                raise Exception(f"CUSIP NOT FOUND - {e}")
            else:
                raise Exception(f"SOMETHING WENT WRONG - {e}")

    def get_ust_cusip_sets(
        self, start_date: datetime, end_date: datetime, assume_otrs: Optional[bool] = False, set_cusips_as_index: Optional[bool] = False
    ) -> Dict[datetime, pd.DataFrame]:
        dict_df = {}
        bdates = pd.date_range(start=start_date, end=end_date, freq=CustomBusinessDay(calendar=USFederalHolidayCalendar()))
        for curr_date in bdates:
            try:
                df = pd.DataFrame(json.loads(self.cusip_set_db.get(curr_date.to_pydatetime().strftime("%Y-%m-%d")))["data"])
                date_cols = ["auction_date", "issue_date", "maturity_date"]
                for date_col in date_cols:
                    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

                if assume_otrs and "original_security_term" in df.columns:
                    df_coups = df.sort_values(by=["issue_date"], ascending=False)
                    df_coups = df_coups[(df_coups["security_type"] == "Note") | (df_coups["security_type"] == "Bond")]
                    df_coups = df_coups.groupby("original_security_term").first().reset_index()

                    df_bills = df.sort_values(by=["maturity_date"], ascending=True)
                    df_bills = df_bills[df_bills["security_type"] == "Bill"]
                    df_bills = df_bills.groupby("security_term").last().reset_index()

                    df_bills.loc[df_bills["security_term"] == "4-Week", "original_security_term"] = "4-Week"
                    df_bills.loc[df_bills["security_term"] == "8-Week", "original_security_term"] = "8-Week"
                    df_bills.loc[df_bills["security_term"] == "13-Week", "original_security_term"] = "13-Week"
                    df_bills.loc[df_bills["security_term"] == "17-Week", "original_security_term"] = "17-Week"
                    df_bills.loc[df_bills["security_term"] == "26-Week", "original_security_term"] = "26-Week"
                    df_bills.loc[df_bills["security_term"] == "52-Week", "original_security_term"] = "52-Week"

                    df = pd.concat([df_bills, df_coups])
                    cusip_to_term_dict = dict(zip(df["cusip"], df["original_security_term"]))
                    df["cusip"] = df["cusip"].replace(cusip_to_term_dict)

                if set_cusips_as_index:
                    df = df.set_index("cusip")

                dict_df[curr_date.to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0)] = df
            except Exception as e:
                self._logger.error(f"Error occured in 'get_ust_cusip_sets' at {curr_date.date()}: {str(e)}")

        return dict_df

    def get_ust_timeseries_by_cusips(
        self,
        start_date: datetime,
        end_date: datetime,
        cusips: List[str],
        cusip_cols: List[str] = ["eod_yield", "eod_price"],
        use_dict_key_df_cols: Optional[bool] = False,
        df_col_delimitter: Optional[str] = "-",
        return_dict: Optional[bool] = False,
    ) -> Dict[datetime, pd.DataFrame] | pd.DataFrame:
        cusip_timeseries_dict = {}
        for cusip in cusips:
            try:
                df = pd.DataFrame(json.loads(self.cusip_timeseries_db.get(cusip)))
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df = df[(df["Date"].dt.date >= start_date.date()) & (df["Date"].dt.date <= end_date.date())]
                cusip_timeseries_dict[cusip] = df
            except Exception as e:
                self._logger.error(f"Error occured in 'get_ust_timeseries_by_cusips' at {cusip}: {str(e)}")

        if return_dict:
            return cusip_timeseries_dict

        renamed_dfs = []
        for key, df in cusip_timeseries_dict.items():
            temp_df = df[["Date"] + cusip_cols].copy()
            if use_dict_key_df_cols:
                temp_df = temp_df.rename(columns={col: f"{key}" for col in temp_df.columns if col != "Date"})
            else:
                temp_df = temp_df.rename(columns={col: f"{key}{df_col_delimitter}{col}" for col in temp_df.columns if col != "Date"})
            renamed_dfs.append(temp_df)

        merged_df = reduce(lambda left, right: pd.merge(left, right, on="Date", how="outer"), renamed_dfs)
        return merged_df

    def fetch_spreads(
        self,
        start_date: datetime,
        end_date: datetime,
        use_bid_side: Optional[bool] = False,
        use_offer_side: Optional[bool] = False,
        use_mid_side: Optional[bool] = False,
        cme_ust_labels: Optional[str] = None,
        ust_label_spread: Optional[str] = None,
        cusip_spread: Optional[str] = None,
        spread_delimtter: Optional[str] = "/",
    ) -> pd.DataFrame:
        spread_label = ""
        if ust_label_spread:
            labels_to_fetch = ust_label_spread.split(spread_delimtter)
            cusips_to_fetch = [self.ust_label_to_cusip(label.strip())["cusip"] for label in labels_to_fetch]
            spread_label = ust_label_spread
        if cme_ust_labels:
            labels_to_fetch = cme_ust_labels.split(spread_delimtter)
            cusips_to_fetch = [self.cme_ust_label_to_cusip(label.strip())["cusip"] for label in labels_to_fetch]
            spread_label = cme_ust_labels
        if cusip_spread:
            cusips_to_fetch = cusip_spread.split(spread_delimtter)
            spread_label = cusip_spread

        if len(cusips_to_fetch) < 2:
            return "not a valid spread"

        yield_col = "eod_yield"
        if use_bid_side:
            yield_col = "bid_yield"
        elif use_offer_side:
            yield_col = "offer_yield"
        elif use_mid_side:
            yield_col = "mid_yield"

        cusip_dict_df = self.get_ust_timeseries_by_cusips(
            start_date=start_date, end_date=end_date, cusips=cusips_to_fetch, return_dict=True, cusip_cols=[yield_col]
        )
        dfs = [df[["Date", yield_col]].rename(columns={yield_col: f"{self.cusip_to_ust_label(key)}"}) for key, df in cusip_dict_df.items()]
        merged_df = reduce(lambda left, right: pd.merge(left, right, on="Date", how="outer"), dfs)
        if len(cusips_to_fetch) == 3:
            merged_df[spread_label] = (merged_df.iloc[:, 2] - merged_df.iloc[:, 1]) - ((merged_df.iloc[:, 3] - merged_df.iloc[:, 2]))
        else:
            merged_df[spread_label] = merged_df.iloc[:, 2] - merged_df.iloc[:, 1]

        return merged_df

    @staticmethod
    def tenor_to_years(tenor):
        if "CMT" in tenor:
            num = tenor.split("CMT")[1]
        elif "CT" in tenor:
            num = tenor.split("CT")[1]
        else:
            raise ValueError(f"Only support CT or CMT Tenors: ", tenor)

        if "M" in num:
            return int(num[:1]) / 30

        return int(num)

    def fetch_ct_yields(
        self,
        start_date: datetime,
        end_date: datetime,
        use_bid_side: Optional[bool] = False,
        use_offer_side: Optional[bool] = False,
        use_mid_side: Optional[bool] = False,
    ) -> pd.DataFrame:
        ytm_type = "eod_yield"
        if use_bid_side:
            ytm_type = "bid_yield"
        elif use_offer_side:
            ytm_type = "offer_yield"
        elif use_mid_side:
            ytm_type = "mid_yield"

        bdates = pd.date_range(start=start_date, end=end_date, freq=CustomBusinessDay(calendar=USFederalHolidayCalendar()))
        ct_yields_dict = {}
        for curr_bdate in bdates:
            try:
                ct_yields_dict[curr_bdate.to_pydatetime()] = self.ct_yields_db[ytm_type].get(curr_bdate.strftime("%Y-%m-%d"))
            except Exception as e:
                self._logger.error(f"Erorr fetching {curr_bdate} CT Yields: {e}")

        df = pd.DataFrame.from_dict(ct_yields_dict, orient="index")
        df.index.name = "Date"

        mapping = {
            # "4-Week": 0.077,
            # "8-Week": 0.15,
            "13-Week": 0.25,
            # "17-Week": 0.33,
            "26-Week": 0.5,
            "52-Week": 1,
            "2-Year": 2,
            "3-Year": 3,
            "5-Year": 5,
            "7-Year": 7,
            "10-Year": 10,
            "20-Year": 20,
            "30-Year": 30,
        }
        existing_cols = [col for col in df.columns if col in mapping]
        cols_sorted = sorted(existing_cols, key=lambda col: mapping[col])
        df = df[cols_sorted]
        df.columns = ["CT3M", "CT6M", "CT1", "CT2", "CT3", "CT5", "CT7", "CT10", "CT20", "CT30"]
        return df

    def fetch_historical_curve_sets(
        self,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        fetch_soma_holdings: Optional[bool] = False,
        fetch_stripping_data: Optional[bool] = False,
        calc_free_float: Optional[bool] = False,
        fitted_curves: Optional[List[Tuple[str, str, Callable]] | List[Tuple[str, str, Callable, Callable]]] = None,
        max_concurrent_tasks: Optional[int] = 128,
        max_connections: Optional[int] = 64,
        sorted_curve_set: Optional[bool] = False,
    ) -> Tuple[Dict[datetime, pd.DataFrame], Dict[datetime, Dict[str, GeneralCurveInterpolator]]]:
        if not end_date:
            end_date = start_date

        keys = set()
        if fetch_soma_holdings:
            keys.add("soma_holdings")
        if fetch_stripping_data:
            keys.add("ust_stripping")
        if calc_free_float:
            fetch_soma_holdings = True
            fetch_stripping_data = True
            keys.add("soma_holdings")
            keys.add("ust_stripping")
        known_keys = list(keys)

        async def gather_tasks(client: httpx.AsyncClient, dates: datetime, max_concurrent_tasks):
            my_semaphore = asyncio.Semaphore(max_concurrent_tasks)
            tasks = []

            if fetch_soma_holdings:
                soma_bwd_date: pd.Timestamp = start_date - BDay(5)
                tasks += await self.curve_data_fetcher.nyfrb_data_fetcher._build_fetch_tasks_historical_soma_holdings(
                    client=client,
                    dates=[soma_bwd_date.to_pydatetime()] + dates,
                    uid="soma_holdings",
                    minimize_api_calls=True,
                    my_semaphore=my_semaphore,
                )

            if fetch_stripping_data:
                strips_bwd_date: pd.Timestamp = start_date - BDay(20)
                tasks += await self.curve_data_fetcher.ust_data_fetcher._build_fetch_tasks_historical_stripping_activity(
                    client=client,
                    dates=[strips_bwd_date.to_pydatetime()] + dates,
                    uid="ust_stripping",
                    minimize_api_calls=True,
                    my_semaphore=my_semaphore,
                )

            return await tqdm.asyncio.tqdm.gather(*tasks, desc="FETCHING CURVE SETS...")

        async def run_fetch_all(dates: datetime, max_concurrent_tasks: int, max_connections: int):
            limits = httpx.Limits(max_connections=max_connections)
            async with httpx.AsyncClient(
                limits=limits,
            ) as client:
                all_data = await gather_tasks(client=client, dates=dates, max_concurrent_tasks=max_concurrent_tasks)
                return all_data

        bdates = [
            bday.to_pydatetime()
            for bday in pd.bdate_range(
                start=start_date,
                end=end_date,
                freq=CustomBusinessDay(calendar=USFederalHolidayCalendar()),
            )
        ]

        results: List[Tuple[datetime, pd.DataFrame, str]] = asyncio.run(
            run_fetch_all(dates=bdates, max_concurrent_tasks=max_concurrent_tasks, max_connections=max_connections)
        )
        grouped_results = {key: {} for key in known_keys}
        for dt, df, group_key in results:
            if group_key in grouped_results:
                grouped_results[group_key][dt] = df
            else:
                raise ValueError(f"Unexpected group key encountered: {group_key}")
        
        curve_sets = self.get_ust_cusip_sets(start_date=start_date, end_date=end_date) 

        auctions_df: pl.DataFrame = pl.from_pandas(self._historical_auctions_df.copy())
        auctions_df = auctions_df.filter(
            (pl.col("security_type") == "Bill") | (pl.col("security_type") == "Note") | (pl.col("security_type") == "Bond")
        )
        auctions_df = auctions_df.with_columns(
            pl.when(pl.col("original_security_term").str.contains("29-Year"))
            .then(pl.lit("30-Year"))
            .when(pl.col("original_security_term").str.contains("30-"))
            .then(pl.lit("30-Year"))
            .otherwise(pl.col("original_security_term"))
            .alias("original_security_term")
        )

        curveset_dict_df: Dict[datetime, List[pd.DataFrame]] = {}
        curveset_intrep_dict: Dict[datetime, Dict[str, GeneralCurveInterpolator]] = {}

        for curr_dt, curve_set_df in tqdm.tqdm(curve_sets.items(), desc="AGGREGATING CURVE SET DFs"):
            last_seen_soma_holdings_df = None
            last_seen_stripping_act_df = None
            
            fetched_soma_holdings_dates = [dt for dt in grouped_results["soma_holdings"].keys() if dt < curr_dt]
            if fetched_soma_holdings_dates:
                closest = max(fetched_soma_holdings_dates)  
                last_seen_soma_holdings_df = pl.from_pandas(grouped_results["soma_holdings"][closest])
            else:
                raise ValueError("Couldnt find valid SOMA holding dates fetched")

            fetched_ust_stripping_dates = [dt for dt in grouped_results["ust_stripping"].keys() if dt < curr_dt]
            if fetched_ust_stripping_dates:
                closest = max(fetched_ust_stripping_dates)  
                last_seen_stripping_act_df = pl.from_pandas(grouped_results["ust_stripping"][closest])
            else:
                raise ValueError("Couldnt find valid UST stripping dates fetched")

            price_df = pl.from_pandas(curve_set_df)
            curr_auctions_df = auctions_df.filter(
                (pl.col("issue_date").dt.date() <= curr_dt.date()) & (pl.col("maturity_date") >= curr_dt)
            ).unique(subset=["cusip"], keep="first")

            merged_df = curr_auctions_df.join(price_df, on="cusip", how="outer")

            if fetch_soma_holdings and last_seen_soma_holdings_df is not None:
                merged_df = merged_df.join(last_seen_soma_holdings_df, on="cusip", how="left")
            if fetch_stripping_data and last_seen_stripping_act_df is not None:
                merged_df = merged_df.join(last_seen_stripping_act_df, on="cusip", how="left")

            merged_df = merged_df.filter(pl.col("cusip").map_elements(is_valid_ust_cusip, return_dtype=pl.Boolean))
            merged_df = merged_df.with_columns(pl.col("maturity_date").cast(pl.Datetime).alias("maturity_date"))
            merged_df = merged_df.with_columns(((pl.col("maturity_date") - curr_dt).dt.total_days() / 365).alias("time_to_maturity"))
            merged_df = merged_df.with_columns(
                pl.col("time_to_maturity").rank(descending=True, method="ordinal").over("original_security_term").sub(1).alias("rank")
            )

            if calc_free_float:
                merged_df = merged_df.with_columns(
                    pl.col("parValue").cast(pl.Float64).fill_null(0).alias("parValue"),
                    (pl.col("portion_stripped_amt").cast(pl.Float64).fill_null(0) * 1000).alias("portion_stripped_amt"),
                    (
                        pl.when((pl.col("est_outstanding_amt").is_not_nan()) & (pl.col("est_outstanding_amt") != 0))
                        .then(pl.col("est_outstanding_amt"))
                        .otherwise(pl.col("outstanding_amt"))
                        .cast(pl.Float64)
                        .fill_null(0)
                        * 1000
                    ).alias("est_outstanding_amt"),
                )
                merged_df = merged_df.with_columns(
                    ((pl.col("est_outstanding_amt") - pl.col("parValue") - pl.col("portion_stripped_amt")) / 1_000_000).alias("free_float")
                )

            curr_curve_set_df = merged_df.to_pandas()
            if sorted_curve_set:
                curr_curve_set_df["sort_key"] = curr_curve_set_df["original_security_term"].apply(ust_sorter)
                curr_curve_set_df = (
                    curr_curve_set_df.sort_values(by=["sort_key", "time_to_maturity"]).drop(columns="sort_key").reset_index(drop=True)
                )

            curveset_dict_df[curr_dt] = curr_curve_set_df

            if fitted_curves:
                for curve_build_params in fitted_curves:
                    if len(curve_build_params) == 3:
                        if callable(curve_build_params[-1]):
                            curve_set_key, quote_type, filter_func = curve_build_params
                            curr_filtered_curve_set_df: pd.DataFrame = filter_func(curr_curve_set_df)

                            if curr_dt not in curveset_intrep_dict:
                                curveset_intrep_dict[curr_dt] = {}
                            if curve_set_key not in curveset_intrep_dict[curr_dt]:
                                curveset_intrep_dict[curr_dt][curve_set_key] = {}

                            curveset_intrep_dict[curr_dt][curve_set_key] = GeneralCurveInterpolator(
                                x=curr_filtered_curve_set_df["time_to_maturity"].to_numpy(), y=curr_filtered_curve_set_df[quote_type].to_numpy()
                            )

                    elif len(curve_build_params) == 4:
                        if callable(curve_build_params[-2]) and callable(curve_build_params[-2]):
                            curve_set_key, quote_type, filter_func, calibrate_func = curve_build_params
                            curr_filtered_curve_set_df: pd.DataFrame = filter_func(curr_curve_set_df)

                            if curr_dt not in curveset_intrep_dict:
                                curveset_intrep_dict[curr_dt] = {}
                            if curve_set_key not in curveset_intrep_dict[curr_dt]:
                                curveset_intrep_dict[curr_dt][curve_set_key] = {}

                            try:
                                curr_filtered_curve_set_df = (
                                    curr_filtered_curve_set_df[["time_to_maturity", quote_type]].dropna().sort_values(by="time_to_maturity")
                                )
                                parameteric_model = calibrate_func(
                                    curr_filtered_curve_set_df["time_to_maturity"].to_numpy(),
                                    curr_filtered_curve_set_df[quote_type].to_numpy(),
                                )
                                assert parameteric_model[1]
                                curveset_intrep_dict[curr_dt][curve_set_key] = parameteric_model[0]

                            except Exception as e:
                                # print(f"{curve_set_key} for {curr_dt} - {str(e)}")
                                curveset_intrep_dict[curr_dt][curve_set_key] = NoneReturningSpline(
                                    curr_filtered_curve_set_df["time_to_maturity"].to_numpy(),
                                    curr_filtered_curve_set_df[quote_type].to_numpy(),
                                )

        if fitted_curves:
            return curveset_dict_df, curveset_intrep_dict

        return curveset_dict_df
