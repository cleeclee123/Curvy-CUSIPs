import asyncio
import multiprocessing as mp
import warnings
from datetime import datetime
from functools import partial, reduce
from typing import Dict, List, Optional, Tuple, Callable
import tqdm
import tqdm.asyncio

import httpx
import numpy as np
import pandas as pd
import polars as pl
from pandas.tseries.offsets import CustomBusinessDay, BDay
from pandas.tseries.holiday import USFederalHolidayCalendar
from scipy.optimize import newton

from CurvyCUSIPs.CurveInterpolator import GeneralCurveInterpolator
from CurvyCUSIPs.DataFetcher.bondsupermart import BondSupermartDataFetcher
from CurvyCUSIPs.DataFetcher.fedinvest import FedInvestDataFetcher
from CurvyCUSIPs.DataFetcher.finra import FinraDataFetcher
from CurvyCUSIPs.DataFetcher.fred import FredDataFetcher
from CurvyCUSIPs.DataFetcher.nyfrb import NYFRBDataFetcher
from CurvyCUSIPs.DataFetcher.public_dotcom import PublicDotcomDataFetcher
from CurvyCUSIPs.DataFetcher.ust import USTreasuryDataFetcher
from CurvyCUSIPs.DataFetcher.wsj import WSJDataFetcher
from CurvyCUSIPs.DataFetcher.yf import YahooFinanceDataFetcher
from CurvyCUSIPs.DataFetcher.bbg_sef import BBGSEF_DataFetcher
from CurvyCUSIPs.DataFetcher.dtcc import DTCCSDR_DataFetcher
from CurvyCUSIPs.utils.QL_BondPricer import QL_BondPricer
from CurvyCUSIPs.utils.RL_BondPricer import RL_BondPricer
from CurvyCUSIPs.utils.ust_utils import get_active_cusips, get_last_n_off_the_run_cusips, is_valid_ust_cusip, ust_sorter, NoneReturningSpline
from CurvyCUSIPs.DataFetcher.anna_dsb import AnnaDSB_DataFetcher

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

import sys

if sys.platform == "win32":
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)


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


class CurveDataFetcher:
    ust_data_fetcher: USTreasuryDataFetcher = None
    fedinvest_data_fetcher: FedInvestDataFetcher = None
    nyfrb_data_fetcher: NYFRBDataFetcher = None
    publicdotcom_data_fetcher: PublicDotcomDataFetcher = None
    fred_data_fetcher: FredDataFetcher = None
    finra_data_fetcher: FinraDataFetcher = None
    bondsupermart_fetcher: BondSupermartDataFetcher = None
    wsj_data_fetcher: WSJDataFetcher = None
    yf_data_fetcher: YahooFinanceDataFetcher = None
    bbg_sef_fetcher: BBGSEF_DataFetcher = None
    dtcc_sdr_fetcher: DTCCSDR_DataFetcher = None
    anna_dsb_fetcher: AnnaDSB_DataFetcher = None

    def __init__(
        self,
        use_ust_issue_date: Optional[bool] = False,
        fred_api_key: Optional[str] = None,
        global_timeout: int = 10,
        proxies: Optional[Dict[str, str]] = None,
        debug_verbose: Optional[bool] = False,
        info_verbose: Optional[bool] = False,
        error_verbose: Optional[bool] = False,
    ):
        self.ust_data_fetcher = USTreasuryDataFetcher(
            use_ust_issue_date=use_ust_issue_date,
            global_timeout=global_timeout,
            proxies=proxies,
            debug_verbose=debug_verbose,
            info_verbose=info_verbose,
            error_verbose=error_verbose,
        )

        self.fedinvest_data_fetcher = FedInvestDataFetcher(
            global_timeout=global_timeout,
            proxies=proxies,
            debug_verbose=debug_verbose,
            info_verbose=info_verbose,
            error_verbose=error_verbose,
        )

        self.nyfrb_data_fetcher = NYFRBDataFetcher(
            global_timeout=global_timeout,
            proxies=proxies,
            debug_verbose=debug_verbose,
            info_verbose=info_verbose,
            error_verbose=error_verbose,
        )

        self.publicdotcom_data_fetcher = PublicDotcomDataFetcher(
            global_timeout=global_timeout,
            proxies=proxies,
            debug_verbose=debug_verbose,
            info_verbose=info_verbose,
            error_verbose=error_verbose,
        )

        self.fred_data_fetcher = FredDataFetcher(
            fred_api_key=fred_api_key,
            global_timeout=global_timeout,
            proxies=proxies,
            debug_verbose=debug_verbose,
            info_verbose=info_verbose,
            error_verbose=error_verbose,
        )

        self.finra_data_fetcher = FinraDataFetcher(
            global_timeout=global_timeout,
            proxies=proxies,
            debug_verbose=debug_verbose,
            info_verbose=info_verbose,
            error_verbose=error_verbose,
        )

        self.bondsupermart_fetcher = BondSupermartDataFetcher(
            global_timeout=global_timeout,
            proxies=proxies,
            debug_verbose=debug_verbose,
            info_verbose=info_verbose,
            error_verbose=error_verbose,
        )

        self.wsj_data_fetcher = WSJDataFetcher(
            global_timeout=global_timeout,
            proxies=proxies,
            debug_verbose=debug_verbose,
            info_verbose=info_verbose,
            error_verbose=error_verbose,
        )

        self.yf_data_fetcher = YahooFinanceDataFetcher(
            global_timeout=global_timeout,
            proxies=proxies,
            debug_verbose=debug_verbose,
            info_verbose=info_verbose,
            error_verbose=error_verbose,
        )
        
        self.bbg_sef_fetcher = BBGSEF_DataFetcher(
            global_timeout=global_timeout,
            proxies=proxies,
            debug_verbose=debug_verbose,
            info_verbose=info_verbose,
            error_verbose=error_verbose,
        )

        self.dtcc_sdr_fetcher = DTCCSDR_DataFetcher(
            global_timeout=global_timeout,
            proxies=proxies,
            debug_verbose=debug_verbose,
            info_verbose=info_verbose,
            error_verbose=error_verbose,
        )

        self.anna_dsb_fetcher = AnnaDSB_DataFetcher(
            global_timeout=global_timeout,
            proxies=proxies,
            debug_verbose=debug_verbose,
            info_verbose=info_verbose,
            error_verbose=error_verbose,
        )

    @staticmethod
    def par_bond_equation(c, maturity, zero_curve_func):
        discounted_cash_flows = sum((c / 2) * np.exp(-(zero_curve_func(t) / 100) * t) for t in np.arange(0.5, maturity + 0.5, 0.5))
        final_payment = 100 * np.exp(-(zero_curve_func(maturity) / 100) * maturity)
        return discounted_cash_flows + final_payment - 100

    @staticmethod
    def par_curve_func(tenor, zero_curve_func):
        init_guess = 4
        return newton(
            CurveDataFetcher.par_bond_equation,
            x0=init_guess,
            args=(tenor, zero_curve_func),
        )

    def async_runner(self, tasks):
        pass

    # one of cme_ust_labels, ust_label_spread, cusip_spread has to be defined
    # default delimtter "/"
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
            cusips_to_fetch = [self.ust_data_fetcher.ust_label_to_cusip(label.strip())["cusip"] for label in labels_to_fetch]
            spread_label = ust_label_spread
        if cme_ust_labels:
            labels_to_fetch = cme_ust_labels.split(spread_delimtter)
            cusips_to_fetch = [self.ust_data_fetcher.cme_ust_label_to_cusip(label.strip())["cusip"] for label in labels_to_fetch]
            spread_label = cme_ust_labels
        if cusip_spread:
            cusips_to_fetch = cusip_spread.split(spread_delimtter)
            spread_label = cusip_spread

        if len(cusips_to_fetch) < 2:
            return "not a valid spread"

        cusip_dict_df = self.fedinvest_data_fetcher.cusips_timeseries(cusips=cusips_to_fetch, start_date=start_date, end_date=end_date)

        yield_col = "eod_yield"
        if use_bid_side:
            yield_col = "bid_yield"
        elif use_offer_side:
            yield_col = "offer_yield"
        elif use_mid_side:
            yield_col = "mid_yield"

        dfs = [
            df[["Date", yield_col]].rename(columns={yield_col: f"{self.ust_data_fetcher.cusip_to_ust_label(key)}"})
            for key, df in cusip_dict_df.items()
        ]
        merged_df = reduce(lambda left, right: pd.merge(left, right, on="Date", how="outer"), dfs)
        if len(cusips_to_fetch) == 3:
            merged_df[spread_label] = (merged_df.iloc[:, 2] - merged_df.iloc[:, 1]) - ((merged_df.iloc[:, 3] - merged_df.iloc[:, 2]))
        else:
            merged_df[spread_label] = merged_df.iloc[:, 2] - merged_df.iloc[:, 1]

        return merged_df

    # max_concurrent_tasks limit applies to the logical tasks that your code wants to execute concurrently
    #    - ensures that we don’t attempt to start too many logical fetching operations at once
    # max_connections limit applies to the physical HTTP requests that can be made concurrently by httpx.AsyncClient
    #    - ensures only a certain number can actually initiate network connections concurrently
    # fitted_curves is tuple of size 3 (interp key, quote yield type, callable curve set df filter function)
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
            
        if calc_free_float:
            fetch_soma_holdings = True
            fetch_stripping_data = True

        async def gather_tasks(client: httpx.AsyncClient, dates: datetime, max_concurrent_tasks):
            my_semaphore = asyncio.Semaphore(max_concurrent_tasks)
            tasks = await self.fedinvest_data_fetcher._build_fetch_tasks_cusip_prices_from_github(
                client=client,
                dates=dates,
                uid="ust_prices",
                cols_to_return=(
                    [
                        "cusip",
                        "offer_price",
                        "offer_yield",
                        "bid_price",
                        "bid_yield",
                        "mid_price",
                        "mid_yield",
                        "eod_price",
                        "eod_yield",
                    ]
                ),
                my_semaphore=my_semaphore,
            )

            if fetch_soma_holdings:
                soma_bwd_date: pd.Timestamp = start_date - BDay(5)
                tasks += await self.nyfrb_data_fetcher._build_fetch_tasks_historical_soma_holdings(
                    client=client,
                    dates=[soma_bwd_date.to_pydatetime()] + dates,
                    uid="soma_holdings",
                    minimize_api_calls=True,
                    my_semaphore=my_semaphore,
                )

            if fetch_stripping_data:
                strips_bwd_date: pd.Timestamp = start_date - BDay(20)
                tasks += await self.ust_data_fetcher._build_fetch_tasks_historical_stripping_activity(
                    client=client,
                    dates=[strips_bwd_date.to_pydatetime()] + dates,
                    uid="ust_stripping",
                    minimize_api_calls=True,
                    my_semaphore=my_semaphore,
                )

            # return await asyncio.gather(*tasks)
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
        sorted_results = sorted(results, key=lambda x: x[0])

        auctions_df: pl.DataFrame = pl.from_pandas(self.ust_data_fetcher._historical_auctions_df.copy())
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

        last_seen_soma_holdings_df = None
        last_seen_stripping_act_df = None
        curveset_dict_df: Dict[datetime, List[pd.DataFrame]] = {}
        curveset_intrep_dict: Dict[datetime, Dict[str, GeneralCurveInterpolator]] = {}

        for tup in tqdm.tqdm(sorted_results, desc="AGGREGATING CURVE SET DFs"):
            curr_dt = tup[0]
            uid = tup[-1]

            if tup[1] is None or tup[1].empty:
                continue

            if uid == "soma_holdings":
                last_seen_soma_holdings_df = pl.from_pandas(tup[1])
                continue

            if uid == "ust_stripping":
                last_seen_stripping_act_df = pl.from_pandas(tup[1])
                continue

            if uid == "ust_prices":
                price_df = pl.from_pandas(tup[1])
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
                    curr_curve_set_df = curr_curve_set_df.sort_values(by=["sort_key", "time_to_maturity"]).drop(columns="sort_key").reset_index(drop=True)
                
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

    def build_curve_set(
        self,
        as_of_date: datetime,
        calc_ytms: Optional[bool] = True,
        use_quantlib: Optional[bool] = False,  # default is rateslib
        include_auction_results: Optional[bool] = False,
        include_soma_holdings: Optional[bool] = False,
        include_stripping_activity: Optional[bool] = False,
        # include_outstanding_amt: Optional[bool] = False,
        # exclude_nonmarketable_outstanding_amt: Optional[bool] = False,
        # auctions_df: Optional[pd.DataFrame] = None,
        sorted: Optional[bool] = False,
        use_github: Optional[bool] = False,
        use_public_dotcom: Optional[bool] = False,
        include_off_the_run_number: Optional[bool] = False,
        include_corpus_cusip: Optional[bool] = False,
        market_cols_to_return: List[str] = None,
        calc_free_float: Optional[bool] = False,
        calc_mod_duration: Optional[bool] = False,
    ):
        if as_of_date.date() > datetime.today().date():
            print(f"crystal ball feature not implemented, yet - {as_of_date} is in the future")
            return

        if use_github or use_public_dotcom:
            calc_ytms = False

        if market_cols_to_return:
            if "cusip" not in market_cols_to_return:
                market_cols_to_return.insert(0, "cusip")

        quote_type = market_cols_to_return[1].split("_")[0] if market_cols_to_return else "eod"
        filtered_free_float_df_col = False
        if calc_free_float:
            if not include_soma_holdings and not include_auction_results:
                filtered_free_float_df_col = True
                include_auction_results = True
                include_soma_holdings = True
                include_stripping_activity = True
                # include_outstanding_amt = True

        async def gather_tasks(client: httpx.AsyncClient, as_of_date: datetime):
            if use_github:
                ust_historical_prices_tasks = await self.fedinvest_data_fetcher._build_fetch_tasks_cusip_prices_from_github(
                    client=client,
                    dates=[as_of_date],
                    uid="ust_prices",
                    cols_to_return=(
                        [
                            "cusip",
                            "offer_price",
                            "offer_yield",
                            "bid_price",
                            "bid_yield",
                            "mid_price",
                            "mid_yield",
                            "eod_price",
                            "eod_yield",
                        ]
                        if not market_cols_to_return
                        else market_cols_to_return
                    ),
                )
            elif use_public_dotcom:
                cusips_to_fetch = get_active_cusips(
                    historical_auctions_df=self.ust_data_fetcher._historical_auctions_df,
                    as_of_date=as_of_date,
                    use_issue_date=True,
                )["cusip"].to_list()
                ust_historical_prices_tasks = await self.publicdotcom_data_fetcher._build_fetch_tasks_cusip_timeseries_public_dotcome(
                    client=client,
                    cusips=cusips_to_fetch,
                    start_date=as_of_date,
                    end_date=as_of_date,
                    uid="ust_prices_public_dotcom",
                )
            else:
                ust_historical_prices_tasks = await self.fedinvest_data_fetcher._build_fetch_tasks_cusip_prices_fedinvest(
                    client=client, dates=[as_of_date], uid="ust_prices"
                )

            tasks = ust_historical_prices_tasks

            if include_soma_holdings:
                tasks += await self.nyfrb_data_fetcher._build_fetch_tasks_historical_soma_holdings(
                    client=client, dates=[as_of_date], uid="soma_holdings"
                )
            if include_stripping_activity:
                tasks += await self.ust_data_fetcher._build_fetch_tasks_historical_stripping_activity(
                    client=client, dates=[as_of_date], uid="ust_stripping"
                )
            # if include_outstanding_amt:
            #     tasks += await self._build_fetch_tasks_historical_amount_outstanding(
            #         client=client,
            #         dates=[as_of_date],
            #         uid="ust_outstanding_amt",
            #         only_marketable=exclude_nonmarketable_outstanding_amt,
            #     )

            return await asyncio.gather(*tasks)

        async def run_fetch_all(as_of_date: datetime):
            limits = httpx.Limits(max_connections=10)
            async with httpx.AsyncClient(
                limits=limits,
            ) as client:
                all_data = await gather_tasks(client=client, as_of_date=as_of_date)
                return all_data

        results = asyncio.run(run_fetch_all(as_of_date=as_of_date))
        auctions_dfs = []
        dfs = []
        public_dotcom_dicts: List[Dict[str, str]] = []
        for tup in results:
            uid = tup[-1]
            if uid == "ust_auctions":
                auctions_dfs.append(tup[0])
                continue

            if uid == "ust_prices" or uid == "soma_holdings" or uid == "ust_stripping" or uid == "ust_outstanding_amt":
                dfs.append(tup[1])
                continue

            if uid == "ust_prices_public_dotcom" and use_public_dotcom:
                if not isinstance(tup[1], pd.DataFrame):
                    continue
                if not tup[1].empty:
                    public_dotcom_dicts.append(
                        {
                            "cusip": tup[0],
                            f"{quote_type}_price": tup[1].iloc[-1]["Price"],
                            f"{quote_type}_yield": tup[1].iloc[-1]["YTM"],
                        }
                    )
                continue

            # should neever get here
            print(f"CURVE SET - unknown UID, Current Tuple: {tup}")

        auctions_df = get_active_cusips(
            historical_auctions_df=self.ust_data_fetcher._historical_auctions_df,
            as_of_date=as_of_date,
            use_issue_date=True,
        )
        otr_cusips_df: pd.DataFrame = get_last_n_off_the_run_cusips(
            auctions_df=auctions_df,
            n=0,
            filtered=True,
            as_of_date=as_of_date,
            use_issue_date=self.ust_data_fetcher._use_ust_issue_date,
        )
        auctions_df["is_on_the_run"] = auctions_df["cusip"].isin(otr_cusips_df["cusip"].to_list())
        auctions_df["time_to_maturity"] = (auctions_df["maturity_date"] - as_of_date).dt.days / 365

        if not include_auction_results or filtered_free_float_df_col:
            default_auction_cols = [
                "cusip",
                "security_type",
                "auction_date",
                "issue_date",
                "maturity_date",
                "time_to_maturity",
                "int_rate",
                "high_investment_rate",
                "is_on_the_run",
                "ust_label",
                "security_term",
                "original_security_term",
            ]
            if include_corpus_cusip:
                default_auction_cols.append("corpus_cusip")
            auctions_df = auctions_df[default_auction_cols]

        merged_df = reduce(lambda left, right: pd.merge(left, right, on="cusip", how="outer"), dfs)
        if not use_public_dotcom:
            merged_df = pd.merge(left=auctions_df, right=merged_df, on="cusip", how="outer")
        else:
            market_df = pd.merge(
                left=auctions_df,
                right=pd.DataFrame(public_dotcom_dicts),
                on="cusip",
                how="outer",
            )
            merged_df = pd.merge(
                left=market_df,
                right=merged_df,
                on="cusip",
                how="outer",
            )
        merged_df = merged_df[merged_df["cusip"].apply(is_valid_ust_cusip)]

        if calc_free_float:
            merged_df["parValue"] = pd.to_numeric(merged_df["parValue"], errors="coerce").fillna(0)
            merged_df["portion_stripped_amt"] = pd.to_numeric(merged_df["portion_stripped_amt"], errors="coerce").fillna(0) * 1000
            merged_df["outstanding_amt"] = pd.to_numeric(merged_df["outstanding_amt"], errors="coerce").fillna(0) * 1000
            merged_df["free_float"] = (merged_df["outstanding_amt"] - merged_df["parValue"] - merged_df["portion_stripped_amt"]) / 1_000_000

        if calc_mod_duration:
            merged_df["mod_dur"] = merged_df.apply(
                lambda row: RL_BondPricer._bond_mod_duration(
                    issue_date=row["issue_date"],
                    maturity_date=row["maturity_date"],
                    as_of=as_of_date,
                    coupon=row["int_rate"] / 100,
                    ytm=(
                        row[
                            next(
                                (item for item in market_cols_to_return if "yield" in item),
                                None,
                            )
                        ]
                        if market_cols_to_return
                        else row["eod_yield"]
                    ),
                ),
                axis=1,
            )

        if not market_cols_to_return:
            if not use_github:
                merged_df["mid_price"] = (merged_df["offer_price"] + merged_df["bid_price"]) / 2
            else:
                merged_df["mid_yield"] = (merged_df["offer_yield"] + merged_df["bid_yield"]) / 2

        if calc_ytms:
            calculate_yields_partial = partial(calculate_yields, as_of_date=as_of_date, use_quantlib=use_quantlib)
            with mp.Pool(mp.cpu_count()) as pool:
                results = pool.map(calculate_yields_partial, [row for _, row in merged_df.iterrows()])
            offer_yields, bid_yields, eod_yields = zip(*results)
            merged_df["offer_yield"] = offer_yields
            merged_df["bid_yield"] = bid_yields
            merged_df["eod_yield"] = eod_yields
            merged_df["mid_yield"] = (merged_df["offer_yield"] + merged_df["bid_yield"]) / 2

        merged_df = merged_df.replace("null", np.nan)
        merged_df = merged_df[merged_df["original_security_term"].notna()]
        if sorted:
            merged_df["sort_key"] = merged_df["original_security_term"].apply(ust_sorter)
            merged_df = merged_df.sort_values(by=["sort_key", "time_to_maturity"]).drop(columns="sort_key").reset_index(drop=True)

        if include_off_the_run_number:
            merged_df["rank"] = merged_df.groupby("original_security_term")["time_to_maturity"].rank(ascending=False, method="first") - 1

        return merged_df
