import asyncio
import multiprocessing as mp
import warnings
from datetime import datetime
from functools import partial, reduce
from typing import Dict, List, Optional

import httpx
import numpy as np
import pandas as pd

from DataFetcher.base import DataFetcherBase
from DataFetcher.bondsupermart import BondSupermartDataFetcher
from DataFetcher.fedinvest import FedInvestDataFetcher
from DataFetcher.finra import FinraDataFetcher
from DataFetcher.fred import FredDataFetcher
from DataFetcher.nyfrb import NYFRBDataFetcher
from DataFetcher.public_dotcom import PublicDotcomDataFetcher
from DataFetcher.ust import USTreasuryDataFetcher
from DataFetcher.wsj import WSJDataFetcher

from utils.QL_BondPricer import QL_BondPricer
from utils.RL_BondPricer import RL_BondPricer
from utils.utils import (
    get_active_cusips,
    get_last_n_off_the_run_cusips,
    is_valid_ust_cusip,
    ust_labeler,
    ust_sorter,
)

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


class CurveDataFetcher():
    ust_data_fetcher: USTreasuryDataFetcher = None
    fedinvest_data_fetcher: FedInvestDataFetcher = None
    nyfrb_data_fetcher: NYFRBDataFetcher = None
    publicdotcom_data_fetcher: PublicDotcomDataFetcher = None
    fred_data_fetcher: FredDataFetcher = None
    finra_data_fetcher: FinraDataFetcher = None
    bondsupermart_fetcher: BondSupermartDataFetcher = None
    wsj_data_fetcher: WSJDataFetcher = None
    
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
    
    def async_runner(self, tasks):
        pass

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
            print(
                f"crystal ball feature not implemented, yet - {as_of_date} is in the future"
            )
            return

        if use_github or use_public_dotcom:
            calc_ytms = False

        if market_cols_to_return:
            if "cusip" not in market_cols_to_return:
                market_cols_to_return.insert(0, "cusip")

        quote_type = (
            market_cols_to_return[1].split("_")[0] if market_cols_to_return else "eod"
        )
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

            if (
                uid == "ust_prices"
                or uid == "soma_holdings"
                or uid == "ust_stripping"
                or uid == "ust_outstanding_amt"
            ):
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

            # ideally should neever get here
            self._logger.warning(f"CURVE SET - unknown UID, Current Tuple: {tup}")

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
        auctions_df["is_on_the_run"] = auctions_df["cusip"].isin(
            otr_cusips_df["cusip"].to_list()
        )
        auctions_df["label"] = auctions_df.apply(lambda row: ust_labeler(row), axis=1)
        auctions_df["time_to_maturity"] = (
            auctions_df["maturity_date"] - as_of_date
        ).dt.days / 365

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
                "label",
                "security_term",
                "original_security_term",
            ]
            if include_corpus_cusip:
                default_auction_cols.append("corpus_cusip")
            auctions_df = auctions_df[default_auction_cols]

        merged_df = reduce(
            lambda left, right: pd.merge(left, right, on="cusip", how="outer"), dfs
        )
        if not use_public_dotcom:
            merged_df = pd.merge(
                left=auctions_df, right=merged_df, on="cusip", how="outer"
            )
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
            merged_df["parValue"] = pd.to_numeric(
                merged_df["parValue"], errors="coerce"
            ).fillna(0)
            merged_df["portion_stripped_amt"] = (
                pd.to_numeric(
                    merged_df["portion_stripped_amt"], errors="coerce"
                ).fillna(0)
                * 1000
            )
            merged_df["outstanding_amt"] = (
                pd.to_numeric(merged_df["outstanding_amt"], errors="coerce").fillna(0)
                * 1000
            )
            merged_df["free_float"] = (
                merged_df["outstanding_amt"]
                - merged_df["parValue"]
                - merged_df["portion_stripped_amt"]
            ) / 1_000_000

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
                                (
                                    item
                                    for item in market_cols_to_return
                                    if "yield" in item
                                ),
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
                merged_df["mid_price"] = (
                    merged_df["offer_price"] + merged_df["bid_price"]
                ) / 2
            else:
                merged_df["mid_yield"] = (
                    merged_df["offer_yield"] + merged_df["bid_yield"]
                ) / 2

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
        merged_df = merged_df[merged_df["original_security_term"].notna()]
        if sorted:
            merged_df["sort_key"] = merged_df["original_security_term"].apply(
                ust_sorter
            )
            merged_df = (
                merged_df.sort_values(by=["sort_key", "time_to_maturity"])
                .drop(columns="sort_key")
                .reset_index(drop=True)
            )

        if include_off_the_run_number:
            merged_df["rank"] = (
                merged_df.groupby("original_security_term")["time_to_maturity"].rank(
                    ascending=False, method="first"
                )
                - 1
            )

        return merged_df
