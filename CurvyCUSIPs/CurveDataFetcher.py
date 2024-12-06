import asyncio
import warnings
from typing import Dict, Optional

import pandas as pd

from CurvyCUSIPs.DataFetcher.anna_dsb import AnnaDSB_DataFetcher
from CurvyCUSIPs.DataFetcher.bbg_sef import BBGSEF_DataFetcher
from CurvyCUSIPs.DataFetcher.bondsupermart import BondSupermartDataFetcher
from CurvyCUSIPs.DataFetcher.dtcc import DTCCSDR_DataFetcher
from CurvyCUSIPs.DataFetcher.erisfutures import ErisFuturesDataFetcher
from CurvyCUSIPs.DataFetcher.fedinvest import FedInvestDataFetcher
from CurvyCUSIPs.DataFetcher.finra import FinraDataFetcher
from CurvyCUSIPs.DataFetcher.fred import FredDataFetcher
from CurvyCUSIPs.DataFetcher.nyfrb import NYFRBDataFetcher
from CurvyCUSIPs.DataFetcher.public_dotcom import PublicDotcomDataFetcher
from CurvyCUSIPs.DataFetcher.ust import USTreasuryDataFetcher
from CurvyCUSIPs.DataFetcher.wsj import WSJDataFetcher
from CurvyCUSIPs.DataFetcher.yf import YahooFinanceDataFetcher
from CurvyCUSIPs.utils.QL_BondPricer import QL_BondPricer
from CurvyCUSIPs.utils.RL_BondPricer import RL_BondPricer

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

import sys

if sys.platform == "win32":
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)


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
        use_ust_issue_date: Optional[bool] = True,
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

        if fred_api_key:
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

        self.eris_data_fetcher = ErisFuturesDataFetcher(
            global_timeout=global_timeout,
            proxies=proxies,
            debug_verbose=debug_verbose,
            info_verbose=info_verbose,
            error_verbose=error_verbose,
        )
