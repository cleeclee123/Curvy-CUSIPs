import asyncio
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Literal
import httpx
import pandas as pd
from functools import reduce

from DataFetcher.base import DataFetcherBase
from DataFetcher.yf_legacy import (
    multi_download_historical_data_yahoofinance,
    download_historical_data_yahoofinance,
)

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

import sys

if sys.platform == "win32":
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)


class YahooFinanceDataFetcher(DataFetcherBase):
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

    def fetch_yf_legacy_multi_tickers(
        self,
        tickers: List[str],
        from_date: datetime,
        to_date: datetime,
        data_dump_dir: Optional[str] = None,
        max_date=False,
        big_wb=False,
    ):
        return multi_download_historical_data_yahoofinance(
            tickers=tickers,
            from_date=from_date,
            to_date=to_date,
            data_dump_dir=data_dump_dir,
            max_date=max_date,
            big_wb=big_wb,
        )

    def fetch_yf_legacy(
        self,
        ticker: str,
        from_date: datetime,
        to_date: datetime,
        raw_path: Optional[str] = None,
        ny_time=False
    ):
        return download_historical_data_yahoofinance(
            ticker=ticker,
            from_date=from_date,
            to_date=to_date,
            raw_path=raw_path,
            ny_time=ny_time
        )

    async def _fetch_cusip_timeseries_yahoofinance(
        self,
        client: httpx.AsyncClient,
        cusip: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: Optional[Literal["1m", "5m", "1h", "1d"]] = "1h",
        exchange: Optional[Literal["SG", "DU", "MU", "TI"]] = "SG",
        max_retries: Optional[int] = 3,
        backoff_factor: Optional[int] = 1,
        uid: Optional[str | int] = None,
    ):
        pass


# start_date = datetime(2024, 5, 1)
# end_date = datetime(2024, 5, 31)
# period1 = int(start_date.timestamp())
# period2 = int(end_date.timestamp())
# isin = "US912810TN81"
# interval = "1h"
# exchange = "SG"
# # SG, DU, MU, TI

# url = f"https://query1.finance.yahoo.com/v8/finance/chart/{isin}.{exchange}?period1={period1}&period2={period2}&interval={interval}"
# print(url)
# headers = {
#     "authority": "query1.finance.yahoo.com",
#     "method": "GET",
#     "path": url.split(".com")[1],
#     "scheme": "https",
#     "accept": "*/*",
#     "accept-encoding": "gzip, deflate, br, zstd",
#     "accept-language": "en-US,en;q=0.9",
#     "cache-control": "no-cache",
#     "dnt": "1",
#     "origin": "https://finance.yahoo.com",
#     "pragma": "no-cache",
#     "priority": "u=1, i",
#     "referer": f"https://finance.yahoo.com/chart/{isin}.{exchange}",
#     "sec-ch-ua": '"Chromium";v="128", "Not;A=Brand";v="24", "Google Chrome";v="128"',
#     "sec-ch-ua-mobile": "?0",
#     "sec-ch-ua-platform": '"Windows"',
#     "sec-fetch-dest": "empty",
#     "sec-fetch-mode": "cors",
#     "sec-fetch-site": "same-site",
#     "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
#     "cookie": r"tbla_id=ce485000-d594-430a-bb0d-106789c83afc-tuctbe93c93; GUC=AQEBCAFm2Z5nAEImMwVs&s=AQAAAAnXOqqR&g=ZthS2Q; A1=d=AQABBA8r7GQCEDl4-HKj5TViLm4-h2a8b8kFEgEBCAGe2WYAZ9w00iMA_eMBAAcIDyvsZGa8b8k&S=AQAAAn93mq3xN8wdsunKFpQ5nYM; A3=d=AQABBA8r7GQCEDl4-HKj5TViLm4-h2a8b8kFEgEBCAGe2WYAZ9w00iMA_eMBAAcIDyvsZGa8b8k&S=AQAAAn93mq3xN8wdsunKFpQ5nYM; A1S=d=AQABBA8r7GQCEDl4-HKj5TViLm4-h2a8b8kFEgEBCAGe2WYAZ9w00iMA_eMBAAcIDyvsZGa8b8k&S=AQAAAn93mq3xN8wdsunKFpQ5nYM; cmp=t=1725546450&j=0&u=1YNN; gpp=DBAA; gpp_sid=-1; PRF=t%3DUS912810UD80.SG%252BUS912810UD8.MU%252BUS912810SX72.SG%252BUS912810UA42.SG%252BUS912810UA4.DU%252B0981.HK%252BUS91282CFG1.BE%252BAMD%252BNVDA%252BJAAA%252BAIR.PA%252BGOVZ%252BIBIT%252BSPY%252BZQQ25.CBT%26newChartbetateaser%3D1%26qke-neo%3Dtrue; axids=gam=y-3cws5sFE2uJzBohwqJrpVs8yxO8XqjLZ~A&dv360=eS1jR0dpUEg1RTJ1RjNjaHF5dnJrckxVVjVhdWJLbG9YY35B&ydsp=y-JF0nspRE2uIqqGlhdNosoSr9syGScSMb~A&tbla=y-gyLskotE2uIge1eqJgtYnhOxvNbazrR3~A",
# }

# res = requests.get(url, headers=headers)
# if res.ok:
#     data = res.json()["chart"]["result"][0]
#     df = pd.DataFrame(
#         {
#             "timestamp": data["timestamp"],
#             "high": data["indicators"]["quote"][0]["high"],
#             "low": data["indicators"]["quote"][0]["low"],
#             "open": data["indicators"]["quote"][0]["open"],
#             "close": data["indicators"]["quote"][0]["close"],
#             "volume": data["indicators"]["quote"][0]["volume"],
#         }
#     )
#     df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, unit="s")
#     df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")

# else:
#     print("rip: ", res.status_code)
#     print(res.content)
