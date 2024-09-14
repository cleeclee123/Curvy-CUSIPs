from datetime import datetime
from typing import Dict, List, Optional

from DataFetcher.base import DataFetcherBase

from utils.fred import Fred


class FredDataFetcher(DataFetcherBase):
    fred: Fred = None

    def __init__(
        self,
        fred_api_key: Optional[str] = None,
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

        self.fred = Fred(api_key=fred_api_key, proxies=self._proxies)

    def get_historical_cmt_yields(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tenors: Optional[List[str]] = None,
    ):
        print("Fetching from FRED...")
        df = self.fred.get_multiple_series(
            series_ids=[
                "DTB3",
                "DTB6",
                "DGS1",
                "DGS2",
                "DGS3",
                "DGS5",
                "DGS7",
                "DGS10",
                "DGS20",
                "DGS30",
            ],
            one_df=True,
            observation_start=start_date,
            observation_end=end_date,
        )
        df.columns = [
            "CMT3M",
            "CMT6M",
            "CMT1",
            "CMT2",
            "CMT3",
            "CMT5",
            "CMT7",
            "CMT10",
            "CMT20",
            "CMT30",
        ]
        if tenors:
            tenors = ["Date"] + tenors
            return df[tenors]
        df = df.dropna()
        df = df.rename_axis("Date").reset_index()
        return df
