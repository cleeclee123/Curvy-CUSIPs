import logging
from datetime import datetime
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import QuantLib as ql
import tqdm

from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import BDay, CustomBusinessDay

from termcolor import colored

from CurvyCUSIPs.utils.ShelveDBWrapper import ShelveDBWrapper
from CurvyCUSIPs.utils.dtcc_swaps_utils import DEFAULT_SWAP_TENORS, datetime_to_ql_date


class S490Swaps:
    s490_nyclose_db: ShelveDBWrapper = None

    _logger = logging.getLogger(__name__)
    _debug_verbose: bool = False
    _error_verbose: bool = False
    _info_verbose: bool = False
    _no_logs_plz: bool = False

    def __init__(
        self,
        s490_curve_db_path: str,
        debug_verbose: Optional[bool] = False,
        info_verbose: Optional[bool] = False,
        error_verbose: Optional[bool] = False,
    ):
        self.setup_s490_nyclose_db(s490_curve_db_path=s490_curve_db_path)

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

    def setup_s490_nyclose_db(self, s490_curve_db_path: str):
        self.s490_nyclose_db = ShelveDBWrapper(s490_curve_db_path) if s490_curve_db_path else None
        self.s490_nyclose_db.open()

        most_recent_db_dt = datetime.fromtimestamp(int(max(self.s490_nyclose_db.keys())))
        self._logger.info(f"Most recent date in db: {most_recent_db_dt}")
        if ((datetime.today() - BDay(1)) - most_recent_db_dt).days >= 1:
            print(
                colored(
                    f"{s490_curve_db_path} is behind --- cd into 'scripts' and run 'update_sofr_ois_db.py' to update --- most recent date in db: {most_recent_db_dt}",
                    "yellow",
                )
            )

    def s490_nyclose_term_structure_ts(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        bdates = pd.date_range(start=start_date, end=end_date, freq=CustomBusinessDay(calendar=USFederalHolidayCalendar()))
        ts_term_structures = []
        for curr_date in bdates:
            try:
                str_ts = str(int(curr_date.to_pydatetime().timestamp()))
                ohlc_df = pd.DataFrame(self.s490_nyclose_db.get(str_ts)["ohlc"])
                curr_term_structure = {"Date": curr_date}
                curr_term_structure = curr_term_structure | dict(zip(DEFAULT_SWAP_TENORS, ohlc_df["Close"] * 100))
                ts_term_structures.append(curr_term_structure)
            except Exception as e:
                self._logger.error(colored(f'"s490_nyclose_term_structure_ts" Something went wrong at {curr_date}: {e}'), "red")

        df = pd.DataFrame(ts_term_structures)
        df = df.set_index("Date")
        return df

    @staticmethod
    def swap_spreads_term_structure(swaps_term_structure_ts_df: pd.DataFrame, cash_term_structure_ts_df: pd.DataFrame, is_cmt=False):
        CT_TENORS = ["CT3M", "CT6M", "CT1", "CT2", "CT3", "CT5", "CT7", "CT10", "CT20", "CT30"]
        CMT_TENORS = ["CMT3M", "CMT6M", "CMT1", "CMT2", "CMT3", "CMT5", "CMT7", "CMT10", "CMT20", "CMT30"]
        SWAP_TENORS = ["3M", "6M", "12M", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
        aligned_index = swaps_term_structure_ts_df.index.intersection(cash_term_structure_ts_df.index)
        swaps_aligned = swaps_term_structure_ts_df.loc[aligned_index, SWAP_TENORS]
        cash_aligned = cash_term_structure_ts_df.loc[aligned_index, CMT_TENORS if is_cmt else CT_TENORS]
        swap_spreads = (swaps_aligned.subtract(cash_aligned.values, axis=0)) * 100
        return swap_spreads

    def s490_nyclose_fwd_curve_matrices(
        self,
        start_date: datetime,
        end_date: datetime,
        fwd_tenors: Optional[List[str]] = ["1D", "1W", "1M", "2M", "3M", "6M", "12M", "18M", "2Y", "3Y", "5Y", "10Y", "15Y"],
        swaption_time_and_sales_dict_for_fwds: Optional[Dict[datetime, pd.DataFrame]] = None,
        ql_piecewise_method: Literal[
            "logLinearDiscount", "logCubicDiscount", "linearZero", "cubicZero", "linearForward", "splineCubicDiscount"
        ] = "logLinearDiscount",
        ql_zero_curve_method: Optional[ql.ZeroCurve] = ql.ZeroCurve,
        ql_compounding=ql.Compounded,
        ql_compounding_freq=ql.Daily,
        use_ql_implied_ts: Optional[bool] = True,
    ) -> Tuple[Dict[datetime, pd.DataFrame], Dict[datetime, ql.DiscountCurve | ql.ZeroCurve | ql.ForwardCurve]]:
        bdates = pd.date_range(start=start_date, end=end_date, freq=CustomBusinessDay(calendar=USFederalHolidayCalendar()))
        fwd_term_structure_grids: Dict[datetime, pd.DataFrame] = {}
        ql_curves: Dict[datetime, ql.DiscountCurve | ql.ZeroCurve | ql.ForwardCurve] = {}
        for curr_date in tqdm.tqdm(bdates, "Building Implied Fwd Curves..."):
            try:
                str_ts = str(int(curr_date.to_pydatetime().timestamp()))
                ql_piecewise_term_struct_nodes: Dict = self.s490_nyclose_db.get(str_ts)[ql_piecewise_method]
                if "Discount" in ql_piecewise_method:
                    ql_curve = ql.DiscountCurve(
                        [datetime_to_ql_date(mat) for mat in ql_piecewise_term_struct_nodes.keys()],
                        ql_piecewise_term_struct_nodes.values(),
                        ql.Actual360(),
                        ql.UnitedStates(ql.UnitedStates.GovernmentBond),
                    )
                elif "Zero" in ql_piecewise_method:
                    ql_curve = ql_zero_curve_method(
                        [datetime_to_ql_date(mat) for mat in ql_piecewise_term_struct_nodes.keys()],
                        ql_piecewise_term_struct_nodes.values(),
                        ql.Actual360(),
                        ql.UnitedStates(ql.UnitedStates.GovernmentBond),
                    )
                elif "Forward" in ql_piecewise_method:
                    ql_curve = ql.ForwardCurve(
                        [datetime_to_ql_date(mat) for mat in ql_piecewise_term_struct_nodes.keys()],
                        ql_piecewise_term_struct_nodes.values(),
                        ql.Actual360(),
                        ql.UnitedStates(ql.UnitedStates.GovernmentBond),
                    )
                else:
                    raise ValueError("Bad ql_piecewise_method method passed in")

                ohlc_df = pd.DataFrame(self.s490_nyclose_db.get(str_ts)["ohlc"])
                ohlc_df["Expiry"] = pd.to_numeric(ohlc_df["Expiry"], errors="coerce")
                ohlc_df["Expiry"] = pd.to_datetime(ohlc_df["Expiry"], errors="coerce", unit="s")
                dict_for_df = {"Tenor": DEFAULT_SWAP_TENORS, "Fixed Rate": ohlc_df["Close"] * 100}

                ql_curve.enableExtrapolation()
                ql.Settings.instance().evaluationDate = datetime_to_ql_date(curr_date)
                ql_curves[curr_date] = ql_curve

                if swaption_time_and_sales_dict_for_fwds is not None:
                    swaption_ts_df = swaption_time_and_sales_dict_for_fwds[curr_date.to_pydatetime()]
                    fwd_tenors = swaption_ts_df["Option Tenor"].unique()

                for fwd_tenor in fwd_tenors:
                    fwd_start_date = ql.UnitedStates(ql.UnitedStates.GovernmentBond).advance(datetime_to_ql_date(curr_date), ql.Period(fwd_tenor))

                    ql_curve_to_use = ql_curve
                    if use_ql_implied_ts:
                        implied_ts = ql.ImpliedTermStructure(ql.YieldTermStructureHandle(ql_curve), fwd_start_date)
                        implied_ts.enableExtrapolation()
                        ql_curve_to_use = implied_ts

                    fwds_list = []
                    for underlying_tenor in ohlc_df["Tenor"]:
                        try:
                            fwd_end_date = ql.UnitedStates(ql.UnitedStates.GovernmentBond).advance(fwd_start_date, ql.Period(underlying_tenor))
                            forward_rate = ql_curve_to_use.forwardRate(
                                fwd_start_date, fwd_end_date, ql.Actual360(), ql_compounding, ql_compounding_freq, True
                            ).rate()
                            fwds_list.append(forward_rate * 100)
                        except Exception as e:
                            self._logger.error(f"Error computing forward rate on {curr_date.date()} for {fwd_tenor}x{underlying_tenor}: {e}")
                            fwds_list.append(float("nan"))

                    dict_for_df[f"{fwd_tenor} Fwd"] = fwds_list

                fwd_term_structure_grids[curr_date] = pd.DataFrame(dict_for_df)

            except Exception as e:
                self._logger.error(colored(f'"s490_nyclose_fwd_grid_term_structures" Something went wrong at {curr_date}: {e}'), "red")
                print(e)

        return fwd_term_structure_grids, ql_curves
