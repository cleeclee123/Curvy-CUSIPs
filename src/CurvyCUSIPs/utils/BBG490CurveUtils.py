import asyncio
import warnings
from datetime import datetime
from typing import Dict, List, Literal, Optional, Tuple
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import QuantLib as ql
import scipy.optimize as optimize
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import BDay, CustomBusinessDay
from termcolor import colored
from pysabr import Hagan2002LognormalSABR as LognormalSABR
from pysabr import hagan_2002_lognormal_sabr as hagan2002LN

from CurvyCUSIPs.CurveDataFetcher import CurveDataFetcher
from CurvyCUSIPs.DataFetcher.erisfutures import ERIS_TENORS
from CurvyCUSIPs.DataFetcher.ShelveDBWrapper import ShelveDBWrapper
from CurvyCUSIPs.utils.dtcc_swaps_utils import DEFAULT_SWAP_TENORS, datetime_to_ql_date, tenor_to_ql_period, tenor_to_years


warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

import sys

if sys.platform == "win32":
    loop = asyncio.ProactorEventLoop()
    asyncio.set_event_loop(loop)


class BBG490CurveUtils:
    _verbose = False
    s490_nyclose_db = None
    _UPI_MIGRATE_DATE = datetime(2024, 1, 29)

    def __init__(self, s490_curve_db_path, verbose=False):
        self.setup_s490_nyclose_db(s490_curve_db_path=s490_curve_db_path)
        self._verbose = verbose

    def setup_s490_nyclose_db(self, s490_curve_db_path: str):
        self.s490_nyclose_db = ShelveDBWrapper(s490_curve_db_path) if s490_curve_db_path else None
        self.s490_nyclose_db.open()

        most_recent_db_dt = datetime.fromtimestamp(int(max(self.s490_nyclose_db.keys())))
        if ((datetime.today() - BDay(1)) - most_recent_db_dt).days > 1:
            print(colored(f"s490_nyclose_db hasnt been updated --- git pull to update --- most recent date in db: {most_recent_db_dt}", "yellow"))

    def s490_nyclose_term_structure_ts(self, start_date: datetime, end_date: datetime, is_eris=False) -> pd.DataFrame:
        bdates = pd.date_range(start=start_date, end=end_date, freq=CustomBusinessDay(calendar=USFederalHolidayCalendar()))
        ts_term_structures = []
        for curr_date in bdates:
            try:
                str_ts = str(int(curr_date.to_pydatetime().timestamp()))
                ohlc_df = pd.DataFrame(self.s490_nyclose_db.get(str_ts)["ohlc"])
                curr_term_structure = {"Date": curr_date}
                curr_term_structure = curr_term_structure | dict(zip(ERIS_TENORS if is_eris else DEFAULT_SWAP_TENORS, ohlc_df["Close"] * 100))
                ts_term_structures.append(curr_term_structure)
            except Exception as e:
                print(colored(f'"s490_nyclose_term_structure_ts" Something went wrong at {curr_date}: {e}'), "red") if self._verbose else None

        return pd.DataFrame(ts_term_structures)

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
        fwd_tenors: Optional[List[str]] = [
            "1D",
            "1W",
            "1M",
            "2M",
            "3M",
            "4M",
            "5M",
            "6M",
            "7M",
            "8M",
            "9M",
            "10M",
            "11M",
            "12M",
            "18M",
            "2Y",
            "3Y",
            "4Y",
            "5Y",
            "7Y",
            "10Y",
            "15Y",
            "20Y",
        ],
        swaption_time_and_sales_dict_for_fwds: Optional[Dict[datetime, pd.DataFrame]] = None,
        ql_piecewise_method: Literal[
            "logLinearDiscount", "logCubicDiscount", "linearZero", "cubicZero", "linearForward", "splineCubicDiscount"
        ] = "logLinearDiscount",
        ql_zero_curve_method: Optional[ql.ZeroCurve] = ql.ZeroCurve,
        ql_compounding=ql.Compounded,
        ql_compounding_freq=ql.Daily,
        bp_fixed_adjustment: Optional[int] = -10,
        is_eris=False,
    ) -> Tuple[Dict[datetime, pd.DataFrame], Dict[datetime, ql.DiscountCurve | ql.ZeroCurve | ql.ForwardCurve]]:
        bdates = pd.date_range(start=start_date, end=end_date, freq=CustomBusinessDay(calendar=USFederalHolidayCalendar()))
        fwd_term_structure_grids: Dict[datetime, pd.DataFrame] = {}
        ql_curves: Dict[datetime, ql.DiscountCurve | ql.ZeroCurve | ql.ForwardCurve] = {}
        for curr_date in bdates:
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
                dict_for_df = {"Tenor": ERIS_TENORS if is_eris else DEFAULT_SWAP_TENORS, "Fixed Rate": ohlc_df["Close"] * 100}

                ql_curve.enableExtrapolation()
                ql.Settings.instance().evaluationDate = datetime_to_ql_date(curr_date)
                ql_curves[curr_date] = ql_curve

                if swaption_time_and_sales_dict_for_fwds is not None:
                    swaption_ts_df = swaption_time_and_sales_dict_for_fwds[curr_date.to_pydatetime()]
                    fwd_tenors = swaption_ts_df["Option Tenor"].unique()

                for fwd_rate_tenor in fwd_tenors:
                    if ql_compounding == ql.Continuous or ql_compounding == ql.Simple:
                        dict_for_df[f"{fwd_rate_tenor} Fwd"] = [
                            (
                                ql_curve.forwardRate(
                                    date,
                                    ql.UnitedStates(ql.UnitedStates.GovernmentBond).advance(date, ql.Period(fwd_rate_tenor)),
                                    ql.Actual360(),
                                    ql_compounding,
                                ).rate()
                            )
                            * 100
                            + bp_fixed_adjustment / 100
                            for date in [datetime_to_ql_date(pd_ts.to_pydatetime()) for pd_ts in ohlc_df["Expiry"].to_list()]
                        ]
                    else:
                        dict_for_df[f"{fwd_rate_tenor} Fwd"] = [
                            (
                                ql_curve.forwardRate(
                                    date,
                                    date + ql.Period(fwd_rate_tenor),
                                    ql.Actual360(),
                                    ql_compounding,
                                    ql_compounding_freq,
                                    True,
                                ).rate()
                            )
                            * 100
                            + bp_fixed_adjustment / 100
                            for date in [datetime_to_ql_date(pd_ts.to_pydatetime()) for pd_ts in ohlc_df["Expiry"].to_list()]
                        ]

                fwd_term_structure_grids[curr_date] = pd.DataFrame(dict_for_df)

            except Exception as e:
                print(colored(f'"s490_nyclose_fwd_grid_term_structures" Something went wrong at {curr_date}: {e}'), "red") if self._verbose else None

        return fwd_term_structure_grids, ql_curves

    def detect_and_merge_straddles(self, df: pd.DataFrame):

        group_columns = [
            "Execution Timestamp",
            "Option Tenor",
            "Underlying Tenor",
            "Strike Price",
            "Option Premium Amount",
            "Notional Amount",
            "Option Premium per Notional",
            "IV",
            "IV bp/day",
        ]
        grouped = df.groupby(group_columns)
        new_rows = []

        def is_straddle(group_df):
            if len(group_df) == 2:
                styles = group_df["Style"].unique()
                if set(styles) == set(["payer", "receiver"]):
                    return True
            return False

        for _, group_df in grouped:
            if is_straddle(group_df):
                base_row = group_df.iloc[0].copy()
                base_row["Style"] = "straddle"
                new_rows.append(base_row)
            else:
                for _, row in group_df.iterrows():
                    new_rows.append(row)

        straddle_df = pd.DataFrame(new_rows)
        straddle_df.reset_index(drop=True, inplace=True)
        return straddle_df

    def split_straddles(self, df: pd.DataFrame):
        straddle_rows = df[df["Style"] == "straddle"]
        non_straddle_rows = df[df["Style"] != "straddle"]
        new_rows = []
        quantity_columns = ["Option Premium Amount", "Notional Amount", "Option Premium per Notional", "IV", "IV bp/day"]
        for _, row in straddle_rows.iterrows():
            row_payer = row.copy()
            row_receiver = row.copy()
            for col in quantity_columns:
                if pd.notnull(row[col]):
                    row_payer[col] = row[col] / 2
                    row_receiver[col] = row[col] / 2
            row_payer["Style"] = "payer"

            row_receiver["Style"] = "receiver"
            new_rows.append(row_payer)
            new_rows.append(row_receiver)

        result_df = pd.concat([non_straddle_rows, pd.DataFrame(new_rows)], ignore_index=True)
        return result_df

    def format_swaption_premium_close(self, df: pd.DataFrame, normalize_premium=True):
        df = df.copy()
        df["Swaption Tenor"] = df["Option Tenor"] + df["Underlying Tenor"]
        df = df.sort_values(by=["Swaption Tenor", "Strike Price", "IV bp/day", "ATMF", "Event timestamp"])
        if normalize_premium:
            df["Premium"] = df["Option Premium Amount"] / df["Notional Amount"]
        else:
            df["Premium"] = df["Option Premium Amount"]

        ohlc_df = (
            df.groupby(["Swaption Tenor", "Strike Price"])
            .agg(
                Premium=("Premium", "last"),
                IV_Daily_BPs=("IV bp/day", "last"),
                ATMF=("ATMF", "last"),
            )
            .reset_index()
        )
        return ohlc_df

    def create_s490_swaption_time_and_sales(
        self, curve_data_fetcher: CurveDataFetcher, start_date: datetime, end_date: datetime, calc_greeks: Optional[bool] = False
    ):
        calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
        day_count = ql.Actual360()
        swaption_dict: Dict[datetime, pd.DataFrame] = curve_data_fetcher.dtcc_sdr_fetcher.fetch_historical_swaption_time_and_sales(
            start_date=start_date,
            end_date=end_date,
            underlying_swap_types=["Fixed_Float_OIS", "Fixed_Float"],
            underlying_reference_floating_rates=["USD-SOFR-COMPOUND", "USD-SOFR-OIS Compound"],
            underlying_ccy="USD",
            underlying_reference_floating_rate_term_value=1,
            underlying_reference_floating_rate_term_unit="DAYS",
            underlying_notional_schedule="Constant",
            underlying_delivery_types=["CASH", "PHYS"],
        )

        fwd_dict, ql_curves_dict = self.s490_nyclose_fwd_curve_matrices(
            start_date=start_date,
            end_date=end_date,
            swaption_time_and_sales_dict_for_fwds=swaption_dict,
            ql_piecewise_method="logLinearDiscount",
            ql_compounding=ql.Simple,
        )

        if len(fwd_dict.keys()) == 0:
            raise ValueError("Forward rates calc error")

        modifed_swaption_time_and_sales = {}
        ohlc_premium = {}
        for bdate in pd.date_range(start=start_date, end=end_date, freq=CustomBusinessDay(calendar=USFederalHolidayCalendar())):
            ql.Settings.instance().evaluationDate = datetime_to_ql_date(bdate.to_pydatetime())
            curr_ql_curve = ql_curves_dict[bdate.to_pydatetime()]
            curr_ql_curve.enableExtrapolation()
            curr_curve_handle = ql.YieldTermStructureHandle(curr_ql_curve)
            # bachelier_engine = ql.BachelierSwaptionEngine(
            #     curr_curve_handle, ql.QuoteHandle(ql.SimpleQuote(0.0)), ql.ActualActual(ql.ActualActual.ISMA)
            # )
            black_engine = ql.BlackSwaptionEngine(curr_curve_handle, ql.QuoteHandle(ql.SimpleQuote(0.0)), ql.ActualActual(ql.ActualActual.ISMA))

            curr_swaption_time_and_sales_with_iv = []
            curr_swaption_time_and_sales = swaption_dict[bdate.to_pydatetime()].to_dict("records")
            err_count = 0
            for i, swaption_trade_dict in enumerate(curr_swaption_time_and_sales):
                if swaption_trade_dict["Fwd"] != "0D":
                    continue

                try:
                    atmf = (
                        fwd_dict[bdate.to_pydatetime()]
                        .loc[
                            fwd_dict[bdate.to_pydatetime()]["Tenor"] == swaption_trade_dict["Underlying Tenor"],
                            f"{swaption_trade_dict["Option Tenor"]} Fwd",
                        ]
                        .values[0]
                        / 100
                    )
                    swaption_trade_dict["ATMF"] = atmf
                    swaption_trade_dict["OTM"] = (swaption_trade_dict["Strike Price"] - swaption_trade_dict["ATMF"]) * 100 * 100
                    option_premium = swaption_trade_dict["Option Premium Amount"] / swaption_trade_dict["Notional Amount"]

                    curr_sofr_ois_swap = ql.MakeOIS(
                        ql.Period(swaption_trade_dict["Underlying Tenor"]),
                        ql.OvernightIndex("SOFR", 1, ql.USDCurrency(), calendar, day_count, curr_curve_handle),
                        swaption_trade_dict["Strike Price"],
                        ql.Period(swaption_trade_dict["Option Tenor"]),
                        paymentLag=2,
                        settlementDays=2,
                        calendar=calendar,
                        convention=ql.ModifiedFollowing,
                    )
                    curr_sofr_ois_swaption = ql.Swaption(
                        curr_sofr_ois_swap,
                        # ql.EuropeanExercise(datetime_to_ql_date(swaption_trade_dict["Expiration Date"])),
                        ql.EuropeanExercise(
                            calendar.advance(
                                datetime_to_ql_date(swaption_trade_dict["Effective Date"]),
                                ql.Period(swaption_trade_dict["Option Tenor"]),
                                ql.ModifiedFollowing,
                            )
                        ),
                    )
                    # curr_sofr_ois_swaption.setPricingEngine(bachelier_engine)
                    curr_sofr_ois_swaption.setPricingEngine(black_engine)
                    implied_vol = curr_sofr_ois_swaption.impliedVolatility(
                        price=option_premium,
                        discountCurve=curr_curve_handle,
                        guess=0.01,
                        accuracy=1e-1,
                        maxEvaluations=750,
                        minVol=0,
                        maxVol=5.0,
                        # type=ql.Normal,
                    )
                    swaption_trade_dict["IV"] = implied_vol * 100
                    swaption_trade_dict["IV bp/day"] = (swaption_trade_dict["IV"] * atmf / np.sqrt(252)) * 100
                    curr_swaption_time_and_sales_with_iv.append(swaption_trade_dict)

                    if calc_greeks:
                        pass

                except Exception as e:
                    print(
                        colored(
                            f"Error at {i} {swaption_trade_dict["Option Tenor"]}x{swaption_trade_dict["Underlying Tenor"]} @ {round(swaption_trade_dict["OTM"])} for {round(option_premium, 3)}: {str(e)}",
                            "red",
                        )
                    )
                    err_count += 1

            curr_swaption_time_and_sales_with_iv_df = pd.DataFrame(curr_swaption_time_and_sales_with_iv)
            # modifed_swaption_time_and_sales[bdate.to_pydatetime()] = curr_swaption_time_and_sales_with_iv_df
            # ohlc_premium[bdate.to_pydatetime()] = self.format_swaption_premium_close(curr_swaption_time_and_sales_with_iv_df)

            # avoid double counting straddles after UPI migration (see below for info):
            # https://www.clarusft.com/swaption-volumes-by-strike-q1-2024/
            with_straddles_df = self.detect_and_merge_straddles(curr_swaption_time_and_sales_with_iv_df)
            split_straddles_df = self.split_straddles(with_straddles_df)
            modifed_swaption_time_and_sales[bdate.to_pydatetime()] = split_straddles_df
            ohlc_premium[bdate.to_pydatetime()] = self.format_swaption_premium_close(split_straddles_df)

        print(colored(f"Errors Count: {err_count}", "red"))
        return modifed_swaption_time_and_sales, ohlc_premium

    @staticmethod
    def sabr_implied_vol(F, K, T, alpha, beta, rho, nu):
        if K <= 0:
            return np.nan  # Return NaN for invalid strikes
        if F == K:
            # ATM formula
            FK_beta = F ** (1 - beta)
            vol = (alpha / FK_beta) * (
                1
                + (((1 - beta) ** 2 / 24) * (alpha**2) / (FK_beta**2) + (rho * beta * nu * alpha) / (4 * FK_beta) + ((2 - 3 * rho**2) * nu**2 / 24))
                * T
            )
        else:
            logFK = np.log(F / K)
            FK_beta = (F * K) ** ((1 - beta) / 2)
            z = (nu / alpha) * FK_beta * logFK
            x_z = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
            if abs(z) < 1e-12 or x_z == 0.0:
                z_over_x_z = 1.0  # Avoid division by zero
            else:
                z_over_x_z = z / x_z
            vol = (
                (alpha / FK_beta)
                * z_over_x_z
                * (1 + (((1 - beta) ** 2 / 24) * (logFK) ** 2) + (((1 - beta) ** 4 / 1920) * (logFK) ** 4))
                * (
                    1
                    + (
                        ((1 - beta) ** 2 / 24) * (alpha**2) / (FK_beta**2)
                        + (rho * beta * nu * alpha) / (4 * FK_beta)
                        + ((2 - 3 * rho**2) * nu**2 / 24)
                    )
                    * T
                )
            )
        return vol

    def create_sabr_smile_matplotlib(
        self,
        swaption_time_and_sales: pd.DataFrame,
        option_tenor: str,
        underlying_tenor: str,
        offsets_bps=np.array([-200, -150, -100, -50, -25, 0, 25, 50, 100, 150, 200]),
        beta=0.5,
    ):

        selected_swaption_df = swaption_time_and_sales[
            (swaption_time_and_sales["Option Tenor"] == option_tenor) & (swaption_time_and_sales["Underlying Tenor"] == underlying_tenor)
        ]
        print(
            selected_swaption_df[
                ["Strike Price", "Notional Amount", "Option Premium per Notional", "Direction", "Style", "IV", "IV bp/day", "ATMF", "OTM"]
            ]
        )

        if selected_swaption_df.empty:
            raise ValueError(f"DID NOT SEE {option_tenor}{underlying_tenor} SWAPTION TRADE")

        grouped = selected_swaption_df.groupby("Strike Price")
        average_vols = grouped["IV"].mean().reset_index()
        unique_strikes = average_vols["Strike Price"].values
        average_vols = average_vols["IV"].values / 100

        F = selected_swaption_df["ATMF"].to_numpy()[-1]
        T = tenor_to_years(option_tenor)

        def sabr_objective(params, F, K, T, market_vols, beta):
            alpha, rho, nu = params
            model_vols = np.array([self.sabr_implied_vol(F, k, T, alpha, beta, rho, nu) for k in K])
            return np.sum((model_vols - market_vols) ** 2)

        initial_params = [0.5, 0.1, 0.5]
        bounds = [(0.0001, 2.0), (0.0, 0.999), (0.0001, 2.0)]
        result = optimize.minimize(sabr_objective, initial_params, args=(F, unique_strikes, T, average_vols, beta), bounds=bounds, method="L-BFGS-B")
        if not result.success:
            raise RuntimeError(f"SABR calibration failed: {result.message}")
        alpha_calibrated, rho_calibrated, nu_calibrated = result.x
        print(f"Calibrated Parameters:\nAlpha: {alpha_calibrated}\nRho: {rho_calibrated}\nNu: {nu_calibrated}")

        offsets_decimal = offsets_bps / 10000
        strikes = F + offsets_decimal
        valid_indices = strikes > 0
        valid_strikes = strikes[valid_indices]
        valid_offsets_bps = offsets_bps[valid_indices]

        vols = []
        for K in valid_strikes:
            vol = self.sabr_implied_vol(F, K, T, alpha_calibrated, beta, rho_calibrated, nu_calibrated)
            vols.append(vol * 100)  # Convert to percentage

        atmf_mean = selected_swaption_df["ATMF"].mean()
        swaption_row = {
            "reference": "SOFR",
            "instrument": "swaption",
            "model": "SABR",
            "date": selected_swaption_df["Effective Date"].iloc[-1],
            "expiration": option_tenor,
            "tenor": underlying_tenor,
            "ATMF": atmf_mean,
        }
        for idx, offset in enumerate(valid_offsets_bps):
            swaption_row[offset] = vols[idx]

        plt.figure(figsize=(18, 8))
        plt.plot(valid_offsets_bps, np.array(vols), label="SABR Smile")
        for _, row in selected_swaption_df.iterrows():
            # plt.scatter((row["Strike Price"] - F) * 10000, row["IV"], label=f"{row["Event timestamp"]}", color="red")
            plt.scatter((row["Strike Price"] - F) * 10000, row["IV"], color="red")

        atm_vol = self.sabr_implied_vol(F, F, T, alpha_calibrated, beta, rho_calibrated, nu_calibrated) * 100
        plt.axvline(
            x=0,
            color="green",
            linestyle="--",
            label=f"ATMF Strike: {atmf_mean * 100:.3f}%\nSABR ATM IV: {atm_vol:.2f} bps\nSABR ATM IV: {atm_vol * atmf_mean / np.sqrt(252) * 100:.3f} bp/d",
        )
        plt.scatter(0, atm_vol, color="green", zorder=6)

        plt.title(f"SABR Volatility Smile - Black Vol (bps, Annual)")
        plt.xlabel("Strike Price K")
        plt.ylabel("Implied Volatility")
        plt.legend(fontsize="large")
        plt.grid(True)
        plt.show()

        return swaption_row

    # TODO
    # - recalc ATM vol on param change
    # - migrate to pysabr
    def create_sabr_smile_interactive(
        self,
        swaption_time_and_sales: pd.DataFrame,
        option_tenor: str,
        underlying_tenor: str,
        offsets_bps=np.array([-200, -150, -100, -50, -25, 0, 25, 50, 100, 150, 200]),
        initial_beta=1,
        show_trades=False,
        scale_by_notional=False,
        drop_trades_idxs: Optional[List[int]] = None,
    ):
        selected_swaption_df = swaption_time_and_sales[
            (swaption_time_and_sales["Option Tenor"] == option_tenor) & (swaption_time_and_sales["Underlying Tenor"] == underlying_tenor)
        ]
        selected_swaption_df = selected_swaption_df.reset_index(drop=True)

        if drop_trades_idxs:
            selected_swaption_df = selected_swaption_df.drop(drop_trades_idxs)

        if show_trades:
            display(
                selected_swaption_df[
                    [
                        "Event timestamp",
                        "Direction",
                        "Style",
                        "Notional Amount",
                        "Strike Price",
                        "OTM",
                        "Option Premium per Notional",
                        "IV",
                        "IV bp/day",
                        "ATMF",
                    ]
                ]
            )

        if selected_swaption_df.empty:
            raise ValueError(f"No data for option tenor {option_tenor} and underlying tenor {underlying_tenor}")

        grouped = selected_swaption_df.groupby("Strike Price")
        average_vols = grouped["IV"].mean().reset_index()
        unique_strikes = average_vols["Strike Price"].values
        average_vols = average_vols["IV"].values / 100  # Convert to decimal

        F = selected_swaption_df["ATMF"].iloc[-1]
        T = tenor_to_years(option_tenor)

        def sabr_objective(params, F, K, T, market_vols, beta):
            alpha, rho, nu = params
            model_vols = np.array([self.sabr_implied_vol(F, k, T, alpha, beta, rho, nu) for k in K])
            return np.sum((model_vols - market_vols) ** 2)

        initial_params = [0.5, 0.1, 0.5]
        bounds = [(0.0001, 2.0), (-0.999, 0.999), (0.0001, 2.0)]
        result = optimize.minimize(
            sabr_objective, initial_params, args=(F, unique_strikes, T, average_vols, initial_beta), bounds=bounds, method="L-BFGS-B"
        )
        if not result.success:
            raise RuntimeError(f"SABR calibration failed: {result.message}")
        alpha_calibrated, rho_calibrated, nu_calibrated = result.x
        # print(f"Calibrated Parameters:\nAlpha: {alpha_calibrated}\nRho: {rho_calibrated}\nNu: {nu_calibrated}")

        offsets_decimal = offsets_bps / 10000
        strikes = F + offsets_decimal
        valid_indices = strikes > 0
        valid_strikes = strikes[valid_indices]
        valid_offsets_bps = offsets_bps[valid_indices]

        def compute_sabr_vols(F, K, T, alpha, beta, rho, nu):
            vols = []
            for k in K:
                vol = self.sabr_implied_vol(F, k, T, alpha, beta, rho, nu)
                vols.append(vol * 100)
            return vols

        vols = compute_sabr_vols(F, valid_strikes, T, alpha_calibrated, initial_beta, rho_calibrated, nu_calibrated)

        fig = go.FigureWidget()
        fig.add_trace(go.Scatter(x=valid_offsets_bps, y=vols, mode="lines", name="SABR Smile"))

        direction_color_map = {
            "buyer": "green",
            "underwritter": "red",
        }

        if scale_by_notional:
            min_size = 5
            max_size = 20
            notional_values = selected_swaption_df["Notional Amount"]
            notional_values = notional_values.replace(0, np.nan)  # Avoid log(0) issues
            log_scaled_sizes = np.log(notional_values)
            log_scaled_sizes = min_size + (log_scaled_sizes - log_scaled_sizes.min()) * (max_size - min_size) / (
                log_scaled_sizes.max() - log_scaled_sizes.min()
            )

        for idx, row in selected_swaption_df.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[(row["Strike Price"] - F) * 10000],
                    y=[row["IV"]],
                    name=f"{row['Event timestamp'].time()}-{row['Style']} {row['Direction']}",
                    mode="markers",
                    marker=dict(size=log_scaled_sizes[idx] if scale_by_notional else 8, color=direction_color_map[row["Direction"]]),
                    hovertemplate=(
                        "Strike Offset: %{x:.3f} bps<br>"
                        "IV bps/annual: " + str(round(row["IV"], 3)) + "<br>"
                        "IV bps/day: " + str(round(row["IV bp/day"], 3)) + "<br>"
                        "Strike Price: " + str(round(row["Strike Price"] * 100, 3)) + "<br>"
                        "ATMF: " + str(round(row["ATMF"] * 100, 3)) + "<br>"
                        "Notional Amount ($): " + str(row["Notional Amount"]) + "<br>"
                        "Option Premium Amount ($): " + str(row["Option Premium Amount"]) + "<br>"
                        "Option Premium per Notional ($): " + str(round(row["Option Premium per Notional"], 3)) + "<br>"
                        "Execution Timestamp: " + str(row["Event timestamp"]) + "<br>"
                        "Direction: " + str(row["Direction"]) + "<br>"
                        "Style: " + str(row["Style"]) + "<br>"
                        "<extra></extra>"
                    ),
                )
            )

        atm_vol = self.sabr_implied_vol(F, F, T, alpha_calibrated, initial_beta, rho_calibrated, nu_calibrated) * 100
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[atm_vol],
                mode="markers",
                marker=dict(color="blue", size=10),
                name="ATM Vol",
                hovertemplate=("ATM Strike Offset: 0 bps<br>" + "ATM Implied Volatility: %{y:.2f} <br>" + "<extra></extra>"),
            )
        )

        fig.update_layout(
            title=(
                f"""{option_tenor} x {underlying_tenor} SABR Black Vol Smile --- DTCC Reported Trades From {selected_swaption_df["Event timestamp"].iloc[1].to_pydatetime().date()}<br>
                ---------------------------------------------------------------------------<br>
                ATMF Strike: {F * 100:.3f}%<br>
                SABR ATM IV: {atm_vol:.2f} bps<br>
                SABR ATM IV: {atm_vol * F / np.sqrt(252) * 100:.3f} bps/day"""
            ),
            xaxis_title="Strike Offset from ATMF (bps)",
            yaxis_title="Implied Volatility (%)",
            legend=dict(font=dict(size=12)),
            hovermode="closest",
            template="plotly_dark",
            height=1200,
            title_x=0.5,
        )
        fig.update_xaxes(
            showgrid=True,
            showspikes=True,
            spikecolor="white",
            spikesnap="cursor",
            spikemode="across",
        )
        fig.update_yaxes(showgrid=True, showspikes=True, spikecolor="white", spikesnap="cursor", spikethickness=0.5)

        slider_layout = widgets.Layout(width="1250px")
        beta_slider = widgets.FloatSlider(
            value=initial_beta, min=0.0, max=1.0, step=0.00001, description="β", continuous_update=True, readout_format=".5f", layout=slider_layout
        )
        alpha_slider = widgets.FloatSlider(
            value=alpha_calibrated,
            min=0.00001,
            max=2.0,
            step=0.00001,
            description="α",
            continuous_update=True,
            readout_format=".5f",
            layout=slider_layout,
        )
        rho_slider = widgets.FloatSlider(
            value=rho_calibrated,
            min=-0.999,
            max=0.999,
            step=0.00001,
            description="Ρ",
            continuous_update=True,
            readout_format=".5f",
            layout=slider_layout,
        )
        nu_slider = widgets.FloatSlider(
            value=nu_calibrated,
            min=0.0001,
            max=2.0,
            step=0.00001,
            description="ν",
            continuous_update=True,
            readout_format=".5f",
            layout=slider_layout,
        )

        def update_sabr_smile(beta, alpha, rho, nu):
            vols = compute_sabr_vols(F, valid_strikes, T, alpha, beta, rho, nu)
            atm_vol = self.sabr_implied_vol(F, F, T, alpha, beta, rho, nu) * 100
            with fig.batch_update():
                fig.data[0].y = vols
                fig.data[-1].y = [atm_vol]
                title_lines = [
                    f"{option_tenor} x {underlying_tenor} SABR Black Vol Smile --- DTCC Reported Trades From {selected_swaption_df['Event timestamp'].iloc[1].to_pydatetime().date()}",
                    "-" * 75,
                    f"ATMF Strike: {F * 100:.3f}%, SABR ATM Black Vol: {atm_vol:.2f} bps",
                ]
                for offset, vol in dict(zip(valid_offsets_bps, vols)).items():
                    bps_per_day = round(vol * F / np.sqrt(252) * 100, 3)
                    title_lines.append(f"{offset:+4} bps: {bps_per_day:.3f} b/d")

                fig.layout.title.text = "<br>".join(title_lines)

        ui = widgets.VBox([beta_slider, alpha_slider, rho_slider, nu_slider])
        out = widgets.interactive_output(update_sabr_smile, {"beta": beta_slider, "alpha": alpha_slider, "rho": rho_slider, "nu": nu_slider})

        display(ui, fig, out)

        swaption_row = {
            "reference": "SOFR",
            "instrument": "swaption",
            "model": "SABR - Black Vol",
            "date": selected_swaption_df["Effective Date"].iloc[-1],
            "expiration": option_tenor,
            "tenor": underlying_tenor,
            "ATMF": F,
        }
        for idx, offset in enumerate(valid_offsets_bps):
            swaption_row[offset] = vols[idx]
        return swaption_row
