from datetime import datetime
from typing import Dict, List, Literal, Optional, Tuple

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import QuantLib as ql
import scipy.optimize as optimize
from IPython.display import HTML, display
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import BDay, CustomBusinessDay
from pysabr import Hagan2002LognormalSABR, Hagan2002NormalSABR
from termcolor import colored

from CurvyCUSIPs.utils.ShelveDBWrapper import ShelveDBWrapper
from CurvyCUSIPs.utils.dtcc_swaps_utils import DEFAULT_SWAP_TENORS, datetime_to_ql_date, tenor_to_years
from CurvyCUSIPs.CurveDataFetcher import CurveDataFetcher 
from CurvyCUSIPs.S490Swaps import S490Swaps


class S490Swaptions:
    s490_swaps: S490Swaps

    def __init__(self, s490_swaps: S490Swaps):
        self.s490_swaps = s490_swaps

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
            row_payer["UPI FISN"] = "NA/O Call Epn OIS USD"
            row_receiver["Style"] = "receiver"
            row_receiver["UPI FISN"] = "NA/O P Epn OIS USD"

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
        self,
        data_fetcher: CurveDataFetcher,
        start_date: datetime,
        end_date: datetime,
        model: Optional[Literal["lognormal", "normal"]] = "normal",
        calc_greeks: Optional[bool] = False,
    ):
        if model != "normal" and model != "lognormal":
            raise ValueError(f"Bad Model Param: {model}")

        calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
        day_count = ql.Actual360()
        swaption_dict: Dict[datetime, pd.DataFrame] = data_fetcher.dtcc_sdr_fetcher.fetch_historical_swaption_time_and_sales(
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

        fwd_dict, ql_curves_dict = self.s490_swaps.s490_nyclose_fwd_curve_matrices(
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
            swaption_pricing_engine = ql.BlackSwaptionEngine(
                curr_curve_handle, ql.QuoteHandle(ql.SimpleQuote(0.0)), ql.ActualActual(ql.ActualActual.ISMA)
            )

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
                        ql.EuropeanExercise(
                            # calendar.advance(
                            #     datetime_to_ql_date(swaption_trade_dict["Effective Date"]),
                            #     ql.Period(swaption_trade_dict["Option Tenor"]),
                            #     ql.ModifiedFollowing,
                            # )
                            datetime_to_ql_date(swaption_trade_dict["Effective Date"])
                            + ql.Period(swaption_trade_dict["Option Tenor"])
                        ),
                    )
                    curr_sofr_ois_swaption.setPricingEngine(swaption_pricing_engine)

                    if model == "normal":
                        implied_vol = curr_sofr_ois_swaption.impliedVolatility(
                            price=option_premium,
                            discountCurve=curr_curve_handle,
                            guess=0.5,
                            accuracy=1e-1,
                            maxEvaluations=750,
                            minVol=0,
                            maxVol=5.0,
                            type=ql.Normal,
                        )
                        swaption_trade_dict["IV"] = implied_vol * 100 * 100
                        swaption_trade_dict["IV bp/day"] = swaption_trade_dict["IV"] / np.sqrt(252)
                    elif model == "lognormal":
                        implied_vol = curr_sofr_ois_swaption.impliedVolatility(
                            price=option_premium,
                            discountCurve=curr_curve_handle,
                            guess=0.01,
                            accuracy=1e-1,
                            maxEvaluations=750,
                            minVol=0,
                            maxVol=5.0,
                        )
                        swaption_trade_dict["IV"] = implied_vol * 100
                        # TODO:
                        # - need to implement Le Floc’h's since approx only works for near ATM
                        swaption_trade_dict["IV bp/day"] = (swaption_trade_dict["IV"] * atmf / np.sqrt(252)) * 100
                    else:
                        raise ValueError("Not reachable")

                    if calc_greeks:
                        pass

                    curr_swaption_time_and_sales_with_iv.append(swaption_trade_dict)

                except Exception as e:
                    self.s490_swaps._logger.error(
                        colored(
                            f"Error at {i} {swaption_trade_dict["Effective Date"]} - {swaption_trade_dict["Option Tenor"]}x{swaption_trade_dict["Underlying Tenor"]}: {str(e)}",
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

        self.s490_swaps._logger.error(colored(f"Errors Count: {err_count}", "red"))
        return modifed_swaption_time_and_sales, ohlc_premium

    # Hagan, P. S., Kumar, D., Lesniewski, A. S., & Woodward, D. E. (2002). Managing Smile Risk. Risk Magazine.
    @staticmethod
    def sabr_implied_vol(F: float, K: float, T: float, alpha: float, beta: float, rho: float, nu: float):
        r"""
        Calculate the SABR implied volatility using the Hagan et al. (2002) approximation.

        Parameters:
        F (float): Forward price
        K (float): Strike price
        T (float): Time to maturity
        alpha (float): Volatility parameter
        beta (float): Elasticity parameter
        rho (float): Correlation parameter
        nu (float): Volatility of volatility

        Returns:
        float: Implied Black-Scholes volatility

        ATM Case (F == K)
        $$
        \sigma_{\text{BS}} = \frac{\alpha}{F^{1-\beta}} \left(1 + \left[\frac{(1-\beta)^2}{24} \frac{\alpha^2}{F^{2(1-\beta)}} + \frac{\rho \beta \nu \alpha}{4 F^{1-\beta}} + \frac{(2-3\rho^2)\nu^2}{24}\right] T \right)
        $$

        Non-ATM Case (F != K)
        $$
        \sigma_{\text{BS}} = \frac{\alpha}{(F K)^{\frac{1-\beta}{2}}} \cdot \frac{z}{x(z)} \left(1 + \frac{(1-\beta)^2}{24} \log^2\left(\frac{F}{K}\right) + \frac{(1-\beta)^4}{1920} \log^4\left(\frac{F}{K}\right)\right) \left(1 + \left[\frac{(1-\beta)^2}{24} \frac{\alpha^2}{(F K)^{1-\beta}} + \frac{\rho \beta \nu \alpha}{4 (F K)^{\frac{1-\beta}{2}}} + \frac{(2-3\rho^2)\nu^2}{24}\right] T \right)
        $$
        where,
        $$
        z = \frac{\nu}{\alpha} (F K)^{\frac{1-\beta}{2}} \log\left(\frac{F}{K}\right)
        $$
        $$
        x(z) = \log\left(\frac{\sqrt{1 - 2 \rho z + z^2} + z - \rho}{1 - \rho}\right)
        $$

        Invalid Strike Edge Case:
        $$
        K \leq 0 \quad \Rightarrow \quad \text{Implied Volatility} = \text{NaN}
        $$
        """

        # TODO handle negative strikes
        # Return NaN for invalid (negative) strikes
        if K <= 0:
            return np.nan

        if F == K:
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

            # division by zero or very small z
            if abs(z) < 1e-12 or x_z == 0.0:
                z_over_x_z = 1.0
            else:
                z_over_x_z = z / x_z

            vol = (
                (alpha / FK_beta)
                * z_over_x_z
                * (1 + ((1 - beta) ** 2 / 24) * logFK**2 + ((1 - beta) ** 4 / 1920) * logFK**4)
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

    def create_sabr_smile_pysabr_matplotlib(
        self,
        swaption_time_and_sales: pd.DataFrame,
        option_tenor: str,
        underlying_tenor: str,
        offsets_bps=np.array([-200, -150, -100, -50, -25, 0, 25, 50, 100, 150, 200]),
        beta=0.5,
        model: Literal["normal", "lognormal"] = "normal",
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

        offsets_decimal = offsets_bps / 10000
        strikes = F + offsets_decimal
        valid_indices = strikes > 0
        valid_strikes = strikes[valid_indices]
        valid_offsets_bps = offsets_bps[valid_indices]

        if model == "normal":
            calibration_normal = Hagan2002NormalSABR(f=F, shift=0, t=T, beta=beta).fit(unique_strikes, average_vols)
            atm_normal_vol = self.sabr_implied_vol(F, F, T, calibration_normal[0], beta, calibration_normal[1], calibration_normal[2]) * 100 * F
            vols = [
                Hagan2002NormalSABR(
                    f=F, shift=0, t=T, v_atm_n=atm_normal_vol, beta=beta, rho=calibration_normal[1], volvol=calibration_normal[2]
                ).normal_vol(strike)
                * 100
                * 100
                for strike in valid_strikes
            ]

        elif model == "lognormal":
            calibration_lognormal = Hagan2002LognormalSABR(f=F, shift=0, t=T, beta=beta).fit(unique_strikes, average_vols)
            atm_normal_vol = (
                self.sabr_implied_vol(F, F, T, calibration_lognormal[0], beta, calibration_lognormal[1], calibration_lognormal[2]) * 100 * F
            )
            vols = [
                Hagan2002LognormalSABR(
                    f=F, shift=0, t=T, v_atm_n=atm_normal_vol, beta=beta, rho=calibration_lognormal[1], volvol=calibration_lognormal[2]
                ).lognormal_vol(strike)
                * 100
                for strike in valid_strikes
            ]

        else:
            raise ValueError(f"bad model param passed in {model}")

        plt.figure(figsize=(18, 10))
        plt.plot(valid_offsets_bps, vols, "r--", label=f"{model.capitalize()} Model")
        plt.axvline(
            x=0,
            color="b",
            label=f"ATM {model.capitalize()} Vol: {atm_normal_vol * 100 * 100 if model == "normal" else atm_normal_vol  * 100 / F}",
        )
        plt.xlabel("Strikes")
        plt.ylabel("Volatility (%)")
        plt.title("SABR Model")
        plt.legend()
        plt.grid(True)
        plt.show()

    def create_sabr_smile_interactive(
        self,
        swaption_time_and_sales: pd.DataFrame,
        option_tenor: str,
        underlying_tenor: str,
        model: Literal["normal", "lognormal"] = "lognormal",
        implementation: Literal["pysabr", "monkey"] = "monkey",
        offsets_bps=np.array([-200, -150, -100, -50, -25, 0, 25, 50, 100, 150, 200]),
        initial_beta=1,
        show_trades=False,
        scale_by_notional=False,
        drop_trades_idxs: Optional[List[int]] = None,
        ploty_height=750,
    ):
        if model == "normal" and implementation == "monkey":
            raise NotImplemented("Normal Vol Monkey Implementaion is in progress")

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
        offsets_decimal = offsets_bps / 10000
        strikes = F + offsets_decimal
        valid_indices = strikes > 0
        valid_strikes = strikes[valid_indices]
        valid_offsets_bps = offsets_bps[valid_indices]

        if implementation == "monkey":

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

            def compute_sabr_vols(F, K, T, alpha, beta, rho, nu):
                vols = []
                for k in K:
                    vol = self.sabr_implied_vol(F, k, T, alpha, beta, rho, nu)
                    vols.append(vol * 100)
                return vols

            atm_vol = self.sabr_implied_vol(F, F, T, alpha_calibrated, initial_beta, rho_calibrated, nu_calibrated) * 100
            vols = compute_sabr_vols(F, valid_strikes, T, alpha_calibrated, initial_beta, rho_calibrated, nu_calibrated)

        elif implementation == "pysabr":
            if model == "lognormal":
                calibration_lognormal = Hagan2002LognormalSABR(f=F, shift=0, t=T, beta=initial_beta).fit(unique_strikes, average_vols)
                alpha_calibrated, rho_calibrated, nu_calibrated = calibration_lognormal
                atm_vol = self.sabr_implied_vol(F, F, T, alpha=alpha_calibrated, beta=initial_beta, rho=rho_calibrated, nu=nu_calibrated) * 100
                vols = [
                    Hagan2002LognormalSABR(
                        f=F, shift=0, t=T, v_atm_n=atm_vol * F, beta=initial_beta, rho=rho_calibrated, volvol=nu_calibrated
                    ).lognormal_vol(strike)
                    * 100
                    for strike in valid_strikes
                ]

            elif model == "normal":
                calibration_normal = Hagan2002NormalSABR(f=F, shift=0, t=T, beta=initial_beta).fit(unique_strikes, average_vols)
                alpha_calibrated, rho_calibrated, nu_calibrated = calibration_normal
                atm_vol = self.sabr_implied_vol(F, F, T, alpha=alpha_calibrated, beta=initial_beta, rho=rho_calibrated, nu=nu_calibrated) * 100
                vols = [
                    Hagan2002NormalSABR(
                        f=F, shift=0, t=T, v_atm_n=atm_vol * F, beta=initial_beta, rho=rho_calibrated, volvol=nu_calibrated
                    ).normal_vol(strike)
                    * 100
                    * 100
                    for strike in valid_strikes
                ]

            else:
                raise ValueError(f"Bad Model Param: {model}")

            atm_vol = atm_vol * 100

        else:
            raise ValueError(f"Bad Implementation Param: {implementation}")

        smile_dict = dict(zip(valid_offsets_bps.astype(str), vols))

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

        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[smile_dict["0"]],
                mode="markers",
                marker=dict(color="blue", size=10),
                name="ATM Vol",
                hovertemplate=("ATM Strike Offset: 0 bps<br>" + "ATM Implied Volatility: %{y:.3f}<br>" "<extra></extra>"),
            )
        )
        fig.update_layout(
            title="",
            xaxis_title="Strike Offset from ATMF (bps)",
            yaxis_title="Implied Volatility (%)",
            legend=dict(font=dict(size=12)),
            hovermode="closest",
            template="plotly_dark",
            height=ploty_height,
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
            value=initial_beta, min=0.0, max=1.0, step=0.000001, description="β", continuous_update=True, readout_format=".6f", layout=slider_layout
        )
        alpha_slider = widgets.FloatSlider(
            value=alpha_calibrated,
            min=0.00001,
            max=1.0,
            step=0.000001,
            description="α",
            continuous_update=True,
            readout_format=".6f",
            layout=slider_layout,
        )
        rho_slider = widgets.FloatSlider(
            value=rho_calibrated,
            min=-0.99999,
            max=0.99999,
            step=0.000001,
            description="Ρ",
            continuous_update=True,
            readout_format=".6f",
            layout=slider_layout,
        )
        nu_slider = widgets.FloatSlider(
            value=nu_calibrated,
            min=0.0001,
            max=2.0,
            step=0.000001,
            description="ν",
            continuous_update=True,
            readout_format=".6f",
            layout=slider_layout,
        )

        smile_output = widgets.Output()

        def update_sabr_smile(beta, alpha, rho, nu):
            atm_vol = self.sabr_implied_vol(F, F, T, alpha, beta, rho, nu) * 100

            if implementation == "monkey":
                vols = compute_sabr_vols(F, valid_strikes, T, alpha, beta, rho, nu)

            elif implementation == "pysabr":
                if model == "lognormal":
                    vols = [
                        Hagan2002LognormalSABR(f=F, shift=0, t=T, v_atm_n=atm_vol * F, beta=beta, rho=rho, volvol=nu).lognormal_vol(strike) * 100
                        for strike in valid_strikes
                    ]
                elif model == "normal":
                    vols = [
                        Hagan2002NormalSABR(f=F, shift=0, t=T, v_atm_n=atm_vol * F, beta=beta, rho=rho, volvol=nu).normal_vol(strike) * 100 * 100
                        for strike in valid_strikes
                    ]

                atm_vol = atm_vol * 100

            else:
                raise ValueError("Unreachable")

            smile_dict = dict(zip(valid_offsets_bps.astype(str), vols))
            smile_dict_bps_vol = dict(
                zip(
                    valid_offsets_bps.astype(str),
                    [round(vol / np.sqrt(252), 3) for vol in smile_dict.values()] if model == "normal" else [None for _ in smile_dict.values()],
                )
            )
            daily_atm_iv = smile_dict["0"] * F / np.sqrt(252) if model == "lognormal" else smile_dict["0"] / np.sqrt(252)
            smile_df = pd.DataFrame(
                {
                    "Offset (bps)": [int(offset) for offset in smile_dict.keys()],
                    "Implied (%/annual)": [round(vol, 3) for vol in smile_dict.values()],
                    "Implied (bps/day)": (
                        [round(vol / np.sqrt(252), 3) for vol in smile_dict.values()] if model == "normal" else [None for vol in smile_dict.values()]
                    ),
                }
            )
            smile_df = smile_df.sort_values("Offset (bps)").reset_index(drop=True)

            with fig.batch_update():
                fig.data[0].y = vols
                fig.data[-1].y = [smile_dict["0"]]
                title_lines = [
                    f"{option_tenor} x {underlying_tenor} SABR {model.capitalize()} Vol Smile - {implementation.upper()} Implementation --- DTCC Reported Trades From {selected_swaption_df['Event timestamp'].iloc[1].to_pydatetime().date()}",
                    "-" * 75,
                    f"ATMF Strike: {F * 100:.3f}%, SABR ATM {model.capitalize()} Vol: {smile_dict["0"]:.3f} {"%" if model == "lognormal" else "bps"}, {str(round(daily_atm_iv, 3))} bps/day",
                    "-" * 75,
                    f"Rec. Skew: {round(smile_dict_bps_vol["-100"] - smile_dict_bps_vol["0"], 3)}, Pay. Skew: {round(smile_dict_bps_vol["100"] - smile_dict_bps_vol["0"], 3)}",
                ]
                # for offset, vol in dict(zip(valid_offsets_bps, vols)).items():
                #     bps_per_day = round(vol * F / np.sqrt(252) * 100, 3) if model == "lognormal" else round(vol / np.sqrt(252), 3)
                #     title_lines.append(f"{offset:+4} bps: {bps_per_day:.3f} b/d")
                fig.layout.title.text = "<br>".join(title_lines)

            with smile_output:
                # smile_output.clear_output()
                # display(smile_dict)
                smile_output.clear_output(wait=True)
                display(smile_df.style)

        ui = widgets.VBox([beta_slider, alpha_slider, rho_slider, nu_slider])
        out = widgets.interactive_output(update_sabr_smile, {"beta": beta_slider, "alpha": alpha_slider, "rho": rho_slider, "nu": nu_slider})
        display(smile_output, ui, fig, out)
