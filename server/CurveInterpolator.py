"""
interpolations:
- linear, log line, Polynomials
- Cosine
- Chebyshev 
- Orthogonal polynomials
- Monomial interpolation
- cubics spline (normal, montone, hermite, bessel, log)
- BSplines
- hagan west monotone convex
- ns, nss
- Vasicek-Fong
- Cox-Ingersoll-Ross 
- Björk-Christensen
- gaussian process regression interpolation
- Gaussian Mixture Models (GMM)
- Principal Component Regression (PCR)
- Nearest neighbour weighted interpolation

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt
from typing import Literal, Optional
import logging
import scipy.interpolate


class CurveInterpolator:
    # all have to exist in curve_set
    _required_cols = [
        "cusip",
        "security_type",
        "auction_date",
        "issue_date",
        "maturity_date",
        "time_to_maturity",
        "is_on_the_run",
        "label",
        "original_security_term",
    ]
    # one has to exist in curve_set
    _required_ytm_cols = [
        "offer_yield",
        "bid_yield",
        "eod_yield",
        "mid_yield",
    ]
    _curve_set_df: pd.DataFrame = pd.DataFrame(columns=_required_cols + ["eod_yield"])
    _yield_to_use: Literal["offer_yield", "bid_yield", "mid_yield", "eod_yield"] = (
        "eod_yield"
    )
    _linspace_x_num: int = 100
    _linspace_x: npt.NDArray[np.float64] = np.zeros(
        shape=_linspace_x_num, dtype=np.float64
    )

    _enable_extrapolate_left_fill: bool = False
    _enable_extrapolate_right_fill: bool = False

    _x: npt.NDArray[np.float64] = np.zeros(
        shape=len(_curve_set_df.index), dtype=np.float64
    )  # ttm
    _y: npt.NDArray[np.float64] = np.zeros(
        shape=len(_curve_set_df.index), dtype=np.float64
    )  # ytm

    _logger = logging.getLogger()
    _debug_verbose: bool = False
    _info_verbose: bool = False  # performance benchmarking mainly
    _no_logs_plz: bool = False

    def __init__(
        self,
        curve_set_df: pd.DataFrame,
        use_bid_side: Optional[bool] = False,
        use_offer_side: Optional[bool] = False,
        use_mid_side: Optional[bool] = False,
        linspace_x_num: int = 100,
        enable_extrapolate_left_fill: Optional[bool] = False,
        enable_extrapolate_right_fill: Optional[bool] = False,
        drop_nans: Optional[bool] = True,
        debug_verbose: Optional[bool] = False,
        info_verbose: Optional[bool] = False,
        no_logs_plz: Optional[bool] = False,
    ):
        is_subset = set(self._required_cols).issubset(curve_set_df.columns)
        self._logger.debug(f"CurveInterpolator Inii - is valid curve set: {is_subset}")
        if not is_subset:
            raise ValueError(
                f"Required Cols Missing: {set(self._required_cols) - set(curve_set_df.columns)}"
            )
        self._curve_set_df = curve_set_df

        if use_bid_side:
            self._yield_to_use = "bid_yield"
        elif use_offer_side:
            self._yield_to_use = "offer_yield"
        elif use_mid_side:
            self._yield_to_use = "mid_yield"

        logging.debug(
            "NaN CUSIPs",
            self._curve_set_df[
                (self._curve_set_df["time_to_maturity"].isna())
                | (self._curve_set_df[self._yield_to_use].isna())
            ][["cusip", "label", "original_security_term"]],
        )
        if drop_nans:
            self._curve_set_df = self._curve_set_df[
                (self._curve_set_df["time_to_maturity"].notna())
                & (self._curve_set_df[self._yield_to_use].notna())
            ]

        self._linspace_x_num = linspace_x_num
        self._enable_extrapolate_left_fill = enable_extrapolate_left_fill
        self._enable_extrapolate_right_fill = enable_extrapolate_right_fill

        min_ttm = (
            0
            if self._enable_extrapolate_left_fill
            else self._curve_set_df["time_to_maturity"].min()
        )
        max_ttm = (
            30
            if self._enable_extrapolate_right_fill
            else self._curve_set_df["time_to_maturity"].max()
        )
        self._linspace_x = np.linspace(min_ttm, max_ttm, num=self._linspace_x_num)

        self._x = self._curve_set_df["time_to_maturity"].to_numpy()
        self._y = self._curve_set_df[self._yield_to_use].to_numpy()

        self._debug_verbose = debug_verbose
        self._info_verbose = info_verbose
        self._no_logs_plz = no_logs_plz
        if self._debug_verbose:
            self._logger.setLevel(self._logger.DEBUG)
        if self._info_verbose:
            self._logger.setLevel(self._logger.INFO)
        if self._no_logs_plz:
            self._logger.disabled = True
            self._logger.propagate = False

    def _linear_interpolation(self):
        func_no_extrap = scipy.interpolate.interp1d(
            self._x, self._y, kind="linear", bounds_error=False
        )
        func_extrap = scipy.interpolate.interp1d(
            self._x,
            self._y,
            axis=0,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        ynew_no_extrap = func_no_extrap(self._linspace_x)
        ynew_extrap = func_extrap(self._linspace_x)
        ynew = ynew_no_extrap.copy()
        if self._enable_extrapolate_left_fill:
            ynew[self._linspace_x < self._x[0]] = ynew_extrap[
                self._linspace_x < self._x[0]
            ]
        if self._enable_extrapolate_right_fill:
            ynew[self._linspace_x > self._x[-1]] = ynew_extrap[
                self._linspace_x > self._x[-1]
            ]

        return ynew

    def plotter(self, linear: Optional[bool] = False):
        if linear:
            ynew = self._linear_interpolation()
            plt.figure(figsize=(17, 6))
            plt.plot(self._linspace_x, ynew, label="Linearly Interpolated")
            plt.plot(self._x, self._y, "o", label="CUSIPs")
            plt.xlabel("Maturity")
            plt.ylabel("Yield")
            plt.title("Linear Interpolation")
            plt.legend()
            plt.show()
