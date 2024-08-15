"""
interpolations:
- linear, log line, Polynomials
- Cosine
- Chebyshev 
- Orthogonal polynomials
- Monomial interpolation
- cubics spline (normal, montone, hermite, bessel, log)
- BSplines
- PCHIP
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
from typing import Literal, Optional, Dict, Callable, List, Tuple
import logging
import scipy.interpolate
from concurrent.futures import ThreadPoolExecutor, as_completed


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

    _cubic_spline_interp_bc_type: Literal[
        "not-a-knot", "periodic", "clamped", "natural"
    ] = "not-a-knot"
    _cubic_spline_interp_custom_knots: npt.NDArray[np.float64] = np.array([])
    _cubic_spline_interp_nu: Literal[0, 1, 2] = 0

    _bspline_k: int = 3

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
        self._logger.debug(f"CurveInterpolator Init - is valid curve set: {is_subset}")
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

    def _linear_interpolation(self) -> np.ndarray:
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

    def _log_linear_interpolation(self) -> np.ndarray:
        log_x = np.log(self._x)
        log_linspace_x = np.log(self._linspace_x)

        func_no_extrap = scipy.interpolate.interp1d(
            log_x, self._y, kind="linear", bounds_error=False
        )
        func_extrap = scipy.interpolate.interp1d(
            log_x,
            self._y,
            axis=0,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )

        ynew_no_extrap = func_no_extrap(log_linspace_x)
        ynew_extrap = func_extrap(log_linspace_x)
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

    def set_cubic_spline_interpolation_bc_type(
        self,
        bc_type: Literal["not-a-knot", "periodic", "clamped", "natural"] = "not-a-knot",
    ):
        self._cubic_spline_interp_bc_type = bc_type

    def set_cubic_spline_interpolation_custom_knots(
        self,
        custom_knots: npt.NDArray[np.float64],
    ):
        self._cubic_spline_interp_custom_knots = custom_knots

    def set_cubic_spline_interpolation_nu(self, nu: int):
        self._cubic_spline_interp_nu = nu

    def _cubic_spline_interpolation(self) -> np.ndarray:
        all_x = np.sort(
            np.concatenate([self._x, self._cubic_spline_interp_custom_knots])
        )
        all_y = np.interp(all_x, self._x, self._y)
        spline = scipy.interpolate.CubicSpline(
            all_x, all_y, bc_type=self._cubic_spline_interp_bc_type
        )
        ynew = spline(self._linspace_x, nu=self._cubic_spline_interp_nu)

        if self._enable_extrapolate_left_fill:
            left_indices = self._linspace_x < min(self._x)
            ynew[left_indices] = self._y[0] + (
                self._linspace_x[left_indices] - self._x[0]
            ) * (self._y[1] - self._y[0]) / (self._x[1] - self._x[0])

        if self._enable_extrapolate_right_fill:
            right_indices = self._linspace_x > max(self._x)
            ynew[right_indices] = self._y[-1] + (
                self._linspace_x[right_indices] - self._x[-1]
            ) * (self._y[-1] - self._y[-2]) / (self._x[-1] - self._x[-2])

        return ynew

    def _pchip_interpolation(self) -> np.ndarray:
        func_no_extrap = scipy.interpolate.PchipInterpolator(
            self._x, self._y, extrapolate=False
        )
        func_extrap = scipy.interpolate.PchipInterpolator(
            self._x, self._y, extrapolate=True
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

    def _akima_interpolation(self) -> np.ndarray:
        akima = scipy.interpolate.Akima1DInterpolator(self._x, self._y)
        ynew = akima(self._linspace_x)
        if self._enable_extrapolate_left_fill:
            left_indices = self._linspace_x < min(self._x)
            ynew[left_indices] = self._y[0] + (
                self._linspace_x[left_indices] - self._x[0]
            ) * (self._y[1] - self._y[0]) / (self._x[1] - self._x[0])

        if self._enable_extrapolate_right_fill:
            right_indices = self._linspace_x > max(self._x)
            ynew[right_indices] = self._y[-1] + (
                self._linspace_x[right_indices] - self._x[-1]
            ) * (self._y[-1] - self._y[-2]) / (self._x[-1] - self._x[-2])

        return ynew

    def set_b_spline_k(self, b: int):
        self._bspline_k = b

    def _b_spline_interpolation(self) -> np.ndarray:
        spline = scipy.interpolate.make_interp_spline(
            self._x, self._y, k=self._bspline_k
        )
        ynew = spline(self._linspace_x)

        if self._enable_extrapolate_left_fill:
            left_indices = self._linspace_x < self._x[0]
            slope_left = (self._y[1] - self._y[0]) / (self._x[1] - self._x[0])
            ynew[left_indices] = self._y[0] + slope_left * (
                self._linspace_x[left_indices] - self._x[0]
            )

        if self._enable_extrapolate_right_fill:
            right_indices = self._linspace_x > self._x[-1]
            slope_right = (self._y[-1] - self._y[-2]) / (self._x[-1] - self._x[-2])
            ynew[right_indices] = self._y[-1] + slope_right * (
                self._linspace_x[right_indices] - self._x[-1]
            )

        return ynew

    def plotter(
        self,
        linear: Optional[bool] = False,
        log_linear: Optional[bool] = False,
        cubic: Optional[bool] = False,
        cubic_bc_types_n_knots_n_nu: Optional[
            List[
                Tuple[
                    Literal["not-a-knot", "periodic", "clamped", "natural"],
                    np.ndarray,
                    int,
                ]
            ]
        ] = None,
        pchip: Optional[bool] = False,
        akima: Optional[bool] = False,
        b_spline: Optional[bool] = False,
        run_parallel: Optional[bool] = False,
    ):
        def plot_helper(ynew: np.ndarray, title: str):
            plt.figure(figsize=(17, 6))
            plt.plot(self._linspace_x, ynew, label="Interpolated")
            plt.plot(self._x, self._y, "o", label="CUSIPs")
            plt.xlabel("Maturity")
            plt.ylabel("Yield")
            plt.title(title)
            plt.legend()
            plt.show()

        plottables: Dict[Callable[[], np.ndarray], str] = {
            self._linear_interpolation: "Linear Interpolation" if linear else None,
            self._log_linear_interpolation: (
                "Log-Linear Interpolation" if log_linear else None
            ),
            self._cubic_spline_interpolation: (
                f"Cubic Spline Interpolation - {self._cubic_spline_interp_bc_type}"
                if cubic
                else None
            ),
            self._pchip_interpolation: "PCHIP Interpolation" if pchip else None,
            self._akima_interpolation: "Akima Interpolation" if akima else None,
            self._b_spline_interpolation: (
                f"B-Spline Interpolation - k = {self._bspline_k}" if b_spline else None
            ),
        }

        if run_parallel:
            to_plot = {func: title for func, title in plottables.items() if title}
            with ThreadPoolExecutor() as executor:
                future_to_title = {
                    executor.submit(func): title for func, title in to_plot.items()
                }
                for future in as_completed(future_to_title):
                    title = future_to_title[future]
                    try:
                        ynew = future.result()
                        plot_helper(ynew, title)
                    except Exception as exc:
                        print(f"{title} generated an exception: {exc}")
        else:
            for func, title in plottables.items():
                if title:
                    ynew = func()
                    plot_helper(ynew, title)

        if cubic_bc_types_n_knots_n_nu:
            for bc_type, knots, nu in cubic_bc_types_n_knots_n_nu:
                if not knots:
                    knots = np.array([])
                if not nu:
                    nu = 0
                self.set_cubic_spline_interpolation_bc_type(bc_type=bc_type)
                self.set_cubic_spline_interpolation_custom_knots(custom_knots=knots)
                self.set_cubic_spline_interpolation_nu(nu=nu)
                ynew = self._cubic_spline_interpolation()
                title = f"Cubic Spline Interpolation - Boundary condition type: {self._cubic_spline_interp_bc_type} - Knots: {self._cubic_spline_interp_custom_knots}"
                plot_helper(ynew, title)
