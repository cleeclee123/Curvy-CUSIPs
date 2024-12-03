"""
interpolations:
- Cosine
- Chebyshev 
- Orthogonal polynomials
- Monomial interpolation
- Vasicek
- Fong-Vasicek (implemented MLESM which was inspired by )
- Cox-Ingersoll-Ross 
- Smith-Wilson
- gaussian process regression interpolation
- Gaussian Mixture Models (GMM)
- Principal Component Regression (PCR)
- Nearest neighbour weighted interpolation

"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.interpolate

from CurvyCUSIPs.models.MonotoneConvex import MonotoneConvex


# TODO
# explore numba
class GeneralCurveInterpolator:
    _x: npt.ArrayLike = None
    _y: npt.ArrayLike = None
    _linspace_x_num: int = 100
    _linspace_x: npt.NDArray[np.float64] = np.zeros(
        shape=_linspace_x_num, dtype=np.float64
    )

    _enable_extrapolate_left_fill: bool = False
    _enable_extrapolate_right_fill: bool = False

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
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        linspace_x_lower_bound: Optional[float | int] = None,
        linspace_x_upper_bound: Optional[float | int] = None,
        linspace_x_num: int = 100,
        enable_extrapolate_left_fill: Optional[bool] = False,
        enable_extrapolate_right_fill: Optional[bool] = False,
        debug_verbose: Optional[bool] = False,
        info_verbose: Optional[bool] = False,
        no_logs_plz: Optional[bool] = False,
    ):
        self._x = x
        self._y = y
        self._linspace_x_num = linspace_x_num
        self._linspace_x = np.linspace(
            linspace_x_lower_bound or min(x),
            linspace_x_upper_bound or max(x),
            num=self._linspace_x_num,
        )
        self._enable_extrapolate_left_fill = enable_extrapolate_left_fill
        self._enable_extrapolate_right_fill = enable_extrapolate_right_fill

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

    def linear_interpolation(
        self, return_func=False
    ) -> np.ndarray | scipy.interpolate.interp1d:
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
        if return_func:
            return func_extrap

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

    def log_linear_interpolation(
        self, return_func=False
    ) -> np.ndarray | scipy.interpolate.interp1d:
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
        if return_func:
            return func_extrap

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

    def cubic_spline_interpolation(
        self,
        bc_type: Optional[
            Literal["not-a-knot", "periodic", "clamped", "natural"]
        ] = None,
        custom_knots: Optional[npt.NDArray[np.float64]] = None,
        nu: Optional[int] = None,
        return_func=False,
    ) -> np.ndarray | scipy.interpolate.CubicSpline:
        if bc_type:
            self._cubic_spline_interp_bc_type = bc_type
        if custom_knots:
            self._cubic_spline_interp_custom_knots = custom_knots
        if nu:
            self._cubic_spline_interp_nu = nu
        all_x = np.sort(
            np.concatenate([self._x, self._cubic_spline_interp_custom_knots])
        )
        all_y = np.interp(all_x, self._x, self._y)
        spline = scipy.interpolate.CubicSpline(
            all_x,
            all_y,
            bc_type=self._cubic_spline_interp_bc_type,
            extrapolate=True,
        )
        if return_func:
            return spline
        
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

    def _calculate_derivatives(self) -> np.ndarray:
        """Calculate derivatives at each point using finite differences."""
        dydx = np.zeros_like(self._y)
        n = len(self._x)

        # For the first point, use forward difference
        dydx[0] = (self._y[1] - self._y[0]) / (self._x[1] - self._x[0])

        # For the last point, use backward difference
        dydx[-1] = (self._y[-1] - self._y[-2]) / (self._x[-1] - self._x[-2])

        # For the interior points, use central difference
        for i in range(1, n - 1):
            dydx[i] = (self._y[i + 1] - self._y[i - 1]) / (
                self._x[i + 1] - self._x[i - 1]
            )

        return dydx

    def cubic_hermite_interpolation(self, return_func=False) -> np.ndarray:
        self._dydx = self._calculate_derivatives()
        func_no_extrap = scipy.interpolate.CubicHermiteSpline(
            self._x, self._y, self._dydx 
        )
        if return_func:
            return func_no_extrap
        
        ynew_no_extrap = func_no_extrap(self._linspace_x)
        ynew = ynew_no_extrap.copy()
        if self._enable_extrapolate_left_fill:
            left_extrap_values = (
                self._dydx[0]
                * (self._linspace_x[self._linspace_x < self._x[0]] - self._x[0])
                + self._y[0]
            )
            ynew[self._linspace_x < self._x[0]] = left_extrap_values
        if self._enable_extrapolate_right_fill:
            right_extrap_values = (
                self._dydx[-1]
                * (self._linspace_x[self._linspace_x > self._x[-1]] - self._x[-1])
                + self._y[-1]
            )
            ynew[self._linspace_x > self._x[-1]] = right_extrap_values

        return ynew

    # monotonic cubic interpolation
    def pchip_interpolation(self, return_func=False) -> np.ndarray:
        func_no_extrap = scipy.interpolate.PchipInterpolator(
            self._x, self._y, extrapolate=False
        )
        func_extrap = scipy.interpolate.PchipInterpolator(
            self._x, self._y, extrapolate=True
        )
        if return_func:
            return func_no_extrap
        
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

    def akima_interpolation(self, return_func=False) -> np.ndarray:
        akima = scipy.interpolate.Akima1DInterpolator(self._x, self._y)
        if return_func:
            return akima
        
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

    def b_spline1_interpolation(
        self, k: Optional[int] = None, return_func=False
    ) -> np.ndarray:
        if k:
            self._bspline_k = k
        spline = scipy.interpolate.make_interp_spline(
            self._x, self._y, k=self._bspline_k
        )
        if return_func:
            return spline
        
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

    def b_spline_with_knots_interpolation(
        self, knots: Optional[npt.ArrayLike], k: Optional[int] = 3, return_func=False
    ) -> np.ndarray:
        tck = scipy.interpolate.splrep(self._x, self._y, t=knots, k=k)
        # bspline = splev(ttm_interp, tck)
        bspline = scipy.interpolate.BSpline(*tck)
        if return_func:
            return bspline

        ynew_no_extrap = bspline(self._linspace_x)
        ynew_extrap = bspline(self._linspace_x)
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

    def univariate_spline(self, s: float, return_func=False):
        unispline = scipy.interpolate.UnivariateSpline(
            x=self._x,
            y=self._y,
            s=s
        )
        if return_func:
            return unispline
        
        ynew_no_extrap = unispline(self._linspace_x)
        ynew_extrap = unispline(self._linspace_x)
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

    
    def ppoly_interpolation(self, return_func=False) -> np.ndarray:
        coeffs = scipy.interpolate.CubicSpline(self._x, self._y).c
        ppoly = scipy.interpolate.PPoly(coeffs, self._x)
        if return_func:
            return ppoly

        ynew_no_extrap = ppoly(self._linspace_x)
        ynew = ynew_no_extrap.copy()
        if self._enable_extrapolate_left_fill:
            left_extrap_values = ppoly(self._linspace_x[self._linspace_x < self._x[0]])
            ynew[self._linspace_x < self._x[0]] = left_extrap_values
        if self._enable_extrapolate_right_fill:
            right_extrap_values = ppoly(
                self._linspace_x[self._linspace_x > self._x[-1]]
            )
            ynew[self._linspace_x > self._x[-1]] = right_extrap_values

        return ynew

    def monotone_convex(self) -> np.ndarray:
        mc_spline = MonotoneConvex(terms=self._x, spots=self._y)
        def mc_spline_func(t):
            return [mc_spline.spot(val) for val in t]
        return mc_spline_func
    
    def smoothing_spline(self, w=None, lam=None):
        return scipy.interpolate.make_smoothing_spline(self._x, self._y, w=w, lam=lam)
    
    def lsq_univariate_soline(self, knots: np.ndarray, k=3):
        return scipy.interpolate.LSQUnivariateSpline(self._x, self._y, t=knots, k=k)

    def plotter(
        self,
        linear: Optional[bool] = False,
        log_linear: Optional[bool] = False,
        cubic: Optional[bool] = False,
        cubic_hermite: Optional[bool] = False,
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
        ppoly: Optional[bool] = False,
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
            self._cubic_hermite_interpolation: (
                f"Cubic Hermite Spline Interpolation" if cubic_hermite else None
            ),
            self._pchip_interpolation: (
                "PCHIP (Monotonic & Hermite Cubic) Interpolation" if pchip else None
            ),
            self._akima_interpolation: "Akima Interpolation" if akima else None,
            self._b_spline_interpolation: (
                f"B-Spline Interpolation - k = {self._bspline_k}" if b_spline else None
            ),
            self._ppoly_interpolation: (f"PPoly Interpolation" if ppoly else None),
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
