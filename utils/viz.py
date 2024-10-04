from datetime import datetime
from typing import Annotated, List, Optional, Tuple, Callable

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import newton
from scipy.interpolate import UnivariateSpline
from scipy.stats import zscore
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import statsmodels.api as sm

sns.set_style("whitegrid", {"grid.linestyle": "--"})

import nest_asyncio

nest_asyncio.apply()

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


def plot_yields(
    df: pd.DataFrame,
    y_cols: List[str],
    x_col="Date",
    max_ticks=10,
    flip=False,
    custom_label_x=None,
    custom_label_y=None,
    custom_title=None,
    date_subset_range: Annotated[List[datetime], 2] | None = None,
    recessions: Annotated[List[List[datetime]], 2] = None,
    bar_plot=False,
):
    copy_df = df.copy()
    copy_df["Date"] = pd.to_datetime(copy_df["Date"])
    if date_subset_range:
        copy_df = copy_df[(copy_df["Date"] >= date_subset_range[0]) & (copy_df["Date"] <= date_subset_range[1])]
    if flip:
        copy_df = copy_df.iloc[::-1]

    fig = go.Figure()
    for y_col in y_cols:
        if bar_plot:
            fig.add_trace(go.Bar(x=copy_df[x_col], y=copy_df[y_col], name=y_col, marker_color="black"))
        else:
            fig.add_trace(go.Scatter(x=copy_df[x_col], y=copy_df[y_col], mode="lines", name=y_col))

    if recessions:
        if date_subset_range:
            start_plot_range, end_plot_range = min(date_subset_range), max(date_subset_range)
        else:
            start_plot_range, end_plot_range = min(copy_df["Date"]), max(copy_df["Date"])

        for recession_dates in recessions:
            start_date, end_date = recession_dates
            if start_date <= end_plot_range and end_date >= start_plot_range:
                fig.add_vrect(
                    x0=start_date,
                    x1=end_date,
                    fillcolor="red",
                    opacity=0.3,
                    layer="below",
                    line_width=0,
                )

    fig.update_layout(
        xaxis_title=custom_label_x or x_col,
        yaxis_title=custom_label_y or ", ".join(y_cols),
        xaxis=dict(nticks=max_ticks),
        title=custom_title or "Yield Plot",
        showlegend=True,
        template="plotly_dark",
        height=700,
    )
    fig.update_xaxes(showspikes=True, spikecolor="white", spikesnap="cursor", spikemode="across")
    fig.update_yaxes(showspikes=True, spikecolor="white", spikesnap="cursor", spikethickness=0.5)
    fig.show(
        config={
            "modeBarButtonsToAdd": [
                "drawline",
                "drawopenpath",
                "drawclosedpath",
                "drawcircle",
                "drawrect",
                "eraseshape",
            ]
        }
    )


def plot_yield_curve_date_range(
    df: pd.DataFrame,
    date_range: List[datetime],
    reverse_mats=False,
    real_yields=False,
    adjusted=False,
    del_mats=None,
):
    maturities = df.columns[1:]
    months = [6, 12, 24, 36, 60, 84, 120, 240, 360]
    months = [50, 84, 120, 240, 360] if real_yields else months
    if adjusted:
        months = [0, 10, 20, 30, 60, 90, 110, 130, 170, 200, 280, 370, 520]
        months = [0, 10, 25, 50, 75] if real_yields else months

    if del_mats:
        df = df.drop(columns=del_mats)
        indexes = [maturities.index(to_del) for to_del in del_mats]
        maturities = [i for j, i in enumerate(maturities) if j not in indexes]
        months = [i for j, i in enumerate(months) if j not in indexes]

    fig = go.Figure()
    for date in date_range:
        try:
            formatted_date = date.strftime("%Y-%m-%d")
            tooltip_formatted_date = date.strftime("%m/%d/%Y")
            yields = df.loc[df["Date"] == formatted_date]
            yields = yields.values.tolist()[0]
            del yields[0]

            if reverse_mats:
                yields.reverse()
                maturities.reverse()

            fig.add_trace(
                go.Scatter(
                    x=months,
                    y=yields,
                    mode="lines+markers",
                    name=f"{tooltip_formatted_date}",
                )
            )
        except:
            print(f"Failed to plot: {date}")

    formatted_title = ""
    for dr in date_range:
        formatted_title += f"{dr.strftime('%Y-%m-%d')} vs "
    formatted_title = formatted_title[:-3]

    fig.update_layout(
        title=formatted_title if not real_yields else f"{formatted_title} (Real)",
        xaxis_title="Maturity (Months)",
        yaxis_title="Yield (%)",
        xaxis=dict(tickvals=months, ticktext=maturities),
        showlegend=True,
        height=700,
        template="plotly_dark",
    )
    fig.update_xaxes(showspikes=True, spikecolor="white", spikesnap="cursor", spikemode="across")
    fig.update_yaxes(showspikes=True, spikecolor="white", spikesnap="cursor", spikethickness=0.5)
    fig.show(
        config={
            "modeBarButtonsToAdd": [
                "drawline",
                "drawopenpath",
                "drawclosedpath",
                "drawcircle",
                "drawrect",
                "eraseshape",
            ]
        }
    )


def run_basic_linear_regression(x_series: pd.Series, y_series: pd.Series, x_label, y_label, title: Optional[str] = None):
    Y = y_series
    X = x_series
    X = sm.add_constant(X)
    model = sm.OLS(Y, X)
    results = model.fit()
    print(results.summary())

    intercept = results.params[0]
    slope = results.params[1]
    r_squared = results.rsquared

    plt.figure(figsize=(20, 10))
    plt.scatter(x_series, y_series)

    regression_line = intercept + slope * x_series
    plt.plot(x_series, regression_line, color="red")

    plt.ylabel(x_label)
    plt.xlabel(y_label)
    plt.title(title or f"{y_label} Regressed on {x_label}")
    equation_text = f"y = {intercept:.3f} + {slope:.3f}x \n R² = {r_squared:.3f} \n SE = {results.bse["const"]:.3f}"
    plt.plot([], [], " ", label=f"{equation_text}")

    plt.legend(fontsize="x-large")
    plt.grid(True)
    plt.show()

    return results


def run_basic_linear_regression_df(df: pd.DataFrame, x_col: str, y_col: str, title: Optional[str] = None):
    if x_col not in df.columns or y_col not in df.columns:
        raise Exception(f"{x_col} or {y_col} not in df cols")

    Y = df[y_col]
    X = df[x_col]
    X = sm.add_constant(X)
    model = sm.OLS(Y, X)
    results = model.fit()
    print(results.summary())

    intercept = results.params[0]
    slope = results.params[1]
    r_squared = results.rsquared
    p_value = results.pvalues[1] if len(results.pvalues) > 1 else None

    plt.figure(figsize=(20, 10))
    plt.scatter(df[x_col], df[y_col])
    most_recent = df["Date"].iloc[-1]
    plt.scatter(
        df[x_col].iloc[-1],
        df[y_col].iloc[-1],
        color="purple",
        s=100,
        label=f"Most Recent: {most_recent}",
    )

    regression_line = intercept + slope * df[x_col]
    plt.plot(df[x_col], regression_line, color="red")

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title or f"{y_col} Regressed on {x_col}", fontdict={"fontsize": "x-large"})
    equation_text = f"y = {intercept:.3f} + {slope:.3f}x\nR² = {r_squared:.3f}\nSE = {results.bse["const"]:.3f}\np-value (x) = {p_value:.3e}"
    plt.plot([], [], " ", label=f"{equation_text}")
    plt.legend(fontsize="x-large")
    plt.grid(True)
    plt.show()

    return results


def plot_residuals_timeseries(
    df: pd.DataFrame, results: sm.regression.linear_model.RegressionResultsWrapper, date_col: str = "Date", plot_zscores: Optional[bool] = False
):
    residuals = results.resid
    zscores = zscore(residuals)
    
    if date_col not in df.columns:
        raise Exception(f"{date_col} not in df columns")

    r_squared = results.rsquared
    intercept = results.params[0]
    slope = results.params[1]
    p_value = results.pvalues[1] if len(results.pvalues) > 1 else None
    dependent_variable = results.model.endog_names
    independent_variables = results.model.exog_names[1]

    if p_value is not None:
        title = f"Residuals of {dependent_variable} Regressed on {independent_variables} Over Time\n"
    else:
        title = f"Residuals of {dependent_variable} Regressed on {independent_variables} Over Time\n"

    plt.figure(figsize=(20, 10))
    plt.plot(df[date_col], residuals if not plot_zscores else zscores, linestyle="-", color="blue")
    plt.axhline(y=0, color="red", linestyle="--")
    equation_text = f"y = {intercept:.3f} + {slope:.3f}x\nR² = {r_squared:.3f}\nSE = {results.bse["const"]:.3f}\np-value (x) = {p_value:.3e}"
    plt.plot([], [], " ", label=f"{equation_text}")
    plt.legend(fontsize="x-large")
    plt.xlabel("Date")
    plt.ylabel("Residuals" if not plot_zscores else "Z-Scores")
    plt.title(title + ", Z-Scroes" if plot_zscores else title, fontdict={"fontsize": "x-large"})
    plt.grid(True)
    plt.show()


def par_curve_func(tenor, zero_curve_func, is_parametric_class=False):
    if is_parametric_class:

        def par_bond_equation(c, maturity, zero_curve_func):
            discounted_cash_flows = sum((c / 2) * np.exp(-(zero_curve_func([t])[0] / 100) * t) for t in np.arange(0.5, maturity + 0.5, 0.5))
            final_payment = 100 * np.exp(-(zero_curve_func([maturity])[0] / 100) * maturity)
            return discounted_cash_flows + final_payment - 100

    else:

        def par_bond_equation(c, maturity, zero_curve_func):
            discounted_cash_flows = sum((c / 2) * np.exp(-(zero_curve_func(t) / 100) * t) for t in np.arange(0.5, maturity + 0.5, 0.5))
            final_payment = 100 * np.exp(-(zero_curve_func(maturity) / 100) * maturity)
            return discounted_cash_flows + final_payment - 100

    init_guess = 4
    return newton(
        par_bond_equation,
        x0=init_guess,
        args=(tenor, zero_curve_func),
    )


def plot_usts(
    curve_set_df: pd.DataFrame,
    ttm_col: Optional[str] = "time_to_maturity",
    ytm_col: Optional[str] = "ytm",
    label_col: Optional[str] = "original_security_term",
    hover_data: Optional[List[str]] = None,
    title: Optional[str] = None,
    custom_x_axis: Optional[str] = "Time to Maturity",
    custom_y_axis: Optional[str] = "Yield to Maturity",
    zero_curves: Optional[List[Tuple[Callable, str]]] = None,
    par_curves: Optional[List[Tuple[Callable, str, bool]]] = None,
    n_yr_fwd_curves: Optional[List[Tuple[Callable, int, str]]] = None,
    impl_spot_n_yr_fwd_curves: Optional[List[Tuple[Callable, int, str]]] = None,
    impl_par_n_yr_fwd_curves: Optional[List[Tuple[Callable, int, str, bool]]] = None,
    cusips_filter: Optional[List[str]] = None,
    ust_labels_filter: Optional[List[str]] = None,
    cusips_hightlighter: Optional[List[str]] = None,
    ust_labels_highlighter: Optional[List[Tuple[str, str] | str]] = None,
    linspace_num: Optional[int] = 1000,
    y_axis_range: Optional[Annotated[List[int], 2]] = [3.33, 5.5],
    plot_height=1000,
):
    curve_set_df = curve_set_df.copy()

    if cusips_filter:
        curve_set_df = curve_set_df[curve_set_df["cusip"].isin(cusips_filter)]
        cusips_filter_set = set(cusips_filter)
        cusips_in_df = set(curve_set_df["cusip"].unique())
        cusips_not_in_df = cusips_filter_set - cusips_in_df
        print("CUSIPs not in Curveset df:", cusips_not_in_df)

    curve_set_df["int_rate"] = pd.to_numeric(curve_set_df["int_rate"], errors="coerce")
    curve_set_df["ust_label"] = curve_set_df.apply(
        lambda row: f"{row['int_rate']:.3f}% {row['maturity_date'].strftime('%b-%y')}",
        axis=1,
    )
    if ust_labels_filter:
        curve_set_df = curve_set_df[curve_set_df["ust_label"].isin(ust_labels_filter)]
        ust_labels_filter_set = set(ust_labels_filter)
        labels_in_df = set(curve_set_df["ust_label"].unique())
        labels_not_in_df = ust_labels_filter_set - labels_in_df
        print("Labels not in Curveset df:", labels_not_in_df)

    otr_mask = curve_set_df["rank"] == 0
    fig = px.scatter(
        curve_set_df[~otr_mask],
        x=ttm_col,
        y=ytm_col,
        color=label_col,
        hover_data=hover_data,
    )

    curve_set_df.loc[otr_mask, label_col] = curve_set_df.loc[otr_mask, label_col].apply(lambda x: f"OTR - {x}")
    otr_fig = px.scatter(
        curve_set_df[otr_mask],
        x=ttm_col,
        y=ytm_col,
        color=label_col,
        hover_data=hover_data,
    )
    for trace in otr_fig.data:
        trace.update(
            marker=dict(
                line=dict(color="white", width=2),
            )
        )
    fig.add_traces(otr_fig.data)

    if cusips_hightlighter:
        if isinstance(cusips_hightlighter[0], tuple):
            for cusip_tuple in cusips_hightlighter:
                if not isinstance(cusip_tuple, tuple):
                    cusip = cusip_tuple
                    label_color = "yellow"
                else:
                    cusip, label_color = cusip_tuple

                if cusip not in curve_set_df["cusip"].values:
                    print(f"{cusip} not in Curveset df!")
                    continue

                cusip_highlight_fig = px.scatter(
                    curve_set_df[curve_set_df["cusip"] == cusip],
                    x=ttm_col,
                    y=ytm_col,
                    color="cusip",
                    hover_data=hover_data,
                )
                for trace in cusip_highlight_fig.data:
                    trace.update(
                        marker=dict(
                            line=dict(color=label_color, width=4),
                        )
                    )
                fig.add_traces(cusip_highlight_fig.data)
        else:
            cusip_highlight_mask = curve_set_df["cusip"].isin(cusips_hightlighter)
            cusip_highlight_fig = px.scatter(
                curve_set_df[cusip_highlight_mask],
                x=ttm_col,
                y=ytm_col,
                color="cusip",
                hover_data=hover_data,
            )
            for trace in cusip_highlight_fig.data:
                trace.update(
                    marker=dict(
                        line=dict(color="yellow", width=4),
                    )
                )
            fig.add_traces(cusip_highlight_fig.data)

    if ust_labels_highlighter:
        if isinstance(ust_labels_highlighter[0], tuple):
            for label_tuple in ust_labels_highlighter:
                if not isinstance(label_tuple, tuple):
                    ust_label = label_tuple
                    label_color = "yellow"
                else:
                    ust_label, label_color = label_tuple

                if ust_label not in curve_set_df["ust_label"].values:
                    print(f"{ust_label} not in Curveset df!")
                    continue

                ust_labels_highlight_fig = px.scatter(
                    curve_set_df[curve_set_df["ust_label"] == ust_label],
                    x=ttm_col,
                    y=ytm_col,
                    color="ust_label",
                    hover_data=hover_data,
                )
                for trace in ust_labels_highlight_fig.data:
                    trace.update(
                        marker=dict(
                            line=dict(color=label_color, width=4),
                        )
                    )
                fig.add_traces(ust_labels_highlight_fig.data)
        else:
            ust_labels_highlight_mask = curve_set_df["ust_label"].isin(ust_labels_highlighter)
            ust_labels_highlight_fig = px.scatter(
                curve_set_df[ust_labels_highlight_mask],
                x=ttm_col,
                y=ytm_col,
                color="ust_label",
                hover_data=hover_data,
            )
            for trace in ust_labels_highlight_fig.data:
                trace.update(
                    marker=dict(
                        line=dict(color="yellow", width=4),
                    )
                )
            fig.add_traces(ust_labels_highlight_fig.data)

    if zero_curves:
        ttm_linspace = np.linspace(0, 30, linspace_num)
        for curve_tup in zero_curves:
            interp_func, label = curve_tup
            fig.add_trace(
                go.Scatter(
                    x=ttm_linspace,
                    y=interp_func(ttm_linspace),
                    mode="lines",
                    name=label,
                )
            )

    if par_curves:
        cfs = np.arange(0.5, 30 + 1, 0.5)
        for curve_tup in par_curves:
            if not len(curve_tup) == 3:
                curve_tup = curve_tup + (False,)
            interp_func, label, is_parametric_class = curve_tup
            fig.add_trace(
                go.Scatter(
                    x=cfs,
                    y=[par_curve_func(t, interp_func, is_parametric_class) for t in cfs],
                    mode="lines",
                    name=label,
                )
            )

    if n_yr_fwd_curves:
        ttm_linspace = np.linspace(0.5, 30, linspace_num)
        for curve_tup in n_yr_fwd_curves:
            interp_func, n, label = curve_tup
            Z_t = interp_func(ttm_linspace)

            if n == -1 or n == np.inf:
                try:
                    interp_func_derivative = interp_func.derivative()
                    dZ_dt = interp_func_derivative(ttm_linspace)
                    F_t = Z_t + ttm_linspace * dZ_dt
                    fig.add_trace(
                        go.Scatter(
                            x=ttm_linspace,
                            y=F_t,
                            mode="lines",
                            name=label,
                        )
                    )
                except Exception as e:
                    print("only scipy interpolation function are supported for inst fwds")
                    print(e)
            else:
                forward_rates = []
                for t1 in ttm_linspace:
                    t2 = t1 + n
                    Z_t1 = interp_func(t1)
                    Z_t2 = interp_func(t2)
                    fwd_rate = (Z_t2 * t2 - Z_t1 * t1) / (t2 - t1)
                    forward_rates.append(fwd_rate)

                fig.add_trace(
                    go.Scatter(
                        x=ttm_linspace,
                        y=forward_rates,
                        mode="lines",
                        name=label,
                    )
                )

    if impl_spot_n_yr_fwd_curves:
        cfs = np.arange(0.5, 30 + 1, 0.5)
        for curve_tup in impl_spot_n_yr_fwd_curves:
            interp_func, n, label = curve_tup
            implied_spot_rates = []
            for t in cfs:
                if t > n:
                    Z_t_temp = interp_func(t)
                    Z_n = interp_func(n)
                    Z_n_t = (Z_t_temp * t - Z_n * n) / (t - n)
                    implied_spot_rates.append(Z_n_t)
                else:
                    implied_spot_rates.append(np.nan)
            implied_spot_rates = np.array(implied_spot_rates)
            fig.add_trace(
                go.Scatter(
                    x=cfs,
                    y=implied_spot_rates,
                    mode="lines",
                    name=label,
                )
            )

    if impl_par_n_yr_fwd_curves:
        cfs = np.arange(0.5, 30 + 1, 0.5)
        for curve_tup in impl_par_n_yr_fwd_curves:
            if not len(curve_tup) == 4:
                curve_tup = curve_tup + (False,)

            interp_func, n, label, is_parametric_class = curve_tup
            implied_spot_rates_n_fwd = []
            if is_parametric_class:
                for t in cfs:
                    if t > n:
                        Z_t_temp = interp_func([t])[0]
                        Z_n = interp_func([n])[0]
                        Z_n_t = (Z_t_temp * t - Z_n * n) / (t - n)
                        implied_spot_rates_n_fwd.append(Z_n_t)
                    else:
                        implied_spot_rates_n_fwd.append(np.nan)
            else:
                for t in cfs:
                    if t > n:
                        Z_t_temp = interp_func(t)
                        Z_n = interp_func(n)
                        Z_n_t = (Z_t_temp * t - Z_n * n) / (t - n)
                        implied_spot_rates_n_fwd.append(Z_n_t)
                    else:
                        implied_spot_rates_n_fwd.append(np.nan)
            implied_spot_rates_n_fwd = np.array(implied_spot_rates_n_fwd)[n * 2 :]
            implied_par_rates_n_fwd_spline = UnivariateSpline(cfs[n * 2 :], implied_spot_rates_n_fwd, s=0, k=3)
            fig.add_trace(
                go.Scatter(
                    x=cfs[n * 2 :],
                    y=[par_curve_func(t, implied_par_rates_n_fwd_spline, is_parametric_class) for t in cfs[n * 2 :]],
                    mode="lines",
                    name=label,
                )
            )

    fig.update_layout(
        xaxis_title=custom_x_axis,
        yaxis_title=custom_y_axis,
        title=title or "Yield Curve",
        showlegend=True,
        template="plotly_dark",
        height=plot_height,
    )
    fig.update_xaxes(showspikes=True, spikecolor="white", spikesnap="cursor", spikemode="across")
    fig.update_yaxes(
        showspikes=True,
        spikecolor="white",
        spikesnap="cursor",
        spikethickness=0.5,
        range=y_axis_range,
    )
    fig.show(
        config={
            "modeBarButtonsToAdd": [
                "drawline",
                "drawopenpath",
                "drawclosedpath",
                "drawcircle",
                "drawrect",
                "eraseshape",
            ]
        }
    )
