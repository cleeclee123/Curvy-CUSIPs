from CurveDataFetcher import CurveDataFetcher
from CurveBuilder import calc_ust_metrics, calc_ust_impl_spot_n_fwd_curve

from datetime import datetime, timedelta
from typing import Annotated, Callable, List, Optional, Tuple, Dict
from termcolor import colored
import numpy.typing as npt

import ujson as json
import scipy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import tstd, zscore, linregress
from scipy.odr import ODR, Model, RealData, Data, Output
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import plotly.express as px
import plotly.graph_objects as go


def ols_hedge_ratio(
    df: pd.DataFrame,
    x_cols: List[str],
    y_col: str,
    run_on_changes: Optional[bool] = False,
    last_n_day_regressions: Optional[List[int]] = None,
    show_last_n_day_points: Optional[bool] = False,
):
    df = df[["Date"] + x_cols + [y_col]].copy()
    if run_on_changes:
        date_col = df["Date"]
        df = df[x_cols + [y_col]].diff()
        df["Date"] = date_col
    df = df.dropna()

    if not last_n_day_regressions:
        last_n_day_regressions = [-1]
    else:
        last_n_day_regressions.append(-1)

    regressions_results: Dict[int, pd.DataFrame | sm.regression.linear_model.RegressionResultsWrapper | int | float | str] = {}
    for last_n_days in last_n_day_regressions:
        curr_df = df.tail(last_n_days)

        Y = curr_df[y_col]
        X = curr_df[x_cols]

        X = sm.add_constant(X)
        model = sm.OLS(Y, X)
        results = model.fit()
        print(
            colored(f"Regression Result for Last {last_n_days} Days:", "green")
            if last_n_days != -1
            else colored("Regression Results - Entire Dateset:\n", "green")
        )
        print(results.summary())
        print("\n\n")

        intercept = results.params["const"]
        slope1 = results.params[x_cols[0]]
        r_squared = results.rsquared
        adj_r_squared = results.rsquared_adj

        slopes = results.params
        equation_text = f"y = {intercept:.3f}" if len(x_cols) == 1 else f"y = {slopes["const"]:.3f}"
        for col in slopes.index:
            if col != "const":
                coef = slopes[col]
                equation_text += f" + {coef:.3f}*{col}"

        if len(x_cols) == 1:
            equation_text_with_r2 = equation_text + f"<br>R² = {r_squared:.3f}"
        if len(x_cols) == 2:
            equation_text_with_r2 = equation_text + f"<br>R² = {r_squared:.3f}, Adjusted R² = {adj_r_squared:.3f}"

        curr_title = (
            f"{y_col} Regressed on {x_cols}<br>{equation_text_with_r2}"
            if not run_on_changes
            else f"Changes in {y_col} Regressed on Changes in {x_cols}<br>{equation_text_with_r2}<br>"
        )

        regressions_results[last_n_days] = {
            "df": curr_df.copy(),
            "results": results,
            "intercept": intercept,
            "slopes": slopes,
            "slope1": slope1,
            "r_squared": r_squared,
            "adj_r_squared": adj_r_squared,
            "equation_text": equation_text,
            "equation_text_with_r2": equation_text_with_r2,
            "curr_title": curr_title,
        }

    if len(x_cols) == 1:
        x_min = df[x_cols[0]].min()
        x_max = df[x_cols[0]].max()
        x_range = np.linspace(x_min, x_max, 100)

        fig = go.Figure()
        for i, (last_n_days, regressions_result) in enumerate(regressions_results.items()):
            curr_df = regressions_result["df"]
            intercept = regressions_result["intercept"]
            slope = regressions_result["slopes"][x_cols[0]]
            curr_title = regressions_result["curr_title"]
            y_range = intercept + slope * x_range

            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_range,
                    mode="lines",
                    name=(
                        f'{regressions_result["equation_text_with_r2"]}, Last N: {last_n_days}'
                        if last_n_days != -1
                        else regressions_result["equation_text_with_r2"]
                    ),
                    line=dict(width=3),
                )
            )
            if show_last_n_day_points or last_n_days == -1:
                fig.add_trace(
                    go.Scatter(
                        x=curr_df[x_cols[0]],
                        y=curr_df[y_col],
                        mode="markers",
                        name=f"Data (Last {last_n_days} days)" if last_n_days != -1 else "Data (Entire Dataset)",
                        hovertext=[
                            f"Date: {row['Date']}<br>{x_cols[0]}: {row[x_cols[0]]:.4f}<br>{y_col}: {row[y_col]:.4f}" for _, row in curr_df.iterrows()
                        ],
                        hoverinfo="text",
                        opacity=0.7,
                    )
                )

        fig.update_layout(
            title=curr_title,
            xaxis_title=x_cols[0],
            yaxis_title=y_col,
            template="plotly_dark",
            height=900,
            showlegend=True,
        )
        fig.update_xaxes(showspikes=True, spikecolor="white", spikesnap="cursor", spikemode="across")
        fig.update_yaxes(showspikes=True, spikecolor="white", spikesnap="cursor", spikethickness=0.5)
        fig.show()

    if len(x_cols) == 2:
        mesh_size = 0.1  # Adjust mesh size for smoother or coarser grid
        margin = 0
        x_min, x_max = X[x_cols[0]].min() - margin, X[x_cols[0]].max() + margin
        y_min, y_max = X[x_cols[1]].min() - margin, X[x_cols[1]].max() + margin
        x_range = np.arange(x_min, x_max, mesh_size)
        y_range = np.arange(y_min, y_max, mesh_size)
        xx, yy = np.meshgrid(x_range, y_range)

        # fig = px.scatter_3d(df, x=x_cols[0], y=x_cols[1], z=y_col)
        # fig.add_traces(go.Surface(x=x_range, y=y_range, z=z_pred, name="OLS Regression Plane", opacity=0.7, showscale=False))
        fig = go.Figure()
        surface_names = []
        for i, (last_n_days, regressions_result) in enumerate(regressions_results.items()):
            curr_df: pd.DataFrame = regressions_result["df"]
            curr_z_pred = regressions_result["intercept"] + regressions_result["slope1"] * xx + regressions_result["results"].params[x_cols[1]] * yy
            scatter = go.Scatter3d(
                x=curr_df[x_cols[0]],
                y=curr_df[x_cols[1]],
                z=curr_df[y_col],
                mode="markers",
                marker=dict(size=4, color=px.colors.qualitative.Plotly[i]),
                name=(
                    f'{regressions_result["equation_text_with_r2"]}, Last N: {last_n_days}'
                    if last_n_days != -1
                    else regressions_result["equation_text_with_r2"]
                ),
                hovertext=[
                    f"Date: {row["Date"]}<br>"
                    + f"{x_cols[0]}: {row[x_cols[0]]:.4f}<br>"
                    + f"{y_col}: {row[y_col]:.4f}<br>"
                    + f"{x_cols[1]}: {row[x_cols[1]]:.4f}"
                    for _, row in curr_df.iterrows()
                ],
                hoverinfo="text",
            )
            surface_name = f"Last {last_n_days} Days - OLS Regression Plane" if not last_n_days == -1 else "Entire Timeframe - OLS Regression Plane"
            surface_names.append(surface_name)
            surface = go.Surface(
                x=x_range,
                y=y_range,
                z=curr_z_pred,
                name=surface_name,
                opacity=0.7,
                showscale=False,
                colorscale=px.colors.named_colorscales()[i],
            )
            # dummy scatter trace for the surface legend entry
            dummy_surface = go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker=dict(size=10, color=px.colors.qualitative.Plotly[i], symbol="square"),
                name=surface_name,
                showlegend=True,
            )
            fig.add_trace(scatter)
            fig.add_trace(surface)
            fig.add_trace(dummy_surface)

        button_all = dict(
            label="Show All",
            method="update",
            args=[{"visible": [True] * len(x_cols) * 3}],
        )
        button_hide = dict(
            label="Hide All",
            method="update",
            args=[{"visible": [False] * len(x_cols) * 3}],
        )

        surface_buttons = []
        for i, (last_n_days, regressions_result) in enumerate(regressions_results.items()):
            visible_props = [False] * 3 * len(surface_names)
            visible_props[i * 3 + 0] = True
            visible_props[i * 3 + 1] = True
            visible_props[i * 3 + 2] = True
            surface_buttons.append(
                dict(
                    label=surface_names[i],
                    method="update",
                    args=[{"visible": visible_props}, {"title": surface_names[i] + "<br>" + regressions_result["curr_title"]}],
                )
            )
        updatemenus = [
            dict(
                buttons=[button_all, button_hide] + surface_buttons,
                direction="down",
                pad={"r": 1, "t": 1},
                showactive=True,
                x=0.0,
                xanchor="right",
                y=0.5,
                yanchor="bottom",
            ),
        ]
        fig.update_layout(
            updatemenus=updatemenus,
            title=regressions_results[-1]["curr_title"],
            scene=dict(xaxis_title=x_cols[0], yaxis_title=x_cols[1], zaxis_title=y_col),
            showlegend=True,
            height=700,
            # width=1400,
            template="plotly_dark",
        )
        fig.update_xaxes(showspikes=True, spikecolor="white", spikesnap="cursor", spikemode="across")
        fig.update_yaxes(showspikes=True, spikecolor="white", spikesnap="cursor", spikethickness=0.5)
        fig.show()

    results = regressions_results[-1]["results"]
    Y_pred = results.fittedvalues
    residuals = results.resid
    Y_pred = np.array(Y_pred)
    residuals = np.array(residuals)
    fig_resid = go.Figure()
    fig_resid.add_trace(
        go.Scatter(
            x=Y_pred,
            y=residuals,
            mode="markers",
            name=f"Residuals (Last {last_n_days} days)" if last_n_days != -1 else "Residuals (Entire Dataset)",
        )
    )
    fig_resid.add_trace(
        go.Scatter(
            x=[Y_pred[-1]],
            y=[residuals[-1]],
            mode="markers",
            marker=dict(color="red", size=10),
            name=f"Most Recent Residual: {curr_df['Date'].iloc[-1]}",
        )
    )
    fig_resid.add_shape(type="line", x0=Y_pred.min(), y0=0, x1=Y_pred.max(), y1=0, line=dict(color="Red", dash="dash"))
    fig_resid.update_layout(
        title="Residuals vs Predicted",
        xaxis_title="Predicted Values",
        yaxis_title="Residuals",
        template="plotly_dark",
        legend=dict(x=0.7, y=1.0, bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
    )
    fig_resid.update_xaxes(showspikes=True, spikecolor="white", spikesnap="cursor", spikemode="across")
    fig_resid.update_yaxes(showspikes=True, spikecolor="white", spikesnap="cursor", spikethickness=0.5)
    fig_resid.show()

    return results


# ols, single and multi
# tls
# odr
# pca


def _run_odr(df: pd.DataFrame, x_cols: List[str], y_col: str, x_errs: Optional[npt.ArrayLike] = None, y_errs: Optional[npt.ArrayLike] = None):
    def orthoregress(
        x: pd.Series | npt.ArrayLike, y: pd.Series | npt.ArrayLike, x_errs: Optional[npt.ArrayLike] = None, y_errs: Optional[npt.ArrayLike] = None
    ):
        # calc weights (inverse variances)
        wd = None
        we = None
        if x_errs is not None:
            wd = 1.0 / np.square(x_errs)
        if y_errs is not None:
            we = 1.0 / np.square(y_errs)

        def f(p, x):
            return (p[0] * x) + p[1]

        od = ODR(Data(x, y, wd=wd, we=we), Model(f), beta0=linregress(x, y)[0:2])
        out = od.run()
        return out

    def orthoregress_multilinear(
        X: pd.DataFrame | pd.Series | npt.ArrayLike,
        y: pd.Series | npt.ArrayLike,
        x_errs: Optional[npt.ArrayLike] = None,
        y_errs: Optional[npt.ArrayLike] = None,
    ):
        # calc weights (inverse variances)
        wd = None
        we = None
        if x_errs is not None:
            x_errs = np.asarray(x_errs)
            wd = 1.0 / np.square(x_errs.T)  # transpose to match ODR shape
        if y_errs is not None:
            we = 1.0 / np.square(y_errs)

        def multilinear_f(p, x):
            return np.dot(p[:-1], x) + p[-1]

        X = np.asarray(X)
        y = np.asarray(y)
        X_odr = X.T
        y_flat = y.flatten()
        X_with_intercept = np.column_stack((X, np.ones(X.shape[0])))
        beta_init, _, _, _ = np.linalg.lstsq(X_with_intercept, y_flat, rcond=None)
        beta0 = np.append(beta_init[:-1], beta_init[-1])
        model = Model(multilinear_f)
        data = Data(X_odr, y, wd=wd, we=we)
        odr_instance = ODR(data, model, beta0=beta0)
        output = odr_instance.run()
        return output

    if len(x_cols) > 1:
        out = orthoregress_multilinear(df[x_cols], df[y_col], x_errs, y_errs)
    else:
        out = orthoregress(df[x_cols[0]], df[y_col], x_errs, y_errs)
    out.beta = np.roll(out.beta, 1)
    return out


def _run_pca(df: pd.DataFrame, x_cols: List[str], y_col: str, run_pca_on_corr_mat: Optional[bool] = False):
    data = df[x_cols + [y_col]]
    if run_pca_on_corr_mat:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        data_to_fit = data_scaled
    else:
        data_to_fit = data.values

    pca = PCA().fit(data_to_fit)
    return pca


def _run_pcr(
    df: pd.DataFrame,
    x_cols: List[str],
    y_col: str,
    n_components: Optional[int] = None,
    run_pca_on_corr_mat: Optional[bool] = False,
):
    X = df[x_cols].values
    y = df[y_col].values

    if run_pca_on_corr_mat:
        scaler = StandardScaler()
        X_to_fit = scaler.fit_transform(X)
    else:
        X_to_fit = X

    if n_components is None:
        n_components = min(len(x_cols), X_to_fit.shape[0])

    pca = PCA(n_components=n_components).fit_transform(X_to_fit)
    X_pca_with_const = sm.add_constant(pca)
    return sm.OLS(y, X_pca_with_const).fit()


def hedge_hog(
    df: pd.DataFrame,
    x_cols: List[str],
    y_col: str,
    run_on_level_changes: Optional[bool] = False,
    run_on_percent_changes: Optional[bool] = False,
    run_pca_on_corr_mat: Optional[bool] = False,
    x_errs: Optional[npt.ArrayLike] = None,
    y_errs: Optional[npt.ArrayLike] = None,
    # rolling_windows: Optional[List[int]] = None,
    # base_notational: Optional[int] = 1_000_000,
) -> Dict[str, sm.regression.linear_model.RegressionResults | Output | pd.DataFrame]:
    df = df[["Date"] + x_cols + [y_col]].copy()
    if run_on_level_changes:
        date_col = df["Date"]
        df = df[x_cols + [y_col]].diff()
        df["Date"] = date_col
    if run_on_percent_changes:
        date_col = df["Date"]
        df = df[x_cols + [y_col]].pct_change()
        df["Date"] = date_col

    df = df.dropna()

    pca = _run_pca(df=df, x_cols=x_cols, y_col=y_col, run_pca_on_corr_mat=run_pca_on_corr_mat)

    regression_results = {
        "ols": sm.OLS(df[y_col], sm.add_constant(df[x_cols])).fit(),
        "tls": _run_odr(df=df, x_cols=x_cols, y_col=y_col, x_errs=None, y_errs=None),
        "odr": _run_odr(df=df, x_cols=x_cols, y_col=y_col, x_errs=x_errs, y_errs=y_errs),  # ODR becomes TLS if errors not specified
        "pca": pca,
        "pca_loadings_df": pd.DataFrame(pca.components_.T, index=x_cols + [y_col], columns=[f"PC_{i+1}" for i in range(len(pca.components_))]),
        "pcr": _run_pcr(df=df, x_cols=x_cols, y_col=y_col, run_pca_on_corr_mat=run_pca_on_corr_mat),
    }

    if len(x_cols) > 1:
        df[f"{x_cols[0]}s{x_cols[1]}s"] = df[x_cols[1]] - df[x_cols[0]]
        df[f"{x_cols[0]}s{y_col}s{x_cols[1]}s"] = (df[y_col] - df[x_cols[0]]) - (df[x_cols[1]] - df[y_col])
        regression_results["mvlsr_ols"] = sm.OLS(
            df[y_col], sm.add_constant(df[[f"{x_cols[0]}s{x_cols[1]}s", f"{x_cols[0]}s{y_col}s{x_cols[1]}s"]])
        ).fit()

    return regression_results


def dv01_neutral_steepener_hegde_ratio(
    as_of_date: datetime,
    front_leg_bond_row: Dict | pd.Series,
    back_leg_bond_row: Dict | pd.Series,
    curve_data_fetcher: CurveDataFetcher,
    scipy_interp_curve: scipy.interpolate.interpolate,
    repo_rate: float,
    quote_type: Optional[str] = "eod",
    front_leg_par_amount: Optional[int] = None,
    back_leg_par_amount: Optional[int] = None,
    verbose: Optional[bool] = True,
    very_verbose: Optional[bool] = False,
):
    front_leg_info = curve_data_fetcher.ust_data_fetcher.cusip_to_ust_label(cusip=front_leg_bond_row["cusip"])
    back_leg_info = curve_data_fetcher.ust_data_fetcher.cusip_to_ust_label(cusip=back_leg_bond_row["cusip"])

    front_leg_metrics = calc_ust_metrics(
        bond_info=front_leg_info,
        curr_price=front_leg_bond_row[f"{quote_type}_price"],
        curr_ytm=front_leg_bond_row[f"{quote_type}_yield"],
        as_of_date=as_of_date,
        scipy_interp=scipy_interp_curve,
        on_rate=repo_rate,
    )
    back_leg_metrics = calc_ust_metrics(
        bond_info=back_leg_info,
        curr_price=back_leg_bond_row[f"{quote_type}_price"],
        curr_ytm=back_leg_bond_row[f"{quote_type}_yield"],
        as_of_date=as_of_date,
        scipy_interp=scipy_interp_curve,
        on_rate=repo_rate,
    )

    front_leg_ttm: float = (front_leg_info["maturity_date"] - as_of_date).days / 365
    back_leg_ttm: float = (back_leg_info["maturity_date"] - as_of_date).days / 365
    impl_spot_3m_fwds = calc_ust_impl_spot_n_fwd_curve(n=0.25, scipy_interp_curve=scipy_interp_curve, return_scipy=True)
    impl_spot_6m_fwds = calc_ust_impl_spot_n_fwd_curve(n=0.5, scipy_interp_curve=scipy_interp_curve, return_scipy=True)
    impl_spot_12m_fwds = calc_ust_impl_spot_n_fwd_curve(n=1, scipy_interp_curve=scipy_interp_curve, return_scipy=True)

    if very_verbose:
        print("Front Leg Info: ")
        print(front_leg_bond_row)
        print(front_leg_metrics)
        print("Back Leg Info: ")
        print(back_leg_bond_row)
        print(back_leg_metrics)

    if front_leg_bond_row["rank"] == 0 and back_leg_bond_row["rank"] == 0:
        print(f"{front_leg_bond_row["original_security_term"].split("-")[0]}s{back_leg_bond_row["original_security_term"].split("-")[0]}s")
    print(f"{back_leg_bond_row["ust_label"]} - {front_leg_bond_row["ust_label"]}") if verbose else None

    hr = back_leg_metrics["bps"] / front_leg_metrics["bps"]
    print("Hedge Ratio (relative to backleg): ", "\x1b[0;30;47m", hr, "\x1b[0m") if verbose else None

    if front_leg_par_amount and back_leg_par_amount:
        print("Both Leg Par Amounts passed in - using backleg par amount")
        front_leg_par_amount = None

    if not front_leg_par_amount and not back_leg_par_amount:
        back_leg_par_amount = 1_000_000

    if back_leg_par_amount:
        print(
            f"{front_leg_bond_row["ust_label"]} === {front_leg_bond_row["original_security_term"]}, TTM = {front_leg_bond_row["time_to_maturity"]:3f} (Frontleg) Par Amount = {back_leg_par_amount * hr:_}"
            if verbose
            else None
        )
        print(
            f"{back_leg_bond_row["ust_label"]} === {back_leg_bond_row["original_security_term"]}, TTM = {back_leg_bond_row["time_to_maturity"]:3f} (Backleg) Par Amount = {back_leg_par_amount:_}"
            if verbose
            else None
        )
        front_leg_par_amount = back_leg_par_amount * hr

    elif front_leg_par_amount:
        print(
            f"{front_leg_bond_row["ust_label"]} === {front_leg_bond_row["original_security_term"]}, TTM = {front_leg_bond_row["time_to_maturity"]:3f} (Frontleg) Par Amount = {front_leg_par_amount:_}"
            if verbose
            else None
        )
        print(
            f"{back_leg_bond_row["ust_label"]} === {back_leg_bond_row["original_security_term"]}, TTM = {back_leg_bond_row["time_to_maturity"]:3f} (Backleg) Par Amount = {front_leg_par_amount / hr:_}"
            if verbose
            else None
        )
        back_leg_par_amount = front_leg_par_amount / hr

    return {
        "curr_spread": (back_leg_bond_row[f"{quote_type}_yield"] - front_leg_bond_row[f"{quote_type}_yield"]) * 100,
        "rough_3m_impl_fwd_spread": (impl_spot_3m_fwds(back_leg_ttm) - impl_spot_3m_fwds(front_leg_ttm)) * 100,
        "rough_6m_impl_fwd_spread": (impl_spot_6m_fwds(back_leg_ttm) - impl_spot_6m_fwds(front_leg_ttm)) * 100,
        "rough_12m_impl_fwd_spread": (impl_spot_12m_fwds(back_leg_ttm) - impl_spot_12m_fwds(front_leg_ttm)) * 100,
        "front_leg_metrics": front_leg_metrics,
        "back_leg_metrics": back_leg_metrics,
        "hedge_ratio": hr,
        "front_leg_par_amount": front_leg_par_amount,
        "back_leg_par_amount": back_leg_par_amount,
        "spread_dv01": np.abs(back_leg_metrics["bps"] * back_leg_par_amount / 100),
        "rough_3m_carry_roll": (back_leg_metrics["rough_carry"] + back_leg_metrics["rough_3m_rolldown"])
        - hr * (front_leg_metrics["rough_carry"] + front_leg_metrics["rough_3m_rolldown"]),
        "rough_6m_carry_roll": (back_leg_metrics["rough_carry"] + back_leg_metrics["rough_6m_rolldown"])
        - hr * (front_leg_metrics["rough_carry"] + front_leg_metrics["rough_6m_rolldown"]),
        "rough_12m_carry_roll": (back_leg_metrics["rough_carry"] + back_leg_metrics["rough_12m_rolldown"])
        - hr * (front_leg_metrics["rough_carry"] + front_leg_metrics["rough_12m_rolldown"]),
    }


def dv01_neutral_butterfly_hegde_ratio(
    as_of_date: datetime,
    front_wing_bond_row: Dict | pd.Series,
    belly_bond_row: Dict | pd.Series,
    back_wing_bond_row: Dict | pd.Series,
    curve_data_fetcher: CurveDataFetcher,
    scipy_interp_curve: scipy.interpolate.interpolate,
    repo_rate: float,
    quote_type: Optional[str] = "eod",
    front_wing_par_amount: Optional[int] = None,
    belly_par_amount: Optional[int] = None,
    back_wing_par_amount: Optional[int] = None,
    verbose: Optional[bool] = True,
    very_verbose: Optional[bool] = False,
):
    front_wing_info = curve_data_fetcher.ust_data_fetcher.cusip_to_ust_label(cusip=front_wing_bond_row["cusip"])
    belly_info = curve_data_fetcher.ust_data_fetcher.cusip_to_ust_label(cusip=belly_bond_row["cusip"])
    back_wing_info = curve_data_fetcher.ust_data_fetcher.cusip_to_ust_label(cusip=back_wing_bond_row["cusip"])

    front_wing_metrics = calc_ust_metrics(
        bond_info=front_wing_info,
        curr_price=front_wing_bond_row[f"{quote_type}_price"],
        curr_ytm=front_wing_bond_row[f"{quote_type}_yield"],
        as_of_date=as_of_date,
        scipy_interp=scipy_interp_curve,
        on_rate=repo_rate,
    )
    belly_metrics = calc_ust_metrics(
        bond_info=belly_info,
        curr_price=belly_bond_row[f"{quote_type}_price"],
        curr_ytm=belly_bond_row[f"{quote_type}_yield"],
        as_of_date=as_of_date,
        scipy_interp=scipy_interp_curve,
        on_rate=repo_rate,
    )
    back_wing_metrics = calc_ust_metrics(
        bond_info=back_wing_info,
        curr_price=back_wing_bond_row[f"{quote_type}_price"],
        curr_ytm=back_wing_bond_row[f"{quote_type}_yield"],
        as_of_date=as_of_date,
        scipy_interp=scipy_interp_curve,
        on_rate=repo_rate,
    )

    front_wing_ttm: float = (front_wing_info["maturity_date"] - as_of_date).days / 365
    belly_ttm: float = (belly_info["maturity_date"] - as_of_date).days / 365
    back_wing_ttm: float = (back_wing_info["maturity_date"] - as_of_date).days / 365
    impl_spot_3m_fwds = calc_ust_impl_spot_n_fwd_curve(n=0.25, scipy_interp_curve=scipy_interp_curve, return_scipy=True)
    impl_spot_6m_fwds = calc_ust_impl_spot_n_fwd_curve(n=0.5, scipy_interp_curve=scipy_interp_curve, return_scipy=True)
    impl_spot_12m_fwds = calc_ust_impl_spot_n_fwd_curve(n=1, scipy_interp_curve=scipy_interp_curve, return_scipy=True)

    if very_verbose:
        print("Front Wing Info: ")
        print(front_wing_info)
        print(front_wing_metrics)

        print("Belly Info: ")
        print(belly_info)
        print(belly_metrics)

        print("Back Wing Info: ")
        print(back_wing_info)
        print(back_wing_metrics)

    hedge_ratios = {
        "front_wing_hr": belly_metrics["bps"] / front_wing_metrics["bps"] / 2,
        "belly_hr": 1,
        "back_wing_hr": belly_metrics["bps"] / back_wing_metrics["bps"] / 2,
    }

    if verbose:
        if front_wing_bond_row["rank"] == 0 and belly_bond_row["rank"] == 0 and back_wing_bond_row["rank"] == 0:
            print(
                f"{front_wing_bond_row["original_security_term"].split("-")[0]}s{belly_bond_row["original_security_term"].split("-")[0]}s{back_wing_bond_row["original_security_term"].split("-")[0]}s"
            )

        print(f"{front_wing_bond_row["ust_label"]} - {belly_bond_row["ust_label"]} - {back_wing_bond_row["ust_label"]} Fly")
        print("\x1b[0;30;47m", "Normalized Belly Fly Hedge Ratio:", "\x1b[0m")
        print(json.dumps(hedge_ratios, indent=4))

        if belly_par_amount:
            front_wing_par_amount = (belly_metrics["bps"] / front_wing_metrics["bps"] / 2) * belly_par_amount
            belly_par_amount = belly_par_amount
            back_wing_par_amount = (belly_metrics["bps"] / back_wing_metrics["bps"] / 2) * belly_par_amount
        elif front_wing_par_amount:
            front_wing_par_amount = front_wing_par_amount
            belly_par_amount = (2 * front_wing_metrics["bps"] / belly_metrics["bps"]) * front_wing_par_amount
            back_wing_par_amount = (
                (belly_metrics["bps"] / back_wing_metrics["bps"] / 2) * (2 * front_wing_metrics["bps"] / belly_metrics["bps"]) * front_wing_par_amount
            )
        elif back_wing_par_amount:
            front_wing_par_amount = (
                (belly_metrics["bps"] / front_wing_metrics["bps"] / 2) * (2 * back_wing_metrics["bps"] / belly_metrics["bps"]) * back_wing_par_amount
            )
            belly_par_amount = (2 * back_wing_metrics["bps"] / belly_metrics["bps"]) * back_wing_par_amount
            back_wing_par_amount = back_wing_par_amount

        print(f"Front Wing Par Amount = {front_wing_par_amount:_}")
        print(f"Belly Par Amount = {belly_par_amount:_}")
        print(f"Back Wing Par Amount = {back_wing_par_amount:_}")

    return {
        "curr_spread": (
            (belly_bond_row[f"{quote_type}_yield"] - front_wing_bond_row[f"{quote_type}_yield"])
            - (back_wing_bond_row[f"{quote_type}_yield"] - belly_bond_row[f"{quote_type}_yield"])
        )
        * 100,
        "rough_3m_impl_fwd_spread": (
            (impl_spot_3m_fwds(belly_ttm) - impl_spot_3m_fwds(front_wing_ttm)) - (impl_spot_3m_fwds(back_wing_ttm) - impl_spot_3m_fwds(belly_ttm))
        )
        * 100,
        "rough_6m_impl_fwd_spread": (
            (impl_spot_6m_fwds(belly_ttm) - impl_spot_6m_fwds(front_wing_ttm)) - (impl_spot_6m_fwds(back_wing_ttm) - impl_spot_6m_fwds(belly_ttm))
        )
        * 100,
        "rough_12m_impl_fwd_spread": (
            (impl_spot_12m_fwds(belly_ttm) - impl_spot_12m_fwds(front_wing_ttm)) - (impl_spot_12m_fwds(back_wing_ttm) - impl_spot_12m_fwds(belly_ttm))
        )
        * 100,
        "front_wing_metrics": front_wing_metrics,
        "belly_metrics": belly_metrics,
        "back_wing_metrics": back_wing_metrics,
        "hedge_ratio": hedge_ratios,
        "front_wing_par_amount": front_wing_par_amount,
        "belly_par_amount": belly_par_amount,
        "back_leg_par_amount": back_wing_par_amount,
        "spread_dv01": np.abs(belly_metrics["bps"] * belly_par_amount / 100),
        "rough_3m_carry_roll": (belly_metrics["rough_carry"] + belly_metrics["rough_3m_rolldown"])
        - (hedge_ratios["front_wing_hr"] * (front_wing_metrics["rough_carry"] + front_wing_metrics["rough_3m_rolldown"]))
        - (hedge_ratios["back_wing_hr"] * (back_wing_metrics["rough_carry"] + back_wing_metrics["rough_3m_rolldown"])),
        "rough_6m_carry_roll": (belly_metrics["rough_carry"] + belly_metrics["rough_6m_rolldown"])
        - (hedge_ratios["front_wing_hr"] * (front_wing_metrics["rough_carry"] + front_wing_metrics["rough_6m_rolldown"]))
        - (hedge_ratios["back_wing_hr"] * (back_wing_metrics["rough_carry"] + back_wing_metrics["rough_6m_rolldown"])),
        "rough_12m_carry_roll": (belly_metrics["rough_carry"] + belly_metrics["rough_12m_rolldown"])
        - (hedge_ratios["front_wing_hr"] * (front_wing_metrics["rough_carry"] + front_wing_metrics["rough_12m_rolldown"]))
        - (hedge_ratios["back_wing_hr"] * (back_wing_metrics["rough_carry"] + back_wing_metrics["rough_12m_rolldown"])),
    }
