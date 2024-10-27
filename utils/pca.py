from __future__ import division, print_function

from datetime import datetime, timedelta
from typing import List, Optional, Annotated, Literal, Dict

import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")

from scipy import signal as sig
from scipy.interpolate import CubicSpline
from scipy.optimize import fmin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import matplotlib.colors as mcolors

from utils.utils import enhanced_plotly_blue_scale


def split_dates_into_ranges(dates: List[datetime]) -> List[List[datetime]]:
    if not dates:
        return []

    sorted_dates = sorted(dates)
    ranges = []
    current_range = [sorted_dates[0]]

    for i in range(1, len(sorted_dates)):
        if sorted_dates[i] - sorted_dates[i - 1] == timedelta(days=1):
            current_range.append(sorted_dates[i])
        else:
            ranges.append(current_range)
            current_range = [sorted_dates[i]]

    if current_range:
        ranges.append(current_range)

    return ranges


def identify_yield_curve_movements(
    pc1: pd.Series,
    pc2: pd.Series,
    date_index: pd.DatetimeIndex,
    resample: Optional[Literal["W", "M", "Y"]] = None,
) -> Annotated[List[List[datetime]], 4]:
    """
    0 => Bull Steepening => lime
    1 => Bear Steepening => skyblue
    2 => Bull Flattening => coral
    4 => Beat Flattening => khaki
    """
    movements = [[], [], [], []]

    if resample:
        df = pd.DataFrame({"PC1": pc1, "PC2": pc2}, index=date_index)
        resample_df = df.resample(resample).last()
        resampled_changes = resample_df.diff()
        for i in range(1, len(resampled_changes)):
            if resampled_changes["PC1"].iloc[i] < 0 and resampled_changes["PC2"].iloc[i] > 0:
                movements[0].append(resampled_changes.index[i])
            elif resampled_changes["PC1"].iloc[i] > 0 and resampled_changes["PC2"].iloc[i] > 0:
                movements[1].append(resampled_changes.index[i])
            elif resampled_changes["PC1"].iloc[i] < 0 and resampled_changes["PC2"].iloc[i] < 0:
                movements[2].append(resampled_changes.index[i])
            elif resampled_changes["PC1"].iloc[i] > 0 and resampled_changes["PC2"].iloc[i] < 0:
                movements[3].append(resampled_changes.index[i])
    else:
        for i in range(1, len(pc1)):
            if pc1[i] < pc1[i - 1] and pc2[i] > pc2[i - 1]:
                movements[0].append(date_index[i])
            elif pc1[i] > pc1[i - 1] and pc2[i] > pc2[i - 1]:
                movements[1].append(date_index[i])
            elif pc1[i] < pc1[i - 1] and pc2[i] < pc2[i - 1]:
                movements[2].append(date_index[i])
            elif pc1[i] > pc1[i - 1] and pc2[i] < pc2[i - 1]:
                movements[3].append(date_index[i])

    return movements


"""
Salomon PCA Note:

    - by def, pca does not recognize time series aspect of term structure data (can scramble and still get same PCs)
    - implicit assumption in PCA is that the data are independent and identically distributed
        - violated if vol is depended on level
        - vol/changes depending on levels is the gist of mean-reverting termstructure models
        - independent and identically distributed changes is consistent with a random walk model (negative yields allowed)
        - simple-minded PCA on yield changes cannot be consistent with a meanreverting framework
            - vol/changes would exhibit positive correlation/trend if levels were either at too high or too a low value in their ranges
        - depends of current macro env => curve shape
        - in typical/normal env,  levels are more likely to take values around this typical shape than around extreme curve shapes
        - change data will likely be dominated by samples with little or no observable serial correlation
        - above two points provides some comfort about the assumption of independence in the change data
        
    - eigenvalue stability decays over time and at higher orders
    - interpretation of the exposure to the fourth PC is provided by viewing the exposure as relative value in a portfolio

    for flies:
    - use pc on levels for flies vs changes (to generate signals and compute beta weights)
    - comparsion with regression beta weights
    - inherent mean reversion assumption
    - pca used for richness and cheapness screener (see Huggins and Schaller)
    - pca used for beta weight calc

    - salomon butterfly model graphs (ex 4s10s27s)
        - long spread vs short spread (10s27s vs 4s10s) and residual
        - butterfly (dd weighted (50-50), diff beta weighted)
        - yield and slope (slope = long wing - short wing) and belly
        - butterfly (diff weightings) vs yield and residual 
        - butterfly (diff weightings) vs curve slope and residual 


CS PCA Note:


"""
# BIG TODO break this function up!

def run_pca_yield_curve(
    df: pd.DataFrame,
    date_subset_range: Annotated[List[datetime], 2] | None = None,
    n_components: Optional[int] = 3,
    run_on_diff: Optional[bool] = False,  # flag to use time step wise changes for PCA
    run_on_scaler: Optional[bool] = False,  # flag to use corr matrix for PCA
    show_cumul_ex_var: Optional[bool] = False,
    show_eigenvectors: Optional[bool] = False,
    show_hedge_ratios: Optional[bool] = False,
    show_3d_plot: Optional[bool] = False,
    show_pc_scores_timeseries: Optional[bool] = False,
    show_recessions: Optional[bool] = False,
    window: Optional[int] = None,
    show_most_recent: Optional[bool] = False,
    show_trend: Optional[bool] = False,
    show_reconstructed: Optional[bool] = False,
    curve_analysis_resampling_window: Optional[Literal["W", "M", "Y"]] = None,
    show_bull_steepening_periods: Optional[bool] = False,
    # show_bear_steepening_periods: Optional[bool] = False,
    # show_bull_flattening_periods: Optional[bool] = False,
    # show_bear_flattening_periods: Optional[bool] = False,
    is_cmt_df=False,
    is_cusips=False,
    html_path: Optional[str] = None,
    show_clusters: Optional[bool] = False,
    num_clusters=8,
    overlay_df: Optional[pd.DataFrame] = None,
    to_overlay_pcs_v_time_cols: Optional[Annotated[List[str], 3]] = None
):
    df = df.copy()
    if date_subset_range:
        df = df.loc[date_subset_range[0] : date_subset_range[1]]

    to_return_dict: Dict[str, pd.DataFrame | List] = {
        "reconstructed_df": None,
        "factor_loadings_df": None,
        "biplot_df": None,
        "cumulative_explained_variance": None,
        "explained_variance": None,
    }

    if run_on_diff:
        df = df.diff()
        df = df.dropna()

    if run_on_scaler:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df)
    else:
        data_scaled = df.values

    if show_cumul_ex_var:
        pca = PCA()
        pca.fit(data_scaled)
        cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
        explained_variance = pca.explained_variance_ratio_
        to_return_dict["cumulative_explained_variance"] = cumulative_explained_variance
        to_return_dict["explained_variance"] = explained_variance
        
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        ax1.plot(cumulative_explained_variance, marker="o")
        ax1.set_xlabel("Number of Components")
        ax1.set_ylabel("Cumulative Explained Variance")
        ax1.set_title("Cumulative Explained Variance by PCA Components")
        ax1.grid(True)
        ax1.set_xticks(range(1, len(explained_variance) + 1))

        ax2.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7)
        ax2.set_xlabel("Number of Components")
        ax2.set_ylabel("Individual Explained Variance")
        ax2.set_title("Individual Explained Variance by PCA Components")
        ax2.grid(True)
        ax2.set_xticks(range(1, len(explained_variance) + 1))

        plt.tight_layout()
        plt.show()

    pca_obj = PCA(n_components=n_components)
    data_fitted_transf_pca = pca_obj.fit_transform(data_scaled)

    if show_eigenvectors:
        tenors_label = df.columns
        fig, axes = plt.subplots(1, 3, figsize=(17, 10))
        pclabels = ["Level", "Slope", "Curvature"]
        for i in range(3):
            axes[i].plot(tenors_label, pca_obj.components_[i, :])
            axes[i].set_title(f"Principal Component {i+1} ({pclabels[i]})")
            axes[i].set_xlabel("Tenor")
            axes[i].set_ylabel("Loading (bps)" if not run_on_diff else "Loading (bps change)")
            axes[i].set_xticks(tenors_label)
            axes[i].tick_params(axis="x", rotation=45)
            axes[i].grid(True)

        plt.tight_layout()
        plt.show()

        factor_loadings_df = pd.DataFrame(pca_obj.components_, columns=tenors_label)
        factor_loadings_df.index = [f"PC{i+1}" for i in range(pca_obj.n_components_)]
        to_return_dict["factor_loadings_df"] = factor_loadings_df

    if show_reconstructed:
        reconstructed_values = pca_obj.inverse_transform(data_fitted_transf_pca)
        reconstructed_df = pd.DataFrame(reconstructed_values, columns=df.columns, index=df.index)
        to_return_dict["reconstructed_df"] = reconstructed_df
        reconstruction_error = mean_squared_error(df, reconstructed_values)
        print("\nReconstruction Error (MSE):", reconstruction_error)

    if show_hedge_ratios:
        pass

    # biplot
    if show_3d_plot:
        biplot_df = pd.DataFrame(
            {
                "Date": df.index,
                "PC1": data_fitted_transf_pca[:, 0],
                # flipping signs here
                # https://stackoverflow.com/questions/44765682/in-sklearn-decomposition-pca-why-are-components-negative
                "PC2": -1 * data_fitted_transf_pca[:, 1],
                "PC3": -1 * data_fitted_transf_pca[:, 2],
            }
        )

        if is_cmt_df:
            default_tenor_vect_color = {
                "CMT3M": "red",
                "CMT6M": "red",
                "CMT1": "red",
                "CMT2": "orange",
                "CMT3": "orange",
                "CMT5": "orange",
                "CMT7": "orange",
                "CMT10": "cornflowerblue",
                "CMT20": "cornflowerblue",
                "CMT30": "cornflowerblue",
            }
        else:
            default_tenor_vect_color = {
                "CT3M": "red",
                "CT6M": "red",
                "CT1": "red",
                "CT2": "red",
                "CT3": "orange",
                "CT5": "orange",
                "CT7": "orange",
                "CT10": "cornflowerblue",
                "CT20": "cornflowerblue",
                "CT30": "cornflowerblue",
            }

        if show_clusters:
            kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(biplot_df.set_index("Date"))
            biplot_df["cluster"] = kmeans.labels_
            to_return_dict["biplot_df"] = biplot_df.copy()
            hover_text = [
                f"Date: {row["Date"]}<br>"
                + f"PC1: {row["PC1"]:.4f}<br>"
                + f"PC2: {row["PC2"]:.4f}<br>"
                + f"PC3: {row["PC3"]:.4f}<br>"
                + f"Cluster: {row["cluster"]}"
                for _, row in biplot_df.iterrows()
            ]
            biplot_df["hover_text"] = hover_text
        else:
            hover_text = [
                f"Date: {row["Date"]}<br>" + f"PC1: {row["PC1"]:.4f}<br>" + f"PC2: {row["PC2"]:.4f}<br>" + f"PC3: {row["PC3"]:.4f}"
                for _, row in biplot_df.iterrows()
            ]
            to_return_dict["biplot_df"] = biplot_df.copy()

        traces = []
        if show_clusters:
            biplot_df = biplot_df.sort_values(by=["cluster"])
            for cluster_id in biplot_df["cluster"].unique():
                cluster_data = biplot_df[biplot_df["cluster"] == cluster_id]
                trace = go.Scatter3d(
                    x=cluster_data["PC1"],
                    y=cluster_data["PC2"],
                    z=cluster_data["PC3"],
                    mode="markers",
                    marker=dict(size=4, color=cluster_id, colorscale="RdBu", opacity=0.8),
                    text=cluster_data["hover_text"],
                    hoverinfo="text",
                    name=f"Cluster {cluster_id}",
                )
                traces.append(trace)
        else:
            scatter = go.Scatter3d(
                x=biplot_df["PC1"],
                y=biplot_df["PC2"],
                z=biplot_df["PC3"],
                mode="markers",
                marker=dict(
                    size=4,
                    color=biplot_df["PC2"],
                    colorscale="RdBu",
                    opacity=0.8,
                ),
                text=hover_text,
                hoverinfo="text",
            )
            traces.append(scatter)

        vectors = []
        for i, tenor in enumerate(df.columns):
            # sign flipping here maybe?
            vectors.append(
                go.Scatter3d(
                    x=[0, pca_obj.components_[0, i]],
                    y=[0, pca_obj.components_[1, i]],
                    z=[0, pca_obj.components_[2, i]],
                    mode="lines+text",
                    line=dict(color=default_tenor_vect_color[tenor], width=3) if not is_cusips else None,
                    text=["", tenor],
                    textposition="top center",
                    textfont=dict(color=default_tenor_vect_color[tenor]) if not is_cusips else None,
                    hoverinfo="text",
                    hoverlabel=dict(bgcolor=default_tenor_vect_color[tenor]) if not is_cusips else None,
                    name=tenor,
                )
            )

        layout = go.Layout(
            scene=dict(
                xaxis_title=f"PC1 ({pca_obj.explained_variance_ratio_[0]:.2%})",
                yaxis_title=f"PC2 ({pca_obj.explained_variance_ratio_[1]:.2%})",
                zaxis_title=f"PC3 ({pca_obj.explained_variance_ratio_[2]:.2%})",
                aspectmode="manual",
                aspectratio=dict(x=2, y=2, z=1),
            ),
            title="Interactive 3D PCA Biplot of CT Yields",
            height=1050,
            width=1400,
            template="plotly_dark",
        )
        data = traces + vectors
        fig = go.Figure(data=data, layout=layout)
        fig.update_xaxes(showspikes=True, spikecolor="white", spikesnap="cursor", spikemode="across")
        fig.update_yaxes(showspikes=True, spikecolor="white", spikesnap="cursor", spikethickness=0.5)
        if html_path:
            fig.write_html(html_path)
        fig.show()

    if show_pc_scores_timeseries:
        pc1 = data_fitted_transf_pca[:, 0]
        # flipping signs
        pc2 = -1 * data_fitted_transf_pca[:, 1]
        pc3 = -1 * data_fitted_transf_pca[:, 2]

        if window:
            moving_avg_pc1 = pd.Series(pc1).rolling(window=window).mean()
            moving_avg_pc2 = pd.Series(pc2).rolling(window=window).mean()
            moving_avg_pc3 = pd.Series(pc3).rolling(window=window).mean()

        time_values = np.arange(len(df.index)).reshape(-1, 1)

        model_pc1 = LinearRegression()
        model_pc2 = LinearRegression()
        model_pc3 = LinearRegression()
        model_pc1.fit(time_values, pc1)
        model_pc2.fit(time_values, pc2)
        model_pc3.fit(time_values, pc3)
        trend_pc1 = model_pc1.predict(time_values)
        trend_pc2 = model_pc2.predict(time_values)
        trend_pc3 = model_pc3.predict(time_values)

        _, axes = plt.subplots(3, 1, figsize=(15, 20))

        pca_container = [
            {
                "label": "PC1 - Level",
                "pc": pc1,
                "model": model_pc1,
                "trend": trend_pc1,
                "ma": moving_avg_pc1 if window else None,
                "color": "blue",
            },
            {
                "label": "PC2 - Slope",
                "pc": pc2,
                "model": model_pc2,
                "trend": trend_pc2,
                "ma": moving_avg_pc2 if window else None,
                "color": "green",
            },
            {
                "label": "PC3 - Curvature",
                "pc": pc3,
                "model": model_pc3,
                "trend": trend_pc3,
                "ma": moving_avg_pc3 if window else None,
                "color": "red",
            },
        ]
        
        opp_colors = {
            "blue": "blueviolet",
            "green": "cyan",
            "red": "orange",
        }

        for i in [0, 1, 2]:
            pca_container[i]["label"]
            axes[i].plot(
                df.index,
                pca_container[i]["pc"],
                color=pca_container[i]["color"],
                label=pca_container[i]["label"],
            )
            if to_overlay_pcs_v_time_cols and overlay_df is not None:
                ax2 = axes[i].twinx()
                ax2.plot(
                    overlay_df.index,
                    overlay_df[to_overlay_pcs_v_time_cols[i]],
                    color=opp_colors[pca_container[i]["color"]],
                    label=to_overlay_pcs_v_time_cols[i],
                )
                ax2.set_ylabel(to_overlay_pcs_v_time_cols[i], color=opp_colors[pca_container[i]["color"]])

            recent = pca_container[i]["trend"][-1]
            if show_most_recent:
                axes[i].axhline(
                    pca_container[i]["pc"][0],
                    color="grey",
                    linestyle="--",
                    label=f"Recent ({recent:.2f})",
                )
            if show_trend:
                axes[i].plot(
                    df.index,
                    pca_container[i]["trend"],
                    color="black",
                    linestyle="--",
                    label="Trend",
                )
            if window:
                rolling_ma = round(pca_container[i]["ma"].tail(1).item(), 3)
                axes[i].plot(
                    df.index,
                    pca_container[i]["ma"].shift(-(window - 1)),
                    color="plum",
                    linestyle="--",
                    label=f"{window} Day Moving Average: {rolling_ma}",
                )

            if show_recessions:
                recessions_list = [
                    [datetime(1990, 7, 1), datetime(1991, 3, 1)],
                    [datetime(2001, 3, 1), datetime(2001, 11, 1)],
                    [datetime(2007, 12, 1), datetime(2009, 6, 1)],
                    [datetime(2020, 2, 1), datetime(2020, 4, 1)],
                ]
                if date_subset_range:
                    start_plot_range, end_plot_range = min(date_subset_range), max(date_subset_range)
                else:
                    start_plot_range, end_plot_range = min(df.index), max(df.index)

                for start, end in recessions_list:
                    if start <= end_plot_range and end >= start_plot_range:
                        axes[i].axvspan(start, end, color="lightcoral", alpha=0.3)

            axes[i].set_title(f"PC{i+1} over Time")
            axes[i].set_ylabel(f"PC{i+1} Score", color=pca_container[i]["color"])
            
            if to_overlay_pcs_v_time_cols and overlay_df is not None: 
                lines, labels = axes[i].get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines + lines2, labels + labels2, loc=0)
            else:
                axes[i].legend()
            
            axes[i].grid(True)

        plt.tight_layout()
        plt.show()

    return to_return_dict


def create_residuals_surface_plot(residuals_df, tick_freq=30):
    fig = go.Figure(
        data=[
            go.Surface(
                z=residuals_df.values,
                x=residuals_df.columns,
                y=residuals_df.index,
                colorscale=enhanced_plotly_blue_scale(),
                cmin=-np.max(np.abs(residuals_df.values)),
                cmax=np.max(np.abs(residuals_df.values)),
                customdata=residuals_df[["CT1"]],
                hovertemplate="Date: %{y}" + "<br>Tenor: %{x}" + "<br>Residual: %{z}<extra></extra>",
            )
        ]
    )
    fig.update_traces(
        contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True),
    )

    fig.update_layout(
        title="UST Yield Curve Residuals - 3D Surface Plot",
        autosize=False,
        width=1400,
        height=1000,
        scene=dict(
            xaxis_title="Tenor",
            xaxis=dict(
                tickmode="array",
                tickvals=residuals_df.columns,
            ),
            yaxis_title="Date",
            zaxis_title="Residual Value",
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.7)),
            aspectmode="manual",
            aspectratio=dict(x=2, y=2, z=1),
        ),
        template="plotly_dark",
    )
    fig.update_traces(colorbar=dict(title="Residual Value", thickness=20, len=0.75, x=0.95))
    fig.update_xaxes(showspikes=True, spikecolor="white", spikesnap="cursor", spikemode="across")
    fig.update_yaxes(showspikes=True, spikecolor="white", spikesnap="cursor", spikethickness=0.5)
    return fig


def plot_residuals_for_date(residuals_df, date):
    if date not in residuals_df.index:
        raise ValueError(f"Date {date} not found in the residuals DataFrame.")

    residuals_for_date = residuals_df.loc[date]
    fig = go.Figure(
        data=[
            go.Bar(
                x=residuals_df.columns,
                y=residuals_for_date,
                marker=dict(
                    color=residuals_for_date,
                    colorscale="RdBu",
                    colorbar=dict(title="Residual Value (bps)"),
                ),
            )
        ]
    )
    fig.update_layout(
        title=f"Residuals for {date}",
        xaxis_title="Tenor",
        yaxis_title="Residual Value",
        template="plotly_dark",
    )
    return fig
