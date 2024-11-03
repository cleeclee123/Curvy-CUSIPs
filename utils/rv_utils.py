from datetime import datetime
from typing import Dict, List, Callable, Optional, Tuple

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.interpolate
import tqdm

params = {
    "axes.titlesize": "x-large",
    "legend.fontsize": "x-large",
    "axes.labelsize": "x-large",
    "xtick.labelsize": "x-large",
    "ytick.labelsize": "x-large",
}
pylab.rcParams.update(params)

import seaborn as sns

from CurveDataFetcher import CurveDataFetcher
from CurveInterpolator import GeneralCurveInterpolator
from CurveBuilder import calc_ust_metrics
from utils.ust_viz import plot_usts
from utils.regression_utils import run_basic_linear_regression, run_basic_linear_regression_df, plot_residuals_timeseries, run_rolling_regression_df

sns.set(style="whitegrid", palette="dark")

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


def cusip_spread_rv_regression(
    curve_data_fetcher: CurveDataFetcher,
    label1: str,
    label2: str,
    cusip_timeseries: Dict[str, List[Dict[str, str | float | int]]],
    fitted_splines_timeseries_dict: Dict[str, Dict[datetime, Callable]],
    benchmark_tenor_1: Optional[int] = None,
    benchmark_tenor_2: Optional[int] = None,
    date_subset: Optional[Tuple[datetime, datetime]] = None,
    main_spline_key: Optional[str] = None,
    ct_yields_df: Optional[pd.DataFrame] = None,
    # rolling_r2_window: Optional[int] = 60,
):
    if not main_spline_key:
        main_spline_key = list(fitted_splines_timeseries_dict.keys())[0]

    cusip1 = curve_data_fetcher.ust_data_fetcher.ust_label_to_cusip(label1)["cusip"]
    cusip2 = curve_data_fetcher.ust_data_fetcher.ust_label_to_cusip(label2)["cusip"]
    cusip1_df = pd.DataFrame(cusip_timeseries[cusip1]).sort_values(by=["Date"])
    cusip2_df = pd.DataFrame(cusip_timeseries[cusip2]).sort_values(by=["Date"])
    if date_subset:
        cusip1_df = cusip1_df[(cusip1_df["Date"] >= date_subset[0]) & (cusip1_df["Date"] <= date_subset[1])]
        cusip2_df = cusip2_df[(cusip2_df["Date"] >= date_subset[0]) & (cusip2_df["Date"] <= date_subset[1])]

    if benchmark_tenor_1 and benchmark_tenor_2:
        benchmark_spline_spreads: Dict[str, List[float]] = {}
        for spline_key, splines_timeseries in fitted_splines_timeseries_dict.items():
            for dt in cusip1_df["Date"].to_list():
                dt: pd.Timestamp = dt
                curr_spread = splines_timeseries[dt.to_pydatetime()](benchmark_tenor_2) - splines_timeseries[dt.to_pydatetime()](benchmark_tenor_1)
                if spline_key not in benchmark_spline_spreads:
                    benchmark_spline_spreads[spline_key] = []

                benchmark_spline_spreads[spline_key].append(curr_spread)

    spread_dict = {
        "Date": cusip1_df["Date"],
        label1: cusip1_df["eod_yield"],
        label2: cusip2_df["eod_yield"],
        f"{label1} / {label2}": cusip2_df["eod_yield"] - cusip1_df["eod_yield"],
        f"{label1}_free_float": cusip1_df["free_float"],
        f"{label2}_free_float": cusip2_df["free_float"],
        f"{label1}_est_outstanding_amount": cusip1_df["est_outstanding_amount"],
        f"{label1}_soma_holdings": cusip1_df["soma_holdings"],
        f"{label1}_soma_holdings_percent_outstanding": cusip1_df["soma_holdings_percent_outstanding"],
        f"{label1}_stripped_amount": cusip1_df["stripped_amount"],
        f"{label1}_reconstituted_amount": cusip1_df["reconstituted_amount"],
        f"{label2}_est_outstanding_amount": cusip2_df["est_outstanding_amount"],
        f"{label2}_soma_holdings": cusip2_df["soma_holdings"],
        f"{label2}_soma_holdings_percent_outstanding": cusip2_df["soma_holdings_percent_outstanding"],
        f"{label2}_stripped_amount": cusip2_df["stripped_amount"],
        f"{label2}_reconstituted_amount": cusip2_df["reconstituted_amount"],
    }
    for spline_key in fitted_splines_timeseries_dict.keys():
        spread_dict[f"{label1}_{spline_key}_spread"] = cusip1_df[f"{spline_key}_spread"]
        spread_dict[f"{label2}_{spline_key}_spread"] = cusip2_df[f"{spline_key}_spread"]
        if benchmark_tenor_1 and benchmark_tenor_2:
            spread_dict[f"{spline_key}_{benchmark_tenor_1}s{benchmark_tenor_2}s"] = benchmark_spline_spreads[spline_key]

    spread_df = pd.DataFrame(spread_dict)
    # f"{main_spline_key}_{benchmark_tenor_1}s{benchmark_tenor_2}s"

    plt.figure(figsize=(20, 10))
    plt.plot(spread_df["Date"], spread_df[label1], label=label1)
    plt.plot(spread_df["Date"], spread_df[label2], label=label2)
    plt.xlabel("Date")
    plt.ylabel("Yield")
    plt.title(f"{label1} vs {label2}", fontdict={"fontsize": "x-large"})
    plt.legend(fontsize="x-large")
    plt.grid(True)
    plt.show()

    for curr_label in [label1, label2]:
        plt.figure(figsize=(20, 10))
        for spline_key in fitted_splines_timeseries_dict.keys():
            curr_spline_spread = spread_df[f"{curr_label}_{spline_key}_spread"]
            plt.plot(spread_df["Date"], curr_spline_spread, label=f"{curr_label} {spline_key} Spread")

            if spline_key == main_spline_key:
                spline_spread_mean = curr_spline_spread.mean()
                spline_spread_std = curr_spline_spread.std()
                plt.axhline(spline_spread_mean, color="green", linestyle="--", label=f"Mean ({curr_label} {main_spline_key} Spread)")
                plt.axhline(0, color="red", linestyle="--")

                for std in [1, 2]:
                    upper = spline_spread_mean + spline_spread_std * std
                    lower = spline_spread_mean - spline_spread_std * std
                    curr = plt.axhline(upper, linestyle="--", label=f"+/- {std} STD ({curr_label} {main_spline_key} Spread)")
                    plt.axhline(lower, linestyle="--", color=curr.get_color())

        plt.xlabel("Date")
        plt.ylabel("Spread (bps)")
        plt.title(f"{curr_label} Spline Spreads", fontdict={"fontsize": "x-large"})
        plt.legend(fontsize="x-large")
        plt.grid(True)
        plt.show()

    plt.figure(figsize=(20, 10))
    plt.plot(spread_df["Date"], spread_df[f"{label1} / {label2}"], label=f"{label1} / {label2}")
    plt.xlabel("Date")
    plt.ylabel("Yield (bps)")
    plt.title(f"{label1} / {label2} Spread", fontdict={"fontsize": "x-large"})
    plt.legend(fontsize="x-large")
    plt.grid(True)
    plt.show()

    if benchmark_tenor_1 and benchmark_tenor_2:
        print(f"{main_spline_key} is Benchmark Spline")
        plt.figure(figsize=(20, 10))
        plt.plot(
            spread_df["Date"],
            spread_df[f"{main_spline_key}_{benchmark_tenor_1}s{benchmark_tenor_2}s"],
            label=f"{benchmark_tenor_1}s{benchmark_tenor_2}s",
        )
        plt.xlabel("Date")
        plt.ylabel("Yield (bps)")
        plt.title(f"{benchmark_tenor_1}s{benchmark_tenor_2}s Benchmark curve", fontdict={"fontsize": "x-large"})
        plt.legend(fontsize="x-large")
        plt.grid(True)
        plt.show()

    for liquidity_metric in [
        "free_float",
        "est_outstanding_amount",
        "soma_holdings",
        "soma_holdings_percent_outstanding",
        "stripped_amount",
        "reconstituted_amount",
    ]:
        fig, ax1 = plt.subplots(figsize=(20, 10))
        (line1,) = ax1.plot(spread_df["Date"], spread_df[f"{label1}_{liquidity_metric}"], label=f"{label1}_{liquidity_metric}", color="b")
        ax1.set_xlabel("Date")
        ax1.set_ylabel(f"{label1} {liquidity_metric} ($mm)", color="b", fontsize=18)
        ax1.tick_params(axis="y", labelcolor="b", labelsize=14)
        ax1.grid(True)
        ax2 = ax1.twinx()
        (line2,) = ax2.plot(spread_df["Date"], spread_df[f"{label2}_{liquidity_metric}"], label=f"{label2}_{liquidity_metric}", color="r")
        ax2.set_ylabel(f"{label2} {liquidity_metric} ($mm)", color="r", fontsize=18)
        ax2.tick_params(axis="y", labelcolor="r", labelsize=14)
        plt.title(f"{label1} vs {label2} {liquidity_metric}", fontsize="x-large")
        plt.legend([line1, line2], [line1.get_label(), line2.get_label()], fontsize="x-large")
        fig.tight_layout()
        plt.show()

    if benchmark_tenor_1 and benchmark_tenor_2:
        r = run_basic_linear_regression_df(
            df=spread_df,
            x_col=f"{main_spline_key}_{benchmark_tenor_1}s{benchmark_tenor_2}s",
            y_col=f"{label1} / {label2}",
        )
        plot_residuals_timeseries(df=spread_df, results=r, stds=[1, 2])
        # plot_residuals_timeseries(df=spread_df, results=r, plot_zscores=True)

    if ct_yields_df is not None:
        ct_yields_df[f"CT{benchmark_tenor_1}-CT{benchmark_tenor_2}"] = ct_yields_df[f"CT{benchmark_tenor_2}"] - ct_yields_df[f"CT{benchmark_tenor_1}"]
        ct_spread_df = pd.merge(left=spread_df[["Date", f"{label1} / {label2}"]], right=ct_yields_df, on="Date", how="inner")
        r = run_basic_linear_regression_df(
            df=ct_spread_df,
            x_col=f"CT{benchmark_tenor_1}-CT{benchmark_tenor_2}",
            y_col=f"{label1} / {label2}",
        )
        plot_residuals_timeseries(df=ct_spread_df, results=r, stds=[1, 2])

    # run_rolling_regression_df(
    #     df=spread_df, x_col=f"{main_spline_key}_{benchmark_tenor_1}s{benchmark_tenor_2}s", y_col=f"{label1} / {label2}", window=rolling_r2_window
    # )

    interp_func_key = list(fitted_splines_timeseries_dict.keys())[0]
    print(f"Using {interp_func_key} for UST Metrics Calcs")
    cusip1_metrics = []
    for _, row in tqdm.tqdm(cusip1_df.iterrows(), desc=f"{label1} Metrics Calc"):
        cusip1_metrics.append(
            calc_ust_metrics(
                bond_info=curve_data_fetcher.ust_data_fetcher.cusip_to_ust_label(cusip=row["cusip"]),
                curr_price=row["eod_price"],
                curr_ytm=row["eod_yield"],
                as_of_date=row["Date"],
                scipy_interp=row[interp_func_key],
            )
        )
    cusip1_metrics_df = pd.DataFrame(cusip1_metrics)

    cusip2_metrics = []
    for _, row in tqdm.tqdm(cusip2_df.iterrows(), desc=f"{label2} Metrics Calc"):
        cusip2_metrics.append(
            calc_ust_metrics(
                bond_info=curve_data_fetcher.ust_data_fetcher.cusip_to_ust_label(cusip=row["cusip"]),
                curr_price=row["eod_price"],
                curr_ytm=row["eod_yield"],
                as_of_date=row["Date"],
                scipy_interp=row[interp_func_key],
            )
        )
    cusip2_metrics_df = pd.DataFrame(cusip2_metrics)

    plt.figure(figsize=(20, 10))
    plt.plot(cusip1_metrics_df["Date"], cusip1_metrics_df["zspread"], label=f"{label1} Z-Spread")
    plt.plot(cusip2_metrics_df["Date"], cusip2_metrics_df["zspread"], label=f"{label2} Z-Spread")
    plt.xlabel("Date")
    plt.ylabel("Yield (bps)")
    plt.title(f"{label1} vs {label2} Z-Spreads", fontdict={"fontsize": "x-large"})
    plt.legend(fontsize="x-large")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(20, 10))
    plt.plot(cusip1_metrics_df["Date"], cusip1_metrics_df["3m_carry_and_roll"], label=f"{label1} 3m Carry & Roll")
    plt.plot(cusip1_metrics_df["Date"], cusip1_metrics_df["12m_carry_and_roll"], label=f"{label1} 12m Carry & Roll")
    plt.plot(cusip2_metrics_df["Date"], cusip2_metrics_df["3m_carry_and_roll"], label=f"{label2} 3m Carry & Roll")
    plt.plot(cusip2_metrics_df["Date"], cusip2_metrics_df["12m_carry_and_roll"], label=f"{label2} 12m Carry & Roll")
    plt.xlabel("Date")
    plt.ylabel("Yield (bps)")
    plt.title(f"{label1} vs {label2} Carry & Roll Profiles", fontdict={"fontsize": "x-large"})
    plt.legend(fontsize="x-large")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(20, 10))
    clean_price_spread_1 = cusip1_metrics_df["clean_price"] - cusip1_metrics_df["zspread_impl_clean_price"]
    clean_price_spread_2 = cusip2_metrics_df["clean_price"] - cusip2_metrics_df["zspread_impl_clean_price"]
    plt.plot(cusip1_metrics_df["Date"], clean_price_spread_1, label=f"{label1} Clean Price Spread")
    plt.plot(cusip1_metrics_df["Date"], clean_price_spread_2, label=f"{label2} Clean Price Spread")
    plt.xlabel("Date")
    plt.ylabel("Yield (bps)")
    plt.title(f"{label1} vs {label2} Clean Price vs Z-Spread Implied Clean Price Spread", fontdict={"fontsize": "x-large"})
    plt.legend(fontsize="x-large")
    plt.grid(True)
    plt.show()
