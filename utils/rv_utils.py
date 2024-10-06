from datetime import datetime
from typing import Dict, List, Callable

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
from utils.viz import plot_residuals_timeseries, plot_usts, run_basic_linear_regression_df

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
    benchmark_tenor_1: int,
    benchmark_tenor_2: int,
):
    cusip1 = curve_data_fetcher.ust_data_fetcher.ust_label_to_cusip(label1)["cusip"]
    cusip2 = curve_data_fetcher.ust_data_fetcher.ust_label_to_cusip(label2)["cusip"]
    cusip1_df = pd.DataFrame(cusip_timeseries[cusip1]).sort_values(by=["Date"])
    cusip2_df = pd.DataFrame(cusip_timeseries[cusip2]).sort_values(by=["Date"])

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
    }
    for spline_key in fitted_splines_timeseries_dict.keys():
        spread_dict[f"{label1}_{spline_key}_spread"] = cusip1_df[f"{spline_key}_spread"]
        spread_dict[f"{label2}_{spline_key}_spread"] = cusip1_df[f"{spline_key}_spread"]
        spread_dict[f"{benchmark_tenor_1}s{benchmark_tenor_2}s"] = benchmark_spline_spreads[spline_key]

    spread_df = pd.DataFrame(spread_dict)

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
            plt.plot(spread_df["Date"], spread_df[f"{curr_label}_{spline_key}_spread"], label=f"{curr_label} {spline_key} Spread")
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

    plt.figure(figsize=(20, 10))
    plt.plot(spread_df["Date"], spread_df[f"{benchmark_tenor_1}s{benchmark_tenor_2}s"], label=f"{benchmark_tenor_1}s{benchmark_tenor_2}s")
    plt.xlabel("Date")
    plt.ylabel("Yield (bps)")
    plt.title(f"{benchmark_tenor_1}s{benchmark_tenor_2}s Benchmark curve", fontdict={"fontsize": "x-large"})
    plt.legend(fontsize="x-large")
    plt.grid(True)
    plt.show()

    fig, ax1 = plt.subplots(figsize=(20, 10))
    (line1,) = ax1.plot(spread_df["Date"], spread_df[f"{label1}_free_float"], label=f"{label1}_free_float", color="b")
    ax1.set_xlabel("Date")
    ax1.set_ylabel(f"{label1} free float ($mm)", color="b", fontsize=18)
    ax1.tick_params(axis="y", labelcolor="b", labelsize=14)
    ax1.grid(True)
    ax2 = ax1.twinx()
    (line2,) = ax2.plot(spread_df["Date"], spread_df[f"{label2}_free_float"], label=f"{label2}_free_float", color="r")
    ax2.set_ylabel(f"{label2} free float ($mm)", color="r", fontsize=18)
    ax2.tick_params(axis="y", labelcolor="r", labelsize=14)
    plt.title(f"{label1} vs {label2} Free Float", fontsize="x-large")
    plt.legend([line1, line2], [line1.get_label(), line2.get_label()], fontsize="x-large")
    fig.tight_layout()
    plt.show()

    r = run_basic_linear_regression_df(
        df=spread_df,
        x_col=f"{benchmark_tenor_1}s{benchmark_tenor_2}s",
        y_col=f"{label1} / {label2}",
    )
    plot_residuals_timeseries(df=spread_df, results=r, stds=[1, 2])
    # plot_residuals_timeseries(df=spread_df, results=r, plot_zscores=True)
    
    interp_func_key = list(fitted_splines_timeseries_dict.keys())[0]
    print(f"Using {interp_func_key} for UST Metrics Calcs")
    cusip1_metrics = []
    for _, row in tqdm.tqdm(cusip1_df.iterrows(), desc=f"{label1} Metrics Calc"):
        cusip1_metrics.append(calc_ust_metrics(
            bond_info=curve_data_fetcher.ust_data_fetcher.cusip_to_ust_label(cusip=row["cusip"]),
            curr_price=row["eod_price"],
            curr_ytm=row["eod_yield"],
            as_of_date=row["Date"],
            scipy_interp=row[interp_func_key],
        ))
    cusip1_metrics_df = pd.DataFrame(cusip1_metrics) 
        
    cusip2_metrics = []
    for _, row in tqdm.tqdm(cusip2_df.iterrows(), desc=f"{label2} Metrics Calc"):
        cusip2_metrics.append(calc_ust_metrics(
            bond_info=curve_data_fetcher.ust_data_fetcher.cusip_to_ust_label(cusip=row["cusip"]),
            curr_price=row["eod_price"],
            curr_ytm=row["eod_yield"],
            as_of_date=row["Date"],
            scipy_interp=row[interp_func_key],
        ))
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

