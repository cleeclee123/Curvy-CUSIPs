import os
import shelve
from datetime import datetime

import time
import tqdm
import pandas as pd
import QuantLib as ql
from dotenv import dotenv_values
from termcolor import colored

from pandas.tseries.offsets import CustomBusinessDay, BDay
from pandas.tseries.holiday import USFederalHolidayCalendar

from CurvyCUSIPs.CurveDataFetcher import CurveDataFetcher
from CurvyCUSIPs.utils.dtcc_swaps_utils import datetime_to_ql_date

env_path = os.path.join(os.getcwd(), "../.env")
config = dotenv_values(env_path)

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


def get_business_days_groups(start_date: datetime, end_date: datetime, group_size=3):
    date_range = pd.date_range(start=start_date, end=end_date, freq=CustomBusinessDay(calendar=USFederalHolidayCalendar()))
    business_day_groups = [[bday.to_pydatetime() for bday in date_range[i : i + group_size].tolist()] for i in range(0, len(date_range), group_size)]
    return business_day_groups


# https://quantlib-users.narkive.com/4mCDwgeS/python-saving-interest-rate-curve-objects-to-file

if __name__ == "__main__":
    t_main = time.time()

    t1 = time.time()
    curve_data_fetcher = CurveDataFetcher(use_ust_issue_date=True, fred_api_key=config["FRED_API_KEY"], error_verbose=True)
    print(f"CurveDataFetcher init took: {time.time() - t1} seconds")

    start_date = datetime(2022, 11, 21)
    end_date = datetime(2024, 11, 18)
    groups = get_business_days_groups(start_date, end_date, group_size=5)

    error_logs = []

    for date_group in groups:
        t1_curr = time.time()

        curr_start_date = min(date_group)
        curr_end_date = max(date_group)

        t1 = time.time()

        print(colored(f"Group: {curr_start_date.date()} - {curr_end_date.date()} --- Starting", "green"))

        calendar = ql.UnitedStates(ql.UnitedStates.FederalReserve)
        day_count = ql.Actual360()
        fixing_lag = 1

        swaps_dict = curve_data_fetcher.dtcc_sdr_fetcher.fetch_historical_swaps_term_structure(
            start_date=curr_start_date,
            end_date=curr_end_date,
            swap_type="Fixed_Float_OIS",
            reference_floating_rates=["USD-SOFR-COMPOUND", "USD-SOFR-OIS Compound"],
            ccy="USD",
            reference_floating_rate_term_value=1,
            reference_floating_rate_term_unit="DAYS",
            notional_schedule="Constant",
            delivery_types=["CASH", "PHYS"],
            tenor_col="Tenor",
            fixed_rate_col="Close",
            settlement_t_plus=2,
            payment_lag=2,
            ql_index=ql.OvernightIndex(
                "SOFR",
                fixing_lag,
                ql.USDCurrency(),
                calendar,
                day_count,
                ql.YieldTermStructureHandle()
            ),
            fixed_leg_frequency=ql.Annual,
            fixed_leg_daycount=day_count,
            fixed_leg_convention=ql.ModifiedFollowing,
            fixed_leg_calendar=calendar,
            logLinearDiscount=True,
            logCubicDiscount=True,
            linearZero=True,
            cubicZero=True,
            linearForward=True,
            splineCubicDiscount=True,
            ny_hours_only=True
        )

        if swaps_dict is None or len(swaps_dict.keys()) == 0:
            print(colored(f"Group: {curr_start_date.date()} - {curr_end_date.date()} --- No Data, skipping", "red"))
            continue

        print(colored(f"Group: {curr_start_date.date()} - {curr_end_date.date()} --- Curve Building/Data fetching took: {time.time() - t1} seconds", "green"))

        quantlib_term_structures = [
            "logLinearDiscount",
            "logCubicDiscount",
            "linearZero",
            "cubicZero",
            "linearForward",
            "splineCubicDiscount",
        ]

        t1 = time.time()
        with shelve.open(r"C:\Users\chris\Project Bond King\ql_swap_curve_objs\nyc_sofr_ois_curve_builds_master") as db:
            for curr_date in tqdm.tqdm(swaps_dict.keys(), desc="Writing to DB..."):
                try:
                    curr_ohlc_df = swaps_dict[curr_date]["ohlc"]
                    curr_ohlc_df["Expiry"] = curr_ohlc_df["Expiry"].apply(lambda x: str(int(x.timestamp())))

                    curr_time_and_sales_df = swaps_dict[curr_date]["time_and_sales"]
                    time_and_sales_date_cols = ["Event timestamp", "Execution Timestamp", "Effective Date", "Expiration Date"]
                    for date_col in time_and_sales_date_cols:
                        curr_time_and_sales_df[date_col] = curr_time_and_sales_df[date_col].apply(lambda x: str(int(x.timestamp())))

                    curr_term_struct_obj = {
                        "ohlc": curr_ohlc_df.to_dict("records"),
                        "time_and_sales": curr_time_and_sales_df.to_dict("records"),
                        "logLinearDiscount": None,
                        "logCubicDiscount": None,
                        "linearZero": None,
                        "cubicZero": None,
                        "linearForward": None,
                        "splineCubicDiscount": None,
                    }

                    for ql_ts in quantlib_term_structures:
                        try:
                            evaluation_date = datetime_to_ql_date(curr_date)
                            ql.Settings.instance().evaluationDate = evaluation_date

                            curr_curve = swaps_dict[curr_date]["ql_curves"][ql_ts]
                            if curr_curve is None:
                                raise ValueError("QuantLib Curve is None") 
                            
                            nodes = curr_curve.nodes()
                            dates = [datetime(node[0].year(), node[0].month(), node[0].dayOfMonth()) for node in nodes]
                            discount_factors = [node[1] for node in nodes]
                            curr_term_struct_obj[ql_ts] = dict(zip(dates, discount_factors))

                        except Exception as e:
                            print(
                                colored(f"Group: {curr_start_date}-{curr_end_date} --- Quantlib bootstrap error at {ql_ts} {curr_date} --- {e}", "red")
                            )
                            error_logs.append(f"Group: {curr_start_date}-{curr_end_date} --- Quantlib bootstrap error at {ql_ts} {curr_date} --- {e}")

                    db[str(int(curr_date.timestamp()))] = curr_term_struct_obj
                    # print(
                    #     colored(f"Group: {curr_start_date}-{curr_end_date} --- QL Term Struct Obj Written", "green")
                    # )

                except Exception as e:
                    print(colored(f"Something went wrong with {curr_date} --- {e}", "red"))
                    error_logs.append(f"Something went wrong with {curr_date} --- {e}")

        print(colored(f"Group: {curr_start_date.date()} - {curr_end_date.date()} --- DB Write took: {time.time() - t1} seconds", "green"))
        print(colored(f"Group: {curr_start_date.date()} - {curr_end_date.date()} --- Current Iteration Took: {time.time() - t1_curr} seconds", "green"))

    print(error_logs)
    print(f"Script Took: {time.time() - t_main}")
