from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
import rateslib as rl
from pandas.tseries.offsets import BDay
from typing import Optional


class RL_BondPricer:

    @staticmethod
    def _pydatetime_to_rldate(date: datetime) -> rl.dt:
        return rl.dt(date.year, date.month, date.day)

    @staticmethod
    def _bill_price_to_ytm(
        issue_date: datetime, maturity_date: datetime, as_of: datetime, price: float
    ) -> float:
        bill_ust = rl.Bill(
            effective=RL_BondPricer._pydatetime_to_rldate(issue_date),
            termination=RL_BondPricer._pydatetime_to_rldate(maturity_date),
            calendar="nyc",
            modifier="NONE",
            currency="usd",
            convention="Act360",
            settle=1,
            notional=-1e6,  # negative notional receives fixed, i.e. buys a bill
            curves="bill_curve",
            calc_mode="us_gbb",
        )
        settle_pd_ts: pd.Timestamp = as_of + BDay(1)
        return bill_ust.ytm(price=price, settlement=settle_pd_ts.to_pydatetime())

    @staticmethod
    def _coupon_bond_price_to_ytm(
        issue_date: datetime,
        maturity_date: datetime,
        as_of: datetime,
        coupon: float,
        price: float,
    ) -> float:
        fxb_ust = rl.FixedRateBond(
            effective=RL_BondPricer._pydatetime_to_rldate(issue_date),
            termination=RL_BondPricer._pydatetime_to_rldate(maturity_date),
            fixed_rate=coupon * 100,
            spec="ust",
            calc_mode="ust_31bii",
        )
        settle_pd_ts: pd.Timestamp = as_of + BDay(1)
        return fxb_ust.ytm(price=price, settlement=settle_pd_ts.to_pydatetime())

    @staticmethod
    def _coupon_bond_ytm_to_price(
        issue_date: datetime,
        maturity_date: datetime,
        as_of: datetime,
        coupon: float,
        ytm: float,
        dirty: Optional[bool] = False,
    ) -> float:
        fxb_ust = rl.FixedRateBond(
            effective=RL_BondPricer._pydatetime_to_rldate(issue_date),
            termination=RL_BondPricer._pydatetime_to_rldate(maturity_date),
            fixed_rate=coupon * 100,
            spec="ust",
            calc_mode="ust_31bii",
        )
        settle_pd_ts: pd.Timestamp = as_of + BDay(1)
        return fxb_ust.price(
            ytm=ytm, settlement=settle_pd_ts.to_pydatetime(), dirty=dirty
        )

    @staticmethod
    def _bond_mod_duration(
        issue_date: datetime,
        maturity_date: datetime,
        as_of: datetime,
        coupon: float,
        ytm: float,
    ) -> float:
        try:
            fxb_ust = rl.FixedRateBond(
                effective=RL_BondPricer._pydatetime_to_rldate(issue_date),
                termination=RL_BondPricer._pydatetime_to_rldate(maturity_date),
                fixed_rate=coupon * 100,
                spec="ust",
            )
            settle_pd_ts: pd.Timestamp = as_of + BDay(1)
            return fxb_ust.duration(settlement=settle_pd_ts, ytm=ytm, metric="modified")
        except:
            return None

    @staticmethod
    def bond_price_to_ytm(
        type: Literal["Bill", "Note", "Bond"],
        issue_date: datetime,
        maturity_date: datetime,
        as_of: datetime,
        coupon: float,
        price: float,
    ):
        try:
            if type == "Bill":
                return RL_BondPricer._bill_price_to_ytm(
                    issue_date=issue_date,
                    maturity_date=maturity_date,
                    as_of=as_of,
                    price=price,
                )

            return RL_BondPricer._coupon_bond_price_to_ytm(
                issue_date=issue_date,
                maturity_date=maturity_date,
                as_of=as_of,
                coupon=coupon,
                price=price,
            )
        except Exception as e:
            # print(e)
            return np.nan
