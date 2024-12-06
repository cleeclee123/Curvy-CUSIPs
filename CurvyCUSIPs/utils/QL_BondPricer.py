from datetime import datetime
from typing import Literal, Optional
from pandas.tseries.offsets import BDay
import numpy as np
import pandas as pd
import rateslib as rl


class QL_BondPricer:

    @staticmethod
    def _pydatetime_to_rldate(date: datetime) -> rl.dt:
        return rl.dt(date.year, date.month, date.day)

    @staticmethod
    def _bill_price_to_ytm(maturity_date: datetime, as_of: datetime, price: float) -> float:
        as_of = as_of.date()
        settlement_date = (pd.to_datetime(as_of) + pd.tseries.offsets.BDay(1)).to_pydatetime()
        t = (maturity_date - settlement_date).days
        F = 100.0
        HPY = (F - price) / price
        BEY = HPY * (360 / t)
        return BEY * 100

    @staticmethod
    def _coupon_bond_price_to_ytm(
        issue_date: datetime,
        maturity_date: datetime,
        as_of: datetime,
        coupon: float,
        price: float,
    ) -> float:
        fxb_ust = rl.FixedRateBond(
            effective=QL_BondPricer._pydatetime_to_rldate(issue_date),
            termination=QL_BondPricer._pydatetime_to_rldate(maturity_date),
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
            effective=QL_BondPricer._pydatetime_to_rldate(issue_date),
            termination=QL_BondPricer._pydatetime_to_rldate(maturity_date),
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
                effective=QL_BondPricer._pydatetime_to_rldate(issue_date),
                termination=QL_BondPricer._pydatetime_to_rldate(maturity_date),
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
        if not price or np.isnan(price):
            return np.nan

        try:
            if type == "Bill":
                return QL_BondPricer._bill_price_to_ytm(
                    maturity_date=maturity_date,
                    as_of=as_of,
                    price=price,
                )

            return QL_BondPricer._coupon_bond_price_to_ytm(
                issue_date=issue_date,
                maturity_date=maturity_date,
                as_of=as_of,
                coupon=coupon,
                price=price,
            )
        except:
            return np.nan