from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
import QuantLib as ql


class QL_BondPricer:

    @staticmethod
    def _pydatetime_to_qldate(date: datetime) -> ql.Date:
        return ql.Date(date.day, date.month, date.year)

    @staticmethod
    def _bill_price_to_ytm(maturity_date: datetime, as_of: datetime, price: float) -> float:
        settlement_date = (pd.to_datetime(as_of) + pd.tseries.offsets.BDay(0)).to_pydatetime()
        t = (maturity_date - settlement_date).days
        if t <= 0:
            return np.nan

        F = 100.0
        HPY = (F - price) / price
        BEY = HPY * (365 / t)
        return BEY * 100

    @staticmethod
    def _coupon_bond_price_to_ytm(
        issue_date: datetime,
        maturity_date: datetime,
        as_of: datetime,
        coupon: float,
        price: float,
    ) -> float:
        issue_ql_date = QL_BondPricer._pydatetime_to_qldate(issue_date)
        maturity_ql_date = QL_BondPricer._pydatetime_to_qldate(maturity_date)
        settlement_date = (pd.to_datetime(as_of) + pd.tseries.offsets.BDay(1)).to_pydatetime()
        settlement_ql_date = QL_BondPricer._pydatetime_to_qldate(settlement_date)

        clean_price = price
        coupon_rate = coupon / 100
        day_count = ql.ActualActual(ql.ActualActual.Bond)
        coupon_frequency = ql.Semiannual
        schedule = ql.Schedule(
            issue_ql_date,
            maturity_ql_date,
            ql.Period(coupon_frequency),
            ql.UnitedStates(ql.UnitedStates.GovernmentBond),
            ql.Unadjusted,
            ql.Unadjusted,
            ql.DateGeneration.Backward,
            False,
        )

        settlement_days = 1
        bond = ql.FixedRateBond(settlement_days, 100.0, schedule, [coupon_rate], day_count)
        bond_price_handle = ql.QuoteHandle(ql.SimpleQuote(clean_price))
        bond_engine = ql.DiscountingBondEngine(ql.YieldTermStructureHandle(ql.FlatForward(settlement_ql_date, 0.0, day_count)))
        bond.setPricingEngine(bond_engine)
        ytm = bond.bondYield(
            bond_price_handle.currentLink().value(),
            day_count,
            ql.Compounded,
            coupon_frequency,
        )
        return ytm * 100

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
