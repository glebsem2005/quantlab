"""Portfolio valuation and reporting."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Account, Instrument, Loan
from . import ledger, margin
from .accrual import accrued_interest

ZERO = Decimal(0)


@dataclass
class PositionView:
    secid: str
    name: str | None
    qty: Decimal
    avg_price: Decimal
    last: Decimal
    value: Decimal
    upnl: Decimal
    aci: Decimal          # НКД per position for bonds, else 0


@dataclass
class PortfolioView:
    cash: Decimal
    debt: Decimal
    positions: list[PositionView]
    nav: Decimal
    margin: margin.MarginReport
    loans: list[Loan]


def snapshot(s: Session, account: Account,
             prices: dict[int, Decimal]) -> PortfolioView:
    views: list[PositionView] = []
    for pos in ledger.positions(s, account.id):
        ins = s.get(Instrument, pos.instrument_id)
        px = Decimal(prices[pos.instrument_id])
        aci = ZERO
        if ins.asset_class in ("bond", "ofz"):
            aci = accrued_interest(s, ins, account.sim_time.date()) * pos.qty
        value = pos.qty * px + aci
        upnl = (px - pos.avg_price) * pos.qty
        views.append(PositionView(
            secid=ins.secid, name=ins.name, qty=pos.qty,
            avg_price=pos.avg_price, last=px, value=value, upnl=upnl, aci=aci))

    rep = margin.compute(s, account, prices)
    loans = list(s.execute(select(Loan).where(
        Loan.account_id == account.id, Loan.status == "open")).scalars())
    cash = ledger.cash_balance(s, account.id)
    debt = ledger.debt_balance(s, account.id)
    nav = cash + sum(v.value for v in views) + debt
    return PortfolioView(cash=cash, debt=debt, positions=views,
                         nav=nav, margin=rep, loans=loans)
