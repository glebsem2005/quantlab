"""Margin engine, Russian-broker conventions (simplified NCC scheme).

Portfolio value (стоимость портфеля) = cash + Σ qty·price + debt
Initial margin  (начальная маржа)   = Σ |qty·price| · D_side
Minimal margin  (минимальная маржа) = Σ |qty·price| · D_side_min,
    with D_min = 1 - sqrt(1 - D)  — the NCC convention relating the
    two discount levels; for small D this is ≈ D/2.

States:
    value >= IM            -> ok          (можно наращивать риск)
    MM <= value < IM       -> restricted  (только снижение риска)
    value <  MM            -> margin_call (принудительное закрытие)

Non-marginal instruments (risk_rate is None) contribute their full
value to IM on the long side and are not shortable.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from sqlalchemy.orm import Session

from ..models import Instrument, Account
from . import ledger

ZERO = Decimal(0)
ONE = Decimal(1)


def _min_rate(d: Decimal) -> Decimal:
    # 1 - sqrt(1 - d), computed in Decimal via float bridge (display-grade)
    return ONE - Decimal(str(float(ONE - d) ** 0.5))


@dataclass
class MarginReport:
    portfolio_value: Decimal
    cash: Decimal
    debt: Decimal
    positions_value: Decimal
    initial_margin: Decimal
    minimal_margin: Decimal
    state: str

    @property
    def free(self) -> Decimal:
        return self.portfolio_value - self.initial_margin


def compute(s: Session, account: Account, prices: dict[int, Decimal]) -> MarginReport:
    cash = ledger.cash_balance(s, account.id)
    debt = ledger.debt_balance(s, account.id)
    pos_value = ZERO
    im = ZERO
    mm = ZERO

    for pos in ledger.positions(s, account.id):
        ins = s.get(Instrument, pos.instrument_id)
        if pos.instrument_id not in prices:
            raise KeyError(f"no price for instrument {ins.secid}")
        px = Decimal(prices[pos.instrument_id])
        value = pos.qty * px
        pos_value += value
        gross = abs(value)
        if pos.qty > 0:
            d = ins.risk_rate_long if ins.risk_rate_long is not None else ONE
        else:
            d = ins.risk_rate_short if ins.risk_rate_short is not None else ONE
        im += gross * d
        mm += gross * _min_rate(d)

    value = cash + pos_value + debt
    if value >= im:
        state = "ok"
    elif value >= mm:
        state = "restricted"
    else:
        state = "margin_call"

    return MarginReport(
        portfolio_value=value, cash=cash, debt=debt,
        positions_value=pos_value, initial_margin=im,
        minimal_margin=mm, state=state,
    )


def update_account_state(s: Session, account: Account,
                         prices: dict[int, Decimal]) -> MarginReport:
    rep = compute(s, account, prices)
    if account.margin_state != rep.state:
        account.margin_state = rep.state
    return rep


def allows_new_risk(rep: MarginReport) -> bool:
    return rep.state == "ok"
