"""Double-entry ledger. Source of truth for all account state.

Invariant: amounts in a batch sum to zero per currency.
Position rows are maintained transactionally alongside postings
that carry a `position` leg.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from ..models import LedgerEntry, Position

ZERO = Decimal(0)


@dataclass
class Entry:
    leg: str
    amount: Decimal
    qty: Decimal = ZERO
    instrument_id: int | None = None
    currency: str = "RUB"
    memo: str | None = None


class UnbalancedBatch(Exception):
    pass


def post_batch(
    s: Session,
    account_id: str,
    ts: datetime,
    entries: list[Entry],
    ref_type: str | None = None,
    ref_id: str | None = None,
) -> str:
    by_ccy: dict[str, Decimal] = {}
    for e in entries:
        by_ccy[e.currency] = by_ccy.get(e.currency, ZERO) + Decimal(e.amount)
    for ccy, total in by_ccy.items():
        if total != ZERO:
            raise UnbalancedBatch(f"{ccy} batch sums to {total}, not 0")

    batch_id = str(uuid.uuid4())
    for e in entries:
        s.add(LedgerEntry(
            account_id=account_id, ts=ts, batch_id=batch_id,
            leg=e.leg, instrument_id=e.instrument_id,
            qty=Decimal(e.qty), amount=Decimal(e.amount),
            currency=e.currency, ref_type=ref_type, ref_id=ref_id,
            memo=e.memo,
        ))
        if e.leg == "position" and e.instrument_id is not None and e.qty != ZERO:
            _apply_position(s, account_id, e.instrument_id, Decimal(e.qty), Decimal(e.amount))
    return batch_id


def _apply_position(s: Session, account_id: str, instrument_id: int,
                    dqty: Decimal, damount: Decimal):
    """Weighted-average-cost position maintenance.

    `damount` is the cost-basis delta carried by the position leg:
    +cost on opening/adding, -released_cost on reduction.
    """
    pos = s.get(Position, (account_id, instrument_id))
    if pos is None:
        pos = Position(account_id=account_id, instrument_id=instrument_id,
                       qty=ZERO, avg_price=ZERO)
        s.add(pos)
    old_qty = pos.qty
    new_qty = old_qty + dqty

    same_sign = (old_qty >= 0 and dqty > 0) or (old_qty <= 0 and dqty < 0)
    if same_sign and new_qty != ZERO:
        # adding to (or opening) a position: re-average cost
        total_cost = pos.avg_price * abs(old_qty) + abs(damount)
        pos.avg_price = total_cost / abs(new_qty)
    elif new_qty == ZERO:
        pos.avg_price = ZERO
    elif (old_qty > 0 > new_qty) or (old_qty < 0 < new_qty):
        # flipped through zero: remainder priced at the new traversal cost
        pos.avg_price = abs(damount / dqty) if dqty != ZERO else ZERO
    # pure reduction keeps avg_price
    pos.qty = new_qty


def _sum_leg(s: Session, account_id: str, leg: str, currency: str) -> Decimal:
    # SQLite SUM() coerces TEXT decimals to float; sum exactly in Python.
    vals = s.execute(
        select(LedgerEntry.amount)
        .where(LedgerEntry.account_id == account_id,
               LedgerEntry.leg == leg,
               LedgerEntry.currency == currency)
    ).scalars()
    return sum((Decimal(v) for v in vals), ZERO)


def cash_balance(s: Session, account_id: str, currency: str = "RUB") -> Decimal:
    return _sum_leg(s, account_id, "cash", currency)


def debt_balance(s: Session, account_id: str, currency: str = "RUB") -> Decimal:
    """Negative when the account owes money."""
    return _sum_leg(s, account_id, "debt", currency)


def positions(s: Session, account_id: str) -> list[Position]:
    return list(s.execute(
        select(Position).where(Position.account_id == account_id,
                               Position.qty != "0")
    ).scalars())


def ledger(s: Session, account_id: str, limit: int = 50) -> list[LedgerEntry]:
    return list(s.execute(
        select(LedgerEntry)
        .where(LedgerEntry.account_id == account_id)
        .order_by(LedgerEntry.ts.desc(), LedgerEntry.id.desc())
        .limit(limit)
    ).scalars())
