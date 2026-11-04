"""Daily accrual clock.

`tick_day(account, to_date)` advances the account's simulation clock
day by day, posting:
  * loan interest, ACT/365, simple daily accrual capitalized into debt;
  * bond coupons / amortizations / maturity redemptions on their dates.

Rate models:
  {"type": "fixed", "rate": "0.21"}
  {"type": "key_rate", "spread": "0.05"}   -> key_rate fixing + spread
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, time
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Account, Loan, RatePoint, BondEvent, Instrument, Position
from . import ledger

ZERO = Decimal(0)
YEAR = Decimal(365)


def _rate_on(s: Session, name: str, d: date) -> Decimal:
    """Latest fixing on or before d."""
    row = s.execute(
        select(RatePoint).where(RatePoint.rate_name == name, RatePoint.ts <= d)
        .order_by(RatePoint.ts.desc()).limit(1)
    ).scalar_one_or_none()
    if row is None:
        raise LookupError(f"no {name} fixing on or before {d}; load rates first")
    return row.value


def _loan_daily_rate(s: Session, loan: Loan, d: date) -> Decimal:
    m = loan.rate_model
    if m["type"] == "fixed":
        annual = Decimal(str(m["rate"]))
    elif m["type"] == "key_rate":
        annual = _rate_on(s, "key_rate", d) + Decimal(str(m["spread"]))
    else:
        raise ValueError(f"unknown rate model {m}")
    return annual / YEAR


def accrue_loans(s: Session, account: Account, d: date):
    loans = s.execute(
        select(Loan).where(Loan.account_id == account.id, Loan.status == "open")
    ).scalars()
    for loan in loans:
        days = (d - loan.last_accrual).days
        if days <= 0:
            continue
        # accrue day by day so floating-rate fixings apply correctly
        cur = loan.last_accrual
        total = ZERO
        principal = loan.principal
        for _ in range(days):
            cur = cur + timedelta(days=1)
            i = (principal * _loan_daily_rate(s, loan, cur)).quantize(Decimal("0.0001"))
            principal += i           # daily capitalization
            total += i
        if total > 0:
            ledger.post_batch(
                s, account.id, datetime.combine(d, time(9, 0)),
                [ledger.Entry("debt", amount=-total, currency=loan.currency,
                              memo=f"interest {loan.id[:8]}"),
                 ledger.Entry("interest", amount=total, currency=loan.currency)],
                ref_type="loan", ref_id=loan.id,
            )
        loan.principal = principal
        loan.last_accrual = d


def process_bond_events(s: Session, account: Account, d: date):
    rows = s.execute(
        select(BondEvent, Position)
        .join(Position, (Position.instrument_id == BondEvent.instrument_id) &
                        (Position.account_id == account.id))
        .where(BondEvent.event_date == d, Position.qty != "0")
    ).all()
    for ev, pos in rows:
        ins = s.get(Instrument, ev.instrument_id)
        total = ev.value * pos.qty
        if total == 0:
            continue
        ts = datetime.combine(d, time(12, 0))
        if ev.event_type == "coupon":
            ledger.post_batch(s, account.id, ts,
                [ledger.Entry("cash", amount=total, currency=ins.currency),
                 ledger.Entry("coupon", amount=-total, instrument_id=ins.id,
                              currency=ins.currency, memo=f"{ins.secid} coupon")],
                ref_type="bond_event")
        elif ev.event_type in ("amort", "maturity"):
            # redeem principal at par fraction `ev.value` per bond:
            # cash in, position cost released pro rata, pnl difference realized
            frac_cost = pos.avg_price * pos.qty if ev.event_type == "maturity" \
                else (ev.value / ins.face_value) * pos.avg_price * pos.qty
            redeem_qty = pos.qty if ev.event_type == "maturity" else ZERO
            realized = total - frac_cost
            entries = [
                ledger.Entry("cash", amount=total, currency=ins.currency),
                ledger.Entry("position", amount=-frac_cost, qty=-redeem_qty,
                             instrument_id=ins.id, currency=ins.currency),
                ledger.Entry("realized_pnl", amount=-realized,
                             instrument_id=ins.id, currency=ins.currency,
                             memo=f"{ins.secid} {ev.event_type}"),
            ]
            ledger.post_batch(s, account.id, ts, entries, ref_type="bond_event")
            if ev.event_type == "amort" and ins.face_value:
                # amortization reduces carrying cost; qty unchanged
                pos.avg_price = pos.avg_price * (1 - ev.value / ins.face_value)


def accrued_interest(s: Session, ins: Instrument, on: date) -> Decimal:
    """Linear НКД per one bond between coupon dates (display/valuation)."""
    events = s.execute(
        select(BondEvent).where(BondEvent.instrument_id == ins.id,
                                BondEvent.event_type == "coupon")
        .order_by(BondEvent.event_date)
    ).scalars().all()
    prev, nxt = None, None
    for ev in events:
        if ev.event_date <= on:
            prev = ev
        elif nxt is None:
            nxt = ev
    if nxt is None:
        return ZERO
    period_start = prev.event_date if prev else nxt.event_date - timedelta(days=182)
    period = (nxt.event_date - period_start).days or 1
    return (nxt.value * (on - period_start).days / period).quantize(Decimal("0.01"))


def tick_to(s: Session, account: Account, to_dt: datetime):
    """Advance the account clock to `to_dt`, day by day."""
    cur = account.sim_time
    while cur.date() < to_dt.date():
        cur = datetime.combine(cur.date() + timedelta(days=1), time(9, 0))
        accrue_loans(s, account, cur.date())
        process_bond_events(s, account, cur.date())
    account.sim_time = to_dt
