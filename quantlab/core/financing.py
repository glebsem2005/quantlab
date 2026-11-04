"""Financing operations."""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy.orm import Session

from ..models import Account, Loan
from . import ledger


def take_loan(s: Session, account: Account, principal: Decimal,
              rate_model: dict, loan_type: str = "margin",
              ts: datetime | None = None) -> Loan:
    ts = ts or account.sim_time
    principal = Decimal(principal)
    loan = Loan(account_id=account.id, loan_type=loan_type,
                principal=principal, rate_model=rate_model,
                opened_ts=ts, last_accrual=ts.date())
    s.add(loan)
    s.flush()
    ledger.post_batch(s, account.id, ts, [
        ledger.Entry("cash", amount=principal),
        ledger.Entry("debt", amount=-principal, memo=f"loan {loan.id[:8]}"),
    ], ref_type="loan", ref_id=loan.id)
    return loan


def repay(s: Session, account: Account, loan: Loan, amount: Decimal,
          ts: datetime | None = None) -> Decimal:
    ts = ts or account.sim_time
    amount = min(Decimal(amount), loan.principal)
    cash = ledger.cash_balance(s, account.id, loan.currency)
    if cash < amount:
        raise ValueError(f"insufficient cash: {cash} < {amount}")
    ledger.post_batch(s, account.id, ts, [
        ledger.Entry("cash", amount=-amount, currency=loan.currency),
        ledger.Entry("debt", amount=amount, currency=loan.currency,
                      memo=f"repay {loan.id[:8]}"),
    ], ref_type="loan", ref_id=loan.id)
    loan.principal -= amount
    if loan.principal == 0:
        loan.status = "closed"
    return amount
