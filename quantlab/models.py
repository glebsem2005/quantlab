"""ORM models. Money is Decimal end-to-end, stored as TEXT in SQLite.

Ledger discipline: every economic event is a batch of entries whose
amounts sum to zero per currency. Legs:
    cash             — денежные средства
    position         — позиция по себестоимости
    realized_pnl     — реализованный P&L (контр-счёт)
    fee              — комиссии (расход)
    interest         — процентный расход по займам
    coupon / dividend— купонный / дивидендный доход (контр-счёт)
    debt             — долг (отрицательный, когда должен)
    vm               — вариационная маржа по фьючерсам (Phase 2)
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime, date
from decimal import Decimal

from sqlalchemy import (
    BIGINT,
    String, Integer, BigInteger, ForeignKey, Date, DateTime, JSON,
    UniqueConstraint, Index, types,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Money(types.TypeDecorator):
    """Exact decimal storage on SQLite (TEXT round-trip)."""
    impl = types.String(40)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        return None if value is None else str(Decimal(value))

    def process_result_value(self, value, dialect):
        return None if value is None else Decimal(value)


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------- instruments

class AssetClass(str, enum.Enum):
    equity = "equity"
    bond = "bond"
    ofz = "ofz"
    future = "future"
    etf = "etf"
    index = "index"
    currency = "currency"


class Instrument(Base):
    __tablename__ = "instruments"
    __table_args__ = (UniqueConstraint("secid", "board"),)

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    secid: Mapped[str] = mapped_column(String(32))
    isin: Mapped[str | None] = mapped_column(String(16))
    name: Mapped[str | None] = mapped_column(String(256))
    board: Mapped[str] = mapped_column(String(8))          # TQBR, TQOB, ...
    asset_class: Mapped[str] = mapped_column(String(16))
    currency: Mapped[str] = mapped_column(String(3), default="RUB")
    lot_size: Mapped[int] = mapped_column(Integer, default=1)
    face_value: Mapped[Decimal | None] = mapped_column(Money)   # bonds
    settlement: Mapped[str] = mapped_column(String(8), default="T+1")
    # NCC-style risk rates (ставки риска); None => instrument is non-marginal
    risk_rate_long: Mapped[Decimal | None] = mapped_column(Money)
    risk_rate_short: Mapped[Decimal | None] = mapped_column(Money)
    # execution-model parameters
    spread_bps: Mapped[Decimal] = mapped_column(Money, default=Decimal("10"))
    commission_bps: Mapped[Decimal] = mapped_column(Money, default=Decimal("5"))
    meta: Mapped[dict] = mapped_column(JSON, default=dict)


class BondEvent(Base):
    """Coupon / amortization / maturity schedule (ISS bondization)."""
    __tablename__ = "bond_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    instrument_id: Mapped[int] = mapped_column(ForeignKey("instruments.id"))
    event_date: Mapped[date] = mapped_column(Date)
    event_type: Mapped[str] = mapped_column(String(16))    # coupon|amort|maturity
    value: Mapped[Decimal] = mapped_column(Money)          # per one bond, RUB
    __table_args__ = (Index("ix_bond_events", "instrument_id", "event_date"),)


class Candle(Base):
    __tablename__ = "candles"
    instrument_id: Mapped[int] = mapped_column(ForeignKey("instruments.id"), primary_key=True)
    tf: Mapped[str] = mapped_column(String(4), primary_key=True)   # '1d','1h','1m'
    ts: Mapped[datetime] = mapped_column(DateTime, primary_key=True)
    o: Mapped[Decimal] = mapped_column(Money)
    h: Mapped[Decimal] = mapped_column(Money)
    l: Mapped[Decimal] = mapped_column(Money)
    c: Mapped[Decimal] = mapped_column(Money)
    volume: Mapped[Decimal | None] = mapped_column(Money)
    value_rub: Mapped[Decimal | None] = mapped_column(Money)


class RatePoint(Base):
    """key_rate, RUONIA, RUSFAR — daily fixings for accrual models."""
    __tablename__ = "rates"
    rate_name: Mapped[str] = mapped_column(String(16), primary_key=True)
    ts: Mapped[date] = mapped_column(Date, primary_key=True)
    value: Mapped[Decimal] = mapped_column(Money)          # 0.16 == 16%


# ------------------------------------------------------------------- accounts

class Account(Base):
    __tablename__ = "accounts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(64), unique=True)
    mode: Mapped[str] = mapped_column(String(16), default="live_paper")  # |replay
    base_ccy: Mapped[str] = mapped_column(String(3), default="RUB")
    sim_time: Mapped[datetime] = mapped_column(DateTime)   # authoritative clock
    margin_state: Mapped[str] = mapped_column(String(16), default="ok")
    # ok | restricted | margin_call | liquidated
    created_at: Mapped[datetime] = mapped_column(DateTime)


class LedgerEntry(Base):
    __tablename__ = "ledger_entries"

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.id"))
    ts: Mapped[datetime] = mapped_column(DateTime)         # simulation time
    batch_id: Mapped[str] = mapped_column(String(36))
    leg: Mapped[str] = mapped_column(String(16))
    instrument_id: Mapped[int | None] = mapped_column(ForeignKey("instruments.id"))
    qty: Mapped[Decimal] = mapped_column(Money, default=Decimal(0))
    amount: Mapped[Decimal] = mapped_column(Money, default=Decimal(0))
    currency: Mapped[str] = mapped_column(String(3), default="RUB")
    ref_type: Mapped[str | None] = mapped_column(String(16))
    ref_id: Mapped[str | None] = mapped_column(String(36))
    memo: Mapped[str | None] = mapped_column(String(256))
    __table_args__ = (Index("ix_ledger_acct_ts", "account_id", "ts"),)


class Position(Base):
    """Materialized from the ledger for read speed; ledger is truth."""
    __tablename__ = "positions"
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.id"), primary_key=True)
    instrument_id: Mapped[int] = mapped_column(ForeignKey("instruments.id"), primary_key=True)
    qty: Mapped[Decimal] = mapped_column(Money, default=Decimal(0))
    avg_price: Mapped[Decimal] = mapped_column(Money, default=Decimal(0))


# --------------------------------------------------------------------- orders

class Order(Base):
    __tablename__ = "orders"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.id"))
    instrument_id: Mapped[int] = mapped_column(ForeignKey("instruments.id"))
    side: Mapped[str] = mapped_column(String(4))           # buy|sell
    order_type: Mapped[str] = mapped_column(String(12), default="market")
    qty: Mapped[Decimal] = mapped_column(Money)            # units (not lots)
    limit_price: Mapped[Decimal | None] = mapped_column(Money)
    status: Mapped[str] = mapped_column(String(12), default="new")
    created_ts: Mapped[datetime] = mapped_column(DateTime)


class Fill(Base):
    __tablename__ = "fills"
    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    order_id: Mapped[str] = mapped_column(ForeignKey("orders.id"))
    ts: Mapped[datetime] = mapped_column(DateTime)
    qty: Mapped[Decimal] = mapped_column(Money)
    price: Mapped[Decimal] = mapped_column(Money)
    commission: Mapped[Decimal] = mapped_column(Money)
    ref_price: Mapped[Decimal] = mapped_column(Money)      # pre-cost reference


# ------------------------------------------------------------------ financing

class Loan(Base):
    __tablename__ = "loans"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.id"))
    loan_type: Mapped[str] = mapped_column(String(12), default="margin")
    principal: Mapped[Decimal] = mapped_column(Money)      # outstanding
    currency: Mapped[str] = mapped_column(String(3), default="RUB")
    # {"type":"fixed","rate":"0.21"} | {"type":"key_rate","spread":"0.05"}
    rate_model: Mapped[dict] = mapped_column(JSON)
    opened_ts: Mapped[datetime] = mapped_column(DateTime)
    last_accrual: Mapped[date] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(12), default="open")
