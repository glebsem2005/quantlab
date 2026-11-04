"""Core engine tests. In-memory SQLite, no network."""
from datetime import datetime, date, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import sessionmaker

from quantlab.models import Base, Account, Instrument, BondEvent, RatePoint, LedgerEntry
from quantlab.core import ledger, execution, margin, financing, accrual

D = Decimal
NOW = datetime(2026, 6, 1, 10, 0)


@pytest.fixture()
def s():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    Session = sessionmaker(bind=eng, expire_on_commit=False)
    sess = Session()
    yield sess
    sess.close()


@pytest.fixture()
def acc(s):
    a = Account(name="t", sim_time=NOW, created_at=NOW)
    s.add(a); s.flush()
    ledger.post_batch(s, a.id, NOW, [
        ledger.Entry("cash", amount=D(1_000_000)),
        ledger.Entry("equity", amount=D(-1_000_000)),
    ], ref_type="init")
    return a


@pytest.fixture()
def sber(s):
    i = Instrument(secid="SBER", board="TQBR", asset_class="equity",
                   lot_size=10, risk_rate_long=D("0.25"), risk_rate_short=D("0.30"),
                   spread_bps=D("0"), commission_bps=D("0"))
    s.add(i); s.flush()
    return i


@pytest.fixture()
def ofz(s):
    i = Instrument(secid="SU26238", board="TQOB", asset_class="ofz",
                   lot_size=1, face_value=D(1000),
                   risk_rate_long=D("0.10"), risk_rate_short=None,
                   spread_bps=D("0"), commission_bps=D("0"))
    s.add(i); s.flush()
    return i


def batches_balanced(s, account_id):
    rows = s.execute(select(LedgerEntry).where(
        LedgerEntry.account_id == account_id)).scalars()
    sums = {}
    for e in rows:
        sums[(e.batch_id, e.currency)] = sums.get((e.batch_id, e.currency), D(0)) + e.amount
    return all(v == 0 for v in sums.values())


# ------------------------------------------------------------------ ledger

def test_unbalanced_batch_rejected(s, acc):
    with pytest.raises(ledger.UnbalancedBatch):
        ledger.post_batch(s, acc.id, NOW, [ledger.Entry("cash", amount=D(1))])


def test_initial_cash(s, acc):
    assert ledger.cash_balance(s, acc.id) == D(1_000_000)


# --------------------------------------------------------------- execution

def test_buy_sell_roundtrip_with_profit(s, acc, sber):
    prices = {sber.id: D(300)}
    execution.execute_market_order(s, acc, sber, "buy", D(100), D(300), prices)
    assert ledger.cash_balance(s, acc.id) == D(970_000)
    pos = ledger.positions(s, acc.id)[0]
    assert pos.qty == D(100) and pos.avg_price == D(300)

    res = execution.execute_market_order(s, acc, sber, "sell", D(100), D(330),
                                         {sber.id: D(330)})
    assert res.realized_pnl == D(3000)
    assert ledger.cash_balance(s, acc.id) == D(1_003_000)
    assert ledger.positions(s, acc.id) == []
    assert batches_balanced(s, acc.id)


def test_lot_size_enforced(s, acc, sber):
    with pytest.raises(execution.OrderRejected):
        execution.execute_market_order(s, acc, sber, "buy", D(15), D(300),
                                       {sber.id: D(300)})


def test_spread_and_commission_costs(s, acc, sber):
    sber.spread_bps = D("20")          # 10 bps each side
    sber.commission_bps = D("5")
    res = execution.execute_market_order(s, acc, sber, "buy", D(100), D(300),
                                         {sber.id: D(300)})
    assert res.fill.price == D(300) * (1 + D("0.001"))
    assert res.fill.commission == D(100) * res.fill.price * D("0.0005")


def test_short_open_and_cover(s, acc, sber):
    prices = {sber.id: D(300)}
    execution.execute_market_order(s, acc, sber, "sell", D(100), D(300), prices)
    assert ledger.cash_balance(s, acc.id) == D(1_030_000)
    pos = ledger.positions(s, acc.id)[0]
    assert pos.qty == D(-100) and pos.avg_price == D(300)

    res = execution.execute_market_order(s, acc, sber, "buy", D(100), D(270),
                                         {sber.id: D(270)})
    assert res.realized_pnl == D(3000)
    assert ledger.cash_balance(s, acc.id) == D(1_003_000)
    assert batches_balanced(s, acc.id)


def test_short_blocked_without_short_rate(s, acc, ofz):
    with pytest.raises(execution.OrderRejected):
        execution.execute_market_order(s, acc, ofz, "sell", D(10), D(700),
                                       {ofz.id: D(700)})


def test_partial_close_keeps_avg(s, acc, sber):
    execution.execute_market_order(s, acc, sber, "buy", D(200), D(300),
                                   {sber.id: D(300)})
    execution.execute_market_order(s, acc, sber, "sell", D(100), D(310),
                                   {sber.id: D(310)})
    pos = ledger.positions(s, acc.id)[0]
    assert pos.qty == D(100) and pos.avg_price == D(300)


# ------------------------------------------------------------------ margin

def test_margin_states_and_leverage(s, acc, sber):
    """1M cash + 350k margin loan, buy 1.35M of SBER, then crash the price."""
    loan = financing.take_loan(s, acc, D(350_000),
                               {"type": "fixed", "rate": "0.21"}, ts=NOW)
    execution.execute_market_order(s, acc, sber, "buy", D(4500), D(300),
                                   {sber.id: D(300)})
    rep = margin.compute(s, acc, {sber.id: D(300)})
    # value = 1.35M cash spent -> 0 cash... cash=1M+350k-1.35M=0; debt=-350k
    assert rep.cash == D(0)
    assert rep.portfolio_value == D(1_000_000)          # 1.35M pos - 350k debt
    assert rep.initial_margin == D(1_350_000) * D("0.25")
    assert rep.state == "ok"

    # -40%: positions 810k, value = 810k - 350k = 460k; IM = 202.5k -> ok still
    rep = margin.compute(s, acc, {sber.id: D(180)})
    assert rep.state == "ok"

    # -65%: positions 472.5k, value 122.5k; IM 118.1k -> ok barely; -70%:
    rep = margin.compute(s, acc, {sber.id: D(90)})      # pos 405k, value 55k
    assert rep.initial_margin == D(405_000) * D("0.25")  # 101.25k > 55k
    assert rep.state in ("restricted", "margin_call")

    rep = margin.compute(s, acc, {sber.id: D(80)})      # pos 360k, value 10k
    # MM ~ 360k * (1-sqrt(0.75)) ~ 48.2k > 10k -> margin call
    assert rep.state == "margin_call"


def test_risk_increasing_order_blocked_in_restricted(s, acc, sber):
    financing.take_loan(s, acc, D(2_000_000), {"type": "fixed", "rate": "0.2"}, ts=NOW)
    execution.execute_market_order(s, acc, sber, "buy", D(9000), D(300),
                                   {sber.id: D(300)})
    # crash so that state is restricted
    px = D(120)
    rep = margin.compute(s, acc, {sber.id: px})
    assert rep.state != "ok"
    with pytest.raises(execution.OrderRejected):
        execution.execute_market_order(s, acc, sber, "buy", D(10), px,
                                       {sber.id: px})
    # risk-reducing sell is allowed
    execution.execute_market_order(s, acc, sber, "sell", D(1000), px,
                                   {sber.id: px})


def test_insufficient_margin_rejected(s, acc, sber):
    with pytest.raises(execution.OrderRejected):
        # IM would be 0.25*6M = 1.5M > 1M portfolio value
        execution.execute_market_order(s, acc, sber, "buy", D(20000), D(300),
                                       {sber.id: D(300)})


# ---------------------------------------------------------------- financing

def test_loan_fixed_accrual_capitalizes(s, acc):
    loan = financing.take_loan(s, acc, D(100_000),
                               {"type": "fixed", "rate": "0.365"}, ts=NOW)
    accrual.tick_to(s, acc, NOW + timedelta(days=10))
    # 0.1% в день с капитализацией: 100000*(1.001^10 - 1) ≈ 1004.51
    assert D("1004.4") < loan.principal - D(100_000) < D("1004.6")
    assert ledger.debt_balance(s, acc.id) == -loan.principal
    assert batches_balanced(s, acc.id)


def test_loan_floating_uses_fixings(s, acc):
    s.add(RatePoint(rate_name="key_rate", ts=date(2026, 1, 1), value=D("0.16")))
    s.add(RatePoint(rate_name="key_rate", ts=date(2026, 6, 6), value=D("0.20")))
    loan = financing.take_loan(s, acc, D(365_000),
                               {"type": "key_rate", "spread": "0.00"}, ts=NOW)
    accrual.tick_to(s, acc, NOW + timedelta(days=10))
    # fixing of Jun 6 applies from Jun 6: 4 days @16% + 6 days @20%
    # = 4*160 + 6*200 = 1840 (+capitalization)
    accrued = loan.principal - D(365_000)
    assert D(1840) < accrued < D(1850)


def test_repay_closes_loan(s, acc):
    loan = financing.take_loan(s, acc, D(100_000),
                               {"type": "fixed", "rate": "0.2"}, ts=NOW)
    financing.repay(s, acc, loan, D(100_000))
    assert loan.status == "closed"
    assert ledger.debt_balance(s, acc.id) == D(0)


# ------------------------------------------------------------------- bonds

def test_coupon_payment(s, acc, ofz):
    s.add(BondEvent(instrument_id=ofz.id, event_date=date(2026, 6, 5),
                    event_type="coupon", value=D("35.40")))
    execution.execute_market_order(s, acc, ofz, "buy", D(100), D(700),
                                   {ofz.id: D(700)})
    cash_before = ledger.cash_balance(s, acc.id)
    accrual.tick_to(s, acc, NOW + timedelta(days=7))
    assert ledger.cash_balance(s, acc.id) == cash_before + D(3540)
    assert batches_balanced(s, acc.id)


def test_maturity_redemption(s, acc, ofz):
    s.add(BondEvent(instrument_id=ofz.id, event_date=date(2026, 6, 5),
                    event_type="maturity", value=D(1000)))
    execution.execute_market_order(s, acc, ofz, "buy", D(100), D(700),
                                   {ofz.id: D(700)})
    accrual.tick_to(s, acc, NOW + timedelta(days=7))
    # bought at 700, redeemed at 1000: +30000 realized, position gone
    assert ledger.positions(s, acc.id) == []
    assert ledger.cash_balance(s, acc.id) == D(1_000_000) - D(70_000) + D(100_000)
    assert batches_balanced(s, acc.id)


def test_accrued_interest_linear(s, ofz):
    s.add(BondEvent(instrument_id=ofz.id, event_date=date(2026, 1, 1),
                    event_type="coupon", value=D("36.50")))
    s.add(BondEvent(instrument_id=ofz.id, event_date=date(2026, 7, 1),
                    event_type="coupon", value=D("36.50")))
    aci = accrual.accrued_interest(s, ofz, date(2026, 4, 1))
    # 90 of 181 days elapsed
    assert D(18) < aci < D("18.3")
