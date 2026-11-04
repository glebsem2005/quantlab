"""Execution simulator shared by paper trading and (later) backtesting.

Fill model v1:
    fill_price = ref_price · (1 ± spread_bps/2e4) · (1 ± impact)
    impact     = k · sqrt(notional / ADV_rub)         (if ADV known)
    commission = notional · commission_bps / 1e4

Margin pre-check: an order is rejected if the post-trade portfolio
would violate initial margin (risk-increasing orders only). Orders
that *reduce* risk are always allowed — this mirrors broker behaviour
in the `restricted` state.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from sqlalchemy.orm import Session

from ..models import Account, Instrument, Order, Fill, Position
from . import ledger, margin

ZERO = Decimal(0)
BPS = Decimal(10000)
IMPACT_K = Decimal("0.1")


class OrderRejected(Exception):
    pass


@dataclass
class ExecutionResult:
    order: Order
    fill: Fill
    realized_pnl: Decimal


def _fill_price(ins: Instrument, side: str, qty: Decimal,
                ref_price: Decimal) -> Decimal:
    half_spread = ref_price * ins.spread_bps / (2 * BPS)
    px = ref_price + half_spread if side == "buy" else ref_price - half_spread
    adv = ins.meta.get("adv_rub")
    if adv:
        notional = qty * ref_price
        impact = IMPACT_K * Decimal(str(float(notional / Decimal(str(adv))) ** 0.5))
        px = px * (1 + impact) if side == "buy" else px * (1 - impact)
    return px


def _would_increase_risk(pos_qty: Decimal, side: str, qty: Decimal) -> bool:
    dqty = qty if side == "buy" else -qty
    return abs(pos_qty + dqty) > abs(pos_qty)


def execute_market_order(
    s: Session,
    account: Account,
    ins: Instrument,
    side: str,
    qty: Decimal,                       # units, must be multiple of lot
    ref_price: Decimal,
    prices: dict[int, Decimal],         # marks for the whole portfolio
    ts: datetime | None = None,
) -> ExecutionResult:
    ts = ts or account.sim_time
    qty = Decimal(qty)
    if qty <= 0:
        raise OrderRejected("qty must be positive")
    if qty % ins.lot_size != 0:
        raise OrderRejected(f"qty must be a multiple of lot size {ins.lot_size}")

    pos = s.get(Position, (account.id, ins.id))
    pos_qty = pos.qty if pos else ZERO

    if side == "sell" and pos_qty - qty < 0 and ins.risk_rate_short is None:
        raise OrderRejected(f"{ins.secid} is not shortable (no short risk rate)")

    price = _fill_price(ins, side, qty, Decimal(ref_price))
    notional = qty * price
    commission = notional * ins.commission_bps / BPS

    # --- margin pre-check on the hypothetical post-trade portfolio
    if _would_increase_risk(pos_qty, side, qty):
        rep = margin.compute(s, account, {**prices, ins.id: Decimal(ref_price)})
        if not margin.allows_new_risk(rep):
            raise OrderRejected(f"margin state is '{rep.state}': risk-increasing orders blocked")
        d = ins.risk_rate_long if side == "buy" else ins.risk_rate_short
        d = d if d is not None else Decimal(1)
        added_im = notional * d
        # cash effect of the trade itself changes portfolio value only by costs
        post_value = rep.portfolio_value - commission
        if post_value < rep.initial_margin + added_im:
            raise OrderRejected(
                f"insufficient margin: portfolio {post_value:.2f} < "
                f"required IM {rep.initial_margin + added_im:.2f}"
            )

    order = Order(account_id=account.id, instrument_id=ins.id, side=side,
                  order_type="market", qty=qty, status="filled", created_ts=ts)
    s.add(order)
    s.flush()
    fill = Fill(order_id=order.id, ts=ts, qty=qty, price=price,
                commission=commission, ref_price=Decimal(ref_price))
    s.add(fill)

    realized = _post_trade(s, account.id, ins, side, qty, price, commission,
                           pos_qty, pos.avg_price if pos else ZERO, ts, order.id)
    return ExecutionResult(order=order, fill=fill, realized_pnl=realized)


def _post_trade(s: Session, account_id: str, ins: Instrument, side: str,
                qty: Decimal, price: Decimal, commission: Decimal,
                pos_qty: Decimal, avg_price: Decimal,
                ts: datetime, order_id: str) -> Decimal:
    dqty = qty if side == "buy" else -qty
    cash_flow = -qty * price if side == "buy" else qty * price
    entries: list[ledger.Entry] = []
    realized = ZERO

    closing = min(abs(dqty), abs(pos_qty)) if (pos_qty * dqty < 0) else ZERO
    opening = abs(dqty) - closing

    if closing > 0:
        released_cost = avg_price * closing
        proceeds = price * closing
        # realized P&L sign depends on which side the position was on
        realized = (proceeds - released_cost) if pos_qty > 0 else (released_cost - proceeds)
        pos_delta_amount = -released_cost if pos_qty > 0 else released_cost
        entries.append(ledger.Entry("position", amount=pos_delta_amount,
                                    qty=(-closing if pos_qty > 0 else closing),
                                    instrument_id=ins.id, currency=ins.currency))
        entries.append(ledger.Entry("realized_pnl", amount=-realized,
                                    instrument_id=ins.id, currency=ins.currency,
                                    memo=f"close {closing} @ {price}"))
        cash_part = proceeds if pos_qty > 0 else -proceeds
        entries.append(ledger.Entry("cash", amount=cash_part, currency=ins.currency))

    if opening > 0:
        sign = Decimal(1) if dqty > 0 else Decimal(-1)
        cost = price * opening
        entries.append(ledger.Entry("position", amount=cost,
                                    qty=sign * opening,
                                    instrument_id=ins.id, currency=ins.currency))
        entries.append(ledger.Entry("cash", amount=-cost if sign > 0 else cost,
                                    currency=ins.currency))
        if sign < 0:
            # short: cash received now, position carried at cost with negative qty
            # cash entry above already credits proceeds; position leg amount is +cost
            # so flip the position amount sign to keep the batch balanced:
            entries[-2].amount = -cost

    if commission > 0:
        entries.append(ledger.Entry("cash", amount=-commission, currency=ins.currency))
        entries.append(ledger.Entry("fee", amount=commission, currency=ins.currency,
                                    memo=f"commission {ins.commission_bps}bps"))

    ledger.post_batch(s, account_id, ts, entries, ref_type="order", ref_id=order_id)
    return realized
