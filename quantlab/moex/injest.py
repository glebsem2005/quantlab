"""Ingest ISS data into the local database."""
from __future__ import annotations

import csv
import io
from datetime import date, datetime
from decimal import Decimal

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Instrument, Candle, BondEvent, RatePoint
from .iss import ISSClient, BOARD_ROUTE

# Conservative default risk rates (ставки риска) when NCC data isn't loaded.
# Real values: https://nationalclearingcentre.ru — update per instrument as needed.
DEFAULT_RISK = {
    "equity": (Decimal("0.25"), Decimal("0.30")),
    "etf":    (Decimal("0.30"), None),
    "ofz":    (Decimal("0.10"), Decimal("0.15")),
    "bond":   (Decimal("0.30"), None),
}


def _classify(desc: dict, board: str) -> str:
    t = (desc.get("TYPE") or desc.get("GROUP") or "").lower()
    if "ofz" in t or board == "TQOB":
        return "ofz"
    if "bond" in t or board in ("TQCB", "TQIR"):
        return "bond"
    if "etf" in t or "ppif" in t or board == "TQTF":
        return "etf"
    if board == "RFUD":
        return "future"
    return "equity"


def add_instrument(s: Session, client: ISSClient, secid: str,
                   board: str | None = None) -> Instrument:
    info = client.security(secid)
    desc, primary = info["description"], info["primary_board"]
    board = board or primary.get("boardid")
    if board not in BOARD_ROUTE:
        raise ValueError(f"board {board} not supported yet (add it to BOARD_ROUTE)")
    existing = s.execute(select(Instrument).where(
        Instrument.secid == secid, Instrument.board == board)).scalar_one_or_none()
    if existing:
        return existing

    cls = _classify(desc, board)
    rr = DEFAULT_RISK.get(cls, (None, None))
    ins = Instrument(
        secid=secid, board=board,
        isin=desc.get("ISIN"),
        name=desc.get("SHORTNAME") or desc.get("NAME"),
        asset_class=cls,
        currency=desc.get("FACEUNIT", "RUB") if cls in ("bond", "ofz") else "RUB",
        lot_size=int(primary.get("lotsize") or desc.get("LOTSIZE") or 1),
        face_value=Decimal(str(desc["FACEVALUE"])) if desc.get("FACEVALUE") else None,
        risk_rate_long=rr[0], risk_rate_short=rr[1],
    )
    s.add(ins)
    s.flush()

    if cls in ("bond", "ofz"):
        sched = client.bondization(secid)
        for c in sched["coupons"]:
            if c.get("coupondate") and c.get("value") is not None:
                s.add(BondEvent(instrument_id=ins.id,
                                event_date=date.fromisoformat(c["coupondate"]),
                                event_type="coupon",
                                value=Decimal(str(c["value"]))))
        for a in sched["amortizations"]:
            if a.get("amortdate") and a.get("value") is not None:
                etype = "maturity" if a.get("data_source") == "maturity" or \
                    Decimal(str(a["value"])) == ins.face_value else "amort"
                s.add(BondEvent(instrument_id=ins.id,
                                event_date=date.fromisoformat(a["amortdate"]),
                                event_type=etype,
                                value=Decimal(str(a["value"]))))
    return ins


def ingest_candles(s: Session, client: ISSClient, ins: Instrument,
                   tf: str = "1d", start: date | None = None,
                   end: date | None = None) -> int:
    engine, market = BOARD_ROUTE[ins.board]
    n = 0
    for row in client.candles(engine, market, ins.board, ins.secid, tf, start, end):
        ts = datetime.fromisoformat(row["begin"])
        if s.get(Candle, (ins.id, tf, ts)):
            continue
        s.add(Candle(instrument_id=ins.id, tf=tf, ts=ts,
                     o=Decimal(str(row["open"])), h=Decimal(str(row["high"])),
                     l=Decimal(str(row["low"])), c=Decimal(str(row["close"])),
                     volume=Decimal(str(row.get("volume") or 0)),
                     value_rub=Decimal(str(row.get("value") or 0))))
        n += 1
    return n


def last_close(s: Session, ins: Instrument) -> Decimal | None:
    row = s.execute(
        select(Candle).where(Candle.instrument_id == ins.id, Candle.tf == "1d")
        .order_by(Candle.ts.desc()).limit(1)).scalar_one_or_none()
    return row.c if row else None


CBR_KEY_RATE_URL = ("https://www.cbr.ru/Queries/UniDbQuery/DownloadExcel/132934"
                    )  # XLSX; simpler: maintain a small CSV manually


def load_key_rate_csv(s: Session, csv_text: str) -> int:
    """Load key-rate fixings from CSV with columns: date,value (value in %).

    The CBR site offers the series as a download; keep a local CSV like
        2023-07-24,8.5
        2023-08-15,12.0
        ...
    Each fixing applies forward until the next one (handled by _rate_on).
    """
    n = 0
    for row in csv.reader(io.StringIO(csv_text.strip())):
        if not row or row[0].startswith("#"):
            continue
        d = date.fromisoformat(row[0].strip())
        v = Decimal(row[1].strip()) / 100
        if not s.get(RatePoint, ("key_rate", d)):
            s.add(RatePoint(rate_name="key_rate", ts=d, value=v))
            n += 1
    return n
