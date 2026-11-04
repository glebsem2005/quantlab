"""MOEX ISS client (no auth; delayed/EOD data, free for personal use).

Docs: https://iss.moex.com/iss/reference/
All methods return plain Python structures; ingestion into the DB
lives in `quantlab.moex.ingest`.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any, Iterator

import httpx

BASE = "https://iss.moex.com/iss"

# interval codes per ISS
TF = {"1m": 1, "10m": 10, "1h": 60, "1d": 24, "1w": 7, "1M": 31}


def _table(block: dict) -> list[dict]:
    cols = block["columns"]
    return [dict(zip(cols, row)) for row in block["data"]]


class ISSClient:
    def __init__(self, client: httpx.Client | None = None, timeout: float = 30.0):
        self._http = client or httpx.Client(timeout=timeout)

    def _get(self, path: str, **params) -> dict:
        params.setdefault("iss.meta", "off")
        r = self._http.get(f"{BASE}{path}.json", params=params)
        r.raise_for_status()
        return r.json()

    # -------------------------------------------------- security metadata
    def security(self, secid: str) -> dict[str, Any]:
        j = self._get(f"/securities/{secid}")
        desc = {row["name"]: row["value"] for row in _table(j["description"])}
        boards = _table(j["boards"])
        primary = next((b for b in boards if b.get("is_primary") == 1), boards[0] if boards else {})
        return {"description": desc, "boards": boards, "primary_board": primary}

    # -------------------------------------------------- candles (paginated)
    def candles(self, engine: str, market: str, board: str, secid: str,
                tf: str = "1d", start: date | None = None,
                end: date | None = None) -> Iterator[dict]:
        offset = 0
        while True:
            params = {"interval": TF[tf], "start": offset}
            if start: params["from"] = start.isoformat()
            if end: params["till"] = end.isoformat()
            j = self._get(
                f"/engines/{engine}/markets/{market}/boards/{board}"
                f"/securities/{secid}/candles", **params)
            rows = _table(j["candles"])
            if not rows:
                return
            yield from rows
            offset += len(rows)

    # -------------------------------------------------- bond schedule
    def bondization(self, secid: str) -> dict[str, list[dict]]:
        coupons, amorts = [], []
        offset = 0
        while True:
            j = self._get(f"/securities/{secid}/bondization",
                          **{"start": offset, "limit": 100})
            c = _table(j["coupons"]); a = _table(j["amortizations"])
            coupons += c; amorts += a
            if len(c) < 100 and len(a) < 100:
                break
            offset += 100
        return {"coupons": coupons, "amortizations": amorts}

    # -------------------------------------------------- last prices
    def marketdata(self, engine: str, market: str, board: str,
                   secids: list[str]) -> dict[str, Decimal]:
        j = self._get(f"/engines/{engine}/markets/{market}/boards/{board}/securities",
                      securities=",".join(secids),
                      **{"iss.only": "marketdata,securities"})
        md = {r["SECID"]: r for r in _table(j["marketdata"])}
        sec = {r["SECID"]: r for r in _table(j["securities"])}
        out = {}
        for sid in secids:
            row = md.get(sid, {})
            px = row.get("LAST") or row.get("MARKETPRICE") or \
                 sec.get(sid, {}).get("PREVPRICE")
            if px is not None:
                out[sid] = Decimal(str(px))
        return out


# board -> (engine, market) routing for common boards
BOARD_ROUTE = {
    "TQBR": ("stock", "shares"),
    "TQTF": ("stock", "shares"),
    "TQOB": ("stock", "bonds"),
    "TQCB": ("stock", "bonds"),
    "TQIR": ("stock", "bonds"),
    "RFUD": ("futures", "forts"),
}
