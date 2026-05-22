"""
Transaction-based portfolio ledger backed by SQLite.

Stores BUY / SELL / DIV transactions and derives current positions, average
cost, and realized P&L. All monetary values are normalized to a base currency
via a per-transaction `fx_rate` (1 unit of trade currency = fx_rate units of
base currency); for single-currency users, fx_rate stays at 1.0.

This module owns the persistence layer; aggregations live in
`portfolio_analytics`, and trade suggestions in `rebalancer`.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..config import config

_EPSILON = 1e-9

VALID_SIDES = ("BUY", "SELL", "DIV")


# ── Connection ───────────────────────────────────────────────────────────────

_db_lock = threading.Lock()


def _db_path() -> Path:
    return config.DATA_DIR / "portfolio.db"


def _connect() -> sqlite3.Connection:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_db_path()), timeout=5, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


_SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker      TEXT    NOT NULL,
    side        TEXT    NOT NULL CHECK (side IN ('BUY','SELL','DIV')),
    trade_date  TEXT    NOT NULL,
    quantity    REAL    NOT NULL,
    price       REAL    NOT NULL,
    currency    TEXT    NOT NULL DEFAULT 'USD',
    fees        REAL    NOT NULL DEFAULT 0,
    fx_rate     REAL    NOT NULL DEFAULT 1.0,
    notes       TEXT    NOT NULL DEFAULT '',
    created_at  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tx_ticker ON transactions(ticker);
CREATE INDEX IF NOT EXISTS idx_tx_date   ON transactions(trade_date);

CREATE TABLE IF NOT EXISTS instruments (
    ticker      TEXT PRIMARY KEY,
    name        TEXT NOT NULL DEFAULT '',
    sector      TEXT NOT NULL DEFAULT '',
    region      TEXT NOT NULL DEFAULT '',
    currency    TEXT NOT NULL DEFAULT 'USD',
    asset_class TEXT NOT NULL DEFAULT 'EQUITY',
    beta        REAL,
    updated_at  TEXT
);

CREATE TABLE IF NOT EXISTS targets (
    ticker        TEXT PRIMARY KEY,
    target_weight REAL NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS watchlist (
    ticker   TEXT PRIMARY KEY,
    name     TEXT NOT NULL DEFAULT '',
    sector   TEXT NOT NULL DEFAULT '',
    currency TEXT NOT NULL DEFAULT 'USD',
    added_at TEXT NOT NULL
);
"""

_initialized = False


def init_db() -> None:
    """Create tables on first use (idempotent)."""
    global _initialized
    if _initialized:
        return
    with _db_lock:
        if _initialized:
            return
        with _connect() as conn:
            conn.executescript(_SCHEMA)
            conn.commit()
        _initialized = True


# ── Domain types ─────────────────────────────────────────────────────────────

@dataclass
class Transaction:
    id: Optional[int]
    ticker: str
    side: str
    trade_date: str
    quantity: float
    price: float
    currency: str = "USD"
    fees: float = 0.0
    fx_rate: float = 1.0
    notes: str = ""
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Position:
    ticker: str
    quantity: float
    avg_cost: float       # in base currency, per share
    cost_basis: float     # quantity * avg_cost (base)
    realized_pnl: float   # cumulative realized P&L (base)
    dividends: float      # cumulative dividends received (base)
    first_buy_date: Optional[str]
    last_trade_date: Optional[str]
    currency: str = "USD"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Transaction CRUD ─────────────────────────────────────────────────────────

def _row_to_tx(row: sqlite3.Row) -> Transaction:
    return Transaction(
        id=row["id"],
        ticker=row["ticker"],
        side=row["side"],
        trade_date=row["trade_date"],
        quantity=row["quantity"],
        price=row["price"],
        currency=row["currency"],
        fees=row["fees"],
        fx_rate=row["fx_rate"],
        notes=row["notes"] or "",
        created_at=row["created_at"],
    )


def list_transactions(ticker: Optional[str] = None) -> List[Transaction]:
    init_db()
    with _connect() as conn:
        if ticker:
            rows = conn.execute(
                "SELECT * FROM transactions WHERE ticker = ? ORDER BY trade_date, id",
                [ticker.upper()],
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM transactions ORDER BY trade_date, id"
            ).fetchall()
    return [_row_to_tx(r) for r in rows]


def add_transaction(
    ticker: str,
    side: str,
    quantity: float,
    price: float,
    trade_date: Optional[str] = None,
    *,
    currency: str = "USD",
    fees: float = 0.0,
    fx_rate: float = 1.0,
    notes: str = "",
) -> Transaction:
    init_db()
    ticker = (ticker or "").strip().upper()
    side = (side or "").strip().upper()
    if not ticker:
        raise ValueError("ticker is required")
    if side not in VALID_SIDES:
        raise ValueError(f"side must be one of {VALID_SIDES}")
    if quantity <= 0:
        raise ValueError("quantity must be > 0")
    if price < 0:
        raise ValueError("price must be >= 0")
    if fx_rate <= 0:
        raise ValueError("fx_rate must be > 0")

    td = (trade_date or "").strip() or date.today().isoformat()
    # Normalize ISO format
    try:
        datetime.fromisoformat(td[:10])
    except ValueError as e:
        raise ValueError(f"invalid trade_date: {td}") from e

    created = datetime.now().isoformat(timespec="seconds")
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO transactions "
            "(ticker, side, trade_date, quantity, price, currency, fees, fx_rate, notes, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            [ticker, side, td, quantity, price, currency.upper(), fees, fx_rate, notes, created],
        )
        tx_id = cur.lastrowid
        conn.commit()
        row = conn.execute("SELECT * FROM transactions WHERE id = ?", [tx_id]).fetchone()
    return _row_to_tx(row)


def update_transaction(tx_id: int, **fields: Any) -> Optional[Transaction]:
    init_db()
    allowed = {"ticker", "side", "trade_date", "quantity", "price",
               "currency", "fees", "fx_rate", "notes"}
    sets = []
    vals: List[Any] = []
    for k, v in fields.items():
        if k not in allowed:
            continue
        if k == "ticker":
            v = (v or "").strip().upper()
        elif k == "side":
            v = (v or "").strip().upper()
            if v not in VALID_SIDES:
                raise ValueError(f"side must be one of {VALID_SIDES}")
        elif k == "currency":
            v = (v or "USD").upper()
        sets.append(f"{k} = ?")
        vals.append(v)
    if not sets:
        return get_transaction(tx_id)
    vals.append(tx_id)
    with _connect() as conn:
        conn.execute(f"UPDATE transactions SET {', '.join(sets)} WHERE id = ?", vals)
        conn.commit()
        row = conn.execute("SELECT * FROM transactions WHERE id = ?", [tx_id]).fetchone()
    return _row_to_tx(row) if row else None


def delete_transaction(tx_id: int) -> bool:
    init_db()
    with _connect() as conn:
        cur = conn.execute("DELETE FROM transactions WHERE id = ?", [tx_id])
        conn.commit()
    return cur.rowcount > 0


def get_transaction(tx_id: int) -> Optional[Transaction]:
    init_db()
    with _connect() as conn:
        row = conn.execute("SELECT * FROM transactions WHERE id = ?", [tx_id]).fetchone()
    return _row_to_tx(row) if row else None


# ── Position derivation (average-cost) ───────────────────────────────────────

def derive_positions(transactions: Optional[Iterable[Transaction]] = None) -> List[Position]:
    """
    Replay transactions in date order to compute current positions, average
    cost basis (in base currency), realized P&L, and cumulative dividends.
    Closed positions (quantity = 0) are dropped.
    """
    txs = list(transactions) if transactions is not None else list_transactions()
    by_ticker: Dict[str, List[Transaction]] = {}
    for tx in txs:
        by_ticker.setdefault(tx.ticker, []).append(tx)

    positions: List[Position] = []
    for ticker, items in by_ticker.items():
        items.sort(key=lambda t: (t.trade_date, t.id or 0))
        qty = 0.0
        avg_cost = 0.0
        realized = 0.0
        divs = 0.0
        first_buy = None
        currency = "USD"
        last_date = None

        for tx in items:
            last_date = tx.trade_date
            if tx.side == "BUY":
                cost_base = (tx.quantity * tx.price + tx.fees) * tx.fx_rate
                new_qty = qty + tx.quantity
                if new_qty > _EPSILON:
                    avg_cost = (avg_cost * qty + cost_base) / new_qty
                qty = new_qty
                if first_buy is None:
                    first_buy = tx.trade_date
                currency = tx.currency
            elif tx.side == "SELL":
                sell_qty = min(tx.quantity, qty)
                proceeds_base = (tx.quantity * tx.price - tx.fees) * tx.fx_rate
                realized += proceeds_base - avg_cost * sell_qty
                qty = max(0.0, qty - tx.quantity)
                if qty <= _EPSILON:
                    qty = 0.0
                    avg_cost = 0.0
            elif tx.side == "DIV":
                # price = dividend per share, quantity = shares on ex-date
                divs += tx.quantity * tx.price * tx.fx_rate

        if qty > _EPSILON:
            positions.append(Position(
                ticker=ticker,
                quantity=round(qty, 8),
                avg_cost=round(avg_cost, 6),
                cost_basis=round(qty * avg_cost, 4),
                realized_pnl=round(realized, 4),
                dividends=round(divs, 4),
                first_buy_date=first_buy,
                last_trade_date=last_date,
                currency=currency,
            ))
        else:
            # Closed position still has realized + divs worth reporting at the portfolio level
            # We surface those via realized_summary(); skip from open positions.
            pass

    positions.sort(key=lambda p: p.ticker)
    return positions


def realized_summary(transactions: Optional[Iterable[Transaction]] = None) -> Dict[str, float]:
    """
    Total realized P&L and dividends across all tickers (including closed ones).
    """
    txs = list(transactions) if transactions is not None else list_transactions()
    by_ticker: Dict[str, List[Transaction]] = {}
    for tx in txs:
        by_ticker.setdefault(tx.ticker, []).append(tx)

    total_realized = 0.0
    total_dividends = 0.0
    for ticker, items in by_ticker.items():
        items.sort(key=lambda t: (t.trade_date, t.id or 0))
        qty = 0.0
        avg_cost = 0.0
        for tx in items:
            if tx.side == "BUY":
                cost_base = (tx.quantity * tx.price + tx.fees) * tx.fx_rate
                new_qty = qty + tx.quantity
                if new_qty > _EPSILON:
                    avg_cost = (avg_cost * qty + cost_base) / new_qty
                qty = new_qty
            elif tx.side == "SELL":
                sell_qty = min(tx.quantity, qty)
                proceeds_base = (tx.quantity * tx.price - tx.fees) * tx.fx_rate
                total_realized += proceeds_base - avg_cost * sell_qty
                qty = max(0.0, qty - tx.quantity)
                if qty <= _EPSILON:
                    qty = 0.0
                    avg_cost = 0.0
            elif tx.side == "DIV":
                total_dividends += tx.quantity * tx.price * tx.fx_rate
    return {
        "realized_pnl": round(total_realized, 4),
        "dividends":    round(total_dividends, 4),
    }


# ── Instruments (metadata cache) ─────────────────────────────────────────────

def upsert_instrument(
    ticker: str,
    *,
    name: str = "",
    sector: str = "",
    region: str = "",
    currency: str = "USD",
    asset_class: str = "EQUITY",
    beta: Optional[float] = None,
) -> None:
    init_db()
    now = datetime.now().isoformat(timespec="seconds")
    with _connect() as conn:
        conn.execute(
            "INSERT INTO instruments (ticker, name, sector, region, currency, asset_class, beta, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?) "
            "ON CONFLICT(ticker) DO UPDATE SET "
            "name=excluded.name, sector=excluded.sector, region=excluded.region, "
            "currency=excluded.currency, asset_class=excluded.asset_class, "
            "beta=COALESCE(excluded.beta, instruments.beta), updated_at=excluded.updated_at",
            [ticker.upper(), name, sector, region, currency.upper(), asset_class, beta, now],
        )
        conn.commit()


def get_instruments(tickers: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    init_db()
    tlist = [t.upper() for t in tickers if t]
    if not tlist:
        return {}
    with _connect() as conn:
        placeholders = ",".join("?" * len(tlist))
        rows = conn.execute(
            f"SELECT * FROM instruments WHERE ticker IN ({placeholders})",
            tlist,
        ).fetchall()
    return {r["ticker"]: dict(r) for r in rows}


# ── Targets (rebalancing) ────────────────────────────────────────────────────

def get_targets() -> Dict[str, float]:
    init_db()
    with _connect() as conn:
        rows = conn.execute("SELECT ticker, target_weight FROM targets").fetchall()
    return {r["ticker"]: float(r["target_weight"]) for r in rows}


def set_targets(weights: Dict[str, float]) -> Dict[str, float]:
    """Replace all targets with the provided mapping. Weights must be in [0, 1]."""
    init_db()
    cleaned: Dict[str, float] = {}
    for k, v in weights.items():
        tk = (k or "").strip().upper()
        if not tk:
            continue
        w = float(v)
        if w < 0 or w > 1:
            raise ValueError(f"target weight for {tk} out of range: {w}")
        cleaned[tk] = w
    total = sum(cleaned.values())
    if total > 1.0 + 1e-6:
        raise ValueError(f"target weights sum to {total:.4f} > 1.0")
    now = datetime.now().isoformat(timespec="seconds")
    with _connect() as conn:
        conn.execute("DELETE FROM targets")
        conn.executemany(
            "INSERT INTO targets (ticker, target_weight, updated_at) VALUES (?,?,?)",
            [(t, w, now) for t, w in cleaned.items()],
        )
        conn.commit()
    return cleaned


# ── Watchlist ────────────────────────────────────────────────────────────────

def list_watchlist() -> List[Dict[str, Any]]:
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT ticker, name, sector, currency FROM watchlist ORDER BY ticker"
        ).fetchall()
    return [dict(r) for r in rows]


def add_to_watchlist(ticker: str, name: str = "", sector: str = "", currency: str = "USD") -> bool:
    init_db()
    ticker = (ticker or "").strip().upper()
    if not ticker:
        raise ValueError("ticker required")
    now = datetime.now().isoformat(timespec="seconds")
    with _connect() as conn:
        try:
            conn.execute(
                "INSERT INTO watchlist (ticker, name, sector, currency, added_at) VALUES (?,?,?,?,?)",
                [ticker, name or ticker, sector, currency.upper(), now],
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False


def remove_from_watchlist(ticker: str) -> bool:
    init_db()
    with _connect() as conn:
        cur = conn.execute("DELETE FROM watchlist WHERE ticker = ?", [ticker.upper()])
        conn.commit()
    return cur.rowcount > 0


# ── Migration from legacy JSON ───────────────────────────────────────────────

def migrate_legacy_json() -> Dict[str, int]:
    """
    Import positions from the legacy `portfolio_positions.json` (and watchlist
    from `portfolio_watchlist.json`) into the SQLite ledger. Each legacy
    position becomes a single BUY transaction with the recorded buy_price and
    buy_date. Safe to run multiple times — checks for duplicates by
    (ticker, trade_date, quantity, price, side='BUY' + notes marker).
    """
    init_db()
    imported_tx = 0
    imported_watch = 0

    # Positions
    candidates = [
        config.DATA_DIR / "portfolio_positions.json",
        Path(__file__).resolve().parents[2] / "portfolio_positions.json",
    ]
    pos_path = next((p for p in candidates if p.exists()), None)
    if pos_path:
        try:
            items = json.loads(pos_path.read_text(encoding="utf-8"))
        except Exception:
            items = []
        for it in items or []:
            ticker = (it.get("ticker") or "").upper()
            qty = float(it.get("quantity", 0))
            price = float(it.get("buy_price", 0))
            tdate = (it.get("buy_date") or "")[:10] or date.today().isoformat()
            if not ticker or qty <= 0 or price <= 0:
                continue
            with _connect() as conn:
                exists = conn.execute(
                    "SELECT 1 FROM transactions WHERE ticker=? AND side='BUY' "
                    "AND trade_date=? AND ABS(quantity-?)<1e-6 AND ABS(price-?)<1e-6",
                    [ticker, tdate, qty, price],
                ).fetchone()
            if exists:
                continue
            add_transaction(
                ticker, "BUY", qty, price, tdate,
                notes=(it.get("notes") or "") + " [migrated]",
            )
            imported_tx += 1

    # Watchlist
    watch_candidates = [
        config.DATA_DIR / "portfolio_watchlist.json",
        Path(__file__).resolve().parents[2] / "portfolio_watchlist.json",
    ]
    watch_path = next((p for p in watch_candidates if p.exists()), None)
    if watch_path:
        try:
            items = json.loads(watch_path.read_text(encoding="utf-8"))
        except Exception:
            items = []
        for it in items or []:
            ticker = (it.get("ticker") or "").upper()
            if not ticker:
                continue
            if add_to_watchlist(
                ticker,
                name=it.get("name", ""),
                sector=it.get("sector", ""),
                currency=it.get("currency", "USD"),
            ):
                imported_watch += 1

    return {"transactions": imported_tx, "watchlist": imported_watch}
