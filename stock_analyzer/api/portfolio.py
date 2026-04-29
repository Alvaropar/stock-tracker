"""
Portfolio API: personal position tracking and watchlist.

Positions are user-entered holdings (ticker, qty, buy price, buy date).
On read, they are enriched with current price, returns, holding period,
and a lightweight technical signal (BUY/HOLD/SELL based on MA + RSI).
"""
from __future__ import annotations

import json
import uuid
import concurrent.futures as cf
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Blueprint, jsonify, request

bp = Blueprint("portfolio", __name__, url_prefix="/api/portfolio")


# ── Storage ──────────────────────────────────────────────────────
def _data_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _positions_path() -> Path:
    return _data_root() / "portfolio_positions.json"


def _watchlist_path() -> Path:
    return _data_root() / "portfolio_watchlist.json"


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ── Market data + signal ─────────────────────────────────────────
def _fetch_ticker_snapshot(ticker: str) -> Dict[str, Any]:
    """Fetch current price + simple technical signal for a ticker."""
    out: Dict[str, Any] = {"ticker": ticker, "current_price": None, "signal": None}
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period="1y", auto_adjust=True)
        if hist is None or hist.empty:
            return out
        close = hist["Close"].dropna()
        if close.empty:
            return out
        current = float(close.iloc[-1])
        out["current_price"] = current
        if len(close) >= 20:
            ma50 = float(close.tail(min(50, len(close))).mean())
            ma200 = float(close.tail(min(200, len(close))).mean())
            # RSI 14
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs_series = gain / loss.replace(0, 1e-9)
            rs = float(rs_series.iloc[-1]) if not rs_series.empty else 1.0
            rsi = 100 - (100 / (1 + rs)) if rs == rs else 50.0

            trend_up = current > ma50 and ma50 > ma200
            trend_down = current < ma50 and ma50 < ma200

            if trend_up and 45 <= rsi <= 70:
                action, color = "BUY", "#22c55e"
            elif trend_up and rsi > 70:
                action, color = "HOLD", "#eab308"
            elif trend_down and rsi < 35:
                action, color = "ACCUMULATE", "#06b6d4"
            elif trend_down or rsi > 78:
                action, color = "SELL", "#ef4444"
            elif current > ma50:
                action, color = "HOLD", "#a78bfa"
            else:
                action, color = "WATCH", "#8b9ab4"

            out["signal"] = {
                "action": action,
                "color": color,
                "rsi": round(rsi, 1),
                "ma50": round(ma50, 2),
                "ma200": round(ma200, 2),
            }
    except Exception:
        pass
    return out


def _days_between(iso_date: str) -> Optional[int]:
    try:
        d = datetime.fromisoformat(iso_date[:10]).date()
        return (date.today() - d).days
    except Exception:
        return None


# ── Positions ────────────────────────────────────────────────────

@bp.route("/positions", methods=["GET"])
def get_positions():
    """Return tracked positions enriched with live prices, returns, signals."""
    positions = _load_json(_positions_path(), [])
    if not positions:
        return jsonify({
            "market_value": 0.0, "cost_basis": 0.0,
            "unrealized_pnl": 0.0, "return_pct": 0.0,
            "positions": [],
        })

    unique_tickers = list({p["ticker"] for p in positions})
    snapshots: Dict[str, Dict[str, Any]] = {}
    with cf.ThreadPoolExecutor(max_workers=min(10, len(unique_tickers))) as pool:
        for snap in pool.map(_fetch_ticker_snapshot, unique_tickers):
            snapshots[snap["ticker"]] = snap

    enriched = []
    total_mv = 0.0
    total_cost = 0.0
    for p in positions:
        snap = snapshots.get(p["ticker"], {})
        qty = float(p.get("quantity", 0))
        buy_price = float(p.get("buy_price", 0))
        current = snap.get("current_price") or buy_price
        market_value = qty * current
        cost_basis = qty * buy_price
        pnl = market_value - cost_basis
        pnl_pct = ((current - buy_price) / buy_price * 100) if buy_price else 0.0
        days_held = _days_between(p.get("buy_date", ""))
        ann_return = None
        if days_held and days_held > 1 and buy_price > 0 and current > 0:
            try:
                ann_return = (((current / buy_price) ** (365 / days_held)) - 1) * 100
            except Exception:
                ann_return = None
        total_mv += market_value
        total_cost += cost_basis
        enriched.append({
            **p,
            "current_price": round(current, 4),
            "market_value": round(market_value, 2),
            "cost_basis": round(cost_basis, 2),
            "unrealized_pnl": round(pnl, 2),
            "unrealized_pct": round(pnl_pct, 2),
            "days_held": days_held,
            "annualized_return_pct": round(ann_return, 2) if ann_return is not None else None,
            "signal": snap.get("signal"),
        })

    total_pnl = total_mv - total_cost
    total_return = (total_pnl / total_cost * 100) if total_cost > 0 else 0.0

    return jsonify({
        "market_value": round(total_mv, 2),
        "cost_basis": round(total_cost, 2),
        "unrealized_pnl": round(total_pnl, 2),
        "return_pct": round(total_return, 2),
        "positions": enriched,
    })


@bp.route("/positions", methods=["POST"])
def add_position():
    body = request.get_json(force=True)
    ticker = (body.get("ticker") or "").strip().upper()
    if not ticker:
        return jsonify({"error": "ticker is required"}), 400
    try:
        quantity = float(body.get("quantity", 0))
        buy_price = float(body.get("buy_price", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "quantity and buy_price must be numbers"}), 400
    if quantity <= 0 or buy_price <= 0:
        return jsonify({"error": "quantity and buy_price must be greater than 0"}), 400

    buy_date = (body.get("buy_date") or "").strip() or date.today().isoformat()

    item = {
        "id": str(uuid.uuid4())[:12],
        "ticker": ticker,
        "name": (body.get("name") or ticker).strip(),
        "quantity": quantity,
        "buy_price": buy_price,
        "buy_date": buy_date,
        "notes": (body.get("notes") or "").strip(),
        "added_at": datetime.now().isoformat(timespec="seconds"),
    }
    items = _load_json(_positions_path(), [])
    items.append(item)
    _save_json(_positions_path(), items)
    return jsonify({"ok": True, "position": item})


@bp.route("/positions/<pos_id>", methods=["PATCH"])
def update_position(pos_id: str):
    body = request.get_json(force=True)
    items = _load_json(_positions_path(), [])
    updated = None
    for p in items:
        if p.get("id") == pos_id:
            for field in ("quantity", "buy_price", "buy_date", "notes", "name"):
                if field in body:
                    p[field] = body[field]
            updated = p
            break
    if not updated:
        return jsonify({"error": "position not found"}), 404
    _save_json(_positions_path(), items)
    return jsonify({"ok": True, "position": updated})


@bp.route("/positions/<pos_id>", methods=["DELETE"])
def remove_position(pos_id: str):
    items = _load_json(_positions_path(), [])
    before = len(items)
    items = [p for p in items if p.get("id") != pos_id]
    _save_json(_positions_path(), items)
    return jsonify({"ok": True, "removed": before - len(items)})


# ── Watchlist ────────────────────────────────────────────────────

@bp.route("/watchlist", methods=["GET"])
def get_watchlist():
    return jsonify(_load_json(_watchlist_path(), []))


@bp.route("/watchlist", methods=["POST"])
def add_to_watchlist():
    body = request.get_json(force=True)
    ticker = (body.get("ticker") or "").strip().upper()
    if not ticker:
        return jsonify({"error": "ticker required"}), 400
    items = _load_json(_watchlist_path(), [])
    if any(i["ticker"] == ticker for i in items):
        return jsonify({"error": f"{ticker} already in watchlist"}), 409
    items.append({
        "ticker": ticker,
        "name": body.get("name", ticker),
        "sector": body.get("sector", ""),
        "currency": body.get("currency", "USD"),
    })
    _save_json(_watchlist_path(), items)
    return jsonify({"ok": True, "count": len(items)})


@bp.route("/watchlist/<ticker>", methods=["DELETE"])
def remove_from_watchlist(ticker: str):
    ticker = ticker.upper()
    items = _load_json(_watchlist_path(), [])
    before = len(items)
    items = [i for i in items if i["ticker"] != ticker]
    _save_json(_watchlist_path(), items)
    return jsonify({"ok": True, "removed": before - len(items)})
