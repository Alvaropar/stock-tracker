"""
Portfolio API: transactions, positions, dashboard, targets, rebalancing.

Position state is derived from a SQLite transaction ledger
(`services.ledger`). Aggregations and per-ticker signals come from
`services.portfolio_analytics`; rebalancing suggestions from
`services.rebalancer`.
"""
from __future__ import annotations

import concurrent.futures as cf
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from flask import Blueprint, jsonify, request

from ..config import config
from ..services import ledger
from ..services import portfolio_analytics as pa
from ..services import rebalancer as rb

bp = Blueprint("portfolio", __name__, url_prefix="/api/portfolio")
log = logging.getLogger("app.portfolio")


# ── Caches (TTL-based, single-process) ───────────────────────────────────────

_QUOTE_TTL = 60          # seconds
_INFO_TTL = 24 * 3600    # 24 h for instrument metadata
_quotes: Dict[str, Tuple[float, Optional[float]]] = {}
_quotes_lock = threading.Lock()


def _fetch_quote(ticker: str) -> Optional[float]:
    """Latest close price. Cached for `_QUOTE_TTL` seconds."""
    now = time.time()
    with _quotes_lock:
        ts, price = _quotes.get(ticker, (0.0, None))
        if now - ts < _QUOTE_TTL and price is not None:
            return price
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period="5d", auto_adjust=True)
        if hist is None or hist.empty:
            return None
        close = float(hist["Close"].dropna().iloc[-1])
    except Exception as e:
        log.warning("quote fetch failed for %s: %s", ticker, e)
        return None
    with _quotes_lock:
        _quotes[ticker] = (now, close)
    return close


def _bulk_quotes(tickers: List[str]) -> Dict[str, Optional[float]]:
    if not tickers:
        return {}
    out: Dict[str, Optional[float]] = {}
    with cf.ThreadPoolExecutor(max_workers=min(10, len(tickers))) as pool:
        futs = {pool.submit(_fetch_quote, t): t for t in tickers}
        for f in cf.as_completed(futs):
            out[futs[f]] = f.result()
    return out


def _ensure_instruments(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Return instrument metadata for the given tickers, fetching from yfinance
    for tickers that are missing or older than `_INFO_TTL`.
    """
    if not tickers:
        return {}
    existing = ledger.get_instruments(tickers)
    now = time.time()
    stale: List[str] = []
    for t in tickers:
        meta = existing.get(t)
        if not meta:
            stale.append(t)
            continue
        updated = meta.get("updated_at") or ""
        try:
            from datetime import datetime as _dt
            age = (now - _dt.fromisoformat(updated).timestamp()) if updated else _INFO_TTL + 1
        except ValueError:
            age = _INFO_TTL + 1
        if age > _INFO_TTL:
            stale.append(t)

    if stale:
        with cf.ThreadPoolExecutor(max_workers=min(8, len(stale))) as pool:
            list(pool.map(_refresh_instrument, stale))
        existing = ledger.get_instruments(tickers)

    return existing


def _refresh_instrument(ticker: str) -> None:
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
    except Exception as e:
        log.debug("info fetch failed for %s: %s", ticker, e)
        info = {}
    name = info.get("longName") or info.get("shortName") or ticker
    sector = info.get("sector") or ""
    currency = (info.get("currency") or "USD").upper()
    beta = info.get("beta")
    try:
        beta = float(beta) if beta is not None else None
    except (TypeError, ValueError):
        beta = None
    region = pa.infer_region(ticker, currency)
    ledger.upsert_instrument(
        ticker, name=name, sector=sector, region=region,
        currency=currency, beta=beta,
    )


def _maybe_auto_migrate() -> None:
    """Run legacy JSON migration once if the ledger is empty."""
    try:
        if not ledger.list_transactions():
            res = ledger.migrate_legacy_json()
            if res.get("transactions") or res.get("watchlist"):
                log.info("auto-migrated legacy portfolio: %s", res)
    except Exception as e:
        log.warning("auto-migration skipped: %s", e)


# ── Transactions ─────────────────────────────────────────────────────────────

@bp.route("/transactions", methods=["GET"])
def list_transactions():
    _maybe_auto_migrate()
    ticker = request.args.get("ticker")
    txs = ledger.list_transactions(ticker)
    return jsonify([t.to_dict() for t in txs])


@bp.route("/transactions", methods=["POST"])
def add_transaction():
    _maybe_auto_migrate()
    body = request.get_json(force=True, silent=True) or {}
    try:
        tx = ledger.add_transaction(
            ticker=body.get("ticker", ""),
            side=body.get("side", "BUY"),
            quantity=float(body.get("quantity", 0)),
            price=float(body.get("price", 0)),
            trade_date=body.get("trade_date"),
            currency=body.get("currency", config.BASE_CURRENCY),
            fees=float(body.get("fees", 0)),
            fx_rate=float(body.get("fx_rate", 1.0)),
            notes=body.get("notes", ""),
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"ok": True, "transaction": tx.to_dict()})


@bp.route("/transactions/<int:tx_id>", methods=["PATCH"])
def patch_transaction(tx_id: int):
    body = request.get_json(force=True, silent=True) or {}
    try:
        tx = ledger.update_transaction(tx_id, **body)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if not tx:
        return jsonify({"error": "transaction not found"}), 404
    return jsonify({"ok": True, "transaction": tx.to_dict()})


@bp.route("/transactions/<int:tx_id>", methods=["DELETE"])
def delete_transaction(tx_id: int):
    removed = ledger.delete_transaction(tx_id)
    if not removed:
        return jsonify({"error": "transaction not found"}), 404
    return jsonify({"ok": True})


# ── Positions (derived) ──────────────────────────────────────────────────────

@bp.route("/positions", methods=["GET"])
def get_positions():
    """
    Derived positions enriched with current quotes, sector/region metadata,
    and aggregate totals. Sentiment signals are NOT computed here — use
    /dashboard for the full view.
    """
    _maybe_auto_migrate()
    positions = ledger.derive_positions()
    if not positions:
        empty_totals = {
            "market_value": 0.0, "cost_basis": 0.0, "unrealized_pnl": 0.0,
            "return_pct": 0.0, "realized_pnl": 0.0, "dividends": 0.0, "total_pnl": 0.0,
        }
        return jsonify({**empty_totals, "positions": []})

    tickers = [p.ticker for p in positions]
    quotes = _bulk_quotes(tickers)
    instruments = _ensure_instruments(tickers)
    enriched, totals = pa.enrich_positions(positions, quotes, instruments)
    return jsonify({
        **totals,
        "positions": [e.to_dict() for e in enriched],
    })


# Legacy compatibility: old frontend POSTs to /positions to add a single buy.
@bp.route("/positions", methods=["POST"])
def add_position_legacy():
    body = request.get_json(force=True, silent=True) or {}
    ticker = (body.get("ticker") or "").strip().upper()
    try:
        quantity = float(body.get("quantity", 0))
        buy_price = float(body.get("buy_price", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "quantity and buy_price must be numbers"}), 400
    try:
        tx = ledger.add_transaction(
            ticker=ticker,
            side="BUY",
            quantity=quantity,
            price=buy_price,
            trade_date=body.get("buy_date"),
            notes=body.get("notes", ""),
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"ok": True, "transaction": tx.to_dict()})


@bp.route("/positions/<pos_id>", methods=["DELETE"])
def remove_position_legacy(pos_id: str):
    """
    Deprecated. Old UUID position IDs no longer map to anything after the
    migration to transactions. Frontends should call /transactions/<tx_id>.
    """
    return jsonify({
        "error": "position deletion has moved to /api/portfolio/transactions/<id>",
        "deprecated": True,
    }), 410


# ── Dashboard ────────────────────────────────────────────────────────────────

def _compute_signal_for(ticker: str, market_ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        from ..services.market_data import fetch_price_history
        hist = fetch_price_history(ticker, period="1y", auto_adjust=True)
        if hist is None or hist.empty:
            return None
        return pa.compute_portfolio_signal(hist, market_ctx)
    except Exception as e:
        log.debug("signal failed for %s: %s", ticker, e)
        return None


@bp.route("/dashboard", methods=["GET"])
def dashboard():
    """
    Full portfolio view: enriched positions + breakdowns (sector / region /
    currency) + portfolio beta + top contributors + per-ticker scoring
    signals. Sentiment is only included when an LLM provider is configured;
    otherwise the signal is technical-only by design.
    """
    _maybe_auto_migrate()
    positions = ledger.derive_positions()
    realized_total = ledger.realized_summary()
    if not positions:
        return jsonify({
            "totals": {
                "market_value": 0.0, "cost_basis": 0.0, "unrealized_pnl": 0.0,
                "return_pct": 0.0,
                "realized_pnl": realized_total["realized_pnl"],
                "dividends": realized_total["dividends"],
                "total_pnl": realized_total["realized_pnl"] + realized_total["dividends"],
            },
            "positions": [],
            "breakdowns": {"sectors": [], "regions": [], "currencies": []},
            "portfolio_beta": None,
            "top_gainers": [], "top_losers": [],
            "concentration_hhi": 0.0,
            "n_positions": 0,
            "llm_available": config.llm_available(),
            "base_currency": config.BASE_CURRENCY,
        })

    tickers = [p.ticker for p in positions]
    quotes = _bulk_quotes(tickers)
    instruments = _ensure_instruments(tickers)

    # Market context (regime, SPY trend, RS reference returns) — shared across signals
    try:
        from ..services.market_data import fetch_market_context
        market_ctx = fetch_market_context(period="1y")
    except Exception as e:
        log.debug("market context unavailable: %s", e)
        market_ctx = {}

    signals: Dict[str, Dict[str, Any]] = {}
    with cf.ThreadPoolExecutor(max_workers=min(8, len(tickers))) as pool:
        for t, sig in zip(tickers, pool.map(lambda x: _compute_signal_for(x, market_ctx), tickers)):
            if sig is not None:
                signals[t] = sig

    enriched, totals = pa.enrich_positions(positions, quotes, instruments, signals)
    aggregates = pa.aggregate(enriched)

    # Merge cumulative realized/dividends from ALL transactions (including closed positions)
    totals["realized_pnl"] = realized_total["realized_pnl"]
    totals["dividends"]    = realized_total["dividends"]
    totals["total_pnl"]    = round(
        totals["market_value"] - totals["cost_basis"]
        + realized_total["realized_pnl"] + realized_total["dividends"],
        2,
    )

    return jsonify({
        "totals":   totals,
        "positions": [e.to_dict() for e in enriched],
        "breakdowns": {
            "sectors":    aggregates["sectors"],
            "regions":    aggregates["regions"],
            "currencies": aggregates["currencies"],
        },
        "portfolio_beta":     aggregates["portfolio_beta"],
        "top_gainers":        aggregates["top_gainers"],
        "top_losers":         aggregates["top_losers"],
        "concentration_hhi":  aggregates["concentration_hhi"],
        "n_positions":        aggregates["n_positions"],
        "market_regime":      market_ctx.get("regime"),
        "spy_trend_bull":     market_ctx.get("spy_trend_bull"),
        "llm_available":      config.llm_available(),
        "base_currency":      config.BASE_CURRENCY,
    })


# ── Targets ──────────────────────────────────────────────────────────────────

@bp.route("/targets", methods=["GET"])
def get_targets():
    return jsonify(ledger.get_targets())


@bp.route("/targets", methods=["PUT"])
def put_targets():
    body = request.get_json(force=True, silent=True) or {}
    if not isinstance(body, dict):
        return jsonify({"error": "expected an object {ticker: weight}"}), 400
    try:
        saved = ledger.set_targets({str(k): float(v) for k, v in body.items()})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"ok": True, "targets": saved})


# ── Rebalance ────────────────────────────────────────────────────────────────

@bp.route("/rebalance", methods=["GET"])
def rebalance():
    """
    Suggested trades to move current positions toward stored targets.

    Query params:
        cash:              Uninvested cash to allocate (default 0)
        drift:             Threshold below which drift is ignored (default 0.01)
        min_trade:         Minimum trade notional (default 0)
        fractional:        '1' to allow fractional shares (default '1')
        lot_size:          Lot size when fractional=0 (default 1)
    """
    _maybe_auto_migrate()
    cash = float(request.args.get("cash", 0))
    drift = float(request.args.get("drift", 0.01))
    min_trade = float(request.args.get("min_trade", 0))
    fractional = request.args.get("fractional", "1") not in ("0", "false", "False")
    lot_size = float(request.args.get("lot_size", 1))

    positions = ledger.derive_positions()
    targets = ledger.get_targets()
    if not positions and not targets:
        return jsonify({"error": "no positions or targets defined"}), 400

    tickers = sorted(set([p.ticker for p in positions]) | set(targets.keys()))
    quotes = _bulk_quotes(tickers)

    holdings: Dict[str, Dict[str, float]] = {}
    for p in positions:
        price = quotes.get(p.ticker) or p.avg_cost
        holdings[p.ticker] = {
            "market_value": p.quantity * price,
            "price": price,
            "quantity": p.quantity,
        }
    # Tickers in targets but not in holdings
    for t in targets:
        if t not in holdings:
            price = quotes.get(t) or 0.0
            holdings[t] = {"market_value": 0.0, "price": price, "quantity": 0.0}

    result = rb.compute_rebalance(
        holdings, targets,
        cash=cash, drift_threshold=drift,
        min_trade_notional=min_trade,
        fractional_shares=fractional, lot_size=lot_size,
    )
    return jsonify(result)


# ── Migration trigger ────────────────────────────────────────────────────────

@bp.route("/migrate", methods=["POST"])
def migrate():
    result = ledger.migrate_legacy_json()
    return jsonify({"ok": True, **result})


# ── Watchlist ────────────────────────────────────────────────────────────────

@bp.route("/watchlist", methods=["GET"])
def get_watchlist():
    _maybe_auto_migrate()
    return jsonify(ledger.list_watchlist())


@bp.route("/watchlist", methods=["POST"])
def add_watch():
    body = request.get_json(force=True, silent=True) or {}
    ticker = (body.get("ticker") or "").strip().upper()
    if not ticker:
        return jsonify({"error": "ticker required"}), 400
    added = ledger.add_to_watchlist(
        ticker,
        name=body.get("name", ticker),
        sector=body.get("sector", ""),
        currency=body.get("currency", "USD"),
    )
    if not added:
        return jsonify({"error": f"{ticker} already in watchlist"}), 409
    return jsonify({"ok": True})


@bp.route("/watchlist/<ticker>", methods=["DELETE"])
def remove_watch(ticker: str):
    removed = ledger.remove_from_watchlist(ticker)
    return jsonify({"ok": True, "removed": int(removed)})
