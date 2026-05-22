"""
Rebalancing engine: compares current portfolio weights to target weights and
emits suggested trades.

Inputs are intentionally simple dicts so this module is easy to unit-test
without any I/O. Quote and price discovery happen in the API layer.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class RebalanceTrade:
    ticker: str
    side: str               # "BUY" | "SELL"
    quantity: float
    price: float
    notional: float         # base currency
    current_weight: float
    target_weight: float
    drift: float            # current - target  (signed)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _round_qty(qty: float, fractional: bool, lot_size: float) -> float:
    if fractional:
        return round(qty, 4)
    if lot_size <= 0:
        lot_size = 1
    # round to nearest whole lot
    n_lots = round(qty / lot_size)
    return float(max(0, n_lots) * lot_size)


def compute_rebalance(
    holdings: Dict[str, Dict[str, float]],
    targets: Dict[str, float],
    *,
    cash: float = 0.0,
    drift_threshold: float = 0.01,   # ignore drift < 1% absolute
    min_trade_notional: float = 0.0,
    fractional_shares: bool = True,
    lot_size: float = 1.0,
) -> Dict[str, Any]:
    """
    Compute the trades that move *holdings* toward *targets*.

    `holdings[ticker]` must include `market_value` and `price` (current).
    `targets[ticker]` is a weight in [0, 1]; tickers absent from targets are
    treated as having target 0 (i.e. should be liquidated). New tickers in
    targets without holdings get a market_value of 0.

    The portfolio total used as the denominator is current market value plus
    any uninvested *cash*.
    """
    tickers = set(holdings) | set(targets)
    portfolio_value = sum(h["market_value"] for h in holdings.values()) + max(0.0, cash)
    if portfolio_value <= 0:
        return {
            "portfolio_value": 0.0,
            "cash": cash,
            "drift_summary": {"max_abs_drift": 0.0, "mean_abs_drift": 0.0, "rebalance_recommended": False},
            "rows": [],
            "trades": [],
        }

    rows: List[Dict[str, Any]] = []
    trades: List[RebalanceTrade] = []
    drifts: List[float] = []

    for ticker in sorted(tickers):
        h = holdings.get(ticker, {})
        price = float(h.get("price", 0.0))
        mv = float(h.get("market_value", 0.0))
        current_w = mv / portfolio_value
        target_w  = float(targets.get(ticker, 0.0))
        drift = current_w - target_w
        target_value = target_w * portfolio_value
        delta_value = target_value - mv   # +ve = buy, -ve = sell
        row = {
            "ticker":         ticker,
            "current_weight": round(current_w, 4),
            "target_weight":  round(target_w, 4),
            "drift":          round(drift, 4),
            "current_value":  round(mv, 2),
            "target_value":   round(target_value, 2),
            "delta_value":    round(delta_value, 2),
            "price":          price,
            "trade":          None,
        }
        drifts.append(abs(drift))

        if abs(drift) < drift_threshold:
            rows.append(row)
            continue
        if abs(delta_value) < min_trade_notional:
            rows.append(row)
            continue
        if price <= 0:
            row["trade"] = {"warning": "missing price"}
            rows.append(row)
            continue

        raw_qty = abs(delta_value) / price
        qty = _round_qty(raw_qty, fractional_shares, lot_size)
        if qty <= 0:
            rows.append(row)
            continue
        side = "BUY" if delta_value > 0 else "SELL"
        notional = qty * price
        trade = RebalanceTrade(
            ticker=ticker,
            side=side,
            quantity=qty,
            price=round(price, 4),
            notional=round(notional, 2),
            current_weight=round(current_w, 4),
            target_weight=round(target_w, 4),
            drift=round(drift, 4),
        )
        row["trade"] = trade.to_dict()
        rows.append(row)
        trades.append(trade)

    rows.sort(key=lambda r: -abs(r["drift"]))
    max_abs = max(drifts) if drifts else 0.0
    mean_abs = sum(drifts) / len(drifts) if drifts else 0.0

    # Cash after trades
    cash_used = sum(t.notional for t in trades if t.side == "BUY") - \
                sum(t.notional for t in trades if t.side == "SELL")
    cash_after = cash - cash_used

    return {
        "portfolio_value": round(portfolio_value, 2),
        "cash":            round(cash, 2),
        "cash_after":      round(cash_after, 2),
        "drift_summary":   {
            "max_abs_drift":         round(max_abs, 4),
            "mean_abs_drift":        round(mean_abs, 4),
            "rebalance_recommended": max_abs >= drift_threshold,
        },
        "rows":   rows,
        "trades": [t.to_dict() for t in trades],
    }
