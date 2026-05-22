"""
Portfolio aggregations and per-position enrichment.

Pure functions that take already-fetched data (quotes, instrument metadata,
optional price history) and produce dashboard payloads: market value, weights,
breakdowns (sector / region / currency), beta, top contributors, and a real
scoring signal computed via `services.scoring`.

I/O (yfinance, parallel fetches) lives in the API layer so this module stays
fast to unit-test.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import scoring
from .ledger import Position

_SUFFIX_REGION = {
    "ST": "EU", "OL": "EU", "CO": "EU", "HE": "EU", "L": "EU",
    "DE": "EU", "PA": "EU", "MI": "EU", "MC": "EU", "BR": "EU",
    "AS": "EU", "LS": "EU", "VI": "EU", "SW": "EU", "IS": "EU",
    "TO": "NA", "V": "NA", "MX": "LATAM",
    "HK": "ASIA", "T": "ASIA", "KS": "ASIA", "KQ": "ASIA",
    "TW": "ASIA", "TWO": "ASIA", "SI": "ASIA", "BO": "ASIA", "NS": "ASIA",
    "SS": "CN", "SZ": "CN",
    "AX": "APAC", "NZ": "APAC",
    "JO": "EMEA", "SA": "LATAM",
}


def infer_region(ticker: str, fallback_currency: str = "USD") -> str:
    """Cheap region inference from ticker suffix; falls back to currency."""
    if "." in ticker:
        suffix = ticker.rsplit(".", 1)[-1].upper()
        if suffix in _SUFFIX_REGION:
            return _SUFFIX_REGION[suffix]
    cur = (fallback_currency or "").upper()
    if cur in ("EUR", "GBP", "CHF", "SEK", "NOK", "DKK"):
        return "EU"
    if cur in ("CNY", "HKD", "JPY", "KRW", "TWD"):
        return "ASIA"
    if cur in ("AUD", "NZD"):
        return "APAC"
    return "NA"


# ── Per-position enrichment ──────────────────────────────────────────────────

@dataclass
class EnrichedPosition:
    ticker: str
    name: str
    quantity: float
    avg_cost: float
    cost_basis: float
    current_price: Optional[float]
    market_value: float
    unrealized_pnl: float
    unrealized_pct: float
    realized_pnl: float
    dividends: float
    weight: float                       # of total market value
    sector: str
    region: str
    currency: str
    beta: Optional[float]
    days_held: Optional[int]
    annualized_return_pct: Optional[float]
    contribution_pct: float             # contribution to overall return %
    signal: Optional[Dict[str, Any]]
    first_buy_date: Optional[str]
    last_trade_date: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _days_between(iso_date: Optional[str]) -> Optional[int]:
    if not iso_date:
        return None
    try:
        d = datetime.fromisoformat(iso_date[:10]).date()
        return (date.today() - d).days
    except ValueError:
        return None


def enrich_positions(
    positions: List[Position],
    quotes: Dict[str, Optional[float]],
    instruments: Dict[str, Dict[str, Any]],
    signals: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List[EnrichedPosition], Dict[str, float]]:
    """
    Combine ledger positions with current quotes, instrument metadata, and
    optional scoring signals into a dashboard-ready list.

    Returns (enriched, totals) where totals carries market_value, cost_basis,
    unrealized_pnl, return_pct, realized_pnl, dividends.
    """
    signals = signals or {}
    enriched: List[EnrichedPosition] = []
    total_mv = 0.0
    total_cost = 0.0
    total_realized = 0.0
    total_divs = 0.0

    # First pass: compute market values to derive weights
    intermediate = []
    for p in positions:
        cur = quotes.get(p.ticker)
        if cur is None:
            cur = p.avg_cost  # fallback so MV ≠ 0
        market_value = p.quantity * cur
        pnl = market_value - p.cost_basis
        pnl_pct = (pnl / p.cost_basis * 100.0) if p.cost_basis > 0 else 0.0
        total_mv += market_value
        total_cost += p.cost_basis
        total_realized += p.realized_pnl
        total_divs += p.dividends
        intermediate.append((p, cur, market_value, pnl, pnl_pct))

    for p, cur, mv, pnl, pnl_pct in intermediate:
        weight = (mv / total_mv) if total_mv > 0 else 0.0
        contribution = (pnl / total_cost * 100.0) if total_cost > 0 else 0.0
        meta = instruments.get(p.ticker, {})
        sector = meta.get("sector") or "Unknown"
        currency = meta.get("currency") or p.currency
        region = meta.get("region") or infer_region(p.ticker, currency)
        beta = meta.get("beta")
        days_held = _days_between(p.first_buy_date)
        ann = None
        if days_held and days_held > 1 and p.avg_cost > 0 and cur > 0:
            try:
                ann = (((cur / p.avg_cost) ** (365.0 / days_held)) - 1) * 100.0
            except (ValueError, ZeroDivisionError):
                ann = None
        enriched.append(EnrichedPosition(
            ticker=p.ticker,
            name=meta.get("name") or p.ticker,
            quantity=p.quantity,
            avg_cost=p.avg_cost,
            cost_basis=p.cost_basis,
            current_price=cur,
            market_value=round(mv, 2),
            unrealized_pnl=round(pnl, 2),
            unrealized_pct=round(pnl_pct, 2),
            realized_pnl=p.realized_pnl,
            dividends=p.dividends,
            weight=round(weight, 4),
            sector=sector,
            region=region,
            currency=currency,
            beta=beta,
            days_held=days_held,
            annualized_return_pct=round(ann, 2) if ann is not None else None,
            contribution_pct=round(contribution, 2),
            signal=signals.get(p.ticker),
            first_buy_date=p.first_buy_date,
            last_trade_date=p.last_trade_date,
        ))

    totals = {
        "market_value":    round(total_mv, 2),
        "cost_basis":      round(total_cost, 2),
        "unrealized_pnl":  round(total_mv - total_cost, 2),
        "return_pct":      round(((total_mv - total_cost) / total_cost * 100.0) if total_cost > 0 else 0.0, 2),
        "realized_pnl":    round(total_realized, 2),
        "dividends":       round(total_divs, 2),
        "total_pnl":       round(total_mv - total_cost + total_realized + total_divs, 2),
    }
    return enriched, totals


# ── Aggregations ─────────────────────────────────────────────────────────────

def _group_by(enriched: List[EnrichedPosition], key: str) -> List[Dict[str, Any]]:
    buckets: Dict[str, Dict[str, float]] = {}
    for e in enriched:
        k = getattr(e, key) or "Unknown"
        b = buckets.setdefault(k, {"market_value": 0.0, "unrealized_pnl": 0.0, "cost_basis": 0.0, "count": 0})
        b["market_value"]   += e.market_value
        b["unrealized_pnl"] += e.unrealized_pnl
        b["cost_basis"]     += e.cost_basis
        b["count"]          += 1
    total_mv = sum(b["market_value"] for b in buckets.values()) or 1.0
    out = []
    for k, b in buckets.items():
        out.append({
            "label":          k,
            "market_value":   round(b["market_value"], 2),
            "weight":         round(b["market_value"] / total_mv, 4),
            "unrealized_pnl": round(b["unrealized_pnl"], 2),
            "cost_basis":     round(b["cost_basis"], 2),
            "count":          int(b["count"]),
        })
    out.sort(key=lambda x: -x["market_value"])
    return out


def aggregate(enriched: List[EnrichedPosition]) -> Dict[str, Any]:
    """Compute portfolio-level breakdowns and contributors."""
    sectors    = _group_by(enriched, "sector")
    regions    = _group_by(enriched, "region")
    currencies = _group_by(enriched, "currency")

    # Portfolio beta (weighted average of available betas; ignores positions with no beta)
    weighted_beta = 0.0
    beta_weight   = 0.0
    for e in enriched:
        if e.beta is not None:
            weighted_beta += e.beta * e.weight
            beta_weight   += e.weight
    portfolio_beta = round(weighted_beta / beta_weight, 3) if beta_weight > 0 else None

    # Top P&L contributors (positive and negative)
    by_pnl = sorted(enriched, key=lambda e: e.unrealized_pnl, reverse=True)
    top_gainers = [
        {"ticker": e.ticker, "unrealized_pnl": e.unrealized_pnl, "unrealized_pct": e.unrealized_pct, "weight": e.weight}
        for e in by_pnl[:5] if e.unrealized_pnl > 0
    ]
    top_losers = [
        {"ticker": e.ticker, "unrealized_pnl": e.unrealized_pnl, "unrealized_pct": e.unrealized_pct, "weight": e.weight}
        for e in by_pnl[::-1][:5] if e.unrealized_pnl < 0
    ]

    # Concentration (Herfindahl-Hirschman)
    hhi = sum((e.weight * 100) ** 2 for e in enriched)

    return {
        "sectors":    sectors,
        "regions":    regions,
        "currencies": currencies,
        "portfolio_beta":  portfolio_beta,
        "top_gainers":     top_gainers,
        "top_losers":      top_losers,
        "concentration_hhi": round(hhi, 1),
        "n_positions":     len(enriched),
    }


# ── Per-ticker scoring signal ────────────────────────────────────────────────

def compute_portfolio_signal(
    hist: pd.DataFrame,
    market_ctx: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Compute a lightweight technical-focused signal using the production
    scoring engine. Returns {action, css_class, score, momentum_score,
    risk_score, rsi, atr_pct} or None if history is too short.

    Fundamentals are skipped (slow yfinance .info call) and sentiment is
    delegated to the dashboard layer (only when LLM is configured). This is
    sufficient for a portfolio-level health check.
    """
    if hist is None or hist.empty or len(hist) < 60:
        return None
    # market_data.compute_indicators is the heavy path — we import locally to
    # avoid pulling its top-level side effects when this module is imported
    # for tests.
    from .market_data import compute_indicators

    enriched = compute_indicators(hist.copy())
    if enriched is None or enriched.empty:
        return None
    latest = enriched.iloc[-1].to_dict()

    rs_block = {"rs_1m": None, "rs_55d": None, "rs_3m": None}
    if market_ctx:
        from .market_data import compute_relative_strength
        rs_block = compute_relative_strength(latest, market_ctx)

    regime = (market_ctx or {}).get("regime") or "NEUTRAL"
    tech_score = scoring.compute_technical_score(
        latest,
        selected=["ma20", "ma50", "ma200", "cross", "rsi", "macd", "bb"],
        regime=regime,
    )
    macd = latest.get("MACD")
    macd_sig = latest.get("MACD_Sig")
    macd_bull = (macd > macd_sig) if (macd is not None and macd_sig is not None) else None
    momentum = scoring.compute_momentum_score(
        adx=latest.get("ADX"),
        rs_1m=rs_block.get("rs_1m"),
        rs_55d=rs_block.get("rs_55d"),
        rs_3m=rs_block.get("rs_3m"),
        vol_ratio=latest.get("Vol_Ratio"),
        macd_bull=macd_bull,
    )
    risk = scoring.compute_risk_score(
        trend_ext=latest.get("Trend_Ext"),
        atr_pct=latest.get("ATR_Pct"),
    )
    dip = scoring.compute_dip_score(
        rsi=latest.get("RSI"),
        fund_score=0.0,
        vol_ratio=latest.get("Vol_Ratio"),
        mkt_regime=regime,
        bb_pct=latest.get("BB_Pct"),
    )

    overall, label, css = scoring.compute_overall_score(
        tech_score=tech_score,
        fund_score=0.0,
        sent_score=None,        # sentiment disabled for fast portfolio view
        weights={"technical": 100, "fundamental": 0, "sentiment": 0},
        vol_ratio=latest.get("Vol_Ratio"),
        rs_1m=rs_block.get("rs_1m"),
        rs_55d=rs_block.get("rs_55d"),
        rs_3m=rs_block.get("rs_3m"),
        atr_pct=latest.get("ATR_Pct"),
        vol_pctl=latest.get("Vol_Pctl"),
        spy_trend_bull=(market_ctx or {}).get("spy_trend_bull"),
        trend_stage=latest.get("Trend_Stage"),
        mkt_regime=regime,
        regime_chg=(market_ctx or {}).get("regime_change"),
        momentum_score=momentum,
        risk_score=risk,
        dip_score=dip,
    )

    return {
        "action":         label,
        "css_class":      css,
        "score":          round(overall, 3),
        "tech_score":     round(tech_score, 3),
        "momentum_score": round(momentum, 3),
        "risk_score":     round(risk, 3),
        "rsi":            _to_float(latest.get("RSI")),
        "atr_pct":        _to_float(latest.get("ATR_Pct")),
        "trend_stage":    latest.get("Trend_Stage"),
    }


def _to_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
        if np.isnan(f):
            return None
        return round(f, 2)
    except (TypeError, ValueError):
        return None
