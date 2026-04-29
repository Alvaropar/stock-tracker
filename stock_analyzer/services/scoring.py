"""
Weighted scoring engine for the app.

Produces normalized scores [-1, 1] for technical, fundamental, and sentiment
components, then combines them via user-specified weights into an overall score.

v4: Momentum continuation, risk interaction, volume-aware differentiation,
    granular signal decision tree, stronger regime transition usage.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


# ── Default indicator breakpoints (tunable via ThresholdConfig) ─────────────
#
# Format: sorted [[x_val, score], ...].  _interp() linearly interpolates
# between adjacent points, clamping outside the outer bounds.
# These are also imported by threshold_config.py to define the built-in default.

DEFAULT_RSI_MR_BP: List[List[float]] = [
    [0.0,   1.0],   # deeply oversold
    [20.0,  1.0],   # fully oversold plateau
    [50.0,  0.0],   # midline — neutral
    [80.0, -1.0],   # fully overbought plateau
    [100.0,-1.0],
]

DEFAULT_RSI_TREND_BP: List[List[float]] = [
    [0.0,  -1.0],   # trend broken
    [30.0, -1.0],
    [40.0, -0.25],  # weakening
    [50.0,  0.0],   # midline
    [60.0,  0.5],   # entering momentum zone
    [75.0,  1.0],   # peak bullish
    [85.0,  0.75],  # possible exhaustion (still bullish but easing)
    [100.0, 0.5],
]

DEFAULT_BB_BP: List[List[float]] = [
    [0.0,   1.0],
    [20.0,  0.5],
    [40.0,  0.0],
    [60.0,  0.0],
    [80.0, -0.5],
    [100.0,-1.0],
]

DEFAULT_MA200_DIST_BP: List[List[float]] = [
    [-20.0, -1.0],
    [-3.0,  -0.3],   # soft lower band edge
    [3.0,    0.3],   # soft upper band edge
    [20.0,   1.0],
]

DEFAULT_SCORE_THRESHOLDS: Dict[str, float] = {
    "strong_buy":   0.5,
    "buy":          0.2,
    "sell":        -0.2,
    "strong_sell": -0.5,
}


def _interp(val: float, bp: List) -> float:
    """
    Piecewise-linear interpolation over a breakpoint table.

    *bp* is a list of [x, y] pairs sorted by x ascending.  Values outside
    the outer bounds are clamped to the endpoint score.  This replaces the
    old step-function scoring (e.g. 'if rsi < 30: +1.0 elif rsi < 45: +0.25')
    with a smooth continuous signal that avoids cliff-edge behaviour and makes
    thresholds directly tunable via ThresholdConfig.
    """
    if val <= bp[0][0]:
        return float(bp[0][1])
    if val >= bp[-1][0]:
        return float(bp[-1][1])
    for i in range(len(bp) - 1):
        x0, y0 = float(bp[i][0]),   float(bp[i][1])
        x1, y1 = float(bp[i+1][0]), float(bp[i+1][1])
        if x0 <= val <= x1:
            t = (val - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return 0.0


# ── Signal mapping (base, kept for backward compat) ─────────────────────────

def score_to_signal(
    score: float,
    thresholds: Optional[Dict[str, Any]] = None,
) -> tuple[str, str]:
    """Map an overall score [-1, 1] to (label, css_class).

    *thresholds* is an optional dict with keys 'strong_buy', 'buy', 'sell',
    'strong_sell' — comes from a ThresholdConfig.  Falls back to defaults.
    """
    t     = thresholds or {}
    sbuy  = float(t.get("strong_buy",   DEFAULT_SCORE_THRESHOLDS["strong_buy"]))
    buy   = float(t.get("buy",          DEFAULT_SCORE_THRESHOLDS["buy"]))
    sell  = float(t.get("sell",         DEFAULT_SCORE_THRESHOLDS["sell"]))
    ssell = float(t.get("strong_sell",  DEFAULT_SCORE_THRESHOLDS["strong_sell"]))
    if   score >= sbuy:  return "STRONG BUY",  "sbuy"
    elif score >= buy:   return "BUY",          "buy"
    elif score >= sell:  return "NEUTRAL",       "neu"
    elif score >= ssell: return "SELL",          "sell"
    else:                return "STRONG SELL",   "ssell"


# ── Regime-aware weight profiles ────────────────────────────────────────────

_REGIME_WEIGHTS: Dict[str, Dict[str, float]] = {
    "TREND": {
        "ma20": 1.0, "ma50": 1.5, "ma200": 2.0, "cross": 1.5,
        "rsi": 0.5, "macd": 2.0, "bb": 0.5,
    },
    "MEAN_REVERSION": {
        "ma20": 0.5, "ma50": 0.5, "ma200": 0.5, "cross": 0.5,
        "rsi": 2.0, "macd": 0.5, "bb": 2.0,
    },
    "NEUTRAL": {
        "ma20": 0.5, "ma50": 1.0, "ma200": 1.5, "cross": 1.0,
        "rsi": 2.0, "macd": 1.5, "bb": 1.0,
    },
}


# ── Technical scoring ─────────────────────────────────────────────────────────

def compute_technical_score(
    latest: Dict,
    selected: List[str],
    regime: str = "NEUTRAL",
    thresholds: Optional[Dict[str, Any]] = None,
    ic_weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute a normalized technical score in [-1, 1].
    Only indicators in *selected* contribute.
    Weights are conditioned on the detected market *regime*.

    *thresholds* is an optional ThresholdConfig dict.  When supplied, the
    continuous breakpoints for RSI, BB, and MA200 are taken from it instead
    of the module-level defaults, enabling user-tunable threshold validation.

    *ic_weights* is an optional dict mapping indicator keys (same names as in
    *selected*: 'rsi', 'macd', 'bb', 'ma20', 'ma50', 'ma200', 'cross') to
    non-negative IC-derived multipliers.  When supplied the regime base weight
    is multiplied by the IC multiplier, making data-driven weighting possible.
    IC multipliers are typically computed by _compute_per_indicator_ic() in the
    backtest API and saved via the threshold config store.
    """
    profile = _REGIME_WEIGHTS.get(regime, _REGIME_WEIGHTS["NEUTRAL"])

    def _w(key: str, base: float) -> float:
        """Apply IC multiplier if provided; fall back to base regime weight."""
        if ic_weights:
            mult = float(ic_weights.get(key, 1.0))
            return base * max(0.0, mult)
        return base

    score = 0.0
    max_score = 0.0
    c = latest.get("Close") or 0

    if "ma20" in selected:
        w = _w("ma20", profile["ma20"])
        ma = latest.get("MA20")
        if ma and w > 0:
            score += w if c > ma else -w
            max_score += w

    if "ma50" in selected:
        w = _w("ma50", profile["ma50"])
        ma = latest.get("MA50")
        if ma and w > 0:
            score += w if c > ma else -w
            max_score += w

    if "ma200" in selected:
        w = _w("ma200", profile["ma200"])
        ma = latest.get("MA200")
        if ma and ma > 0 and w > 0:
            dist_pct = (c - ma) / ma * 100
            _bp = (thresholds or {}).get("ma200_dist_bp", DEFAULT_MA200_DIST_BP)
            score += _interp(dist_pct, _bp) * w
            max_score += w

    if "cross" in selected:
        w = _w("cross", profile["cross"])
        ma50  = latest.get("MA50")
        ma200 = latest.get("MA200")
        if ma50 and ma200 and w > 0:
            score += w if ma50 > ma200 else -w
            max_score += w

    if "rsi" in selected:
        w = _w("rsi", profile["rsi"])
        rsi = latest.get("RSI")
        if rsi is not None and w > 0:
            if regime == "TREND":
                _bp = (thresholds or {}).get("rsi_trend_bp", DEFAULT_RSI_TREND_BP)
            else:
                _bp = (thresholds or {}).get("rsi_mr_bp", DEFAULT_RSI_MR_BP)
            score += _interp(rsi, _bp) * w
            max_score += w

    if "macd" in selected:
        w = _w("macd", profile["macd"])
        macd  = latest.get("MACD")
        msig  = latest.get("MACD_Sig")
        mhist = latest.get("MACD_Hist")
        if w > 0:
            w_line = w * 0.67
            w_hist = w * 0.33
            if macd is not None and msig is not None:
                score += w_line if macd > msig else -w_line
                max_score += w_line
            if mhist is not None:
                score += w_hist if mhist > 0 else -w_hist
                max_score += w_hist

    if "bb" in selected:
        w = _w("bb", profile["bb"])
        bb = latest.get("BB_Pct")
        if bb is not None and w > 0:
            _bp = (thresholds or {}).get("bb_bp", DEFAULT_BB_BP)
            score += _interp(bb, _bp) * w
            max_score += w

    return round(score / max_score, 4) if max_score else 0.0


# ── Sector normalization ──────────────────────────────────────────────────────

_SECTOR_MAP: Dict[str, str] = {
    "technology": "TECH", "information technology": "TECH",
    "software": "TECH", "semiconductors": "TECH", "hardware": "TECH",
    "financial services": "FINANCIALS", "financials": "FINANCIALS",
    "banking": "FINANCIALS", "banks": "FINANCIALS", "insurance": "FINANCIALS",
    "healthcare": "HEALTHCARE", "health care": "HEALTHCARE",
    "biotechnology": "HEALTHCARE", "pharmaceuticals": "HEALTHCARE",
    "consumer cyclical": "CONSUMER_DISC", "consumer discretionary": "CONSUMER_DISC",
    "consumer defensive": "CONSUMER_STAPLES", "consumer staples": "CONSUMER_STAPLES",
    "energy": "ENERGY",
    "utilities": "UTILITIES",
    "real estate": "REAL_ESTATE",
    "basic materials": "MATERIALS", "materials": "MATERIALS",
    "industrials": "INDUSTRIALS",
    "communication services": "COMM_SERVICES",
    "telecommunications": "COMM_SERVICES",
}


def _normalize_sector(sector: str) -> str:
    """Map a raw yfinance sector string to a canonical key, or '' if unknown."""
    return _SECTOR_MAP.get((sector or "").lower().strip(), "")


# ── Fundamental scoring ───────────────────────────────────────────────────────

def compute_fundamental_score(
    fund: Dict,
    selected: List[str],
    sector: str = "",
) -> float:
    """
    Compute a normalized fundamental score in [-1, 1].
    Each selected indicator contributes equally.

    *sector* adjusts thresholds for PE, D/E, and current ratio so that
    structurally high-PE sectors (Tech, Healthcare) or high-leverage sectors
    (Financials, Utilities) are scored against peer-appropriate benchmarks
    rather than universal absolute cutoffs.
    """
    _sk = _normalize_sector(sector or fund.get("sector", ""))
    signals: List[float] = []

    if "pe" in selected:
        pe = fund.get("pe_trail")
        if pe and pe > 0:
            if _sk == "TECH":
                # Growth premium accepted; penalise only at extreme multiples
                if   pe < 15:  signals.append( 1.0)
                elif pe < 25:  signals.append( 0.5)
                elif pe < 40:  signals.append( 0.0)
                elif pe < 70:  signals.append(-0.5)
                else:          signals.append(-1.0)
            elif _sk == "FINANCIALS":
                # Banks/insurers trade at structurally lower multiples
                if   pe <  8:  signals.append( 1.0)
                elif pe < 13:  signals.append( 0.5)
                elif pe < 18:  signals.append( 0.0)
                elif pe < 28:  signals.append(-0.5)
                else:          signals.append(-1.0)
            elif _sk == "UTILITIES":
                if   pe < 12:  signals.append( 1.0)
                elif pe < 18:  signals.append( 0.5)
                elif pe < 28:  signals.append( 0.0)
                elif pe < 40:  signals.append(-0.5)
                else:          signals.append(-1.0)
            elif _sk == "HEALTHCARE":
                # R&D pipelines justify higher multiples
                if   pe < 15:  signals.append( 1.0)
                elif pe < 28:  signals.append( 0.5)
                elif pe < 45:  signals.append( 0.0)
                elif pe < 70:  signals.append(-0.5)
                else:          signals.append(-1.0)
            else:
                # Default: Industrials, Consumer, Energy, Materials, etc.
                if   pe < 10:  signals.append( 1.0)
                elif pe < 20:  signals.append( 0.5)
                elif pe < 30:  signals.append( 0.0)
                elif pe < 50:  signals.append(-0.5)
                else:          signals.append(-1.0)

    if "peg" in selected:
        peg = fund.get("peg")
        if peg and peg > 0:
            if   peg < 0.5: signals.append( 1.0)
            elif peg < 1.0: signals.append( 0.5)
            elif peg < 2.0: signals.append( 0.0)
            elif peg < 3.0: signals.append(-0.5)
            else:           signals.append(-1.0)

    if "pb" in selected:
        pb = fund.get("pb")
        if pb and pb > 0:
            if   pb < 1.0:  signals.append( 1.0)
            elif pb < 2.5:  signals.append( 0.5)
            elif pb < 5.0:  signals.append( 0.0)
            elif pb < 10.0: signals.append(-0.5)
            else:           signals.append(-1.0)

    if "margins" in selected:
        nm = fund.get("net_mgn")
        if nm is not None:
            if   nm >= 25:  signals.append( 1.0)
            elif nm >= 10:  signals.append( 0.5)
            elif nm >=  0:  signals.append( 0.0)
            elif nm >= -10: signals.append(-0.5)
            else:           signals.append(-1.0)

    if "roe" in selected:
        roe = fund.get("roe")
        if roe is not None:
            if   roe >= 25: signals.append( 1.0)
            elif roe >= 15: signals.append( 0.5)
            elif roe >=  5: signals.append( 0.0)
            elif roe >=  0: signals.append(-0.25)
            else:           signals.append(-1.0)

    if "growth" in selected:
        rg = fund.get("rev_growth")
        if rg is not None:
            if   rg >= 20: signals.append( 1.0)
            elif rg >= 10: signals.append( 0.5)
            elif rg >=  0: signals.append( 0.0)
            elif rg >= -5: signals.append(-0.5)
            else:          signals.append(-1.0)

    if "de" in selected:
        de = fund.get("debt_eq")
        if de is not None:
            actual = de / 100  # yfinance stores ×100
            if _sk == "FINANCIALS":
                # Banks routinely operate at D/E 5-15; not a red flag
                if   actual <  5: signals.append( 1.0)
                elif actual < 10: signals.append( 0.5)
                elif actual < 15: signals.append( 0.0)
                elif actual < 20: signals.append(-0.5)
                else:             signals.append(-1.0)
            elif _sk == "UTILITIES":
                # Infrastructure debt is structural, not distress
                if   actual < 0.8: signals.append( 1.0)
                elif actual < 1.5: signals.append( 0.5)
                elif actual < 3.0: signals.append( 0.0)
                elif actual < 5.0: signals.append(-0.5)
                else:              signals.append(-1.0)
            elif _sk == "REAL_ESTATE":
                if   actual < 0.5: signals.append( 1.0)
                elif actual < 1.0: signals.append( 0.5)
                elif actual < 2.0: signals.append( 0.0)
                elif actual < 4.0: signals.append(-0.5)
                else:              signals.append(-1.0)
            else:
                # Default: Tech, Healthcare, Industrials, Consumer, etc.
                if   actual < 0.3: signals.append( 1.0)
                elif actual < 0.7: signals.append( 0.5)
                elif actual < 1.5: signals.append( 0.0)
                elif actual < 3.0: signals.append(-0.5)
                else:              signals.append(-1.0)

    if "cr" in selected:
        # Current ratio is not meaningful for banks/insurers — their liabilities
        # (deposits) are structurally > current assets by design.
        if _sk != "FINANCIALS":
            cr = fund.get("curr_ratio")
            if cr is not None:
                if   cr >= 2.5: signals.append( 1.0)
                elif cr >= 1.5: signals.append( 0.5)
                elif cr >= 1.0: signals.append( 0.0)
                else:           signals.append(-1.0)

    if "analyst" in selected:
        rec = fund.get("rec_mean")
        if rec is not None:
            s = max(-1.0, min(1.0, (3 - rec) / 2))
            signals.append(s)

    return round(sum(signals) / len(signals), 4) if signals else 0.0


# ── NEW v4: Momentum score ───────────────────────────────────────────────────

def compute_momentum_score(
    adx: Optional[float] = None,
    rs_1m: Optional[float] = None,
    rs_55d: Optional[float] = None,
    rs_3m: Optional[float] = None,
    vol_ratio: Optional[float] = None,
    macd_bull: Optional[bool] = None,
) -> float:
    """
    Composite momentum score normalized to [0, 1].

    Components (each contributes 0-1 on its own scale, then averaged):
      - ADX strength:     0/25/40+ → 0.0/0.5/1.0  (trend conviction)
      - RS composite:     ≤0 / 5-20 / 20-50 / 50+% → 0.0 / 0.33 / 0.67 / 1.0
                          Weighted 30% 1M / 40% 55D / 30% 3M (IBD-inspired)
      - Volume ratio:     <0.8 / 0.8-1.2 / 1.2-2 / 2+ → 0.0 / 0.25 / 0.6 / 1.0
      - MACD confirm:     bearish → 0.0 / bullish → 1.0

    Use cases:
      - High momentum (≥ 0.65): trend continuation even in EXTENDED stage
      - Low momentum (< 0.30):  weak confirmation → downgrade signals
    """
    parts: List[float] = []

    # ADX — trend conviction
    if adx is not None:
        if   adx >= 40: parts.append(1.0)
        elif adx >= 30: parts.append(0.75)
        elif adx >= 25: parts.append(0.50)
        elif adx >= 20: parts.append(0.25)
        else:           parts.append(0.0)

    # Relative strength — composite across available periods (IBD-inspired)
    rs_comps: list = []
    if rs_1m  is not None: rs_comps.append((rs_1m,  0.30))
    if rs_55d is not None: rs_comps.append((rs_55d, 0.40))
    if rs_3m  is not None: rs_comps.append((rs_3m,  0.30))
    if rs_comps:
        total_w = sum(w for _, w in rs_comps)
        rs_c = sum(v * w for v, w in rs_comps) / total_w
        if   rs_c >= 50: parts.append(1.0)
        elif rs_c >= 20: parts.append(0.67)
        elif rs_c >=  5: parts.append(0.33)
        elif rs_c >=  0: parts.append(0.15)
        else:            parts.append(0.0)

    # Volume confirmation
    if vol_ratio is not None:
        if   vol_ratio >= 2.0: parts.append(1.0)
        elif vol_ratio >= 1.5: parts.append(0.75)
        elif vol_ratio >= 1.2: parts.append(0.5)
        elif vol_ratio >= 0.8: parts.append(0.25)
        else:                  parts.append(0.0)

    # MACD directional confirmation
    if macd_bull is not None:
        parts.append(1.0 if macd_bull else 0.0)

    if not parts:
        return 0.0
    return round(sum(parts) / len(parts), 4)


# ── NEW v4: Risk interaction score ───────────────────────────────────────────

def compute_risk_score(
    trend_ext: Optional[float] = None,
    atr_pct: Optional[float] = None,
) -> float:
    """
    Composite risk score: RISK = |TREND_EXTENSION| × ATR%.

    Interpretation:
      - < 0.15: low risk
      - 0.15–0.50: moderate risk
      - 0.50–1.00: high risk
      - > 1.00: extreme risk (parabolic + volatile = danger zone)

    Used to:
      - Penalize overall score when risk is high
      - Downgrade SIGNAL from BUY → HOLD or HOLD → AVOID
      - Reduce adjusted confidence
    """
    if trend_ext is None or atr_pct is None:
        return 0.0
    return round(abs(trend_ext) * atr_pct, 4)


# ── NEW v4.1: Oversold dip detection ─────────────────────────────────────────

def compute_dip_score(
    rsi: Optional[float] = None,
    fund_score: float = 0.0,
    vol_ratio: Optional[float] = None,
    mkt_regime: Optional[str] = None,
    regime_chg: Optional[str] = None,
    bb_pct: Optional[float] = None,
    target_gap: Optional[float] = None,
    n_analysts: Optional[int] = None,
) -> float:
    """
    Detect "buy the dip" quality.  Normalized to [0, 1].

    A high dip score means the stock is oversold on technicals but has strong
    fundamentals — the classic mean-reversion buy opportunity.  Volume spikes
    during the dip are GOOD (capitulation = sellers exhausted), unlike for
    trend-following where low volume is penalized.

    Components:
      RSI oversold:  < 20 → 1.0, < 25 → 0.8, < 30 → 0.6, < 35 → 0.35,
                     < 40 → 0.15, < 45 → 0.05  (early/developing dip)
      Fund quality:  > 0.5 → 1.0, > 0.3 → 0.7, > 0.1 → 0.4, else 0.0
      Volume (capitulation): VR > 2.0 → 1.0, > 1.5 → 0.7, > 1.0 → 0.4, else 0.2
      Bollinger:     < 5 → 1.0, < 15 → 0.6, < 25 → 0.3, else 0.0
      Target gap:    stock > 40% below analyst target → 0.9, > 25% → 0.6, > 15% → 0.3
                     (amplifies dip quality when professionals see large upside)

    Anti-falling-knife filters:
      BEARISH REVERSAL → 0 (too early, knife still falling)
      BEARISH market   → ×0.40 (dips in bear markets are riskier)
      TRANSITION       → ×0.70

    Gate lowered from RSI < 40 to RSI < 45 to catch early-stage dips.
    """
    if rsi is None or rsi >= 45:
        return 0.0

    parts: List[float] = []

    # RSI oversold depth
    if   rsi < 20: parts.append(1.0)
    elif rsi < 25: parts.append(0.8)
    elif rsi < 30: parts.append(0.6)
    elif rsi < 35: parts.append(0.35)
    elif rsi < 40: parts.append(0.15)   # mildly oversold
    else:          parts.append(0.05)   # 40-45: early developing dip

    # Fundamental quality (the key differentiator vs value traps)
    if   fund_score >= 0.50: parts.append(1.0)
    elif fund_score >= 0.30: parts.append(0.7)
    elif fund_score >= 0.10: parts.append(0.4)
    else:                    parts.append(0.0)

    # Volume during dip (high vol = capitulation = bullish for dips)
    if vol_ratio is not None:
        if   vol_ratio >= 2.0: parts.append(1.0)
        elif vol_ratio >= 1.5: parts.append(0.7)
        elif vol_ratio >= 1.0: parts.append(0.4)
        else:                  parts.append(0.2)

    # Bollinger Band position (deeply below lower band = stronger dip signal)
    if bb_pct is not None:
        if   bb_pct < 5:  parts.append(1.0)
        elif bb_pct < 15: parts.append(0.6)
        elif bb_pct < 25: parts.append(0.3)
        else:             parts.append(0.0)

    # Analyst target gap — amplifies quality when professionals see large upside
    # target_gap = (target_px − price) / price × 100; positive = stock below target
    # Guard: require ≥ 3 analyst opinions.  Pre-revenue biotech / speculative
    # names often have 1-2 DCF-model targets 200-500% above price; without this
    # guard those would fire massive dip scores even on genuinely broken charts.
    if (target_gap is not None and target_gap > 0
            and (n_analysts is None or n_analysts >= 3)):
        if   target_gap >= 40: parts.append(0.9)
        elif target_gap >= 25: parts.append(0.6)
        elif target_gap >= 15: parts.append(0.3)

    raw = sum(parts) / len(parts) if parts else 0.0

    # Anti-falling-knife filters
    if regime_chg == "BEARISH REVERSAL":
        return 0.0  # knife still falling — don't catch
    if mkt_regime == "BEARISH":
        raw *= 0.40
    elif mkt_regime == "TRANSITION":
        raw *= 0.70

    return round(max(0.0, min(1.0, raw)), 4)


def _dip_adjustment(dip_score: float, score: float, fund_score: float) -> float:
    """
    Boost overall score for quality dip candidates.

    During a dip, the negative tech score drags the overall down.  This
    additive boost compensates, scaling with dip quality and fundamental
    strength.  Only applies when the base score is neutral-ish or below
    (won't boost an already-strong BUY).

    Returns additive adjustment (0 to +0.25).
    """
    if dip_score < 0.35 or score >= 0.35:
        return 0.0

    # Scale boost by dip quality × fundamental quality
    # dip=0.35 → +0.05,  dip=0.60 → +0.12,  dip=0.80 → +0.18,  dip=1.0 → +0.25
    max_boost = 0.25
    boost = dip_score * max_boost

    # Stronger fundamentals → larger share of the max boost
    fund_mult = max(0.3, min(1.0, fund_score * 1.5))
    boost *= fund_mult

    return round(boost, 4)


# ── Confirmation & penalty adjustments ───────────────────────────────────────

def _volume_adjustment(
    vol_ratio: Optional[float],
    score: float = 0.0,
) -> float:
    """
    Volume confirmation boost/penalty.  Returns additive value.

    v4: Stronger penalties for low volume on bullish scores, and stronger
    boosts for high volume confirming the trend direction.
    """
    if vol_ratio is None:
        return 0.0

    # ── High volume — confirms directional move ───────────────────────────
    if vol_ratio >= 2.5:  return  0.12
    if vol_ratio >= 2.0:  return  0.10
    if vol_ratio >= 1.5:  return  0.07
    if vol_ratio >= 1.2:  return  0.03

    # ── Low volume — weakens conviction ───────────────────────────────────
    # Asymmetric: penalize bullish scores more harshly on low volume
    if vol_ratio <= 0.4:
        return -0.12 if score > 0 else -0.06
    if vol_ratio <= 0.6:
        return -0.08 if score > 0 else -0.04
    if vol_ratio <= 0.8:
        return -0.04 if score > 0 else -0.02

    return 0.0


def _relative_strength_adjustment(
    rs_1m:  Optional[float] = None,
    rs_55d: Optional[float] = None,
    rs_3m:  Optional[float] = None,
) -> float:
    """
    Multi-period RS vs SPY adjustment.  Returns additive [-0.10, +0.10].

    Uses a weighted composite of all available RS periods so that stocks
    consolidating after a strong run (positive 55D/3M RS) are not unfairly
    penalised by a flat 1-month window, and one-week spikes don't dominate:
      30 % RS 1M  — short-term momentum
      40 % RS 55D — intermediate (IBD-inspired, highest weight)
      30 % RS 3M  — trend confirmation
    Falls back gracefully when fewer periods are available.
    """
    components: list = []
    if rs_1m  is not None: components.append((rs_1m,  0.30))
    if rs_55d is not None: components.append((rs_55d, 0.40))
    if rs_3m  is not None: components.append((rs_3m,  0.30))
    if not components:
        return 0.0
    total_w = sum(w for _, w in components)
    rs_c = sum(v * w for v, w in components) / total_w
    if rs_c >= 10:   return  0.10
    if rs_c >=  5:   return  0.06
    if rs_c >=  2:   return  0.03
    if rs_c <= -10:  return -0.10
    if rs_c <=  -5:  return -0.06
    if rs_c <=  -2:  return -0.03
    return 0.0


def _volatility_penalty(atr_pct: Optional[float], vol_pctl: Optional[float]) -> float:
    """
    Penalize extreme volatility.
    Returns a multiplier in [0.65, 1.0] — applied to the overall score.
    """
    if atr_pct is None:
        return 1.0
    if vol_pctl is not None and vol_pctl > 90:
        return 0.70
    if vol_pctl is not None and vol_pctl > 80:
        return 0.85
    if atr_pct >= 15.0:  return 0.65   # extreme
    if atr_pct >=  8.0:  return 0.75   # high
    if atr_pct >=  6.0:  return 0.85
    if atr_pct >=  4.0:  return 0.92
    return 1.0


def _market_filter(spy_trend_bull: Optional[bool], score: float) -> float:
    """If SPY below MA200, dampen BUY signals by 20%."""
    if spy_trend_bull is False and score > 0:
        return score * 0.80
    return score


def _analyst_adjustment(
    price:      Optional[float] = None,
    target_px:  Optional[float] = None,
    rec_mean:   Optional[float] = None,
    n_analysts: Optional[int]   = None,
) -> float:
    """
    Analyst consensus adjustment.  Returns additive [-0.10, +0.10].

    Two components:

    1. Target-price gap = (consensus_target − price) / price × 100
         ≥ +40 %:  +0.08   large institutional upside = quality discount
         ≥ +25 %:  +0.05
         ≥ +15 %:  +0.02
         ≤ −25 %:  −0.06   stock trading above consensus = valuation caution
         ≤ −15 %:  −0.03

    2. Recommendation mean  (1 = Strong Buy … 5 = Strong Sell)
         < 1.5:   +0.04
         < 2.0:   +0.02
         > 3.5:   −0.02
         > 4.0:   −0.04

    Requires ≥ 3 analyst opinions; returns 0 for thinly-covered names.
    Combined result is capped at [-0.10, +0.10].
    """
    if n_analysts is not None and n_analysts < 3:
        return 0.0
    adj = 0.0
    if price and price > 0 and target_px and target_px > 0:
        gap = (target_px - price) / price * 100
        if   gap >= 40:  adj += 0.08
        elif gap >= 25:  adj += 0.05
        elif gap >= 15:  adj += 0.02
        elif gap <= -25: adj -= 0.06
        elif gap <= -15: adj -= 0.03
    if rec_mean is not None:
        if   rec_mean < 1.5: adj += 0.04
        elif rec_mean < 2.0: adj += 0.02
        elif rec_mean > 4.0: adj -= 0.04
        elif rec_mean > 3.5: adj -= 0.02
    return round(max(-0.10, min(0.10, adj)), 4)


# ── Trend maturity penalty (v4: momentum-aware) ─────────────────────────────

_TREND_STAGE_MULT: Dict[str, float] = {
    "EARLY":         1.00,   # no penalty — fresh trend
    "HEALTHY":       1.00,   # still fine
    "EXTENDED":      0.88,   # moderate penalty (may be overridden by momentum)
    "OVEREXTENDED":  0.75,   # significant penalty
    "PARABOLIC":     0.60,   # strong penalty — likely blow-off top
}


def _trend_maturity_penalty(
    trend_stage: Optional[str],
    score: float,
    momentum_score: float = 0.0,
) -> float:
    """
    Penalize bullish scores in late-stage / parabolic trends.
    Only applies when score is positive (penalizes buying, not selling).

    v4: High momentum (≥ 0.65) softens the penalty for EXTENDED stage,
    allowing momentum continuation trades.  OVEREXTENDED/PARABOLIC are
    NOT overridden — those are genuine risk zones.
    """
    if trend_stage is None or score <= 0:
        return score
    mult = _TREND_STAGE_MULT.get(trend_stage, 1.0)

    # Momentum continuation: soften EXTENDED penalty if momentum is strong
    if trend_stage == "EXTENDED" and momentum_score >= 0.65:
        # Interpolate: at momentum=0.65 → mult=0.88, at momentum=1.0 → mult=0.97
        boost = min(0.09, (momentum_score - 0.65) * 0.257)
        mult = min(1.0, mult + boost)

    return score * mult


# ── NEW v4: Risk interaction penalty ─────────────────────────────────────────

def _risk_penalty(risk_score: float, score: float) -> float:
    """
    Apply risk interaction penalty (extension × volatility).
    Only penalizes bullish scores — high risk should not make sells weaker.

    Thresholds (v4 recalibrated):
      risk < 0.30:  no penalty  (low extension + low vol)
      risk 0.30-1.0: ×0.94     (moderate — slightly caution)
      risk 1.0-2.0:  ×0.85     (high — meaningful penalty)
      risk 2.0-5.0:  ×0.72     (very high)
      risk > 5.0:    ×0.55     (extreme — parabolic + volatile)
    """
    if score <= 0 or risk_score < 0.30:
        return score
    if risk_score >= 5.0:
        return score * 0.55
    if risk_score >= 2.0:
        return score * 0.72
    if risk_score >= 1.0:
        return score * 0.85
    return score * 0.94


# ── Market regime transition adjustment (v4: stronger) ──────────────────────

def _regime_transition_adjustment(
    mkt_regime: Optional[str],
    regime_chg: Optional[str],
    score: float,
) -> float:
    """
    Adjust score based on market regime and transitions.

    v4: Stronger multipliers for transitions.  BEARISH REVERSAL now also
    penalizes scores in the neutral zone (not just positive), and
    BULLISH REVERSAL gives a bigger boost.
    """
    if regime_chg and regime_chg != "NONE":
        # ── Active transitions (take priority over steady-state) ──────────
        if regime_chg == "BEARISH REVERSAL":
            if score > 0:
                return score * 0.40         # savage cut to bullish conviction
            return score * 1.10             # boost bearish conviction
        if regime_chg == "BEARISH CONFIRMATION":
            if score > 0:
                return score * 0.55
            return score * 1.05
        if regime_chg == "BULLISH REVERSAL":
            if score > 0:
                return score * 1.20         # strong early recovery boost
            return score * 0.80             # reduce bearish conviction
        if regime_chg == "BULLISH CONFIRMATION":
            if score > 0:
                return score * 1.12
            return score
        if regime_chg == "WEAKENING":
            if score > 0:
                return score * 0.75         # was 0.85, now stronger
            return score
        if regime_chg == "POTENTIAL BOTTOM":
            if score < 0:
                return score * 0.70         # strongly reduce bearish conviction
            return score * 1.08             # slight boost to bullish

    # ── Steady-state regime filter (no transition) ─────────────────────────
    if mkt_regime == "BEARISH" and score > 0:
        return score * 0.70                 # was 0.75
    if mkt_regime == "TRANSITION":
        return score * 0.88                 # was 0.90

    return score


# ── Volatility-aware confidence adjustment (v4: includes risk_score) ─────────

def compute_confidence_adjustment(
    base_confidence: Optional[float],
    vol_regime: Optional[str],
    trend_stage: Optional[str],
    mkt_regime: Optional[str],
    risk_score: float = 0.0,
    momentum_score: float = 0.0,
    dip_score: float = 0.0,
) -> float:
    """
    Adjust the buying-checklist confidence % based on volatility regime,
    trend maturity, market regime, risk interaction, momentum, and dip quality.
    Returns adjusted confidence (0-100).

    v4.1: Added dip_score boost — quality oversold conditions partially
    compensate for the mechanical confidence drags.
    """
    if base_confidence is None:
        return 0.0
    conf = base_confidence

    # Volatility drag
    vol_drag = {"LOW": 1.0, "NORMAL": 1.0, "HIGH": 0.85, "EXTREME": 0.65}
    conf *= vol_drag.get(vol_regime or "NORMAL", 1.0)

    # Trend maturity drag
    stage_drag = {"EARLY": 1.0, "HEALTHY": 1.0, "EXTENDED": 0.90,
                  "OVEREXTENDED": 0.75, "PARABOLIC": 0.55}
    conf *= stage_drag.get(trend_stage or "EARLY", 1.0)

    # Market regime drag
    mkt_drag = {"BULLISH": 1.0, "TRANSITION": 0.85, "BEARISH": 0.65}
    conf *= mkt_drag.get(mkt_regime or "BULLISH", 1.0)

    # v4: Risk interaction drag (extension × volatility compound risk)
    if risk_score >= 5.0:
        conf *= 0.45
    elif risk_score >= 2.0:
        conf *= 0.60
    elif risk_score >= 1.0:
        conf *= 0.75
    elif risk_score >= 0.30:
        conf *= 0.90

    # v4: Momentum boost — high momentum can partially recover confidence
    if momentum_score >= 0.75:
        conf *= 1.10  # cap at 100 later
    elif momentum_score >= 0.50:
        conf *= 1.04

    # v4.1: Dip quality boost — oversold fundamentally strong stocks
    # deserve higher confidence even though mechanical checklist scores low
    if dip_score >= 0.65:
        conf *= 1.20
    elif dip_score >= 0.45:
        conf *= 1.12
    elif dip_score >= 0.35:
        conf *= 1.06

    return round(max(0.0, min(100.0, conf)), 1)


# ── Context-aware signal labels (v4: full decision tree) ─────────────────────

def contextual_signal(
    base_label: str,
    base_css: str,
    score: float,
    *,
    regime: Optional[str] = None,
    trend_stage: Optional[str] = None,
    mkt_regime: Optional[str] = None,
    regime_chg: Optional[str] = None,
    rsi: Optional[float] = None,
    vol_regime: Optional[str] = None,
    adx: Optional[float] = None,
    vol_ratio: Optional[float] = None,
    rs_1m: Optional[float] = None,
    rs_55d: Optional[float] = None,
    rs_3m: Optional[float] = None,
    atr_pct: Optional[float] = None,
    momentum_score: float = 0.0,
    risk_score: float = 0.0,
    dip_score: float = 0.0,
    fund_score: float = 0.0,
    target_gap: Optional[float] = None,
) -> Tuple[str, str, str]:
    """
    v4.1 decision tree for context-enriched signal labels.

    Evaluation order (first match wins):
      1.  Extreme risk → AVOID
      2.  Regime transitions → transition-specific labels
      2b. Oversold dip (quality) → BUY (OVERSOLD DIP)        ← NEW
      3.  Momentum continuation → BUY override for EXTENDED
      4.  Trend maturity → HOLD subtypes
      5.  Weak confirmation → HOLD (WEAK MOMENTUM)  [dips exempt]
      6.  Default → base label (possibly with trend/reversion hints)

    Returns (enriched_label, base_css_class, context_hint).
    """
    hint = ""
    label = base_label
    css = base_css

    _adx  = adx or 0
    _rs   = rs_1m or 0
    _vr   = vol_ratio or 1.0
    _atr  = atr_pct or 0
    _tgap = target_gap or 0

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 1: EXTREME RISK — AVOID (overrides everything)
    #  Only PARABOLIC triggers AVOID.  For other stages, the risk penalty
    #  in compute_overall_score already dampens the score; the decision
    #  tree below picks the appropriate HOLD/BUY label.
    # ══════════════════════════════════════════════════════════════════════
    if trend_stage == "PARABOLIC":
        if _atr >= 12 or _vr < 1.0:
            label = "AVOID"
            css = "ssell"
            hint = "PARABOLIC / HIGH RISK"
            return f"{label} ({hint})", css, hint
        # Parabolic but vol is OK — still dangerous
        if risk_score >= 2.0:
            label = "AVOID"
            css = "ssell"
            hint = "PARABOLIC / EXTREME RISK"
            return f"{label} ({hint})", css, hint

    # Non-parabolic extreme risk (e.g. OVEREXTENDED + extremely volatile)
    if risk_score >= 3.0 and score >= -0.2:
        label = "AVOID"
        css = "sell"
        hint = "EXTREME RISK"
        return f"{label} ({hint})", css, hint

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 2: REGIME TRANSITIONS (second priority)
    # ══════════════════════════════════════════════════════════════════════
    if regime_chg and regime_chg != "NONE":

        if regime_chg == "BEARISH REVERSAL":
            if score >= 0.2:
                # Was BUY but market just flipped bearish → downgrade to HOLD
                label = "HOLD"
                css = "neu"
                hint = "BEARISH REVERSAL"
                return f"{label} ({hint})", css, hint
            if score >= -0.2:
                label = "SELL"
                css = "sell"
                hint = "BEARISH REVERSAL"
                return f"{label} ({hint})", css, hint
            # Already SELL/STRONG SELL — boost the label
            hint = "TREND REVERSAL"
            return f"{label} ({hint})", css, hint

        if regime_chg == "BEARISH CONFIRMATION":
            if score > 0:
                label = "HOLD"
                css = "neu"
                hint = "BEAR CONFIRMED"
                return f"{label} ({hint})", css, hint
            hint = "BEAR CONFIRMED"
            return f"{label} ({hint})", css, hint

        if regime_chg == "BULLISH REVERSAL":
            if score >= 0.2:
                hint = "EARLY REVERSAL"
                return f"{label} ({hint})", css, hint
            if score >= -0.2:
                # Neutral score but bullish reversal → upgrade to BUY
                label = "BUY"
                css = "buy"
                hint = "EARLY REVERSAL"
                return f"{label} ({hint})", css, hint

        if regime_chg == "WEAKENING":
            if score >= 0.2:
                label = "HOLD"
                css = "neu"
                hint = "WEAKENING TREND"
                return f"{label} ({hint})", css, hint

        if regime_chg == "POTENTIAL BOTTOM":
            if score >= -0.2 and regime == "MEAN_REVERSION" and rsi is not None and rsi < 35:
                label = "BUY"
                css = "buy"
                hint = "POTENTIAL BOTTOM"
                return f"{label} ({hint})", css, hint

        if regime_chg == "BULLISH CONFIRMATION":
            if score >= 0.2:
                hint = "BULL CONFIRMED"
                return f"{label} ({hint})", css, hint

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 2b: OVERSOLD DIP — BUY THE DIP (quality oversold + strong fund)
    #
    #  This must come AFTER regime transitions (we don't buy dips during
    #  BEARISH REVERSAL) but BEFORE the weak-momentum filter (which would
    #  incorrectly suppress dip signals since momentum is low by definition
    #  during a dip).
    # ══════════════════════════════════════════════════════════════════════
    _rsi = rsi or 50

    if dip_score >= 0.55 and fund_score >= 0.30:
        # Strong dip candidate — override to BUY
        if _rsi < 25 and fund_score >= 0.45:
            label = "BUY"
            css = "sbuy" if score >= 0.3 else "buy"
            hint = "OVERSOLD DIP"
            return f"{label} ({hint})", css, hint
        if _rsi < 30:
            label = "BUY"
            css = "buy"
            hint = "OVERSOLD DIP"
            return f"{label} ({hint})", css, hint
        if _rsi < 35:
            label = "BUY"
            css = "buy"
            hint = "OVERSOLD – EARLY DIP"
            return f"{label} ({hint})", css, hint

    if dip_score >= 0.35 and fund_score >= 0.20 and _rsi < 30:
        # Moderate dip candidate — still a buy but lower conviction
        label = "BUY"
        css = "buy"
        hint = "MEAN REVERSION DIP"
        return f"{label} ({hint})", css, hint

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 2c: ABOVE ANALYST CONSENSUS TARGET
    #
    #  Stocks > 25% above the professional consensus price target are pricing
    #  in more than analysts think is justified.  This is a valuation caution
    #  flag — not a fundamental short, but a "wait for the price to earn its
    #  valuation" signal.  Catches AXTI/AEHR-type situations where the stock
    #  has already run past what the fundamentals support.
    # ══════════════════════════════════════════════════════════════════════
    if _tgap < -25 and score >= -0.3 and momentum_score < 0.55:
        # Guard: strong momentum (≥ 0.55) exempts breakout stocks whose
        # analysts simply haven't updated stale price targets yet.  Without
        # this guard a legitimate breakout (NVDA post-earnings, etc.) would
        # get wrongly capped as HOLD (ABOVE ANALYST TARGET).
        label = "HOLD" if score >= -0.2 else "SELL"
        css   = "neu"  if score >= -0.2 else "sell"
        hint  = "ABOVE ANALYST TARGET"
        return f"{label} ({hint})", css, hint

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 3: MOMENTUM CONTINUATION (overrides EXTENDED penalty)
    # ══════════════════════════════════════════════════════════════════════
    if (trend_stage == "EXTENDED"
            and momentum_score >= 0.65
            and score >= 0.1):       # at least weakly bullish
        # Strong momentum in an extended trend → allow continuation trade
        label = "BUY"
        css = "buy"
        hint = "MOMENTUM CONTINUATION"
        return f"{label} ({hint})", css, hint

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 4: TREND MATURITY → HOLD subtypes
    # ══════════════════════════════════════════════════════════════════════
    if score >= -0.2:  # NEUTRAL or better

        # 4a. Strong trend but EXTENDED/OVEREXTENDED → wait for pullback
        if (trend_stage in ("EXTENDED", "OVEREXTENDED")
                and _adx >= 25 and _rs >= 5):
            label = "HOLD"
            css = "neu"
            hint = "STRONG TREND – WAIT FOR PULLBACK"
            return f"{label} ({hint})", css, hint

        # 4b. PARABOLIC that wasn't caught by step 1 (ATR and vol OK)
        if trend_stage == "PARABOLIC":
            label = "HOLD"
            css = "sell"
            hint = "PARABOLIC – REDUCE EXPOSURE"
            return f"{label} ({hint})", css, hint

        # 4c. OVEREXTENDED without strong trend → more generic wait
        if trend_stage == "OVEREXTENDED":
            label = "HOLD"
            css = "neu"
            hint = "OVEREXTENDED – WAIT FOR PULLBACK"
            return f"{label} ({hint})", css, hint

        # 4d. EXTENDED without strong momentum → mild caution
        if trend_stage == "EXTENDED" and score >= -0.2:
            label = "HOLD"
            css = "neu"
            hint = "EXTENDED – WAIT FOR PULLBACK"
            return f"{label} ({hint})", css, hint

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 5: WEAK CONFIRMATION → HOLD (WEAK MOMENTUM)
    #  Exempt: dip candidates (dip_score ≥ 0.35) — during dips, low
    #  momentum / negative RS are expected, not disqualifying.
    # ══════════════════════════════════════════════════════════════════════
    if score >= -0.2 and score < 0.5 and dip_score < 0.35:
        weak_count = 0
        if _vr < 0.8:
            weak_count += 1
        if _rs < -2:
            weak_count += 1
        if momentum_score < 0.30:
            weak_count += 1

        if weak_count >= 2:
            # ── Consolidation exemption ───────────────────────────────────
            # Strong intermediate/long-term RS + strong fundamentals means
            # the stock is digesting a big move, not reversing.  A bearish
            # MACD and low 1M volume in this context are consolidation noise,
            # not a sell signal.  (Fixes MU-type situations post-earnings.)
            _rs55 = rs_55d or 0
            _rs3  = rs_3m  or 0
            if _rs55 > 10 and _rs3 > 10 and fund_score >= 0.50 and _rs > -15:
                # Exempt: short-term weakness in a sustained outperformer
                # (consolidation after a big move, not a reversal).
                # Guard _rs > -15: if 1M RS is catastrophic the stock is breaking
                # down, not consolidating — the exemption must not fire for a
                # stock that is -20% vs SPY in the past month even if its
                # 55D/3M windows still look positive from an older run.
                pass
            else:
                label = "HOLD"
                css   = "neu"
                # Quality-discount hint: weak technically but large analyst upside
                if fund_score >= 0.30 and _tgap > 20:
                    hint = "QUALITY DISCOUNT"
                else:
                    hint = "WEAK MOMENTUM"
                return f"{label} ({hint})", css, hint

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 6: DEFAULT — standard labels with optional context
    # ══════════════════════════════════════════════════════════════════════

    # ── BUY variants ─────────────────────────────────────────────────────
    if score >= 0.2:
        if regime == "TREND" and trend_stage in ("EARLY", "HEALTHY"):
            hint = "EARLY TREND"
        elif regime == "MEAN_REVERSION" and rsi is not None and rsi < 30:
            hint = "MEAN REVERSION"
        elif regime == "TREND":
            hint = "TREND"
        elif vol_regime in ("HIGH", "EXTREME"):
            hint = "HIGH VOLATILITY"
        elif momentum_score >= 0.65:
            hint = "STRONG MOMENTUM"
        elif regime == "MEAN_REVERSION":
            hint = "MOMENTUM"

    # ── NEUTRAL (nothing caught above) ────────────────────────────────────
    elif score >= -0.2:
        if mkt_regime == "TRANSITION":
            hint = "CONSOLIDATION"

    # ── SELL variants ─────────────────────────────────────────────────────
    else:
        if rsi is not None and rsi > 70:
            hint = "OVERBOUGHT"
        elif mkt_regime == "BEARISH":
            hint = "WEAKNESS"

    # Build enriched label
    if hint:
        enriched = f"{label} ({hint})"
    else:
        enriched = label

    return enriched, css, hint


# ── Overall weighted score (v4: momentum + risk aware) ───────────────────────

def compute_overall_score(
    tech_score: float,
    fund_score: float,
    sent_score: Optional[float],
    weights: Dict[str, float],
    *,
    vol_ratio:       Optional[float] = None,
    rs_1m:           Optional[float] = None,
    rs_55d:          Optional[float] = None,
    rs_3m:           Optional[float] = None,
    atr_pct:         Optional[float] = None,
    vol_pctl:        Optional[float] = None,
    spy_trend_bull:  Optional[bool]  = None,
    trend_stage:     Optional[str]   = None,
    mkt_regime:      Optional[str]   = None,
    regime_chg:      Optional[str]   = None,
    momentum_score:  float           = 0.0,
    risk_score:      float           = 0.0,
    dip_score:       float           = 0.0,
    price:           Optional[float] = None,
    target_px:       Optional[float] = None,
    rec_mean:        Optional[float] = None,
    n_analysts:      Optional[int]   = None,
    thresholds:      Optional[Dict[str, Any]] = None,
) -> tuple[float, str, str]:
    """
    Combine component scores using user weights, then apply adjustment
    pipeline in order:

      1. Base weighted score
      2. Volume confirmation (additive, direction-aware)
      3. Relative strength — multi-period composite (additive)
      3.5 Analyst consensus — target gap + rec_mean (additive)  ← NEW
      4. Dip adjustment (additive, oversold + strong fund)
      5. Trend maturity penalty (multiplicative, momentum-aware)
      6. Volatility penalty (multiplicative)
      7. Risk interaction penalty (multiplicative)
      8. Market regime transition (multiplicative)
      9. Market filter (multiplicative, SPY trend)

    Returns (overall_score, signal_label, css_class).
    """
    tw = weights.get("technical",   40) / 100
    fw = weights.get("fundamental", 40) / 100
    sw = weights.get("sentiment",   20) / 100

    if sent_score is None:
        total = tw + fw
        if total > 0:
            tw = tw / total
            fw = fw / total
        sw = 0.0

    total_w = tw + fw + sw
    if total_w == 0:
        return 0.0, "NEUTRAL", "neu"

    # Step 1: Base weighted score
    overall = (tech_score * tw + fund_score * fw + (sent_score or 0) * sw) / total_w

    # Step 2: Volume confirmation (v4: direction-aware)
    overall += _volume_adjustment(vol_ratio, overall)

    # Step 3: Relative strength (multi-period composite)
    overall += _relative_strength_adjustment(rs_1m, rs_55d, rs_3m)

    # Step 3.5: Analyst consensus — target gap + recommendation mean
    overall += _analyst_adjustment(price, target_px, rec_mean, n_analysts)

    # Step 4: Dip adjustment (NEW v4.1 — boost oversold quality dips)
    overall += _dip_adjustment(dip_score, overall, fund_score)

    # Step 5: Trend maturity penalty (v4: momentum-aware)
    overall = _trend_maturity_penalty(trend_stage, overall, momentum_score)

    # Step 6: Volatility penalty
    overall *= _volatility_penalty(atr_pct, vol_pctl)

    # Step 7: Risk interaction penalty
    overall = _risk_penalty(risk_score, overall)

    # Step 8: Market regime transition
    overall = _regime_transition_adjustment(mkt_regime, regime_chg, overall)

    # Step 9: Market filter (SPY)
    overall = _market_filter(spy_trend_bull, overall)

    overall = round(max(-1.0, min(1.0, overall)), 4)
    label, cls = score_to_signal(overall, thresholds)
    return overall, label, cls
