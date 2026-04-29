"""
Market data fetching and indicator computation.
Produces a data dict compatible with stock_tracker.py's format.

Includes:
  - Classic indicators (MA, RSI, MACD, Bollinger, ATR)
  - Elder Impulse System (daily + weekly)
  - Buying Checklist with confidence % (inspired by top-down approach)
  - Market context (VIX, NYSE breadth)
"""
from __future__ import annotations

import logging
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("app.market_data")

_MIN_VOL_PCTL_HISTORY = 20
_NEUTRAL_VOL_PCTL = 50.0


# ── Polygon.io fallback ───────────────────────────────────────────────────────

def _load_polygon_key() -> str:
    """Load POLYGON_API_KEY from project .env."""
    if getattr(sys, "frozen", False):
        env_path = Path(sys.executable).resolve().parent / ".env"
    else:
        env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("POLYGON_API_KEY"):
                _, _, v = line.partition("=")
                return v.strip().strip('"').strip("'")
    return ""


_POLYGON_KEY: str = ""   # lazily loaded


def _fetch_polygon_history(ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
    """
    Fetch daily OHLCV from Polygon.io (free tier, 15-min delayed).
    Used as fallback when yfinance fails.
    Returns DataFrame matching yfinance format or None.
    """
    global _POLYGON_KEY
    if not _POLYGON_KEY:
        _POLYGON_KEY = _load_polygon_key()
    if not _POLYGON_KEY:
        return None

    # Map yfinance period strings to day counts
    period_days = {
        "3mo": 92, "6mo": 183, "1y": 365, "2y": 730,
        "3y": 1095, "5y": 1825, "10y": 3650, "max": 3650,
    }
    days = period_days.get(period, 365)
    end   = datetime.now()
    start = end - timedelta(days=days)

    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker.upper()}/range/1/day/"
        f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
        f"?adjusted=true&sort=asc&limit=50000&apiKey={_POLYGON_KEY}"
    )
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return None
        df = pd.DataFrame(results)
        df["Date"] = pd.to_datetime(df["t"], unit="ms").dt.normalize()
        df = df.rename(columns={"o": "Open", "h": "High", "l": "Low",
                                 "c": "Close", "v": "Volume"})
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].set_index("Date")
        df.index.name = None
        df = df.sort_index()
        log.info("Polygon fallback succeeded for %s (%d bars)", ticker, len(df))
        return df
    except Exception as e:
        log.warning("Polygon fallback failed for %s: %s", ticker, e)
        return None


def _safe(v) -> Optional[float]:
    try:
        if v is None:
            return None
        f = float(v)
        return None if np.isnan(f) or np.isinf(f) else f
    except (TypeError, ValueError):
        return None


def _pct(v) -> Optional[float]:
    """yfinance returns ratios (0-1) for margins/rates; convert to %."""
    r = _safe(v)
    return None if r is None else round(r * 100, 2)


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _safe_vol_percentile(atr_pct: pd.Series, lookback: int) -> pd.Series:
    """
    Time-safe rolling volatility percentile.

    Very short histories cannot support a 20-bar rolling percentile window.
    In that case we return a neutral percentile rather than raising or
    backfilling with future information.
    """
    if lookback <= 0:
        return pd.Series(np.nan, index=atr_pct.index, dtype=float)
    if lookback < _MIN_VOL_PCTL_HISTORY:
        neutral = pd.Series(_NEUTRAL_VOL_PCTL, index=atr_pct.index, dtype=float)
        neutral[atr_pct.isna()] = np.nan
        return neutral
    return atr_pct.rolling(lookback, min_periods=_MIN_VOL_PCTL_HISTORY).rank(pct=True) * 100


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicator columns to a raw OHLCV DataFrame."""
    df = df.copy()
    if df.empty:
        return df
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    # Moving averages
    df["MA20"]  = df["Close"].rolling(20).mean()
    df["MA50"]  = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()

    # RSI
    df["RSI"] = _rsi(df["Close"])

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]      = ema12 - ema26
    df["MACD_Sig"]  = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Sig"]

    # Bollinger Bands
    bma = df["Close"].rolling(20).mean()
    bsd = df["Close"].rolling(20).std()
    df["BB_Upper"] = bma + 2 * bsd
    df["BB_Lower"] = bma - 2 * bsd
    df["BB_Pct"]   = ((df["Close"] - df["BB_Lower"]) /
                      (df["BB_Upper"] - df["BB_Lower"]).replace(0, np.nan) * 100)

    # ATR
    hi_lo = df["High"] - df["Low"]
    hi_cp = (df["High"] - df["Close"].shift(1)).abs()
    lo_cp = (df["Low"]  - df["Close"].shift(1)).abs()
    df["ATR"] = pd.concat([hi_lo, hi_cp, lo_cp], axis=1).max(axis=1).rolling(14).mean()

    # Volume ratio
    df["Vol_Ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

    # ── ADX (Average Directional Index, 14-period) ───────────────────────────
    adx_period = 14
    _plus_dm_raw  = df["High"].diff()
    _minus_dm_raw = -df["Low"].diff()
    _plus_dm  = pd.Series(
        np.where((_plus_dm_raw > _minus_dm_raw) & (_plus_dm_raw > 0), _plus_dm_raw, 0.0),
        index=df.index)
    _minus_dm = pd.Series(
        np.where((_minus_dm_raw > _plus_dm_raw) & (_minus_dm_raw > 0), _minus_dm_raw, 0.0),
        index=df.index)
    # Wilder smoothing (EWM with alpha=1/period)
    _tr_raw     = pd.concat([hi_lo, hi_cp, lo_cp], axis=1).max(axis=1)
    _tr_smooth  = _tr_raw.ewm(alpha=1/adx_period, min_periods=adx_period).mean()
    _plus_di    = 100 * _plus_dm.ewm(alpha=1/adx_period, min_periods=adx_period).mean() / _tr_smooth.replace(0, np.nan)
    _minus_di   = 100 * _minus_dm.ewm(alpha=1/adx_period, min_periods=adx_period).mean() / _tr_smooth.replace(0, np.nan)
    _dx         = 100 * (_plus_di - _minus_di).abs() / (_plus_di + _minus_di).replace(0, np.nan)
    df["ADX"]   = _dx.ewm(alpha=1/adx_period, min_periods=adx_period).mean()

    # ATR as percentage of price
    df["ATR_Pct"] = (df["ATR"] / df["Close"].replace(0, np.nan)) * 100

    # Volatility percentile (ATR% rank over lookback window)
    vol_lookback = min(252, len(df))
    df["Vol_Pctl"] = _safe_vol_percentile(df["ATR_Pct"], vol_lookback)

    # ── OBV (On-Balance Volume) ──────────────────────────────────────────────
    obv_sign = np.sign(df["Close"].diff()).fillna(0)
    df["OBV"] = (df["Volume"] * obv_sign).cumsum()

    # ── ADX Regime detection ──────────────────────────────────────────────────
    adx_val = df["ADX"]
    df["Regime"] = np.where(
        adx_val > 25, "TREND",
        np.where(adx_val < 20, "MEAN_REVERSION", "NEUTRAL")
    )

    # ── Trend Extension & Stage ──────────────────────────────────────────────
    _ma50 = df["MA50"]
    df["Trend_Ext"] = (df["Close"] - _ma50) / _ma50.replace(0, np.nan)
    _te = df["Trend_Ext"].abs()
    df["Trend_Stage"] = np.select(
        [_te >= 1.0, _te >= 0.5, _te >= 0.3, _te >= 0.1],
        ["PARABOLIC", "OVEREXTENDED", "EXTENDED", "HEALTHY"],
        default="EARLY",
    )

    # ── Volatility Regime ────────────────────────────────────────────────────
    _atr_pct = df["ATR_Pct"]
    df["Vol_Regime"] = np.select(
        [_atr_pct >= 15, _atr_pct >= 8, _atr_pct >= 3],
        ["EXTREME", "HIGH", "NORMAL"],
        default="LOW",
    )

    # ── Market Regime (Bull / Bear / Transition) ─────────────────────────────
    _ma200 = df["MA200"]
    _above_200 = df["Close"] > _ma200
    _ma50_above_200 = _ma50 > _ma200
    df["Mkt_Regime"] = np.select(
        [_above_200 & _ma50_above_200,
         ~_above_200 & ~_ma50_above_200],
        ["BULLISH", "BEARISH"],
        default="TRANSITION",
    )

    # ── Regime Change (transition detection) ─────────────────────────────────
    _prev_mkt = df["Mkt_Regime"].shift(1)
    df["Regime_Chg"] = np.select(
        [(_prev_mkt == "BULLISH")  & (df["Mkt_Regime"] == "BEARISH"),
         (_prev_mkt == "BEARISH")  & (df["Mkt_Regime"] == "BULLISH"),
         (_prev_mkt == "BULLISH")  & (df["Mkt_Regime"] == "TRANSITION"),
         (_prev_mkt == "BEARISH")  & (df["Mkt_Regime"] == "TRANSITION"),
         (_prev_mkt == "TRANSITION") & (df["Mkt_Regime"] == "BULLISH"),
         (_prev_mkt == "TRANSITION") & (df["Mkt_Regime"] == "BEARISH")],
        ["BEARISH REVERSAL", "BULLISH REVERSAL",
         "WEAKENING", "POTENTIAL BOTTOM",
         "BULLISH CONFIRMATION", "BEARISH CONFIRMATION"],
        default="NONE",
    )

    # Performance
    df["Ret_1D"]  = df["Close"].pct_change(1)  * 100
    df["Ret_5D"]  = df["Close"].pct_change(5)  * 100
    df["Ret_21D"] = df["Close"].pct_change(21) * 100
    df["Ret_55D"] = df["Close"].pct_change(55) * 100
    df["Ret_63D"] = df["Close"].pct_change(63) * 100

    # 52-week range
    w52h = df["High"].rolling(252, min_periods=1).max()
    w52l = df["Low"].rolling(252, min_periods=1).min()
    df["W52_Hi"]  = w52h
    df["W52_Lo"]  = w52l
    df["W52_Pct"] = ((df["Close"] - w52l) /
                     (w52h - w52l).replace(0, np.nan) * 100)

    # ── New: EMAs for buying checklist ─────────────────────────────────────────
    df["EMA8"]  = df["Close"].ewm(span=8,  adjust=False).mean()
    df["EMA13"] = df["Close"].ewm(span=13, adjust=False).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()

    # ── Elder Impulse System (daily) ───────────────────────────────────────────
    # Green = 13-EMA rising AND MACD-Histogram rising  → strong bulls
    # Red   = 13-EMA falling AND MACD-Histogram falling → strong bears
    # Blue  = mixed → neutral / transition
    ema13_rising = df["EMA13"] > df["EMA13"].shift(1)
    hist_rising  = df["MACD_Hist"] > df["MACD_Hist"].shift(1)
    df["Elder_D"] = np.where(
        ema13_rising & hist_rising,   "green",
        np.where(~ema13_rising & ~hist_rising, "red", "blue"),
    )

    # MACD histogram direction flag (used by buying checklist)
    df["MACD_Hist_Rising"] = hist_rising

    # ── Liquidity metrics (used by ML engine for quality filtering) ────────
    # Average daily dollar volume (20-day rolling)
    df["Dollar_Volume"] = df["Close"] * df["Volume"]
    df["Avg_Dollar_Vol"] = df["Dollar_Volume"].rolling(20).mean()
    # Spread proxy: (High - Low) / Close as a bid-ask proxy (lower = more liquid)
    df["Spread_Proxy"] = (df["High"] - df["Low"]) / df["Close"].replace(0, np.nan)

    return df


# ── Weekly-timeframe indicators ───────────────────────────────────────────────

def compute_weekly_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Resample daily OHLCV data to weekly bars and compute weekly-timeframe
    indicators used by the buying checklist and Elder Impulse.
    """
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)

    wk = df.resample("W").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    }).dropna(subset=["Close"])

    if len(wk) < 35:
        return {}

    c = wk["Close"]

    # Weekly moving averages
    ma13w  = c.rolling(13).mean()
    ma34w  = c.rolling(34).mean()
    ema13w = c.ewm(span=13, adjust=False).mean()
    ema34w = c.ewm(span=34, adjust=False).mean()

    # Weekly MACD (standard 12,26,9)
    ema12w  = c.ewm(span=12, adjust=False).mean()
    ema26w  = c.ewm(span=26, adjust=False).mean()
    macd_w  = ema12w - ema26w
    sig_w   = macd_w.ewm(span=9, adjust=False).mean()
    hist_w  = macd_w - sig_w

    # Weekly Elder Impulse
    ema13w_rising = ema13w.iloc[-1] > ema13w.iloc[-2]
    hist_w_rising = hist_w.iloc[-1] > hist_w.iloc[-2]
    elder_w = ("green" if ema13w_rising and hist_w_rising
               else "red" if not ema13w_rising and not hist_w_rising
               else "blue")

    latest = wk.iloc[-1]
    prev   = wk.iloc[-2]

    return {
        "ma13w":              _safe(ma13w.iloc[-1]),
        "ma34w":              _safe(ma34w.iloc[-1]),
        "ema13w":             _safe(ema13w.iloc[-1]),
        "ema34w":             _safe(ema34w.iloc[-1]),
        "ma13w_rising":       bool(ma13w.iloc[-1] > ma13w.iloc[-2]),
        "ma34w_rising":       bool(ma34w.iloc[-1] > ma34w.iloc[-2]),
        "ema13w_gt_ema34w":   bool(ema13w.iloc[-1] > ema34w.iloc[-1]),
        "macd_hist_w":        _safe(hist_w.iloc[-1]),
        "macd_hist_w_rising": bool(hist_w.iloc[-1] > hist_w.iloc[-2]),
        "price_gt_ema13w":    bool(latest["Close"] > ema13w.iloc[-1]),
        "elder_w":            elder_w,
        "elder_w_not_red":    elder_w != "red",
    }


# ── Buying Checklist ──────────────────────────────────────────────────────────

CHECKLIST_ITEMS = [
    "13W MA Rising",
    "34W MA Rising",
    "13W EMA > 34W EMA",
    "W MACD Hist Rising",
    "D MACD Hist Rising",
    "8D EMA > 20D EMA",
    "Price > 13W EMA",
    "Price > 50D SMA",
    "MACD Positive",
    "Elder W Not Red",
    "Elder D Not Red",
]


def compute_buying_checklist(
    latest: Dict[str, Any],
    weekly: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate the 11-item buying checklist and return a confidence %.

    The checklist mirrors the top-down approach from
    annualizethis.substack.com — combining weekly trend (items 1-4),
    daily momentum (5-6), price position (7-8), and Elder Impulse (9-11).
    """
    price = latest.get("Close")
    checks: List[Tuple[str, Optional[bool]]] = []

    # 1. 13W MA Rising
    checks.append(("13W MA Rising", weekly.get("ma13w_rising")))

    # 2. 34W MA Rising
    checks.append(("34W MA Rising", weekly.get("ma34w_rising")))

    # 3. 13W EMA > 34W EMA  (weekly golden cross)
    checks.append(("13W EMA > 34W EMA", weekly.get("ema13w_gt_ema34w")))

    # 4. Weekly MACD Histogram Rising
    checks.append(("W MACD Hist Rising", weekly.get("macd_hist_w_rising")))

    # 5. Daily MACD Histogram Rising
    checks.append(("D MACD Hist Rising", latest.get("MACD_Hist_Rising")))

    # 6. 8D EMA > 20D EMA
    ema8, ema20 = latest.get("EMA8"), latest.get("EMA20")
    checks.append(("8D EMA > 20D EMA",
                    bool(ema8 > ema20) if ema8 is not None and ema20 is not None else None))

    # 7. Price > 13W EMA
    ema13w = weekly.get("ema13w")
    checks.append(("Price > 13W EMA",
                    bool(price > ema13w) if price and ema13w else None))

    # 8. Price > 50D SMA
    ma50 = latest.get("MA50")
    checks.append(("Price > 50D SMA",
                    bool(price > ma50) if price and ma50 else None))

    # 9. MACD Positive (daily)
    macd = latest.get("MACD")
    checks.append(("MACD Positive",
                    bool(macd > 0) if macd is not None else None))

    # 10. Elder W Not Red (weekly Elder Impulse is not bearish)
    checks.append(("Elder W Not Red", weekly.get("elder_w_not_red")))

    # 11. Elder D Not Red (daily Elder Impulse is not bearish)
    elder_d = latest.get("Elder_D")
    checks.append(("Elder D Not Red",
                    elder_d != "red" if elder_d is not None else None))

    # Confidence = passed / total (excluding None)
    valid  = [(n, p) for n, p in checks if p is not None]
    passed = sum(1 for _, p in valid if p)
    total  = len(valid)
    confidence = round(passed / total * 100, 2) if total > 0 else 0.0

    return {
        "checks":     checks,           # [(name, bool|None), ...]
        "passed":     passed,
        "total":      total,
        "confidence": confidence,        # 0-100
    }


# ── Market context (top-down layer) ──────────────────────────────────────────

_market_ctx_cache: Dict[str, Any] = {}


def fetch_market_context(force: bool = False, period: str = "1y") -> Dict[str, Any]:
    """
    Fetch market-wide health indicators (called once per analysis run).

    Returns:
        vix, vix_safe, breadth_safe, nyse_new_lows,
        spy_trend_bull: True if SPY > MA200 (market is in uptrend),
        spy_ret_1m:     SPY 1-month return (%),
        spy_df:         SPY DataFrame with computed indicators (for relative strength),
    """
    import time
    now = time.time()

    if not force and _market_ctx_cache.get("_ts") and now - _market_ctx_cache["_ts"] < 300:
        return _market_ctx_cache

    ctx: Dict[str, Any] = {
        "vix": None, "vix_safe": None,
        "breadth_safe": None, "nyse_new_lows": None,
        "spy_trend_bull": None, "spy_ret_1m": None, "spy_df": None,
    }

    # ── VIX ──────────────────────────────────────────────────
    try:
        import concurrent.futures as _cf
        def _fetch_vix():
            return yf.Ticker("^VIX").history(period="5d")
        with _cf.ThreadPoolExecutor() as _pool:
            hist = _pool.submit(_fetch_vix).result(timeout=10)
        if hist is not None and not hist.empty:
            v = float(hist["Close"].iloc[-1])
            ctx["vix"]      = round(v, 2)
            ctx["vix_safe"] = v < 25
    except Exception:
        pass

    # ^NYLW (NYSE New Lows) is delisted — skip to avoid hanging.
    # VIX and SPY trend provide sufficient market-level context.
    ctx["breadth_safe"] = True

    # ── SPY benchmark (for relative strength + market filter) ────────────
    try:
        import concurrent.futures as _cf
        def _fetch_spy():
            return yf.Ticker("SPY").history(period=period, auto_adjust=True)
        with _cf.ThreadPoolExecutor() as _pool:
            spy_hist = _pool.submit(_fetch_spy).result(timeout=30)
        if spy_hist is not None and not spy_hist.empty and len(spy_hist) > 200:
            spy_hist = spy_hist.copy()
            spy_ma200 = spy_hist["Close"].rolling(200).mean()
            spy_close = float(spy_hist["Close"].iloc[-1])
            spy_ma200_val = _safe(spy_ma200.iloc[-1])
            ctx["spy_trend_bull"] = spy_close > spy_ma200_val if spy_ma200_val else None

            # SPY returns for relative-strength calculation
            spy_hist["Ret_21D"] = spy_hist["Close"].pct_change(21) * 100
            spy_hist["Ret_55D"] = spy_hist["Close"].pct_change(55) * 100
            spy_hist["Ret_63D"] = spy_hist["Close"].pct_change(63) * 100
            ctx["spy_ret_1m"]  = _safe(spy_hist["Ret_21D"].iloc[-1])
            ctx["spy_ret_55d"] = _safe(spy_hist["Ret_55D"].iloc[-1])
            ctx["spy_ret_3m"]  = _safe(spy_hist["Ret_63D"].iloc[-1])
            ctx["spy_df"] = spy_hist
    except Exception:
        pass

    ctx["_ts"] = now
    _market_ctx_cache.update(ctx)
    try:
        from .pit_data import PointInTimeStore
        PointInTimeStore().record_market_context(pd.Timestamp.utcnow(), ctx)
    except Exception:
        pass
    return ctx


def fetch_fundamentals(tk: yf.Ticker) -> Dict[str, Any]:
    """Fetch fundamental metrics from yfinance."""
    try:
        import concurrent.futures as _cf
        with _cf.ThreadPoolExecutor() as _pool:
            info = _pool.submit(lambda: tk.info).result(timeout=15)
    except Exception:
        return {}

    # ── PEG: try stored values first, then compute ──
    peg = _safe(info.get("pegRatio")) or _safe(info.get("trailingPegRatio"))
    if peg is None:
        _pe = _safe(info.get("trailingPE"))
        _eg = _safe(info.get("earningsGrowth"))  # ratio, e.g. 0.15
        if _pe and _eg and _eg > 0:
            peg = round(_pe / (_eg * 100), 2)

    # ── ROA: try direct, then compute ──
    roa = _pct(info.get("returnOnAssets"))
    if roa is None:
        _ni = _safe(info.get("netIncomeToCommon"))
        _ta = _safe(info.get("totalAssets"))
        if _ni is not None and _ta and _ta > 0:
            roa = round((_ni / _ta) * 100, 2)

    # ── EPS Growth: try primary, then quarterly ──
    eps_growth = _pct(info.get("earningsGrowth"))
    if eps_growth is None:
        eps_growth = _pct(info.get("earningsQuarterlyGrowth"))

    # ── Dividend Yield: always convert ratio → % ──
    div_yield = _pct(info.get("dividendYield"))
    if div_yield is None:
        div_yield = _pct(info.get("trailingAnnualDividendYield"))
    if div_yield is None:
        # Some tickers provide it as a direct percentage already
        _dy = _safe(info.get("yield"))
        if _dy is not None:
            div_yield = round(_dy * 100, 2) if _dy < 1 else round(_dy, 2)

    return {
        "pe_trail":    _safe(info.get("trailingPE")),
        "pe_fwd":      _safe(info.get("forwardPE")),
        "peg":         peg,
        "pb":          _safe(info.get("priceToBook")),
        "ps":          _safe(info.get("priceToSalesTrailing12Months")),
        "ev_ebitda":   _safe(info.get("enterpriseToEbitda")),
        "ev_rev":      _safe(info.get("enterpriseToRevenue")),
        "gross_mgn":   _pct(info.get("grossMargins")),
        "op_mgn":      _pct(info.get("operatingMargins")),
        "net_mgn":     _pct(info.get("profitMargins")),
        "roe":         _pct(info.get("returnOnEquity")),
        "roa":         roa,
        "rev_growth":  _pct(info.get("revenueGrowth")),
        "eps_growth":  eps_growth,
        "debt_eq":     _safe(info.get("debtToEquity")),
        "curr_ratio":  _safe(info.get("currentRatio")),
        "quick_ratio": _safe(info.get("quickRatio")),
        "fcf":         _safe(info.get("freeCashflow")),
        "mkt_cap":     _safe(info.get("marketCap")),
        "beta":        _safe(info.get("beta")),
        "div_yield":   div_yield,
        "short_float": _pct(info.get("shortPercentOfFloat")),
        "target_px":   _safe(info.get("targetMeanPrice")),
        "rec_mean":    _safe(info.get("recommendationMean")),
        "n_analysts":  _safe(info.get("numberOfAnalystOpinions")),
        "short_name":  info.get("shortName", ""),
        "long_name":   info.get("longName", ""),
        "sector":      info.get("sector", ""),
        "currency":    info.get("currency", "USD"),
    }


def compute_relative_strength(
    asset_latest: Dict[str, Any],
    market_ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Optional[float]]:
    """
    Compute relative strength of an asset vs the SPY benchmark.
    Returns RS_1M (21-day), RS_55D (55-day, IBD-inspired), and RS_3M (63-day).
    All values are asset return − SPY return in percentage points.
    Positive = outperforming SPY; negative = underperforming.
    """
    rs: Dict[str, Optional[float]] = {"rs_1m": None, "rs_55d": None, "rs_3m": None}
    if not market_ctx:
        return rs
    spy_1m   = market_ctx.get("spy_ret_1m")
    spy_55d  = market_ctx.get("spy_ret_55d")
    spy_3m   = market_ctx.get("spy_ret_3m")
    asset_1m  = asset_latest.get("Ret_21D")
    asset_55d = asset_latest.get("Ret_55D")
    asset_3m  = asset_latest.get("Ret_63D")
    if asset_1m is not None and spy_1m is not None:
        rs["rs_1m"] = round(asset_1m - spy_1m, 2)
    if asset_55d is not None and spy_55d is not None:
        rs["rs_55d"] = round(asset_55d - spy_55d, 2)
    if asset_3m is not None and spy_3m is not None:
        rs["rs_3m"] = round(asset_3m - spy_3m, 2)
    return rs


def fetch_price_history(
    ticker: str,
    period: str = "1y",
    auto_adjust: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Fetch raw OHLCV history for a ticker.
    Primary source: yfinance.  Fallback: Polygon.io (if POLYGON_API_KEY set in .env).
    """
    hist = _yf_history(ticker, period, auto_adjust)
    if hist is not None:
        return hist
    # yfinance failed → try Polygon
    log.warning("yfinance failed for %s, trying Polygon fallback", ticker)
    return _fetch_polygon_history(ticker, period)


def _yf_history(ticker: str, period: str, auto_adjust: bool) -> Optional[pd.DataFrame]:
    """Isolated yfinance fetch with timeout."""
    try:
        import concurrent.futures as _cf
        tk = yf.Ticker(ticker)
        def _fetch():
            return tk.history(period=period, auto_adjust=auto_adjust)
        with _cf.ThreadPoolExecutor() as _pool:
            hist = _pool.submit(_fetch).result(timeout=30)
    except Exception:
        return None

    if hist is None or hist.empty or len(hist) < 5:
        return None
    hist = hist.copy()
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = [c[0] for c in hist.columns]
    if hist.index.tz is not None:
        hist.index = hist.index.tz_localize(None)
    if not hist.index.is_monotonic_increasing:
        hist = hist.sort_index()
    return hist


def fetch_asset_data(
    ticker: str,
    period: str = "1y",
    meta: Optional[Dict] = None,
    include_fundamentals: bool = True,
    market_ctx: Optional[Dict] = None,
) -> Optional[Dict[str, Any]]:
    """
    Fetch OHLCV history, compute indicators, and optionally load fundamentals.
    Returns a dict compatible with stock_tracker.py's data format.
    Returns None if the ticker has no data.
    """
    hist = fetch_price_history(ticker, period=period, auto_adjust=True)
    if hist is None:
        return None

    df = compute_indicators(hist)
    latest_row = df.iloc[-1]
    _STR_COLS = {"Elder_D", "Regime", "Trend_Stage", "Vol_Regime",
                  "Mkt_Regime", "Regime_Chg"}
    latest = {col: _safe(latest_row[col]) for col in df.columns
              if col not in _STR_COLS}
    # String columns — copy separately
    latest["Elder_D"]      = str(latest_row.get("Elder_D", "blue"))
    latest["Regime"]       = str(latest_row.get("Regime", "NEUTRAL"))
    latest["Trend_Stage"]  = str(latest_row.get("Trend_Stage", "EARLY"))
    latest["Vol_Regime"]   = str(latest_row.get("Vol_Regime", "NORMAL"))
    latest["Mkt_Regime"]   = str(latest_row.get("Mkt_Regime", "TRANSITION"))
    latest["Regime_Chg"]   = str(latest_row.get("Regime_Chg", "NONE"))
    # MACD_Hist_Rising is a bool
    latest["MACD_Hist_Rising"] = bool(latest_row.get("MACD_Hist_Rising", False))

    # Compute technical signal using stock_tracker.py's scoring formula
    score, sig = _score_technical(latest)

    # Weekly-timeframe indicators & buying checklist
    weekly = compute_weekly_indicators(df)
    checklist = compute_buying_checklist(latest, weekly)

    # Relative strength vs SPY
    rs = compute_relative_strength(latest, market_ctx)

    result: Dict[str, Any] = {
        "meta": meta or {
            "name": ticker.split(".")[0],
            "full": ticker,
            "cur": "USD",
            "sector": "Unknown",
        },
        "df":         df,
        "latest":     latest,
        "sig":        sig,
        "score":      score,
        "weekly":     weekly,
        "checklist":  checklist,
        "rs":         rs,
    }

    if include_fundamentals:
        tk = yf.Ticker(ticker)
        fund = fetch_fundamentals(tk)
        result.update(fund)  # merge all fundamental fields at top level
        result["fundamentals"] = fund  # also available as sub-dict
        try:
            from .pit_data import PointInTimeStore
            PointInTimeStore().record_fundamentals(ticker, df.index[-1], fund)
        except Exception:
            pass

    return result


def _score_technical(latest: Dict) -> tuple[float, str]:
    """
    Replicate stock_tracker.py's score_stock() scoring.
    Returns (score, sig_key) where score ∈ [-9, +9].
    """
    score = 0.0
    c = latest.get("Close") or 0

    ma20  = latest.get("MA20")
    ma50  = latest.get("MA50")
    ma200 = latest.get("MA200")

    if ma20:   score += 0.5  if c > ma20  else -0.5
    if ma50:   score += 1.0  if c > ma50  else -1.0
    if ma200:  score += 1.5  if c > ma200 else -1.5

    if ma50 and ma200:
        score += 1.0 if ma50 > ma200 else -1.0

    rsi = latest.get("RSI")
    if rsi is not None:
        if   rsi < 30:  score += 2.0
        elif rsi < 45:  score += 0.5
        elif rsi > 70:  score -= 2.0
        elif rsi > 55:  score -= 0.5

    macd  = latest.get("MACD")
    msig  = latest.get("MACD_Sig")
    mhist = latest.get("MACD_Hist")
    if macd is not None and msig is not None:
        score += 1.0 if macd > msig else -1.0
    if mhist is not None:
        score += 0.5 if mhist > 0 else -0.5

    bb = latest.get("BB_Pct")
    if bb is not None:
        if   bb < 15:  score += 1.0
        elif bb > 85:  score -= 1.0

    if   score >= 4.5: sig = "SBUY"
    elif score >= 2.0: sig = "BUY"
    elif score >= -2.0: sig = "NEU"
    elif score >= -4.5: sig = "SELL"
    else:              sig = "SSELL"

    return round(score, 1), sig


def search_assets(query: str, limit: int = 10) -> list:
    """Search for tickers via yfinance. Returns list of {ticker, name, exchange, type}."""
    try:
        import concurrent.futures as _cf
        def _search():
            return yf.Search(query, max_results=limit)
        with _cf.ThreadPoolExecutor() as _pool:
            results = _pool.submit(_search).result(timeout=10)
        quotes = results.quotes if hasattr(results, "quotes") else []
        out = []
        for q in quotes[:limit]:
            out.append({
                "ticker":   q.get("symbol", ""),
                "name":     q.get("longname") or q.get("shortname", ""),
                "exchange": q.get("exchDisp") or q.get("exchange", ""),
                "sector":   q.get("sectorDisp") or q.get("sector", ""),
                "type":     q.get("quoteType", "EQUITY").lower(),
                "currency": q.get("currency", "USD"),
            })
        return out
    except Exception:
        return []
