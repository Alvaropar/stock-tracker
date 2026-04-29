"""
Backtest API.

POST /api/backtest/run  →  replay the historical scoring engine on price data.

For every trading day in the requested period (after a 200-bar warmup so MA200 is
valid), the full scoring pipeline is run:

  compute_technical_score   — uses actual historical OHLCV-derived indicators
  compute_fundamental_score — optional static overlay, disabled by default
  compute_momentum_score    — uses historical ADX, vol_ratio, MACD, and RS vs SPY
  compute_risk_score        — uses historical trend_ext × ATR_pct
  compute_dip_score         — uses historical RSI, BB_pct, vol_ratio
  compute_overall_score     — full pipeline including regime adjustments
  contextual_signal         — produces the same verbal labels as the live analysis

Returned time series per ticker:
  dates          — ISO date strings
  prices         — closing price
  scores         — overall_score ∈ [-1, 1]
  verbal_signals — e.g. "BUY (EARLY TREND)", "AVOID (EXTREME RISK)"
  css_classes    — sbuy / buy / neu / sell / ssell
  fund_score     — single static value (same for all dates)
  n_points       — number of valid rows returned

Limitations / caveats:
  - By default the backtest is TECHNICAL-ONLY. Fundamentals are excluded unless
    `use_fundamentals=true` is explicitly supplied.
  - If fundamentals are enabled, the score is STATIC. The backtest reuses the
    latest available analysis or PIT-store fundamentals across the whole
    horizon because daily historical fundamentals are not available from
    yfinance.
  - The analyst target_gap is also computed relative to the current analyst
    target and the historical price — meaningful for identifying when the stock
    was deeply discounted, but the target itself is not historically adjusted.
  - SPY data is fetched separately so that per-date RS vs SPY can be
    computed historically (asset rolling return − SPY rolling return).
"""
from __future__ import annotations

import traceback
from typing import Any, Dict, List, Optional, Tuple

from flask import Blueprint, jsonify, request

bp = Blueprint("backtest", __name__, url_prefix="/api/backtest")

import numpy as np
import pandas as pd

_FUND_NUMERIC_KEYS = {
    "pe_trail", "pe_fwd", "peg", "pb", "ps", "ev_ebitda", "ev_rev",
    "gross_mgn", "op_mgn", "net_mgn", "roe", "roa", "rev_growth",
    "eps_growth", "debt_eq", "curr_ratio", "quick_ratio", "fcf",
    "mkt_cap", "beta", "div_yield", "short_float", "target_px",
    "rec_mean", "n_analysts",
}
_FUND_TEXT_KEYS = {"short_name", "long_name", "sector", "currency"}
_SIGNAL_QUALITY_HORIZONS: Tuple[Tuple[int, str, float], ...] = (
    (5, "1w", 0.2),
    (21, "1m", 0.5),
    (63, "3m", 0.3),
)
_ACTIONABLE_SCORE_THRESHOLD = 0.2


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_float(v) -> Optional[float]:
    """Convert a value to float, returning None on NaN or failure."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if (f != f) else f   # NaN → None
    except (TypeError, ValueError):
        return None


def _row_to_latest(row) -> Dict[str, Any]:
    """
    Convert a DataFrame row (pd.Series) to the 'latest' dict format consumed
    by the scoring engine.  Numeric values are cast to float; regime string
    columns (Regime, Trend_Stage, …) are preserved as-is; NaN / None omitted.
    """
    result: Dict[str, Any] = {}
    for col, val in row.items():
        sf = _safe_float(val)
        if sf is not None:
            result[col] = sf
        elif isinstance(val, str):
            result[col] = val
        elif isinstance(val, bool):
            result[col] = val
        # NaN / None / other → skip
    return result


def _clean_fundamentals(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}

    cleaned: Dict[str, Any] = {}
    for key in _FUND_NUMERIC_KEYS:
        if key in raw:
            value = _safe_float(raw.get(key))
            if value is not None:
                cleaned[key] = value

    for key in _FUND_TEXT_KEYS:
        value = raw.get(key)
        if isinstance(value, str) and value:
            cleaned[key] = value

    return cleaned


def _load_fundamentals_for_backtest(
    ticker: str,
    supplied_map: Any,
) -> Tuple[Dict[str, Any], str]:
    supplied = supplied_map.get(ticker) if isinstance(supplied_map, dict) else None
    cleaned = _clean_fundamentals(supplied)
    if cleaned:
        return cleaned, "analysis"

    try:
        import pandas as pd

        from ..services.pit_data import PointInTimeStore

        snapshot = PointInTimeStore().snapshot_asof(
            "fundamentals",
            ticker,
            pd.Timestamp.utcnow(),
        )
        cleaned = _clean_fundamentals(snapshot)
        if cleaned:
            return cleaned, "pit_store"
    except Exception:
        pass

    return {}, "missing"


def _compute_signal_quality_metrics(
    prices: List[float],
    scores: List[float],
) -> Dict[str, Any]:
    import numpy as np
    import pandas as pd

    price_s = pd.Series(prices, dtype="float64")
    score_s = pd.Series(scores, dtype="float64")

    horizon_rows: List[Dict[str, Any]] = []
    weighted_ic_sum = 0.0
    weighted_ic_den = 0.0

    for days, label, weight in _SIGNAL_QUALITY_HORIZONS:
        forward_ret = price_s.shift(-days) / price_s - 1.0
        valid = score_s.notna() & forward_ret.notna()

        ic = None
        if int(valid.sum()) >= max(10, days // 2):
            ic_val = score_s[valid].rank(method="average").corr(
                forward_ret[valid].rank(method="average")
            )
            if pd.notna(ic_val):
                ic = float(ic_val)
                weighted_ic_sum += ic * weight
                weighted_ic_den += weight

        horizon_rows.append({
            "days": days,
            "label": label,
            "information_coefficient": round(ic, 4) if ic is not None else None,
        })

    composite_ic = (
        weighted_ic_sum / weighted_ic_den
        if weighted_ic_den > 0
        else None
    )

    forward_21d = price_s.shift(-21) / price_s - 1.0
    actionable = score_s.abs() >= _ACTIONABLE_SCORE_THRESHOLD
    valid_actionable = actionable & forward_21d.notna()

    actionable_n = int(valid_actionable.sum())
    actionable_hit_rate = None
    avg_signed_return_21d = None

    if actionable_n:
        signed_returns = np.sign(score_s[valid_actionable]) * forward_21d[valid_actionable]
        actionable_hit_rate = float((signed_returns > 0).mean())
        avg_signed_return_21d = float(signed_returns.mean())

    buy_mask = (score_s >= _ACTIONABLE_SCORE_THRESHOLD) & forward_21d.notna()
    sell_mask = (score_s <= -_ACTIONABLE_SCORE_THRESHOLD) & forward_21d.notna()

    avg_buy_return_21d = (
        float(forward_21d[buy_mask].mean()) if int(buy_mask.sum()) else None
    )
    avg_sell_return_21d = (
        float(forward_21d[sell_mask].mean()) if int(sell_mask.sum()) else None
    )

    signal_quality_score = None
    if composite_ic is not None:
        signal_quality_score = float(np.clip(50.0 + 100.0 * composite_ic, 0.0, 100.0))

    # ── ICIR: annualised IC / std(IC) across non-overlapping monthly windows ──
    # We split the 21d-horizon IC into monthly (21-bar) non-overlapping windows,
    # compute Spearman IC per window, then:
    #   ICIR = mean(window_ICs) / std(window_ICs) * sqrt(12)   (annualised)
    # Requires >= 6 windows (~6 months) for a meaningful estimate.
    icir_21d = None
    _win = 21
    _fwd21 = price_s.shift(-_win) / price_s - 1.0
    _win_ics: List[float] = []
    n_bars = len(price_s)
    for start in range(0, n_bars - _win * 2, _win):
        end = start + _win
        s_w = score_s.iloc[start:end]
        f_w = _fwd21.iloc[start:end]
        valid_w = s_w.notna() & f_w.notna()
        if int(valid_w.sum()) >= 10:
            ic_w = s_w[valid_w].rank(method="average").corr(
                f_w[valid_w].rank(method="average")
            )
            if pd.notna(ic_w):
                _win_ics.append(float(ic_w))

    if len(_win_ics) >= 6:
        _ic_arr = np.array(_win_ics)
        _ic_std = float(_ic_arr.std())
        if _ic_std > 1e-9:
            icir_21d = round(float(_ic_arr.mean() / _ic_std) * (12 ** 0.5), 4)

    return {
        "signal_quality_score": round(signal_quality_score, 2) if signal_quality_score is not None else None,
        "composite_ic": round(composite_ic, 4) if composite_ic is not None else None,
        "icir_21d": icir_21d,
        "actionable_threshold": _ACTIONABLE_SCORE_THRESHOLD,
        "actionable_n": actionable_n,
        "actionable_hit_rate_21d": round(actionable_hit_rate, 4) if actionable_hit_rate is not None else None,
        "avg_signed_return_21d": round(avg_signed_return_21d, 4) if avg_signed_return_21d is not None else None,
        "avg_buy_return_21d": round(avg_buy_return_21d, 4) if avg_buy_return_21d is not None else None,
        "avg_sell_return_21d": round(avg_sell_return_21d, 4) if avg_sell_return_21d is not None else None,
        "horizons": horizon_rows,
    }


def _build_backtest_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    quality_scores: List[float] = []
    composite_ics: List[float] = []
    icirs: List[float] = []
    hit_rates: List[float] = []

    for payload in results.values():
        if not isinstance(payload, dict) or payload.get("error"):
            continue
        quality = payload.get("signal_quality") or {}
        if quality.get("signal_quality_score") is not None:
            quality_scores.append(float(quality["signal_quality_score"]))
        if quality.get("composite_ic") is not None:
            composite_ics.append(float(quality["composite_ic"]))
        if quality.get("icir_21d") is not None:
            icirs.append(float(quality["icir_21d"]))
        if quality.get("actionable_hit_rate_21d") is not None:
            hit_rates.append(float(quality["actionable_hit_rate_21d"]))

    def _avg(values: List[float], digits: int) -> Optional[float]:
        if not values:
            return None
        return round(sum(values) / len(values), digits)

    return {
        "n_tickers": len(results),
        "n_quality_tickers": len(quality_scores),
        "avg_signal_quality_score": _avg(quality_scores, 2),
        "avg_composite_ic": _avg(composite_ics, 4),
        "avg_icir_21d": _avg(icirs, 4),
        "avg_actionable_hit_rate_21d": _avg(hit_rates, 4),
    }


# ── Quant analysis helpers ────────────────────────────────────────────────────

# Indicator columns analysed for per-indicator IC and correlation matrix.
# These are the raw indicator values from compute_indicators().
_QUANT_INDICATOR_COLS = [
    "RSI", "BB_Pct", "MACD", "ADX", "Vol_Ratio", "ATR_Pct",
    "Ret_21D", "Ret_63D", "MA200",
]


def _compute_per_indicator_ic(
    prices: List[float],
    df_slice: pd.DataFrame,
    forward_days: int = 21,
) -> Dict[str, Any]:
    """
    Spearman Information Coefficient for each raw indicator independently
    against `forward_days`-day forward returns.
    """
    price_s = pd.Series(prices, dtype="float64").reset_index(drop=True)
    fwd = price_s.shift(-forward_days) / price_s - 1.0

    records: List[Dict] = []
    for col in _QUANT_INDICATOR_COLS:
        if col not in df_slice.columns:
            continue
        ind_s = df_slice[col].reset_index(drop=True)
        if len(ind_s) != len(price_s):
            continue
        # MA200: convert to % distance from price so direction is intuitive
        if col == "MA200":
            ma_s = ind_s.copy()
            valid_ma = ma_s.notna() & (ma_s > 0)
            ind_s = (price_s - ma_s) / ma_s.where(ma_s > 0) * 100
        valid = ind_s.notna() & fwd.notna()
        n_valid = int(valid.sum())
        if n_valid < 20:
            continue
        ic_val = float(
            ind_s[valid].rank(method="average").corr(
                fwd[valid].rank(method="average")
            )
        )
        if not np.isnan(ic_val):
            records.append({"name": col, "ic": round(ic_val, 4), "n_obs": n_valid})

    records.sort(key=lambda x: abs(x["ic"]), reverse=True)
    return {
        "indicators": records,
        "forward_days": forward_days,
        "n_indicators": len(records),
    }


def _walk_forward_oos_ic(
    prices: List[float],
    scores: List[float],
    n_train: int = 126,
    n_test: int = 21,
) -> Dict[str, Any]:
    """
    Walk-forward out-of-sample IC validation.

    Partitions the history into non-overlapping test windows of `n_test` bars,
    each preceded by a `n_train`-bar burn-in.  Spearman IC of `scores` vs
    forward-`n_test` returns is computed on every test window.

    This separates in-sample from OOS IC — a large drop reveals over-fitting.
    """
    n = len(prices)
    if n < n_train + n_test * 2:
        return {"windows": [], "n_windows": 0, "oos_mean_ic": None, "oos_icir": None}

    price_s = pd.Series(prices, dtype="float64")
    score_s = pd.Series(scores, dtype="float64")
    fwd = price_s.shift(-n_test) / price_s - 1.0

    windows: List[Dict] = []
    start = n_train
    while start + n_test <= n - n_test:
        end = start + n_test
        s_w = score_s.iloc[start:end]
        f_w = fwd.iloc[start:end]
        valid = s_w.notna() & f_w.notna()
        n_valid = int(valid.sum())
        if n_valid >= max(10, n_test // 3):
            ic_w = float(
                s_w[valid].rank(method="average").corr(
                    f_w[valid].rank(method="average")
                )
            )
            if not np.isnan(ic_w):
                windows.append({
                    "window_idx": len(windows),
                    "bar_start": start,
                    "bar_end": end,
                    "ic": round(ic_w, 4),
                    "n_obs": n_valid,
                })
        start += n_test

    if not windows:
        return {"windows": [], "n_windows": 0, "oos_mean_ic": None, "oos_icir": None}

    ic_arr = np.array([w["ic"] for w in windows])
    mean_ic = float(ic_arr.mean())
    std_ic = float(ic_arr.std()) if len(ic_arr) > 1 else 0.0
    oos_icir = mean_ic / std_ic * np.sqrt(12) if std_ic > 1e-9 else None

    return {
        "windows": windows,
        "n_windows": len(windows),
        "oos_mean_ic": round(mean_ic, 4),
        "oos_std_ic": round(std_ic, 4),
        "oos_icir": round(float(oos_icir), 4) if oos_icir is not None else None,
        "pct_positive_windows": round(float((ic_arr > 0).mean()), 3),
        "n_train_bars": n_train,
        "n_test_bars": n_test,
    }


def _compute_signal_correlation_matrix(
    df_slice: pd.DataFrame,
    min_valid: int = 40,
) -> Dict[str, Any]:
    """
    Spearman pairwise correlation between raw indicator time series.
    High off-diagonal correlations reveal redundant signals that reduce
    effective diversification across indicators.
    """
    cols = [c for c in _QUANT_INDICATOR_COLS if c in df_slice.columns]
    if len(cols) < 2:
        return {"columns": cols, "matrix": []}

    sub = df_slice[cols].copy().reset_index(drop=True)
    if "MA200" in sub.columns and "Close" in df_slice.columns:
        close_s = df_slice["Close"].reset_index(drop=True)
        ma_s = sub["MA200"]
        sub["MA200"] = np.where(ma_s > 0, (close_s - ma_s) / ma_s * 100, np.nan)

    corr_df = sub.corr(method="spearman", min_periods=min_valid)

    matrix: List[List] = []
    for c1 in cols:
        row: List = []
        for c2 in cols:
            if c1 in corr_df.index and c2 in corr_df.columns:
                val = corr_df.loc[c1, c2]
                row.append(round(float(val), 3) if not np.isnan(val) else None)
            else:
                row.append(None)
        matrix.append(row)

    return {"columns": cols, "matrix": matrix}


def _compute_factor_exposure(
    prices: List[float],
    scores: List[float],
    forward_days: int = 21,
) -> Dict[str, Any]:
    """
    OLS regression of forward returns on three style factors:
      - Momentum (12m-1m cross-sectional proxy)
      - Medium-term trend (50d price return)
      - Low-volatility (negative 21d realised vol)

    Coefficient signs reveal which factor bias the signal implicitly carries.
    R² shows how much of return predictability is explained by these factors.
    """
    price_s = pd.Series(prices, dtype="float64")
    score_s = pd.Series(scores, dtype="float64")

    fwd = price_s.shift(-forward_days) / price_s - 1.0
    mom = price_s.shift(21) / price_s.shift(252) - 1.0   # 12m-1m momentum
    trend = price_s / price_s.shift(50) - 1.0             # 50-day trend
    log_r = np.log(price_s / price_s.shift(1))
    rvol = log_r.rolling(21).std() * np.sqrt(252)
    low_vol = -rvol                                        # high = less volatile

    valid = (
        fwd.notna() & mom.notna() & trend.notna() &
        low_vol.notna() & score_s.notna()
    )
    n_valid = int(valid.sum())

    if n_valid < 30:
        return {
            "n_obs": n_valid, "factors": {},
            "r_squared": None, "note": "insufficient data",
        }

    Y = fwd[valid].values
    X_c = np.column_stack([
        np.ones(n_valid),
        mom[valid].values,
        trend[valid].values,
        low_vol[valid].values,
    ])
    try:
        betas, _, _, _ = np.linalg.lstsq(X_c, Y, rcond=None)
    except Exception:
        return {"n_obs": n_valid, "factors": {}, "r_squared": None}

    Y_hat = X_c @ betas
    ss_res = float(np.sum((Y - Y_hat) ** 2))
    ss_tot = float(np.sum((Y - float(Y.mean())) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    return {
        "n_obs": n_valid,
        "factors": {
            "momentum_12_1m": round(float(betas[1]), 5),
            "trend_50d":      round(float(betas[2]), 5),
            "low_vol":        round(float(betas[3]), 5),
        },
        "intercept": round(float(betas[0]), 5),
        "r_squared":  round(float(r2), 4),
    }


# ── Route ──────────────────────────────────────────────────────────────────────

@bp.route("/run", methods=["POST"])
def run():
    """
    Run a historical signal backtest for one or more tickers.

    Request JSON:
      {
        "tickers":     ["NVDA", "MU"],
        "period":      "2y",          // yfinance period string
        "technical":   [...],         // indicator selection (same as step 2)
        "fundamental": [...],
        "weights":     {"technical": 60, "fundamental": 40},
        "use_fundamentals": false
      }
    """
    config   = request.get_json(force=True, silent=True) or {}
    tickers = config.get("tickers", [])
    period = config.get("period",  "2y")
    tech_sel = config.get("technical",   ["ma20", "ma50", "ma200", "cross", "rsi", "macd", "bb"])
    use_fundamentals = bool(config.get("use_fundamentals", False))
    fund_sel = (
        config.get("fundamental", ["pe", "margins", "roe", "growth", "analyst"])
        if use_fundamentals
        else []
    )
    weights = config.get("weights", {"technical": 60, "fundamental": 40})
    supplied_fundamentals = config.get("fundamentals_map", {})

    # ── Threshold config (optional) ───────────────────────────────────────────
    # The caller may supply either a full inline config dict or just a config_id.
    # Inline dict takes priority; config_id falls back to the stored config.
    from ..services import threshold_config as tc
    _threshold_cfg: Optional[Dict] = None
    _inline = config.get("threshold_config")
    _cfg_id = config.get("threshold_config_id")
    if isinstance(_inline, dict) and _inline:
        _threshold_cfg = _inline
    elif _cfg_id:
        _threshold_cfg = tc.get_config(_cfg_id)
    # Extract the sub-dicts scoring functions expect
    _score_thresholds = (_threshold_cfg or {}).get("score_thresholds") or None
    _ic_weights       = (_threshold_cfg or {}).get("ic_weights") or None
    _calib_ticker     = (_threshold_cfg or {}).get("ic_calibration_ticker", "")

    if not tickers:
        return jsonify({"ok": False, "error": "No tickers provided"}), 400

    from ..services import market_data as md
    from ..services import scoring as sc

    # ── Historical SPY data for per-date RS computation ───────────────────────
    spy_df = None
    try:
        raw = md.fetch_price_history("SPY", period=period, auto_adjust=True)
        if raw is not None:
            spy_df = md.compute_indicators(raw)
    except Exception:
        spy_df = None

    results: Dict[str, Any] = {}

    for ticker in tickers:
        try:
            raw = md.fetch_price_history(ticker, period=period, auto_adjust=True)
            if raw is None:
                results[ticker] = {"error": "No data returned"}
                continue

            df = md.compute_indicators(raw)
            if use_fundamentals:
                fund_d, fund_source = _load_fundamentals_for_backtest(ticker, supplied_fundamentals)
            else:
                fund_d, fund_source = {}, "disabled"

            # Static fund score — same value used for all historical dates
            fund_score = sc.compute_fundamental_score(
                fund_d, fund_sel, sector=fund_d.get("sector", "")
            )

            # Static analyst metadata (target_gap varies by historical price)
            target_px  = fund_d.get("target_px")
            rec_mean   = fund_d.get("rec_mean")
            n_analysts = fund_d.get("n_analysts")

            dates:          List[str]   = []
            prices:         List[float] = []
            scores:         List[float] = []
            verbal_signals: List[str]   = []
            css_classes:    List[str]   = []
            _valid_rows:    List        = []  # raw df rows used (for quant analysis)

            # Skip first WARMUP rows so that MA200 and long-window indicators
            # have enough history to be meaningful.
            WARMUP = 200
            spy_aligned = (
                spy_df.reindex(df.index, method="ffill")
                if spy_df is not None and not spy_df.empty
                else None
            )

            for i in range(len(df)):
                if i < WARMUP:
                    continue

                row   = df.iloc[i]
                idx   = df.index[i]
                close = _safe_float(row.get("Close"))
                if close is None or close <= 0:
                    continue

                latest = _row_to_latest(row)

                # ── Historical SPY context at this date ───────────────────
                spy_trend_bull: Optional[bool] = None
                rs_1m = rs_55d = rs_3m = None

                if spy_aligned is not None:
                    spy_row = spy_aligned.iloc[i]
                    spy_ma200v = _safe_float(spy_row.get("MA200"))
                    spy_closev = _safe_float(spy_row.get("Close"))
                    if spy_ma200v is not None and spy_closev is not None:
                        spy_trend_bull = spy_closev > spy_ma200v

                    spy_r21 = _safe_float(spy_row.get("Ret_21D"))
                    spy_r55 = _safe_float(spy_row.get("Ret_55D"))
                    spy_r63 = _safe_float(spy_row.get("Ret_63D"))

                    a21 = _safe_float(row.get("Ret_21D"))
                    a55 = _safe_float(row.get("Ret_55D"))
                    a63 = _safe_float(row.get("Ret_63D"))

                    if a21 is not None and spy_r21 is not None:
                        rs_1m  = round(a21 - spy_r21, 2)
                    if a55 is not None and spy_r55 is not None:
                        rs_55d = round(a55 - spy_r55, 2)
                    if a63 is not None and spy_r63 is not None:
                        rs_3m  = round(a63 - spy_r63, 2)

                # Per-date target gap (analyst target is static, price varies)
                target_gap: Optional[float] = None
                if target_px and close > 0:
                    target_gap = round((target_px - close) / close * 100, 1)

                # ── Regime fields ─────────────────────────────────────────
                regime      = latest.get("Regime",      "NEUTRAL")
                trend_stage = latest.get("Trend_Stage")
                mkt_regime  = latest.get("Mkt_Regime")
                regime_chg  = latest.get("Regime_Chg")
                vol_regime  = latest.get("Vol_Regime")
                # "NONE" string → None (matches analysis.py behaviour)
                if regime_chg == "NONE":
                    regime_chg = None

                # ── Score pipeline ────────────────────────────────────────
                # Apply IC weights if config was calibrated for this ticker
                _use_ic = (
                    _ic_weights and
                    (not _calib_ticker or _calib_ticker.upper() == ticker.upper())
                )
                tech = sc.compute_technical_score(
                    latest, tech_sel, regime=regime,
                    thresholds=_threshold_cfg,
                    ic_weights=_ic_weights if _use_ic else None,
                )

                macd_v    = latest.get("MACD")
                macd_s    = latest.get("MACD_Sig")
                macd_bull = (macd_v > macd_s) if (
                    macd_v is not None and macd_s is not None
                ) else None

                momentum_score = sc.compute_momentum_score(
                    adx=latest.get("ADX"),
                    rs_1m=rs_1m, rs_55d=rs_55d, rs_3m=rs_3m,
                    vol_ratio=latest.get("Vol_Ratio"),
                    macd_bull=macd_bull,
                )
                risk_score = sc.compute_risk_score(
                    trend_ext=latest.get("Trend_Ext"),
                    atr_pct=latest.get("ATR_Pct"),
                )
                dip_score = sc.compute_dip_score(
                    rsi=latest.get("RSI"),
                    fund_score=fund_score,
                    vol_ratio=latest.get("Vol_Ratio"),
                    mkt_regime=mkt_regime,
                    regime_chg=regime_chg,
                    bb_pct=latest.get("BB_Pct"),
                    target_gap=target_gap,
                    n_analysts=n_analysts,
                )
                overall, signal, css = sc.compute_overall_score(
                    tech, fund_score, None, weights,
                    vol_ratio=latest.get("Vol_Ratio"),
                    rs_1m=rs_1m, rs_55d=rs_55d, rs_3m=rs_3m,
                    atr_pct=latest.get("ATR_Pct"),
                    vol_pctl=latest.get("Vol_Pctl"),
                    spy_trend_bull=spy_trend_bull,
                    trend_stage=trend_stage,
                    mkt_regime=mkt_regime,
                    regime_chg=regime_chg,
                    momentum_score=momentum_score,
                    risk_score=risk_score,
                    dip_score=dip_score,
                    price=close,
                    target_px=target_px,
                    rec_mean=rec_mean,
                    n_analysts=n_analysts,
                    thresholds=_score_thresholds,
                )
                ctx_label, ctx_css, _ = sc.contextual_signal(
                    signal, css, overall,
                    regime=regime,
                    trend_stage=trend_stage,
                    mkt_regime=mkt_regime,
                    regime_chg=regime_chg,
                    rsi=latest.get("RSI"),
                    vol_regime=vol_regime,
                    adx=latest.get("ADX"),
                    vol_ratio=latest.get("Vol_Ratio"),
                    rs_1m=rs_1m, rs_55d=rs_55d, rs_3m=rs_3m,
                    atr_pct=latest.get("ATR_Pct"),
                    momentum_score=momentum_score,
                    risk_score=risk_score,
                    dip_score=dip_score,
                    fund_score=fund_score,
                    target_gap=target_gap,
                )

                date_str = (
                    str(idx.date()) if hasattr(idx, "date") else str(idx)[:10]
                )
                dates.append(date_str)
                prices.append(round(close, 4))
                scores.append(round(overall, 4))
                verbal_signals.append(ctx_label)
                css_classes.append(ctx_css)
                _valid_rows.append(row)

            signal_quality = _compute_signal_quality_metrics(prices, scores)

            # ── Quant analysis (per-indicator IC, walk-forward, correlation, factor) ──
            per_ind_ic   = {}
            walk_fwd     = {}
            corr_matrix  = {}
            factor_exp   = {}
            try:
                if _valid_rows and len(prices) > 50:
                    _df_valid = pd.DataFrame(_valid_rows).reset_index(drop=True)
                    per_ind_ic  = _compute_per_indicator_ic(prices, _df_valid)
                    walk_fwd    = _walk_forward_oos_ic(prices, scores)
                    corr_matrix = _compute_signal_correlation_matrix(_df_valid)
                    factor_exp  = _compute_factor_exposure(prices, scores)
            except Exception as _qe:
                per_ind_ic = {"error": str(_qe)}

            results[ticker] = {
                "dates":          dates,
                "prices":         prices,
                "scores":         scores,
                "verbal_signals": verbal_signals,
                "css_classes":    css_classes,
                "fund_score":     round(fund_score, 4),
                "fundamentals_source": fund_source,
                "n_points":       len(dates),
                "signal_quality": signal_quality,
                "per_indicator_ic":    per_ind_ic,
                "walk_forward":        walk_fwd,
                "signal_correlation":  corr_matrix,
                "factor_exposure":     factor_exp,
            }

        except Exception as e:
            results[ticker] = {
                "error":     str(e),
                "traceback": traceback.format_exc(),
            }

    return jsonify({
        "ok": True,
        "results": results,
        "summary": _build_backtest_summary(results),
    })
