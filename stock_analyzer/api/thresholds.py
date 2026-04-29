"""
Threshold configuration API.

GET    /api/thresholds               → list all saved configs (summaries)
GET    /api/thresholds/default       → full default config (breakpoints)
GET    /api/thresholds/<id>          → full config by ID
POST   /api/thresholds               → save / upsert a config
DELETE /api/thresholds/<id>          → delete a config
POST   /api/thresholds/calibrate-ic  → compute IC weights for a ticker and save

The 'default' config is always present and cannot be deleted.
"""
from __future__ import annotations

import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, request

from ..services import threshold_config as tc

bp = Blueprint("thresholds", __name__, url_prefix="/api/thresholds")


@bp.route("", methods=["GET"])
def list_configs():
    return jsonify({"ok": True, "configs": tc.list_configs()})


@bp.route("/default", methods=["GET"])
def get_default():
    return jsonify({"ok": True, "config": tc.default_config()})


@bp.route("/<config_id>", methods=["GET"])
def get_config(config_id: str):
    cfg = tc.get_config(config_id)
    if cfg is None:
        return jsonify({"ok": False, "error": f"Config '{config_id}' not found"}), 404
    return jsonify({"ok": True, "config": cfg})


@bp.route("", methods=["POST"])
def save_config():
    payload = request.get_json(force=True, silent=True) or {}
    if not payload.get("name"):
        return jsonify({"ok": False, "error": "name is required"}), 400
    try:
        saved = tc.save_config(payload)
        return jsonify({"ok": True, "config": saved})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@bp.route("/<config_id>", methods=["DELETE"])
def delete_config(config_id: str):
    if config_id == "default":
        return jsonify({"ok": False, "error": "Built-in default cannot be deleted"}), 400
    deleted = tc.delete_config(config_id)
    if not deleted:
        return jsonify({"ok": False, "error": f"Config '{config_id}' not found"}), 404
    return jsonify({"ok": True})


# ── IC weight calibration ──────────────────────────────────────────────────────

@bp.route("/calibrate-ic", methods=["POST"])
def calibrate_ic():
    """
    Compute per-indicator IC weights from historical data for a single ticker,
    then save the result as a named ThresholdConfig.

    The calibration window is STRICTLY BEFORE the backtest window so there is
    zero data leakage.  Example: backtest_period="2y" + calib_period="5y" means
    calibration uses data from ~5 years ago up to ~2 years ago, and the separate
    backtest covers only the most recent 2 years.

    Request JSON:
        {
          "ticker":           "NVDA",
          "backtest_period":  "2y",    // evaluation window (excluded from calib)
          "calib_period":     "5y",    // how far back to fetch for calibration
          "tech_sel":         ["rsi","macd","bb","ma50","ma200","cross"],
          "name":             "NVDA IC weights",   // config display name (optional)
          "config_id":        "abc123"             // update existing config (optional)
        }

    Response: the saved ThresholdConfig including ic_weights.
    """
    import datetime as _dt

    body             = request.get_json(force=True, silent=True) or {}
    ticker           = (body.get("ticker") or "").upper()
    backtest_period  = body.get("backtest_period", "2y")
    calib_period     = body.get("calib_period", "5y")
    tech_sel         = body.get("tech_sel", ["ma20","ma50","ma200","cross","rsi","macd","bb"])
    name             = body.get("name") or f"{ticker} IC weights (calib {calib_period}, bt {backtest_period})"
    config_id        = body.get("config_id")

    if not ticker:
        return jsonify({"ok": False, "error": "ticker required"}), 400

    # Map period strings to approximate calendar days
    _PERIOD_DAYS: Dict[str, int] = {
        "6mo": 183, "1y": 365, "2y": 730, "3y": 1095, "5y": 1825, "10y": 3650,
    }
    backtest_days = _PERIOD_DAYS.get(backtest_period, 730)

    # Backtest window starts approximately this many calendar days ago
    backtest_start = _dt.date.today() - _dt.timedelta(days=backtest_days)

    try:
        from ..services import market_data as md

        # Fetch the longer calibration history
        raw = md.fetch_price_history(ticker, period=calib_period, auto_adjust=True)
        if raw is None or len(raw) < 100:
            return jsonify({"ok": False, "error": "Insufficient price history"}), 400

        df = md.compute_indicators(raw)

        # Keep only rows BEFORE the backtest window to ensure zero overlap.
        # df.index is a DatetimeIndex from yfinance; .normalize() drops the time component.
        df_pre = df[df.index.normalize() < pd.Timestamp(backtest_start)]

        # Map scoring indicator keys → DataFrame column names for IC computation
        _IND_COLS = {
            "rsi":   "RSI",
            "macd":  "MACD",
            "bb":    "BB_Pct",
            "ma20":  "MA20",    # will be converted to dist %
            "ma50":  "MA50",
            "ma200": "MA200",
            "cross": None,      # derived: MA50 > MA200
        }

        WARMUP   = 200
        FORWARD  = 21

        if len(df_pre) <= WARMUP + FORWARD + 20:
            return jsonify({
                "ok":    False,
                "error": (
                    f"Calibration window too short. "
                    f"Only {len(df_pre)} rows found before backtest start "
                    f"({backtest_start}). Try a longer calib_period or shorter backtest_period."
                ),
            }), 400

        df_calib  = df_pre.iloc[WARMUP:].reset_index(drop=True)
        price_s   = df_calib["Close"].reset_index(drop=True)
        fwd       = price_s.shift(-FORWARD) / price_s - 1.0
        n_calib   = len(df_calib)

        ic_results: Dict[str, float] = {}

        for key in tech_sel:
            col = _IND_COLS.get(key)
            if col is None and key == "cross":
                # MA50 > MA200 cross: 1 if golden, -1 if death
                if "MA50" in df_calib.columns and "MA200" in df_calib.columns:
                    ind_s = pd.Series(
                        np.where(df_calib["MA50"].values > df_calib["MA200"].values, 1.0, -1.0),
                        dtype="float64"
                    )
                else:
                    continue
            elif col and col in df_calib.columns:
                ind_s = df_calib[col].reset_index(drop=True).astype("float64")
                # Convert MA columns to % distance from price
                if key in ("ma20", "ma50", "ma200"):
                    close_s = price_s.copy()
                    ind_s = np.where(
                        ind_s.notna() & (ind_s > 0),
                        (close_s - ind_s) / ind_s * 100,
                        np.nan,
                    )
                    ind_s = pd.Series(ind_s, dtype="float64")
            else:
                continue

            valid = pd.Series(ind_s).notna() & fwd.notna()
            if int(valid.sum()) < 20:
                continue

            ic_val = float(
                pd.Series(ind_s)[valid].rank(method="average").corr(
                    fwd[valid].rank(method="average")
                )
            )
            if not np.isnan(ic_val):
                ic_results[key] = round(ic_val, 4)

        if not ic_results:
            return jsonify({"ok": False, "error": "Could not compute IC for any indicator"}), 400

        # Convert raw IC values to multipliers:
        # Positive IC → proportional weight; negative IC → 0 (suppressed)
        # Normalised so mean positive multiplier = 1.0
        pos_ics = {k: v for k, v in ic_results.items() if v > 0}
        ic_weights: Dict[str, float] = {}
        if pos_ics:
            sum_pos = sum(pos_ics.values())
            n_pos   = len(pos_ics)
            for key in tech_sel:
                raw_ic = ic_results.get(key, 0.0)
                if raw_ic > 0:
                    ic_weights[key] = round(raw_ic / sum_pos * n_pos, 4)
                else:
                    ic_weights[key] = 0.0
        else:
            # All negative — fall back to equal weights
            for key in tech_sel:
                ic_weights[key] = 1.0

        # Record the last date in the calibration window
        last_calib_date = ""
        try:
            last_calib_date = str(df_pre.index[-1])[:10]
        except Exception:
            pass

        payload = {
            "config_id":             config_id,
            "name":                  name,
            "description": (
                f"IC-calibrated weights for {ticker}. "
                f"Calib: {calib_period} of history ending {last_calib_date} "
                f"(before backtest window of {backtest_period}). "
                f"Positive-IC indicators: {sorted(pos_ics)}"
            ),
            "ic_weights":               ic_weights,
            "ic_calibration_ticker":    ticker,
            "ic_calibration_meta": {
                "calib_period":    calib_period,
                "backtest_period": backtest_period,
                "backtest_start":  str(backtest_start),
                "n_calib_bars":    n_calib,
                "calib_end_date":  last_calib_date,
                "forward_days":    FORWARD,
                "raw_ic":          ic_results,
            },
        }

        saved = tc.save_config(payload)
        return jsonify({
            "ok":         True,
            "config":     saved,
            "ic_results": ic_results,
            "ic_weights": ic_weights,
            "calib_end_date": last_calib_date,
            "backtest_start": str(backtest_start),
            "n_calib_bars":   n_calib,
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "traceback": traceback.format_exc()}), 500
