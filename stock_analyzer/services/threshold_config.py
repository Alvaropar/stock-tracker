"""
Threshold configuration store.

Saves and loads user-customisable signal threshold configurations — the
breakpoint tables for continuous indicator scoring (RSI, BB, MA200) and the
score-to-label cutoffs.  Users tune these in the backtest page, save configs
by name, then apply a saved config when running a new analysis.

Storage: a single JSON file  `threshold_configs.json`  placed in the same
directory as settings.json so it survives app restarts and executable packaging.
"""
from __future__ import annotations

import copy
import datetime
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("app.threshold_config")

# Import the canonical defaults from scoring.py so there is exactly one source
# of truth.  scoring.py does NOT import from this module, so no circular import.
from .scoring import (
    DEFAULT_BB_BP,
    DEFAULT_MA200_DIST_BP,
    DEFAULT_RSI_MR_BP,
    DEFAULT_RSI_TREND_BP,
    DEFAULT_SCORE_THRESHOLDS,
)


# ── Built-in default ──────────────────────────────────────────────────────────

def default_config() -> Dict[str, Any]:
    """Return the built-in default ThresholdConfig as a plain dict (read-only)."""
    return {
        "config_id":         "default",
        "name":              "Default",
        "description":       "Built-in thresholds — read only",
        "created_at":        "",
        "rsi_mr_bp":         copy.deepcopy(DEFAULT_RSI_MR_BP),
        "rsi_trend_bp":      copy.deepcopy(DEFAULT_RSI_TREND_BP),
        "bb_bp":             copy.deepcopy(DEFAULT_BB_BP),
        "ma200_dist_bp":     copy.deepcopy(DEFAULT_MA200_DIST_BP),
        "score_thresholds":  copy.deepcopy(DEFAULT_SCORE_THRESHOLDS),
        # IC-derived indicator weights (populated by the calibration feature)
        "ic_weights":        {},     # {indicator_key: multiplier}
        "ic_calibration_ticker": "", # ticker this config was calibrated on
        "ic_calibration_meta":   {}, # {period, n_bars, calib_end_date}
    }


def _breakpoint_keys() -> List[str]:
    return [
        "rsi_mr_bp", "rsi_trend_bp", "bb_bp", "ma200_dist_bp",
        "score_thresholds", "ic_weights",
        "ic_calibration_ticker", "ic_calibration_meta",
    ]


# ── File path (mirrors settings.py pattern) ───────────────────────────────────

def _store_path() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "threshold_configs.json"
    return Path(__file__).resolve().parents[2] / "data" / "threshold_configs.json"


def _load_store() -> Dict[str, Any]:
    p = _store_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("Failed to read threshold_configs.json: %s", e)
    return {}


def _save_store(store: Dict[str, Any]) -> None:
    p = _store_path()
    try:
        p.write_text(json.dumps(store, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        log.error("Failed to write threshold_configs.json: %s", e)
        raise


# ── CRUD ──────────────────────────────────────────────────────────────────────

def list_configs() -> List[Dict[str, Any]]:
    """Return summary rows (no breakpoint data) for all saved configs."""
    rows = [
        {
            "config_id":   "default",
            "name":        "Default",
            "description": "Built-in thresholds — read only",
            "created_at":  "",
        }
    ]
    for cfg in _load_store().values():
        rows.append({
            "config_id":             cfg.get("config_id"),
            "name":                  cfg.get("name", ""),
            "description":           cfg.get("description", ""),
            "created_at":            cfg.get("created_at", ""),
            # Include IC fields so the Step 4 weights UI can filter and group by ticker
            "ic_calibration_ticker": cfg.get("ic_calibration_ticker", ""),
            "ic_weights":            cfg.get("ic_weights", {}),
            "ic_calibration_meta":   cfg.get("ic_calibration_meta", {}),
        })
    return rows


def get_config(config_id: str) -> Optional[Dict[str, Any]]:
    """Load a full config by ID, or None if not found."""
    if config_id == "default":
        return default_config()
    return _load_store().get(config_id)


def save_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Upsert a threshold config.

    Assigns a new UUID-based ID if none is present.  Unknown keys in *payload*
    are ignored; missing breakpoint keys fall back to defaults, so older saved
    configs continue to work after a new breakpoint is added to scoring.py.
    """
    store = _load_store()

    config_id = payload.get("config_id")
    if not config_id or config_id == "default":
        config_id = str(uuid.uuid4())[:8]

    merged = copy.deepcopy(default_config())
    merged.update({k: payload[k] for k in _breakpoint_keys() if k in payload})
    merged["config_id"]   = config_id
    merged["name"]        = payload.get("name", "Unnamed")
    merged["description"] = payload.get("description", "")
    merged["created_at"]  = (
        payload.get("created_at")
        or datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    )

    store[config_id] = merged
    _save_store(store)
    return merged


def delete_config(config_id: str) -> bool:
    if config_id == "default":
        return False
    store = _load_store()
    if config_id not in store:
        return False
    del store[config_id]
    _save_store(store)
    return True
