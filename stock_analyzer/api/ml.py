"""
ML Classifier API endpoints.

POST /api/ml/train          → train model for a ticker (or universe)
GET  /api/ml/status         → backend info, GPU availability
GET  /api/ml/models         → list cached models
POST /api/ml/models/clear   → clear all cached models

Dashboard endpoints:
POST /api/ml/dashboard/train          → train with full metrics + streaming
GET  /api/ml/dashboard/stream/<id>    → SSE epoch-by-epoch progress
POST /api/ml/dashboard/predict-series → timeseries predictions for charting
POST /api/ml/dashboard/data-info      → date ranges, row counts, splits
"""
from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from typing import Dict

from flask import Blueprint, Response, jsonify, request, stream_with_context

bp = Blueprint("ml", __name__, url_prefix="/api/ml")

# ── Task store for SSE streaming ────────────────────────────────────────────
_tasks: Dict[str, Dict] = {}
_tasks_lock = threading.Lock()


def _push(task_id: str, event: Dict):
    with _tasks_lock:
        if task_id in _tasks:
            _tasks[task_id]["events"].append(event)


def _finish(task_id: str, result: Dict):
    with _tasks_lock:
        _tasks[task_id]["status"] = "done"
        _tasks[task_id]["result"] = result
        _tasks[task_id]["events"].append({"type": "complete"})


def _fail(task_id: str, error: str):
    with _tasks_lock:
        _tasks[task_id]["status"] = "error"
        _tasks[task_id]["error"] = error
        _tasks[task_id]["events"].append({"type": "error", "message": error})


# ── In-memory store for most-recently trained engine per ticker ──────────────
# Populated after successful training so backtest/save don't need to retrain.
_trained_engines: Dict[str, object] = {}
_trained_results: Dict[str, Dict] = {}
_trained_dfs: Dict[str, object] = {}
_trained_result_objs: Dict[str, object] = {}


# ── Existing endpoints ──────────────────────────────────────────────────────

@bp.route("/status")
def status():
    """Return ML backend availability and GPU info."""
    info = {
        "sklearn_available": False,
        "pytorch_available": False,
        "cuda_available": False,
        "gpu_name": None,
        "recommended_backend": "sklearn",
    }
    try:
        import sklearn  # noqa: F401
        info["sklearn_available"] = True
    except ImportError:
        pass

    try:
        import torch
        info["pytorch_available"] = True
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["recommended_backend"] = "pytorch"
    except ImportError:
        pass

    from ..services.ml_engine import get_available_backend
    info["recommended_backend"] = get_available_backend()

    return jsonify(info)


@bp.route("/models")
def models():
    """List all cached ML models."""
    from ..services.ml_engine import list_cached_models
    return jsonify(list_cached_models())


@bp.route("/models/clear", methods=["POST"])
def clear_models():
    """Clear all cached ML models."""
    from ..services.ml_engine import clear_cached_models
    count = clear_cached_models()
    return jsonify({"cleared": count})


# ── Dashboard: Train with streaming ─────────────────────────────────────────

@bp.route("/dashboard/train", methods=["POST"])
def dashboard_train():
    """
    Train a model for one ticker with full metrics.
    Returns task_id for SSE streaming of epoch progress.
    """
    body = request.get_json(force=True)
    ticker = body.get("ticker", "").upper()
    if not ticker:
        return jsonify({"error": "ticker required"}), 400

    task_id = str(uuid.uuid4())[:8]
    with _tasks_lock:
        _tasks[task_id] = {
            "status": "running", "events": [], "result": None, "error": None,
        }

    def _run():
        try:
            from ..services.ml_engine import MLEngine, MLConfig, clear_cached_models
            import yfinance as yf
            from ..services import market_data as md

            # ── Parse validation mode ──────────────────────────────────────
            val_mode      = body.get("validation_mode", "walkforward")
            start_date    = body.get("start_date")
            end_date      = body.get("end_date")
            train_start   = body.get("train_start")
            train_end     = body.get("train_end")
            test_start    = body.get("test_start")
            test_end      = body.get("test_end")
            period        = body.get("period", body.get("training_period", "5y"))
            cv_splits     = int(body.get("cv_splits", 5))
            wf_gap        = int(body.get("wf_gap", 5))
            wf_window     = body.get("wf_window", "expanding")
            wf_rolling_sz = body.get("wf_rolling_size")
            if wf_rolling_sz is not None:
                wf_rolling_sz = int(wf_rolling_sz)
            holdout_months = int(body.get("holdout_months", 0))

            # ── Build MLConfig ─────────────────────────────────────────────
            model_type = body.get("model_type", "lightgbm")
            cfg = MLConfig(
                model_type=model_type,
                backend="auto" if model_type == "mlp" else body.get("backend", "auto"),
                training_period=period,
                forward_horizon=int(body.get("forward_horizon", 21)),
                feature_set=body.get("feature_set", "full"),
                n_trees=int(body.get("n_trees", 300)),
                max_depth=int(body.get("max_depth", 5)),
                num_leaves=int(body.get("num_leaves", 31)),
                cv_splits=cv_splits,
                wf_gap=wf_gap if wf_gap > 5 else 21,  # enforce minimum gap
                wf_window=wf_window,
                wf_rolling_size=wf_rolling_sz,
                epochs=int(body.get("epochs", 100)),
                dropout=float(body.get("dropout", 0.3)),
                target_annual_vol=float(body.get("target_vol", 0.15)),
                max_drawdown_trigger=float(body.get("max_dd_trigger", 0.15)),
                wf_trade_cost=float(body.get("wf_trade_cost", 0.001)),
            )

            # Clear cache files for this ticker (hex-hash names only, preserve registry _v* versions)
            import re as _re
            from pathlib import Path
            model_dir = Path(__file__).resolve().parent.parent.parent / "ml_models"
            _cache_pat = _re.compile(rf"^{_re.escape(ticker)}_[0-9a-f]{{8,}}\.")
            for f in model_dir.glob(f"{ticker}_*"):
                if _cache_pat.match(f.name):
                    f.unlink(missing_ok=True)

            engine = MLEngine(cfg)

            # Wire epoch callback to SSE
            def on_epoch(entry):
                _push(task_id, {"type": "epoch", **entry})

            engine._epoch_callback = on_epoch

            _push(task_id, {"type": "status", "message": f"Fetching data for {ticker}..."})

            # ── Fetch data based on validation mode ────────────────────────
            if val_mode == "chrono":
                # Fetch train + test windows, concatenate for indicators
                hist_train = yf.Ticker(ticker).history(start=train_start, end=train_end, auto_adjust=True)
                hist_test  = yf.Ticker(ticker).history(start=test_start,  end=test_end,  auto_adjust=True)
                if hist_train.empty or hist_test.empty:
                    raise ValueError("No data for specified train or test range")
                import pandas as pd
                hist_all = pd.concat([hist_train, hist_test])
                hist_all = hist_all[~hist_all.index.duplicated(keep="last")].sort_index()
                df_all   = md.compute_indicators(hist_all)
                # Mark the chrono split boundary on the engine
                engine._chrono_train_end = train_end
                engine._chrono_test_start = test_start
            elif start_date and end_date:
                hist = yf.Ticker(ticker).history(start=start_date, end=end_date, auto_adjust=True)
                if hist is None or hist.empty:
                    raise ValueError(f"No data for {ticker} in {start_date} – {end_date}")
                df_all = md.compute_indicators(hist)
            else:
                # Fetch extra warmup history (200+ days) for indicator computation,
                # then trim to the requested period so walk-forward folds are correct.
                _PERIOD_TO_DAYS = {"1y": 365, "2y": 730, "3y": 1095, "5y": 1825, "10y": 3650}
                _target_days = _PERIOD_TO_DAYS.get(period, 1825)
                _fetch_days = _target_days + 250  # 250 trading-day warmup for MA200 etc.
                import datetime as _dt
                _end_dt = _dt.date.today()
                _start_dt = _end_dt - _dt.timedelta(days=_fetch_days)
                hist = yf.Ticker(ticker).history(start=str(_start_dt), end=str(_end_dt), auto_adjust=True)
                if hist is None or hist.empty or len(hist) < 100:
                    raise ValueError(f"Insufficient data for {ticker}")
                df_all_full = md.compute_indicators(hist)
                # Trim to requested period window (drop warmup rows)
                _trim_start = _end_dt - _dt.timedelta(days=_target_days)
                import pandas as _pd
                df_all = df_all_full[df_all_full.index.date >= _trim_start]
                if len(df_all) < 50:
                    df_all = df_all_full  # fallback: use all data

            if len(df_all) < 100:
                raise ValueError(f"Too few rows after computing indicators: {len(df_all)}")

            # ── Holdout split: reserve last N months as out-of-sample ──────
            holdout_start_str = None
            holdout_end_str   = None
            df_holdout        = None
            if holdout_months > 0:
                import datetime as _dt2
                _last_date = df_all.index[-1]
                _last_dt   = _last_date.date() if hasattr(_last_date, "date") else _last_date
                import pandas as _pd2
                _holdout_start = (_pd2.Timestamp(_last_dt) - _pd2.DateOffset(months=holdout_months)).date()
                df_train_only  = df_all[df_all.index.date < _holdout_start]
                df_holdout     = df_all[df_all.index.date >= _holdout_start]
                if len(df_train_only) < 50:
                    _push(task_id, {"type": "status", "message": "Warning: holdout leaves too few training rows — ignoring holdout"})
                else:
                    holdout_start_str = str(_holdout_start)
                    holdout_end_str   = str(_last_dt)
                    df_all = df_train_only

            _push(task_id, {
                "type": "status",
                "message": f"Training {model_type} on {len(df_all)} rows, {len(cfg.feature_names())} features "
                           f"({'chrono split' if val_mode == 'chrono' else f'walk-forward {cv_splits} folds'})"
                           f"{f'  |  holdout: {holdout_start_str} → {holdout_end_str}' if holdout_start_str else ''}...",
            })

            result = engine.train(ticker, df=df_all)

            # Get timeseries predictions
            ts = engine.predict_timeseries(df=df_all)

            final = result.to_dict()
            final["timeseries"] = ts
            if holdout_start_str:
                final["holdout_start"] = holdout_start_str
                final["holdout_end"]   = holdout_end_str

            # Store engine for backtest/save without retraining
            _trained_engines[ticker] = engine
            _trained_results[ticker] = final
            # Store the full df (including holdout) for backtest so the date range is accessible
            _trained_dfs[ticker] = df_holdout if df_holdout is not None else df_all
            _trained_result_objs[ticker] = result

            _finish(task_id, final)

        except Exception as e:
            _fail(task_id, traceback.format_exc())

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    return jsonify({"task_id": task_id})


@bp.route("/dashboard/stream/<task_id>")
def dashboard_stream(task_id):
    """SSE stream of training progress."""
    if task_id not in _tasks:
        return jsonify({"error": "Not found"}), 404

    def generate():
        last = 0
        while True:
            with _tasks_lock:
                task = _tasks.get(task_id, {})
                events = task.get("events", [])
                task_status = task.get("status", "running")

            for ev in events[last:]:
                yield f"data: {json.dumps(ev)}\n\n"
            last = len(events)

            if task_status in ("done", "error"):
                break
            time.sleep(0.2)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@bp.route("/dashboard/result/<task_id>")
def dashboard_result(task_id):
    """Get final training result after completion."""
    with _tasks_lock:
        task = _tasks.get(task_id)
    if not task:
        return jsonify({"error": "Not found"}), 404
    if task["status"] == "running":
        return jsonify({"status": "running"}), 202
    if task["status"] == "error":
        return jsonify({"status": "error", "error": task["error"]}), 500
    return jsonify({"status": "done", "result": task["result"]})


@bp.route("/dashboard/predict-series", methods=["POST"])
def dashboard_predict_series():
    """
    Run predictions across entire history for charting.
    Requires a trained (cached) model.
    """
    body = request.get_json(force=True)
    ticker = body.get("ticker", "").upper()
    if not ticker:
        return jsonify({"error": "ticker required"}), 400

    try:
        from ..services.ml_engine import MLEngine, MLConfig
        from ..services import market_data as md
        import yfinance as yf

        cfg = MLConfig(
            backend=body.get("backend", "auto"),
            feature_set=body.get("feature_set", "full"),
        )
        engine = MLEngine(cfg)

        # Load cached model
        cached = engine._load_cached(ticker)
        if cached is None:
            return jsonify({"error": f"No trained model for {ticker}. Train first."}), 404
        engine._models = cached

        # Fetch data
        period = body.get("period", "5y")
        hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        df = md.compute_indicators(hist)

        ts = engine.predict_timeseries(df=df)
        return jsonify(ts)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/dashboard/data-info", methods=["POST"])
def dashboard_data_info():
    """Return data stats for a ticker: row count, date range, indicators available."""
    body = request.get_json(force=True)
    ticker = body.get("ticker", "").upper()
    if not ticker:
        return jsonify({"error": "ticker required"}), 400

    try:
        import yfinance as yf
        import pandas as pd
        from ..services import market_data as md

        val_mode   = body.get("validation_mode", "walkforward")
        period     = body.get("period", "5y")
        start_date = body.get("start_date")
        end_date   = body.get("end_date")
        train_start = body.get("train_start")
        train_end   = body.get("train_end")
        test_start  = body.get("test_start")
        test_end    = body.get("test_end")

        if val_mode == "chrono" and train_start and test_end:
            h_tr = yf.Ticker(ticker).history(start=train_start, end=train_end, auto_adjust=True)
            h_te = yf.Ticker(ticker).history(start=test_start,  end=test_end,  auto_adjust=True)
            if h_tr.empty or h_te.empty:
                return jsonify({"error": "No data for specified date ranges"}), 404
            hist = pd.concat([h_tr, h_te])
            hist = hist[~hist.index.duplicated(keep="last")].sort_index()
        elif start_date and end_date:
            hist = yf.Ticker(ticker).history(start=start_date, end=end_date, auto_adjust=True)
        else:
            hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)

        if hist is None or hist.empty:
            return jsonify({"error": f"No data for {ticker}"}), 404

        df = md.compute_indicators(hist)
        _iso = lambda d: d.isoformat()[:10] if hasattr(d, 'isoformat') else str(d)
        n = len(df)

        # Determine split boundary for display
        if val_mode == "chrono" and test_start:
            date_strs = [str(d)[:10] for d in df.index]
            split = next((i for i, d in enumerate(date_strs) if d >= test_start), int(n * 0.8))
        else:
            split = int(n * 0.8)

        cv_splits = int(body.get("cv_splits", 5))
        fold_size = n // (cv_splits + 1) if val_mode == "walkforward" else None

        info = {
            "ticker": ticker,
            "validation_mode": val_mode,
            "total_rows": n,
            "date_range": [_iso(df.index[0]), _iso(df.index[-1])],
            "train_rows": split,
            "test_rows": n - split,
            "train_range": [_iso(df.index[0]), _iso(df.index[split - 1])],
            "test_range": [_iso(df.index[split]), _iso(df.index[-1])],
            "cv_splits": cv_splits,
            "approx_fold_size": fold_size,
            "columns_available": sorted(df.columns.tolist()),
            "price_range": [round(float(df["Close"].min()), 2), round(float(df["Close"].max()), 2)],
        }
        return jsonify(info)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/dashboard/backtest", methods=["POST"])
def dashboard_backtest():
    """Run a full backtest on the most-recently trained model for a ticker."""
    body = request.get_json(force=True)
    ticker = body.get("ticker", "").upper()
    if not ticker:
        return jsonify({"error": "ticker required"}), 400

    engine = _trained_engines.get(ticker)
    df = _trained_dfs.get(ticker)
    if engine is None or df is None:
        return jsonify({"error": f"No trained model in memory for {ticker}. Train first."}), 400

    try:
        from ..services.backtest import Backtester, BacktestConfig
        import pandas as _pd

        _bt_entry_thr = float(body["entry_threshold"]) if body.get("entry_threshold") not in (None, "") else None
        _bt_exit_thr  = float(body["exit_threshold"])  if body.get("exit_threshold")  not in (None, "") else None

        bt_cfg = BacktestConfig(
            initial_capital=float(body.get("initial_capital", 100_000)),
            commission_pct=float(body.get("commission_pct", 0.001)),
            slippage_pct=float(body.get("slippage_pct", 0.0005)),
            stop_loss_pct=float(body.get("stop_loss_pct")) if body.get("stop_loss_pct") else 0.08,
            take_profit_pct=float(body.get("take_profit_pct")) if body.get("take_profit_pct") else None,
            entry_threshold=_bt_entry_thr,
            exit_threshold=_bt_exit_thr,
            target_annual_vol=float(body.get("target_vol", 0.15)),
            max_drawdown_trigger=float(body.get("max_dd_trigger", 0.15)),
        )

        # Apply optional date range filter
        start_date = body.get("start_date")
        end_date   = body.get("end_date")

        if start_date or end_date:
            # Check if the requested range is covered by the stored df
            df_start = str(df.index[0])[:10]
            df_end   = str(df.index[-1])[:10]
            need_fetch = (
                (start_date and start_date < df_start) or
                (end_date   and end_date   > df_end)   or
                # Also re-fetch if the slice would yield < 10 rows
                len(df.loc[start_date:end_date] if (start_date and end_date) else
                    df.loc[start_date:] if start_date else df.loc[:end_date]) < 10
            )

            if need_fetch:
                # Fetch fresh data for the requested window (+250 days warmup for indicators)
                import yfinance as _yf
                import datetime as _dt
                from ..services import market_data as _md
                _ws = _dt.date.fromisoformat(start_date) if start_date else _dt.date.fromisoformat(df_start)
                _we = _dt.date.fromisoformat(end_date)   if end_date   else _dt.date.today()
                _fetch_start = _ws - _dt.timedelta(days=250)
                hist = _yf.Ticker(ticker).history(start=str(_fetch_start), end=str(_we + _dt.timedelta(days=1)), auto_adjust=True)
                if hist is None or hist.empty:
                    return jsonify({"error": f"No market data available for {ticker} in {start_date}–{end_date}"}), 400
                bt_df = _md.compute_indicators(hist)
            else:
                bt_df = df.copy()

            # Slice to requested range
            if start_date and end_date:
                bt_df = bt_df.loc[start_date:end_date]
            elif start_date:
                bt_df = bt_df.loc[start_date:]
            elif end_date:
                bt_df = bt_df.loc[:end_date]
        else:
            bt_df = df.copy()

        if len(bt_df) < 10:
            return jsonify({"error": f"Too few rows in selected date range ({len(bt_df)}). Widen the period."}), 400

        bt = Backtester(bt_cfg)
        result = bt.run(engine, bt_df)

        def _sanitize(obj):
            """Replace inf/nan with null and coerce numpy scalars to Python primitives."""
            import math
            import numpy as _np
            if isinstance(obj, _np.integer):
                return int(obj)
            if isinstance(obj, _np.floating):
                v = float(obj)
                return None if (math.isinf(v) or math.isnan(v)) else v
            if isinstance(obj, _np.ndarray):
                return [_sanitize(x) for x in obj.tolist()]
            if isinstance(obj, float):
                return None if (math.isinf(obj) or math.isnan(obj)) else obj
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_sanitize(v) for v in obj]
            return obj

        return jsonify({"status": "ok", "result": _sanitize(result.to_dict())})
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@bp.route("/dashboard/registry", methods=["GET"])
def dashboard_registry():
    """List all saved model versions, optionally filtered by ticker."""
    ticker = request.args.get("ticker")
    try:
        from ..services.ml_engine import get_registry
        reg = get_registry()
        entries = reg.list(ticker=ticker if ticker else None)
        return jsonify({"versions": entries})
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@bp.route("/dashboard/registry/save", methods=["POST"])
def dashboard_registry_save():
    """Save the currently trained model to the versioned registry."""
    body = request.get_json(force=True)
    ticker = body.get("ticker", "").upper()
    notes = body.get("notes", "")
    if not ticker:
        return jsonify({"error": "ticker required"}), 400

    engine = _trained_engines.get(ticker)
    df = _trained_dfs.get(ticker)
    if engine is None:
        return jsonify({"error": f"No trained model in memory for {ticker}. Train first."}), 400

    try:
        from ..services.ml_engine import get_registry, TrainResult
        from dataclasses import asdict
        reg = get_registry()
        raw = _trained_results.get(ticker, {})

        # Reconstruct a minimal TrainResult from stored dict
        # We need the actual TrainResult object - store it separately
        tr = _trained_result_objs.get(ticker)
        if tr is None:
            return jsonify({"error": "Training result object not available. Retrain the model."}), 400

        mv = reg.save(engine, tr, df=df, notes=notes)
        return jsonify({"status": "ok", "version_id": mv.version_id, "version": mv.to_dict()})
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@bp.route("/dashboard/registry/load", methods=["POST"])
def dashboard_registry_load():
    """Load a saved model version into memory so it can be backtested."""
    body = request.get_json(force=True)
    version_id = body.get("version_id", "")
    if not version_id:
        return jsonify({"error": "version_id required"}), 400

    try:
        from ..services.ml_engine import get_registry, MLEngine, MLConfig
        reg = get_registry()

        if version_id not in reg._registry:
            return jsonify({"error": f"Version '{version_id}' not found"}), 404

        entry = reg._registry[version_id]
        ticker = entry["ticker"]
        model_type = entry["model_type"]
        cfg_dict = entry.get("config", {})
        from ..services.ml_engine import MLConfig as _MC
        valid_fields = set(_MC.__dataclass_fields__.keys())
        cfg = _MC(**{k: v for k, v in cfg_dict.items() if k in valid_fields})

        engine = MLEngine(cfg)
        engine._models = reg.load(version_id)
        engine._ticker = ticker
        engine._policy_opt = entry.get("policy_opt")
        engine._range_regime_sharpe = entry.get("range_regime_sharpe")
        engine._readiness = entry.get("readiness")

        import yfinance as yf
        from ..services import market_data as md
        period = cfg.training_period or "5y"
        hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if hist is None or hist.empty:
            return jsonify({"error": f"Could not fetch data for {ticker}"}), 400
        df = md.compute_indicators(hist)

        _trained_engines[ticker] = engine
        _trained_dfs[ticker] = df

        # Reconstruct a minimal TrainResult so Save works after Load
        from ..services.ml_engine import TrainResult as _TR
        tr = _TR(
            ticker=ticker,
            backend=model_type,
            model_type=model_type,
            n_samples=entry.get("n_samples", 0),
            n_features=entry.get("n_features", 0),
            regime_accuracy=entry.get("regime_accuracy", 0.0),
            regime_f1=entry.get("regime_f1", {}),
            entry_mae=entry.get("entry_mae", 0.0),
            exit_mae=entry.get("exit_mae", 0.0),
            feature_importances={},
            cv_scores=[],
            training_time_s=0.0,
        )
        _trained_result_objs[ticker] = tr

        return jsonify({
            "status": "ok",
            "ticker": ticker,
            "version_id": version_id,
            "model_type": model_type,
            "train_period": entry.get("train_period"),
            "metrics": {
                "regime_accuracy": entry.get("regime_accuracy"),
                "sharpe_ratio": entry.get("sharpe_ratio"),
                "max_drawdown": entry.get("max_drawdown"),
                "cagr": entry.get("cagr"),
            }
        })
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@bp.route("/dashboard/ready", methods=["GET"])
def dashboard_ready():
    """Return which tickers have a trained ML Lab model in memory (ready for analysis)."""
    return jsonify({"tickers": list(_trained_engines.keys())})


@bp.route("/dashboard/infer", methods=["POST"])
def dashboard_infer():
    """
    Run inference with the in-memory trained model on the latest available data.
    Returns current regime, entry/exit scores, decision, and raw indicator values
    for the most recent trading day.
    """
    body = request.get_json(force=True)
    ticker = body.get("ticker", "").upper()
    if not ticker:
        return jsonify({"error": "ticker required"}), 400

    engine = _trained_engines.get(ticker)
    if engine is None:
        return jsonify({"error": f"No trained model for {ticker}. Train or load first."}), 400

    try:
        import yfinance as _yf
        import datetime as _dt
        from ..services import market_data as _md

        # Always fetch fresh data so we get today's bar
        _end = _dt.date.today() + _dt.timedelta(days=1)
        _start = _end - _dt.timedelta(days=300)   # 300-day window for indicator warmup
        hist = _yf.Ticker(ticker).history(start=str(_start), end=str(_end), auto_adjust=True)
        if hist is None or hist.empty:
            return jsonify({"error": f"No recent data for {ticker}"}), 400

        df_fresh = _md.compute_indicators(hist)
        if df_fresh.empty:
            return jsonify({"error": "Could not compute indicators"}), 400

        # Fetch fundamentals for quality overlay display (best-effort — not in ML features)
        _fund_quality_score = None
        try:
            from ..services.ml_engine import compute_fund_quality_score as _cfqs
            _info = _yf.Ticker(ticker).info
            _fundamentals = {
                "debt_eq":  _info.get("debtToEquity"),
                "curr_ratio": _info.get("currentRatio"),
                "roe":      (_info.get("returnOnEquity") or 0.0) * 100 or None,
                "net_mgn":  (_info.get("profitMargins") or 0.0) * 100 or None,
            }
            _fund_quality_score = _cfqs(_fundamentals)
        except Exception:
            pass  # fundamentals unavailable — shown as N/A in UI

        pred = engine.predict_from_df(df_fresh)

        # Pull latest indicator values from the last row for the Excel export
        last = df_fresh.iloc[-1]
        last_date = str(df_fresh.index[-1])[:10]

        def _f(col, default=None):
            v = last.get(col, default)
            return None if (v is None or (hasattr(v, '__float__') and __import__('math').isnan(float(v)))) else float(v)

        indicators = {
            "date":          last_date,
            "close":         _f("Close"),
            "rsi":           _f("RSI"),
            "bb_pct":        _f("BB_Pct"),
            "macd_hist":     _f("MACD_Hist"),
            "atr_pct":       _f("ATR_Pct"),
            "adx":           _f("ADX"),
            "vol_ratio":     _f("Vol_Ratio"),
            "obv_slope":     _f("obv_slope"),
            "ma50":          _f("MA50"),
            "ma200":         _f("MA200"),
        }

        result = {
            "ticker":             ticker,
            "as_of":              last_date,
            "regime":             pred.regime,
            "regime_confidence":  pred.regime_confidence,
            "regime_probs":       pred.regime_probs,
            "entry_score":        pred.entry_score,
            "exit_score":         pred.exit_score,
            "ml_signal":          pred.ml_signal,
            "decision":           pred.decision,
            "uncertainty":        pred.uncertainty,
            "indicators":         indicators,
            "fund_quality_score": round(float(_fund_quality_score), 3) if _fund_quality_score is not None else None,
            "range_gate_active":  bool(getattr(engine, "_range_regime_sharpe", 0.0) is not None
                                       and (getattr(engine, "_range_regime_sharpe", 0.0) or 0.0) < 0),
        }
        return jsonify({"status": "ok", "result": result})
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@bp.route("/dashboard/registry/<version_id>", methods=["DELETE"])
def dashboard_registry_delete(version_id):
    """Delete a model version from the registry."""
    delete_weights = request.args.get("weights", "false").lower() == "true"
    try:
        from ..services.ml_engine import get_registry
        reg = get_registry()
        reg.delete(version_id, delete_weights=delete_weights)
        return jsonify({"status": "ok", "deleted": version_id})
    except KeyError:
        return jsonify({"error": f"Version '{version_id}' not found"}), 404
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500
