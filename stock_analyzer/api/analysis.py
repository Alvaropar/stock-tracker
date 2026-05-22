"""
Analysis pipeline API.

POST /api/analysis/start  → spawn background task, return {task_id}
GET  /api/analysis/stream/<task_id> → SSE progress stream
GET  /api/analysis/results/<task_id> → final results JSON
"""
from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from typing import Any, Dict, List, Optional

from flask import Blueprint, Response, jsonify, request, stream_with_context

from ..services import scoring as sc

bp = Blueprint("analysis", __name__, url_prefix="/api/analysis")

# In-memory task store  {task_id: {"status", "events", "results", "error"}}
_tasks: Dict[str, Dict] = {}
_tasks_lock = threading.Lock()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _push(task_id: str, event: Dict):
    with _tasks_lock:
        _tasks[task_id]["events"].append(event)


def _finish(task_id: str, results: List):
    with _tasks_lock:
        _tasks[task_id]["status"]  = "done"
        _tasks[task_id]["results"] = results
        _tasks[task_id]["events"].append({"type": "complete"})


def _fail(task_id: str, error: str):
    with _tasks_lock:
        _tasks[task_id]["status"] = "error"
        _tasks[task_id]["error"]  = error
        _tasks[task_id]["events"].append({"type": "error", "message": error})


def _fmt_result(ticker: str, asset: Dict, data: Dict,
                tech: float, fund: float, sent_score: Optional[float],
                sent_obj, weights: Dict,
                market_ctx: Optional[Dict] = None,
                ml_pred=None,
                score_thresholds: Optional[Dict] = None) -> Dict:
    """Format one asset's analysis into a JSON-serialisable dict."""
    latest = data.get("latest", {})
    fund_d = data.get("fundamentals", {})
    meta   = data.get("meta", {})
    cur    = meta.get("cur", asset.get("currency", "USD"))
    sym    = "€" if cur == "EUR" else "$"
    rs     = data.get("rs", {})

    def v(k): return latest.get(k)

    ma50  = v("MA50")
    ma200 = v("MA200")
    close = v("Close")
    weekly    = data.get("weekly", {})
    checklist = data.get("checklist", {})

    # ── Regime-conditioned overall score ───────────────────────────────────
    trend_stage = latest.get("Trend_Stage")
    mkt_regime  = latest.get("Mkt_Regime")
    regime_chg  = latest.get("Regime_Chg")
    vol_regime  = latest.get("Vol_Regime")

    # v4: Composite momentum + risk scores
    _macd_bull = (v("MACD") > v("MACD_Sig")
                  if v("MACD") is not None and v("MACD_Sig") is not None
                  else None)
    momentum_score = sc.compute_momentum_score(
        adx=v("ADX"),
        rs_1m=rs.get("rs_1m"), rs_55d=rs.get("rs_55d"), rs_3m=rs.get("rs_3m"),
        vol_ratio=v("Vol_Ratio"), macd_bull=_macd_bull,
    )
    risk_score = sc.compute_risk_score(
        trend_ext=v("Trend_Ext"), atr_pct=v("ATR_Pct"),
    )

    # v4.1: Oversold dip detection
    _target_px  = fund_d.get("target_px")
    _target_gap = (
        round((_target_px - close) / close * 100, 1)
        if _target_px and close and close > 0 else None
    )
    dip_score = sc.compute_dip_score(
        rsi=v("RSI"), fund_score=fund,
        vol_ratio=v("Vol_Ratio"),
        mkt_regime=mkt_regime, regime_chg=regime_chg,
        bb_pct=v("BB_Pct"),
        target_gap=_target_gap,
        n_analysts=fund_d.get("n_analysts"),
    )

    overall, signal, css = sc.compute_overall_score(
        tech, fund, sent_score, weights,
        vol_ratio=v("Vol_Ratio"),
        rs_1m=rs.get("rs_1m"), rs_55d=rs.get("rs_55d"), rs_3m=rs.get("rs_3m"),
        atr_pct=v("ATR_Pct"),
        vol_pctl=v("Vol_Pctl"),
        spy_trend_bull=market_ctx.get("spy_trend_bull") if market_ctx else None,
        trend_stage=trend_stage,
        mkt_regime=mkt_regime,
        regime_chg=regime_chg,
        momentum_score=momentum_score,
        risk_score=risk_score,
        dip_score=dip_score,
        price=close,
        target_px=_target_px,
        rec_mean=fund_d.get("rec_mean"),
        n_analysts=fund_d.get("n_analysts"),
        thresholds=score_thresholds,
    )

    # ── Context-aware signal enrichment ──────────────────────────────────
    regime = latest.get("Regime", "NEUTRAL")
    ctx_label, ctx_css, ctx_hint = sc.contextual_signal(
        signal, css, overall,
        regime=regime,
        trend_stage=trend_stage,
        mkt_regime=mkt_regime,
        regime_chg=regime_chg,
        rsi=v("RSI"),
        vol_regime=vol_regime,
        adx=v("ADX"),
        vol_ratio=v("Vol_Ratio"),
        rs_1m=rs.get("rs_1m"),
        rs_55d=rs.get("rs_55d"),
        rs_3m=rs.get("rs_3m"),
        atr_pct=v("ATR_Pct"),
        momentum_score=momentum_score,
        risk_score=risk_score,
        dip_score=dip_score,
        fund_score=fund,
        target_gap=_target_gap,
    )

    # ── Volatility-adjusted confidence ───────────────────────────────────
    base_confidence = checklist.get("confidence")
    adj_confidence = sc.compute_confidence_adjustment(
        base_confidence, vol_regime, trend_stage, mkt_regime,
        risk_score=risk_score, momentum_score=momentum_score,
        dip_score=dip_score,
    )

    return {
        "ticker":        ticker,
        "name":          asset.get("name", ticker),
        "sector":        asset.get("sector", meta.get("sector", "")),
        "currency":      cur,
        "symbol":        sym,
        "price":         round(close, 4) if close else None,
        "ret_1d":        v("Ret_1D"),
        "ret_1w":        v("Ret_5D"),
        "ret_1m":        v("Ret_21D"),
        "ret_3m":        v("Ret_63D"),
        "w52_pct":       v("W52_Pct"),
        "w52_hi":        v("W52_Hi"),
        "w52_lo":        v("W52_Lo"),
        "rsi":           v("RSI"),
        "ma_cross":      ("golden" if ma50 and ma200 and ma50 > ma200
                          else "death" if ma50 and ma200 else None),
        "macd_bull":     (v("MACD") > v("MACD_Sig")
                          if v("MACD") is not None and v("MACD_Sig") is not None
                          else None),
        "bb_pct":        v("BB_Pct"),
        "vol_ratio":     v("Vol_Ratio"),
        "atr":           v("ATR"),
        # ── New regime + quant indicators ──────────────────────
        "adx":           v("ADX"),
        "regime":        latest.get("Regime", "NEUTRAL"),
        "atr_pct":       v("ATR_Pct"),
        "vol_pctl":      v("Vol_Pctl"),
        "rs_1m":         rs.get("rs_1m"),
        "rs_55d":        rs.get("rs_55d"),
        "rs_3m":         rs.get("rs_3m"),
        # indicators
        "ma20":          v("MA20"),
        "ma50":          ma50,
        "ma200":         ma200,
        "macd":          v("MACD"),
        "macd_sig":      v("MACD_Sig"),
        # ── Buying checklist & Elder Impulse ──────────────
        "elder_d":       latest.get("Elder_D"),
        "elder_w":       weekly.get("elder_w"),
        "confidence":    base_confidence,
        "adj_confidence": adj_confidence,
        "checklist":     checklist.get("checks", []),
        "chk_passed":    checklist.get("passed", 0),
        "chk_total":     checklist.get("total", 0),
        # ── New quant fields ─────────────────────────────────────
        "trend_ext":     v("Trend_Ext"),
        "trend_stage":   trend_stage,
        "vol_regime":    vol_regime,
        "mkt_regime":    mkt_regime,
        "regime_chg":    regime_chg if regime_chg != "NONE" else None,
        "momentum_score": momentum_score,
        "risk_score":    risk_score,
        "dip_score":     dip_score,
        # scores
        "tech_score":    tech,
        "fund_score":    fund,
        "sent_score":    sent_score,
        "overall_score": overall,
        "signal":        signal,
        "signal_css":    css,
        "ctx_signal":    ctx_label,
        "ctx_hint":      ctx_hint,
        "raw_score":     data.get("score"),     # -9..+9 tech score for Excel
        "raw_sig":       data.get("sig"),        # "BUY" etc. for Excel
        # fundamentals
        "pe_trail":      fund_d.get("pe_trail"),
        "pe_fwd":        fund_d.get("pe_fwd"),
        "peg":           fund_d.get("peg"),
        "pb":            fund_d.get("pb"),
        "gross_mgn":     fund_d.get("gross_mgn"),
        "op_mgn":        fund_d.get("op_mgn"),
        "net_mgn":       fund_d.get("net_mgn"),
        "roe":           fund_d.get("roe"),
        "roa":           fund_d.get("roa"),
        "rev_growth":    fund_d.get("rev_growth"),
        "eps_growth":    fund_d.get("eps_growth"),
        "debt_eq":       fund_d.get("debt_eq"),
        "curr_ratio":    fund_d.get("curr_ratio"),
        "quick_ratio":   fund_d.get("quick_ratio"),
        "fcf":           fund_d.get("fcf"),
        "mkt_cap":       fund_d.get("mkt_cap"),
        "beta":          fund_d.get("beta"),
        "div_yield":     fund_d.get("div_yield"),
        "target_px":     fund_d.get("target_px"),
        "rec_mean":      fund_d.get("rec_mean"),
        "n_analysts":    fund_d.get("n_analysts"),
        "short_float":   fund_d.get("short_float"),
        "ev_ebitda":     fund_d.get("ev_ebitda"),
        "ps":            fund_d.get("ps"),
        # sentiment
        "sent_signal":     sent_obj.signal      if sent_obj else None,
        "n_articles":      sent_obj.n_articles  if sent_obj else None,
        "n_positive":      sent_obj.n_positive  if sent_obj else None,
        "n_negative":      sent_obj.n_negative  if sent_obj else None,
        "n_neutral":       sent_obj.n_neutral   if sent_obj else None,
        "sent_momentum":   sent_obj.momentum    if sent_obj else None,
        "sent_weekly":     sent_obj.weekly_score if sent_obj else None,
        "sent_monthly":    sent_obj.monthly_score if sent_obj else None,
        "sent_vol_trend":  sent_obj.volume_trend if sent_obj else None,
        "headlines":       [(h, s) for h, s in sent_obj.headlines] if sent_obj else [],
        "articles":        sent_obj.all_articles if sent_obj else [],
        # ── ML Classifier results ──────────────────────────────────
        "ml_regime":       ml_pred.regime if ml_pred else None,
        "ml_regime_conf":  ml_pred.regime_confidence if ml_pred else None,
        "ml_regime_probs": ml_pred.regime_probs if ml_pred else None,
        "ml_entry":        ml_pred.entry_score if ml_pred else None,
        "ml_exit":         ml_pred.exit_score if ml_pred else None,
        "ml_signal":       ml_pred.ml_signal if ml_pred else None,
        "ml_decision":     ml_pred.decision if ml_pred else None,
        "ml_uncertainty":  ml_pred.uncertainty if ml_pred else None,
        # store raw data for Excel export
        "_data": data,
    }


# ── Background worker ─────────────────────────────────────────────────────────

def _run_task(task_id: str, config: Dict):
    from ..services import market_data as md
    from ..services import threshold_config as tc
    assets      = config.get("assets", [])
    ind_cfg     = config.get("indicators", {})
    sent_cfg    = config.get("sentiment", {})
    weights     = config.get("weights", {"technical": 40, "fundamental": 40, "sentiment": 20})
    period      = ind_cfg.get("period", "1y")
    tech_sel    = ind_cfg.get("technical", ["ma20","ma50","ma200","cross","rsi","macd","bb"])
    fund_sel    = ind_cfg.get("fundamental", ["pe","margins","roe","growth","analyst"])
    n           = len(assets)

    # ── Threshold config (optional) ───────────────────────────────────────────
    _threshold_cfg: Optional[Dict] = None
    _inline = config.get("threshold_config")
    _cfg_id = config.get("threshold_config_id")
    if isinstance(_inline, dict) and _inline:
        _threshold_cfg = _inline
    elif _cfg_id:
        _threshold_cfg = tc.get_config(_cfg_id)
    _score_thresholds = (_threshold_cfg or {}).get("score_thresholds") or None

    _push(task_id, {"type": "start", "total": n})

    # ── Phase 0: Market context (VIX, breadth) — once for all stocks ─────────
    _push(task_id, {"type": "progress", "stage": "market_context", "pct": 2,
                     "msg": "Fetching market context (VIX, breadth)…"})
    try:
        market_ctx = md.fetch_market_context(period=period)
    except Exception:
        market_ctx = {}

    # ── Phase 1: Market data ─────────────────────────────────────────────────
    asset_data: Dict[str, Any] = {}

    for i, asset in enumerate(assets):
        ticker = asset["ticker"]
        _push(task_id, {
            "type": "progress", "ticker": ticker,
            "stage": "market_data", "done": i, "total": n,
            "pct": int(i / n * 55),
        })
        try:
            data = md.fetch_asset_data(
                ticker, period=period,
                meta={
                    "name":   asset.get("name", ticker),
                    "full":   asset.get("name", ticker),
                    "cur":    asset.get("currency", "USD"),
                    "sector": asset.get("sector", ""),
                },
                include_fundamentals=bool(fund_sel),
                market_ctx=market_ctx,
            )
            if data is None:
                _push(task_id, {"type": "warn", "ticker": ticker, "msg": "No data"})
                continue
            asset_data[ticker] = data
        except Exception as e:
            _push(task_id, {"type": "warn", "ticker": ticker, "msg": str(e)})

    _push(task_id, {"type": "progress", "stage": "scoring", "pct": 60})

    # ── Phase 2: Technical + fundamental scores (regime-conditioned) ─────────
    results_raw = {}
    for ticker, data in asset_data.items():
        asset = next((a for a in assets if a["ticker"] == ticker), {})
        latest = data.get("latest", {})
        fund_d = data.get("fundamentals", {})
        regime = latest.get("Regime", "NEUTRAL")
        tech   = sc.compute_technical_score(
            latest, tech_sel, regime=regime, thresholds=_threshold_cfg
        )
        fund   = sc.compute_fundamental_score(
            fund_d, fund_sel,
            sector=fund_d.get("sector", "") or asset.get("sector", ""),
        )
        results_raw[ticker] = (asset, data, tech, fund, None, None)

    # ── Phase 2.5: ML Classifier (optional) ──────────────────────────────────
    ml_cfg_raw = config.get("ml", {})
    ml_predictions: Dict[str, Any] = {}

    if ml_cfg_raw.get("enabled"):
        try:
            from ..services.ml_engine import MLConfig, MLEngine, UniverseMLEngine
            ml_cfg = MLConfig(
                model_type=ml_cfg_raw.get("model_type", "lightgbm"),
                backend=ml_cfg_raw.get("backend", "auto"),
                training_period=ml_cfg_raw.get("training_period", "5y"),
                forward_horizon=int(ml_cfg_raw.get("forward_horizon", 21)),
                strong_threshold=float(ml_cfg_raw.get("strong_threshold", 0.06)),
                weak_threshold=float(ml_cfg_raw.get("weak_threshold", 0.02)),
                feature_set=ml_cfg_raw.get("feature_set", "full"),
                train_mode=ml_cfg_raw.get("train_mode", "per_ticker"),
                n_trees=int(ml_cfg_raw.get("n_trees", 300)),
                max_depth=int(ml_cfg_raw.get("max_depth", 5)),
                learning_rate=float(ml_cfg_raw.get("learning_rate", 0.03)),
                wf_gap=int(ml_cfg_raw.get("wf_gap", 21)),
                # PyTorch MLP
                epochs=int(ml_cfg_raw.get("epochs", 100)),
                dropout=float(ml_cfg_raw.get("dropout", 0.3)),
                # Risk management
                target_annual_vol=float(ml_cfg_raw.get("target_vol", 0.15)),
                max_drawdown_trigger=float(ml_cfg_raw.get("max_dd_trigger", 0.15)),
            )

            _push(task_id, {"type": "progress", "stage": "ml_training",
                            "pct": 60, "msg": "Training ML classifier…"})

            # Build context maps for fundamentals & sentiment (available now)
            _fund_map = {t: d.get("fundamentals", {}) for t, d in asset_data.items()}
            # Sentiment not available yet at this phase — will be None during
            # training (model learns without it) but supplied at predict time
            # if sentiment runs later. For now pass empty.
            _sent_map: Dict[str, Any] = {}

            if ml_cfg.train_mode == "universe" and len(asset_data) > 1:
                engine = UniverseMLEngine(ml_cfg)
                dfs = {t: d["df"] for t, d in asset_data.items() if "df" in d}
                engine.train_universe(
                    list(dfs.keys()), dfs,
                    market_ctx=market_ctx,
                    fundamentals_map=_fund_map,
                    sentiment_map=_sent_map,
                )
                for ticker, data in asset_data.items():
                    try:
                        pred = engine.predict_from_df(
                            data["df"], market_ctx=market_ctx,
                            fundamentals=_fund_map.get(ticker),
                        )
                        ml_predictions[ticker] = pred
                    except Exception as e:
                        _push(task_id, {"type": "warn", "ticker": ticker,
                                        "msg": f"ML predict failed: {e}"})
            else:
                # If analysis period is short, let ML engine fetch its own
                # longer history for training (needs ~500+ rows ideally).
                _period_months = {"6mo": 6, "1y": 12, "2y": 24, "5y": 60, "10y": 120}
                _analysis_months = _period_months.get(period, 12)
                _ml_months = _period_months.get(ml_cfg.training_period, 60)
                _use_analysis_df = _analysis_months >= _ml_months

                tickers_list = list(asset_data.keys())
                for i, ticker in enumerate(tickers_list):
                    data = asset_data[ticker]
                    _push(task_id, {
                        "type": "progress", "ticker": ticker,
                        "stage": "ml_training", "done": i,
                        "total": len(tickers_list),
                        "pct": 60 + int(i / max(len(tickers_list), 1) * 5),
                        "msg": f"ML: {ticker}",
                    })
                    try:
                        _fund = _fund_map.get(ticker, {})

                        # Check if the ML Lab already has a trained engine for this ticker.
                        # If so, skip re-training and use it directly — preserves the
                        # validated model the user built in the ML Lab.
                        from .ml import _trained_engines as _lab_engines, _trained_dfs as _lab_dfs
                        _lab_engine = _lab_engines.get(ticker)

                        if _lab_engine is not None:
                            _push(task_id, {
                                "type": "progress", "ticker": ticker,
                                "stage": "ml_training", "done": i,
                                "total": len(tickers_list),
                                "pct": 60 + int(i / max(len(tickers_list), 1) * 5),
                                "msg": f"ML: {ticker} (using ML Lab model)",
                            })
                            _lab_df = _lab_dfs.get(ticker)
                            _df_for_pred = _lab_df if _lab_df is not None else data["df"]
                            pred = _lab_engine.predict_from_df(
                                _df_for_pred, market_ctx=market_ctx,
                                fundamentals=_fund,
                            )
                        else:
                            engine = MLEngine(ml_cfg)
                            if _use_analysis_df:
                                engine.train(ticker, df=data["df"],
                                             market_ctx=market_ctx,
                                             fundamentals=_fund)
                            else:
                                engine.train(ticker, period=ml_cfg.training_period,
                                             market_ctx=market_ctx,
                                             fundamentals=_fund)
                            pred = engine.predict_from_df(
                                data["df"], market_ctx=market_ctx,
                                fundamentals=_fund,
                            )
                        ml_predictions[ticker] = pred
                    except Exception as e:
                        _push(task_id, {"type": "warn", "ticker": ticker,
                                        "msg": f"ML failed: {e}"})

        except Exception as e:
            _push(task_id, {"type": "warn", "msg": f"ML classifier failed: {e}"})

    # ── Phase 3: Sentiment (optional) ────────────────────────────────────────
    sent_results: Dict[str, Any] = {}

    # Guard: skip cleanly when no LLM is available. compute_overall_score
    # renormalizes weights when sent_score is None.
    if sent_cfg.get("enabled") and results_raw:
        from ..services.sentiment import sentiment_available
        if not sentiment_available(sent_cfg.get("provider", "local"), sent_cfg):
            _push(task_id, {"type": "warn",
                            "msg": "Sentiment skipped: no LLM provider configured."})
            sent_cfg = {**sent_cfg, "enabled": False}

    if sent_cfg.get("enabled") and results_raw:
        provider = sent_cfg.get("provider", "local")
        if provider == "local":
            _push(task_id, {"type": "progress", "stage": "loading_model", "pct": 62,
                             "msg": "Loading local sentiment model..."})
        else:
            _push(task_id, {"type": "progress", "stage": "loading_model", "pct": 62,
                             "msg": f"Connecting to {provider} API..."})
        try:
            from ..services.sentiment import SentimentAnalyzer
            analyzer = SentimentAnalyzer(
                provider=provider,
                api_key=sent_cfg.get("api_key", ""),
                model=sent_cfg.get("model", ""),
                model_path=sent_cfg.get("model_path", ""),
                adapter_path=sent_cfg.get("adapter_path", ""),
                max_articles=sent_cfg.get("max_articles", 50),
                days=sent_cfg.get("days", 15),
            )

            tickers = list(results_raw.keys())
            for i, ticker in enumerate(tickers):
                asset = results_raw[ticker][0]
                _push(task_id, {
                    "type": "progress", "ticker": ticker,
                    "stage": "sentiment", "done": i, "total": len(tickers),
                    "pct": 62 + int(i / len(tickers) * 33),
                })
                sent = analyzer.analyze_asset(
                    ticker, company_name=asset.get("name", ticker)
                )
                sent_results[ticker] = sent

            analyzer.unload()

        except Exception as e:
            _push(task_id, {"type": "warn", "msg": f"Sentiment failed: {e}"})

    # ── Phase 4: Assemble final results ──────────────────────────────────────
    _push(task_id, {"type": "progress", "stage": "assembling", "pct": 97})

    final = []
    for ticker, (asset, data, tech, fund, _, _) in results_raw.items():
        sent_obj   = sent_results.get(ticker)
        sent_score = sent_obj.score if sent_obj and sent_obj.n_articles > 0 else None
        ml_pred    = ml_predictions.get(ticker)
        r = _fmt_result(ticker, asset, data, tech, fund, sent_score, sent_obj, weights,
                        market_ctx=market_ctx, ml_pred=ml_pred,
                        score_thresholds=_score_thresholds)
        final.append(r)

    # ── Phase 4.5: Cross-sectional z-scoring + correlation regime ────────────
    if len(final) > 1:
        try:
            import numpy as np
            import pandas as pd

            # Cross-sectional z-score of overall_score across the universe
            _raw_scores = [r.get("overall_score") or 0.0 for r in final]
            _arr = np.array(_raw_scores, dtype=float)
            _mean = float(_arr.mean())
            _std  = float(_arr.std()) if _arr.std() > 1e-9 else 1.0
            _n    = len(_arr)
            for i, r in enumerate(final):
                r["cs_z_score"]   = round((_arr[i] - _mean) / _std, 3)
                r["cs_rank_pct"]  = round(
                    float(np.searchsorted(np.sort(_arr), _arr[i])) / _n, 3
                )

            # Correlation regime: pairwise correlation of recent 21d returns
            _ret_series: Dict[str, Any] = {}
            for ticker, data in asset_data.items():
                _df = data.get("df")
                if _df is not None and "Close" in _df.columns and len(_df) > 42:
                    _ret_series[ticker] = (
                        _df["Close"].pct_change().tail(63).reset_index(drop=True)
                    )

            if len(_ret_series) >= 2:
                _ret_df = pd.DataFrame(_ret_series).dropna()
                if len(_ret_df) >= 15:
                    _corr_mx = _ret_df.corr(method="spearman")
                    _upper = _corr_mx.where(
                        np.triu(np.ones(_corr_mx.shape), k=1).astype(bool)
                    ).stack()
                    _avg_corr = float(_upper.mean()) if len(_upper) else 0.0
                    _corr_regime = (
                        "HIGH"   if _avg_corr >= 0.60 else
                        "MEDIUM" if _avg_corr >= 0.35 else
                        "LOW"
                    )
                    # Attach to all results
                    for r in final:
                        r["universe_avg_corr"]   = round(_avg_corr, 3)
                        r["universe_corr_regime"] = _corr_regime
        except Exception:
            pass  # non-critical

    # Sort by overall score descending
    final.sort(key=lambda x: (x.get("overall_score") or 0), reverse=True)

    portfolio_plan = None
    if ml_predictions:
        try:
            from ..services.portfolio_construction import (
                PortfolioCandidate,
                PortfolioConfig,
                build_portfolio_plan,
            )
            candidates = []
            for ticker, pred in ml_predictions.items():
                decision = pred.decision or {}
                action = decision.get("action")
                position_size = float(decision.get("position_size") or 0.0)
                score_spread = float(
                    pred.score_spread
                    if getattr(pred, "score_spread", None) is not None
                    else (pred.entry_score - pred.exit_score)
                )
                score = max(0.0, score_spread) * float(pred.regime_confidence or 0.0) * max(position_size, 0.0)
                sector = (
                    results_raw.get(ticker, ({}, {}, 0, 0, None, None))[1]
                    .get("meta", {})
                    .get("sector", "Unknown")
                )
                candidates.append(
                    PortfolioCandidate(
                        ticker=ticker,
                        sector=sector,
                        score=score,
                        signal=action or "HOLD",
                        conviction=str(decision.get("conviction") or "NONE"),
                        position_size=position_size,
                        regime=str(pred.regime),
                    )
                )
            portfolio_plan = build_portfolio_plan(candidates, PortfolioConfig())
            alloc_map = {a.ticker: a for a in portfolio_plan.allocations}
            for r in final:
                alloc = alloc_map.get(r["ticker"])
                r["ml_portfolio_weight"] = alloc.weight if alloc else 0.0
                r["ml_portfolio_score"] = alloc.score if alloc else 0.0
        except Exception as e:
            _push(task_id, {"type": "warn", "msg": f"Portfolio construction failed: {e}"})

    # Store raw data objects for Excel export (not JSON-serialisable)
    with _tasks_lock:
        _tasks[task_id]["_raw_data"] = {
            r["ticker"]: r.pop("_data", None) for r in final
        }
        _tasks[task_id]["_market_ctx"] = market_ctx
        if portfolio_plan is not None:
            _tasks[task_id]["_portfolio_plan"] = portfolio_plan.to_dict()

    _finish(task_id, final)


# ── Routes ────────────────────────────────────────────────────────────────────

@bp.route("/start", methods=["POST"])
def start():
    config = request.get_json(force=True, silent=True) or {}
    if not config.get("assets"):
        return jsonify({"ok": False, "error": "No assets provided"}), 400

    task_id = str(uuid.uuid4())
    with _tasks_lock:
        _tasks[task_id] = {"status": "running", "events": [], "results": None, "error": None}

    t = threading.Thread(target=_run_task_safe, args=(task_id, config), daemon=True)
    t.start()
    return jsonify({"ok": True, "task_id": task_id})


def _run_task_safe(task_id, config):
    try:
        _run_task(task_id, config)
    except Exception as e:
        _fail(task_id, traceback.format_exc())


@bp.route("/stream/<task_id>")
def stream(task_id):
    if task_id not in _tasks:
        return jsonify({"error": "Not found"}), 404

    def generate():
        last = 0
        while True:
            with _tasks_lock:
                task   = _tasks.get(task_id, {})
                events = task.get("events", [])
                status = task.get("status", "running")

            for ev in events[last:]:
                yield f"data: {json.dumps(ev)}\n\n"
            last = len(events)

            if status in ("done", "error"):
                break
            time.sleep(0.3)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@bp.route("/results/<task_id>")
def results(task_id):
    with _tasks_lock:
        task = _tasks.get(task_id)
    if not task:
        return jsonify({"error": "Not found"}), 404
    if task["status"] == "running":
        return jsonify({"status": "running"}), 202
    if task["status"] == "error":
        return jsonify({"status": "error", "error": task.get("error")}), 500
    return jsonify({"status": "done", "results": task["results"]})
