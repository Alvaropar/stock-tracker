"""
Local web application for the sentiment pipeline.

Serves a browser-based UI on localhost and exposes JSON API endpoints
that the frontend calls.  Auto-opens in the default browser on startup.

Filter and sentiment run in a background thread — the frontend polls
``/api/task`` to track progress and collect results when done.

Start with::

    python -m pipeline.client.local_app            # default port 5000
    python -m pipeline.client.local_app --port 8080
"""
from __future__ import annotations

import argparse
import atexit
import logging
import threading
import webbrowser
from typing import Any, Dict

from ._state import PipelineState
from ..exceptions import ConfigError, PipelineError

log = logging.getLogger("pipeline.app")


def _create_app(state: PipelineState | None = None):
    """Factory that returns a Flask app wired to a PipelineState backend."""
    try:
        from flask import Flask, jsonify, request, render_template_string, Response
    except ImportError:
        raise ImportError(
            "Flask is required for the local app. Install with: pip install flask"
        )

    import os as _os
    _static_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "static")
    app = Flask(__name__, static_folder=_static_dir, static_url_path="/static")
    _state = state or PipelineState()
    atexit.register(_state.cleanup)

    def _json_body() -> Dict[str, Any]:
        return request.get_json(force=True, silent=True) or {}

    # ── Security headers ──────────────────────────────────────────
    @app.after_request
    def _security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        # Only allow localhost requests
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "connect-src 'self'"
        )
        return response

    # ── Localhost-only guard ──────────────────────────────────────
    @app.before_request
    def _localhost_only():
        """Reject requests that don't originate from localhost."""
        remote = request.remote_addr
        if remote not in ("127.0.0.1", "::1", None):
            log.warning("Blocked non-localhost request from %s", remote)
            return jsonify({"ok": False, "error": "Forbidden"}), 403

    # ── Global error handlers ─────────────────────────────────────
    @app.errorhandler(ConfigError)
    def handle_config_error(e):
        return jsonify({"ok": False, "error": str(e)}), 400

    @app.errorhandler(PipelineError)
    def handle_pipeline_error(e):
        log.error("Pipeline error: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500

    @app.errorhandler(KeyError)
    def handle_key_error(e):
        return jsonify({"ok": False, "error": f"Missing required field: {e}"}), 400

    @app.errorhandler(ValueError)
    def handle_value_error(e):
        return jsonify({"ok": False, "error": str(e)}), 400

    # ── HTML frontend ────────────────────────────────────────────────

    @app.route("/", methods=["GET"])
    def index():
        return render_template_string(_INDEX_HTML)

    # ── JSON API ─────────────────────────────────────────────────────

    @app.route("/api/config", methods=["GET"])
    def config():
        cfg = _state.get_config()
        cfg["last_config"] = _state.get_last_config()
        return jsonify(cfg)

    @app.route("/api/status", methods=["GET"])
    def status():
        return jsonify(_state.get_status())

    @app.route("/api/task", methods=["GET"])
    def task_status():
        return jsonify(_state.get_task_status())

    @app.route("/api/market", methods=["POST"])
    def set_market():
        body = _json_body()
        _state.set_market(body["market"])
        return jsonify({"ok": True, "market": _state.market})

    @app.route("/api/asset", methods=["POST"])
    def set_asset():
        body = _json_body()
        _state.set_asset(body["asset_type"], body["asset_id"])
        return jsonify({"ok": True, "asset_type": _state.asset_type,
                        "asset_id": _state.asset_id})

    @app.route("/api/filter_model", methods=["POST"])
    def set_filter_model():
        body = _json_body()
        _state.set_filter_model(body["base_model_path"],
                                body.get("adapter_path"))
        return jsonify({"ok": True})

    @app.route("/api/sentiment_model", methods=["POST"])
    def set_sentiment_model():
        body = _json_body()
        _state.set_sentiment_model(body["base_model_path"],
                                   body.get("adapter_path"))
        return jsonify({"ok": True})

    @app.route("/api/browse", methods=["GET"])
    def browse():
        path = request.args.get("path", "")
        return jsonify(_state.browse_directory(path or None))

    @app.route("/api/price", methods=["GET"])
    def price():
        market = request.args.get("market", "")
        asset_type = request.args.get("asset_type", "")
        asset_id = request.args.get("asset_id", "")
        if not all([market, asset_type, asset_id]):
            return jsonify({"error": "market, asset_type, asset_id required"}), 400
        return jsonify(_state.get_price_data(market, asset_type, asset_id))

    @app.route("/api/price/live", methods=["GET"])
    def price_live():
        market = request.args.get("market", "")
        asset_type = request.args.get("asset_type", "")
        asset_id = request.args.get("asset_id", "")
        if not all([market, asset_type, asset_id]):
            return jsonify({"error": "market, asset_type, asset_id required"}), 400
        return jsonify(_state.get_live_price(market, asset_type, asset_id))

    @app.route("/api/fetch", methods=["POST"])
    def fetch():
        try:
            started = _state.start_fetch()
            if not started:
                return jsonify({"ok": False,
                                "error": "A task is already running"}), 409
            return jsonify({"ok": True, "message": "Fetch started"})
        except Exception as e:
            log.warning("Fetch failed: %s", e)
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.route("/api/filter", methods=["POST"])
    def filter_news():
        try:
            started = _state.start_filter()
            if not started:
                return jsonify({"ok": False,
                                "error": "A task is already running"}), 409
            return jsonify({"ok": True, "message": "Filter started"})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.route("/api/sentiment", methods=["POST"])
    def sentiment():
        try:
            started = _state.start_sentiment()
            if not started:
                return jsonify({"ok": False,
                                "error": "A task is already running"}), 409
            return jsonify({"ok": True, "message": "Sentiment started"})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    @app.route("/api/results", methods=["GET"])
    def results():
        data = _state.get_results()
        return jsonify({"ok": True, "count": len(data), "articles": data})

    @app.route("/api/export/csv", methods=["GET"])
    def export_csv():
        csv_str = _state.export_csv()
        if not csv_str:
            return jsonify({"ok": False, "error": "No data to export"}), 404
        return Response(
            csv_str,
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=sentiment_results.csv"},
        )

    @app.route("/api/history", methods=["GET"])
    def history():
        return jsonify({"ok": True, "sessions": _state.get_history()})

    @app.route("/api/sentiment_history", methods=["GET"])
    def sentiment_history():
        days = request.args.get("days", "30")
        try:
            days = int(days)
        except ValueError:
            days = 30
        return jsonify({"ok": True, "data": _state.get_sentiment_history(days=days)})

    return app


# ─── HTML template ───────────────────────────────────────────────────────────

_INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sentiment Pipeline</title>
<script src="/static/lightweight-charts.standalone.production.js"></script>
<style>
  :root {
    --bg: #0f1117; --surface: #1a1d27; --border: #2a2d3a;
    --text: #e4e4e7; --muted: #71717a; --accent: #3b82f6;
    --green: #22c55e; --red: #ef4444; --yellow: #eab308;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
         background: var(--bg); color: var(--text); min-height: 100vh; }
  .container { max-width: 1020px; margin: 0 auto; padding: 24px 16px; }
  h1 { font-size: 1.5rem; font-weight: 600; margin-bottom: 4px; }
  .subtitle { color: var(--muted); font-size: 0.85rem; margin-bottom: 24px; }

  .card { background: var(--surface); border: 1px solid var(--border);
          border-radius: 10px; padding: 20px; margin-bottom: 16px; }
  .card h2 { font-size: 1rem; font-weight: 600; margin-bottom: 12px; }

  .row { display: flex; gap: 10px; flex-wrap: wrap; align-items: end; }
  .field { display: flex; flex-direction: column; gap: 4px; flex: 1; min-width: 140px; }
  .field label { font-size: 0.7rem; color: var(--muted); text-transform: uppercase;
                 letter-spacing: 0.05em; }
  select, input[type=text] {
    background: var(--bg); border: 1px solid var(--border); color: var(--text);
    border-radius: 6px; padding: 8px 10px; font-size: 0.85rem; width: 100%;
  }
  select:focus, input:focus { outline: none; border-color: var(--accent); }

  .btn { border: none; border-radius: 6px; padding: 9px 18px; font-size: 0.85rem;
         font-weight: 500; cursor: pointer; transition: opacity .15s; white-space: nowrap; }
  .btn:hover { opacity: 0.85; }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-primary { background: var(--accent); color: #fff; }
  .btn-green   { background: var(--green); color: #fff; }
  .btn-sm { padding: 6px 12px; font-size: 0.78rem; }
  .btn-browse { background: var(--border); color: var(--text); padding: 8px 12px; }

  #status-bar { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
  .badge { font-size: 0.75rem; padding: 3px 10px; border-radius: 999px;
           background: var(--border); color: var(--muted); }
  .badge.active { background: #1e3a5f; color: var(--accent); }

  #log { font-family: 'Cascadia Code', 'Consolas', monospace; font-size: 0.8rem;
         color: var(--muted); max-height: 140px; overflow-y: auto;
         white-space: pre-wrap; padding: 10px; background: var(--bg);
         border-radius: 6px; border: 1px solid var(--border); }

  .tbl { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  .tbl th { text-align: left; color: var(--muted); font-weight: 500;
            padding: 8px 10px; border-bottom: 1px solid var(--border); }
  .tbl td { padding: 7px 10px; border-bottom: 1px solid var(--border); }
  .tbl tr:hover td { background: rgba(255,255,255,0.02); }
  .sent { font-weight: 600; text-transform: uppercase; font-size: 0.75rem; }
  .sent-positive { color: var(--green); }
  .sent-negative { color: var(--red); }
  .sent-neutral  { color: var(--yellow); }

  .spinner { display: inline-block; width: 14px; height: 14px;
             border: 2px solid var(--border); border-top-color: var(--accent);
             border-radius: 50%; animation: spin .6s linear infinite;
             vertical-align: middle; }
  @keyframes spin { to { transform: rotate(360deg); } }

  .empty { color: var(--muted); text-align: center; padding: 32px 0; }

  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 640px) { .grid-2 { grid-template-columns: 1fr; } }
  .note { font-size: 0.7rem; color: var(--muted); margin-top: 2px; }

  /* ── input + browse button row ── */
  .input-browse { display: flex; gap: 6px; align-items: stretch; }
  .input-browse input { flex: 1; min-width: 0; }

  /* ── File Browser Modal ── */
  .modal-overlay {
    display: none; position: fixed; inset: 0;
    background: rgba(0,0,0,0.65); z-index: 1000;
    align-items: center; justify-content: center;
  }
  .modal-overlay.active { display: flex; }
  .modal {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; width: 560px; max-width: 95vw;
    max-height: 80vh; display: flex; flex-direction: column;
  }
  .modal-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 18px; border-bottom: 1px solid var(--border);
  }
  .modal-header h3 { font-size: 0.95rem; }
  .modal-close { background: none; border: none; color: var(--muted);
                 font-size: 1.2rem; cursor: pointer; padding: 4px 8px; }
  .modal-close:hover { color: var(--text); }

  .modal-breadcrumb {
    padding: 10px 18px; font-size: 0.78rem; color: var(--muted);
    border-bottom: 1px solid var(--border); word-break: break-all;
    background: var(--bg);
  }
  .modal-body {
    flex: 1; overflow-y: auto; padding: 0;
  }
  .modal-footer {
    padding: 12px 18px; border-top: 1px solid var(--border);
    display: flex; gap: 8px; justify-content: flex-end;
  }

  .fb-item {
    display: flex; align-items: center; gap: 10px;
    padding: 9px 18px; cursor: pointer; font-size: 0.85rem;
    border-bottom: 1px solid var(--border);
    transition: background .1s;
  }
  .fb-item:hover { background: rgba(59,130,246,0.08); }
  .fb-icon { font-size: 1rem; width: 20px; text-align: center; }
  .fb-name { flex: 1; }
  .fb-tag {
    font-size: 0.65rem; padding: 2px 6px; border-radius: 4px;
    font-weight: 600; text-transform: uppercase;
  }
  .fb-tag-model { background: #1e3a5f; color: var(--accent); }
  .fb-tag-adapter { background: #2a1e3f; color: #a78bfa; }

  /* ── Price card ── */
  .price-header { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }
  .price-big { font-size: 2rem; font-weight: 700; }
  .price-change { font-size: 1rem; font-weight: 600; }
  .price-change.up { color: var(--green); }
  .price-change.down { color: var(--red); }
  .price-currency { font-size: 0.8rem; color: var(--muted); }
  #chart-container { width: 100%; height: 380px; margin-top: 12px; border-radius: 8px; overflow: hidden; }
  .price-loading { color: var(--muted); padding: 20px 0; font-size: 0.85rem; }

  /* ── Onboarding wizard ── */
  .wizard-overlay {
    display: none; position: fixed; inset: 0;
    background: rgba(0,0,0,0.75); z-index: 2000;
    align-items: center; justify-content: center;
  }
  .wizard-overlay.active { display: flex; }
  .wizard {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 16px; width: 520px; max-width: 95vw;
    padding: 32px; text-align: center;
  }
  .wizard h2 { font-size: 1.3rem; margin-bottom: 8px; }
  .wizard p { color: var(--muted); font-size: 0.85rem; margin-bottom: 20px; line-height: 1.6; }
  .wizard .step-indicator { display: flex; justify-content: center; gap: 8px; margin-bottom: 24px; }
  .wizard .step-dot {
    width: 10px; height: 10px; border-radius: 50%;
    background: var(--border); transition: background 0.2s;
  }
  .wizard .step-dot.active { background: var(--accent); }
  .wizard .step-dot.done { background: var(--green); }
  .wizard-content { text-align: left; margin-bottom: 20px; }
  .wizard-btns { display: flex; gap: 10px; justify-content: center; }
  .wizard-icon { font-size: 2.5rem; margin-bottom: 16px; }

  /* ── Dashboard ── */
  .dashboard-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 16px; }
  @media (max-width: 640px) { .dashboard-grid { grid-template-columns: 1fr; } }
  .stat-card {
    background: var(--bg); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px; text-align: center;
  }
  .stat-card .stat-value { font-size: 1.8rem; font-weight: 700; }
  .stat-card .stat-label { font-size: 0.75rem; color: var(--muted); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }
  .sentiment-bar { display: flex; height: 28px; border-radius: 6px; overflow: hidden; margin-top: 12px; }
  .sentiment-bar .seg { transition: width 0.5s; }
  .sentiment-bar .seg-pos { background: var(--green); }
  .sentiment-bar .seg-neg { background: var(--red); }
  .sentiment-bar .seg-neu { background: var(--yellow); }
  .sentiment-bar-labels { display: flex; justify-content: space-between; margin-top: 6px; font-size: 0.75rem; color: var(--muted); }
  #sentiment-chart-container { width: 100%; height: 200px; margin-top: 12px; border-radius: 8px; overflow: hidden; }

  /* ── Toast notification ── */
  .toast-container { position: fixed; top: 16px; right: 16px; z-index: 3000; display: flex; flex-direction: column; gap: 8px; }
  .toast {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 12px 18px; font-size: 0.85rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4); animation: toastIn 0.3s ease;
    max-width: 360px;
  }
  .toast.success { border-left: 3px solid var(--green); }
  .toast.error { border-left: 3px solid var(--red); }
  .toast.info { border-left: 3px solid var(--accent); }
  @keyframes toastIn { from { opacity: 0; transform: translateX(40px); } to { opacity: 1; transform: translateX(0); } }

  /* ── Pagination ── */
  .pagination { display: flex; align-items: center; gap: 8px; justify-content: center; margin-top: 12px; }
  .pagination button { background: var(--border); color: var(--text); border: none; border-radius: 6px; padding: 6px 12px; font-size: 0.8rem; cursor: pointer; }
  .pagination button:hover { opacity: 0.8; }
  .pagination button:disabled { opacity: 0.3; cursor: not-allowed; }
  .pagination span { font-size: 0.8rem; color: var(--muted); }

  /* ── Search box ── */
  .search-bar { margin-bottom: 12px; }
  .search-bar input { width: 100%; background: var(--bg); border: 1px solid var(--border); color: var(--text); border-radius: 6px; padding: 8px 12px; font-size: 0.85rem; }
  .search-bar input::placeholder { color: var(--muted); }
</style>
</head>
<body>
<div class="container">
  <h1>Commodity Trading Sentiment Analyzer</h1>
  <p class="subtitle">Scrape &rarr; Filter &rarr; Analyze &mdash; all running locally <span id="app-version" style="opacity:0.5"></span></p>

  <!-- Toast container -->
  <div class="toast-container" id="toasts"></div>

  <!-- ── Configuration ── -->
  <div class="card">
    <h2>Configuration</h2>
    <div class="row">
      <div class="field">
        <label>Market</label>
        <select id="sel-market"></select>
      </div>
      <div class="field">
        <label>Asset Type</label>
        <select id="sel-asset-type">
          <option value="commodity">Commodity</option>
          <option value="stock">Stock</option>
        </select>
      </div>
      <div class="field">
        <label>Asset</label>
        <select id="sel-asset"></select>
        <input type="text" id="inp-asset" placeholder="e.g. AAPL" style="display:none">
      </div>
      <button class="btn btn-primary" id="btn-configure">Set</button>
    </div>
  </div>

  <!-- ── Market Data (price + chart) ── -->
  <div class="card" id="price-card" style="display:none">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
      <h2 style="margin-bottom:0">Market Data</h2>
      <div style="display:flex;align-items:center;gap:8px">
        <span id="price-updated" style="font-size:0.7rem;color:var(--muted)"></span>
        <select id="sel-refresh-interval" style="width:auto;padding:4px 8px;font-size:0.75rem;">
          <option value="0">Manual</option>
          <option value="15">15s</option>
          <option value="30" selected>30s</option>
          <option value="60">1m</option>
          <option value="300">5m</option>
        </select>
        <button class="btn btn-sm" style="background:var(--border);color:var(--text);padding:5px 10px" title="Refresh now">&#8635;</button>
      </div>
    </div>
    <div id="price-info">
      <p class="price-loading"><span class="spinner"></span> Loading price data...</p>
    </div>
    <div id="chart-container"></div>
  </div>

  <!-- ── Model Selection ── -->
  <div class="card">
    <h2>Models</h2>
    <div class="grid-2">
      <div>
        <h3 style="font-size:0.85rem;margin-bottom:8px;color:var(--muted);">Filter Model</h3>
        <div class="field" style="margin-bottom:6px">
          <label>Base Model</label>
          <div class="input-browse">
            <input type="text" id="inp-filter-base" placeholder="Path to base model...">
            <button class="btn btn-browse btn-sm" data-target="inp-filter-base">Browse</button>
          </div>
        </div>
        <div class="field">
          <label>LoRA Adapter</label>
          <div class="input-browse">
            <input type="text" id="inp-filter-adapter" placeholder="None (base only)">
            <button class="btn btn-browse btn-sm" data-target="inp-filter-adapter">Browse</button>
          </div>
          <span class="note">Leave empty to use base model without fine-tuning</span>
        </div>
      </div>
      <div>
        <h3 style="font-size:0.85rem;margin-bottom:8px;color:var(--muted);">Sentiment Model</h3>
        <div class="field" style="margin-bottom:6px">
          <label>Base Model</label>
          <div class="input-browse">
            <input type="text" id="inp-sent-base" placeholder="Path to base model...">
            <button class="btn btn-browse btn-sm" data-target="inp-sent-base">Browse</button>
          </div>
        </div>
        <div class="field">
          <label>LoRA Adapter</label>
          <div class="input-browse">
            <input type="text" id="inp-sent-adapter" placeholder="None (base only)">
            <button class="btn btn-browse btn-sm" data-target="inp-sent-adapter">Browse</button>
          </div>
          <span class="note">Leave empty to use base model without fine-tuning</span>
        </div>
      </div>
    </div>
    <div style="margin-top:10px">
      <button class="btn btn-primary" id="btn-apply-models">Apply Model Settings</button>
    </div>
  </div>

  <!-- ── Pipeline controls ── -->
  <div class="card">
    <h2>Pipeline</h2>
    <div id="status-bar" style="margin-bottom: 12px;">
      <span class="badge" id="badge-market">No market</span>
      <span class="badge" id="badge-asset">No asset</span>
      <span class="badge" id="badge-articles">0 articles</span>
      <span class="badge" id="badge-filtered">0 filtered</span>
    </div>
    <div class="row">
      <button class="btn btn-primary" id="btn-fetch" disabled>
        Fetch News
      </button>
      <button class="btn btn-primary" id="btn-filter" disabled>
        Filter
      </button>
      <button class="btn btn-green" id="btn-sentiment" disabled>
        Analyze Sentiment
      </button>
    </div>
    <div id="log" style="margin-top: 12px;"></div>
  </div>

  <!-- ── Sentiment Dashboard ── -->
  <div class="card" id="dashboard-card" style="display:none">
    <h2>Sentiment Dashboard</h2>
    <div class="dashboard-grid" id="dashboard-stats"></div>
    <div>
      <h3 style="font-size:0.85rem;color:var(--muted);margin-bottom:4px">Sentiment Distribution</h3>
      <div class="sentiment-bar" id="sentiment-bar"></div>
      <div class="sentiment-bar-labels" id="sentiment-labels"></div>
    </div>
    <div>
      <h3 style="font-size:0.85rem;color:var(--muted);margin-top:16px;margin-bottom:4px">30-Day Trend</h3>
      <div id="sentiment-chart-container"></div>
    </div>
  </div>

  <!-- ── Results ── -->
  <div class="card">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
      <h2 style="margin-bottom:0">Results</h2>
      <div style="display:flex;gap:8px">
        <button class="btn btn-sm" id="btn-export" style="background:var(--border);color:var(--text);display:none">&#128190; Export CSV</button>
        <button class="btn btn-sm" id="btn-history" style="background:var(--border);color:var(--text)">&#128218; History</button>
      </div>
    </div>
    <div class="search-bar" id="search-bar" style="display:none">
      <input type="text" id="inp-search" placeholder="Search headlines...">
    </div>
    <div id="results-area"><p class="empty">No results yet. Run the pipeline above.</p></div>
    <div class="pagination" id="pagination" style="display:none"></div>
  </div>

  <!-- ── History ── -->
  <div class="card" id="history-card" style="display:none">
    <h2>Analysis History</h2>
    <div id="history-area"><p class="empty">No history yet.</p></div>
  </div>
</div>

<!-- ── Onboarding Wizard ── -->
<div class="wizard-overlay" id="wizard-overlay">
  <div class="wizard">
    <div class="step-indicator" id="wizard-steps"></div>
    <div id="wizard-body"></div>
    <div class="wizard-btns" id="wizard-btns"></div>
  </div>
</div>

<!-- ── File Browser Modal ── -->
<div class="modal-overlay" id="fb-modal">
  <div class="modal">
    <div class="modal-header">
      <h3>Select Model Directory</h3>
      <button class="modal-close">&times;</button>
    </div>
    <div class="modal-breadcrumb" id="fb-path">Loading...</div>
    <div class="modal-body" id="fb-body"></div>
    <div class="modal-footer">
      <button class="btn btn-sm" id="fb-cancel-btn" style="background:var(--border);color:var(--text)">Cancel</button>
      <button class="btn btn-sm btn-primary" id="fb-select-btn">Select This Directory</button>
    </div>
  </div>
</div>

<script src="/static/app.js"></script>
</body>
</html>
"""


# ─── entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sentiment Pipeline — local web app")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't auto-open the browser")
    args = parser.parse_args()

    # Initialize structured logging for web mode
    try:
        from ..logging_config import setup_logging
        setup_logging(level="INFO", log_to_file=True)
    except Exception:
        logging.basicConfig(level=logging.INFO)

    app = _create_app()
    url = f"http://{args.host}:{args.port}"

    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    log.info("Starting sentiment pipeline on %s", url)
    print(f"Starting sentiment pipeline on {url}")
    print("Press Ctrl+C to stop.\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
