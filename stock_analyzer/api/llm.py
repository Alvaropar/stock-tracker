"""
LLM + RAG validation API.

GET  /api/llm/status     → is the CompactifAI API key configured and reachable?
POST /api/llm/validate   → validate one analysis result against live web search
                           Body: {ticker, company_name?, analysis}
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request

from ..services import llm_validator as lv

bp = Blueprint("llm", __name__, url_prefix="/api/llm")


@bp.route("/status", methods=["GET"])
def status():
    return jsonify(lv.api_status())


@bp.route("/validate", methods=["POST"])
def validate():
    payload = request.get_json(force=True, silent=True) or {}
    ticker = (payload.get("ticker") or "").strip().upper()
    if not ticker:
        return jsonify({"ok": False, "error": "ticker is required"}), 400
    analysis = payload.get("analysis") or {}
    if not isinstance(analysis, dict):
        return jsonify({"ok": False, "error": "analysis must be an object"}), 400

    try:
        result = lv.validate_analysis(
            ticker,
            analysis,
            company_name=payload.get("company_name", ""),
            model=payload.get("model"),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@bp.route("/quant-rank", methods=["POST"])
def quant_rank():
    """Cross-sectional AI ranking of analyzed stocks."""
    payload = request.get_json(force=True, silent=True) or {}
    results = payload.get("results", [])
    if not isinstance(results, list) or len(results) == 0:
        return jsonify({"ok": False, "error": "results array is required"}), 400
    try:
        return jsonify(lv.quant_rank(results, model=payload.get("model")))
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@bp.route("/portfolio-review", methods=["POST"])
def portfolio_review():
    """AI portfolio manager review of current holdings."""
    payload = request.get_json(force=True, silent=True) or {}
    positions = payload.get("positions", [])
    if not isinstance(positions, list) or len(positions) == 0:
        return jsonify({"ok": False, "error": "positions array is required"}), 400
    try:
        return jsonify(lv.portfolio_review(positions, model=payload.get("model")))
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
