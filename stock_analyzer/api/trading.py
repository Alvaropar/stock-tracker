"""
Trading API: guarded execution and paper-trading state.
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request

from .settings import load_settings
from ..services.execution import (
    ExecutionRequest,
    GuardedExecutionService,
    PaperExecutionAdapter,
    ShadowExecutionAdapter,
)
from ..services.live_risk import LiveRiskControls, LiveRiskManager
from ..services.paper_trading import PaperTradingConfig, PaperTradingEngine

bp = Blueprint("trading", __name__, url_prefix="/api/trading")


def _risk_manager() -> LiveRiskManager:
    settings = load_settings()
    live_cfg = settings.get("live_trading", {})
    return LiveRiskManager(LiveRiskControls(**live_cfg))


def _paper_engine() -> PaperTradingEngine:
    settings = load_settings()
    paper_cfg = settings.get("paper_trading", {})
    return PaperTradingEngine(
        config=PaperTradingConfig(**paper_cfg) if paper_cfg else PaperTradingConfig(),
        risk_manager=_risk_manager(),
    )


@bp.route("/status", methods=["GET"])
def trading_status():
    engine = _paper_engine()
    return jsonify(engine.status())


@bp.route("/paper/reset", methods=["POST"])
def reset_paper():
    engine = _paper_engine()
    ledger = engine.reset()
    return jsonify(ledger.to_dict())


@bp.route("/paper/mark", methods=["POST"])
def mark_paper():
    body = request.get_json(force=True)
    prices = body.get("prices", {})
    as_of = body.get("as_of")
    if not as_of or not isinstance(prices, dict):
        return jsonify({"error": "as_of and prices are required"}), 400
    engine = _paper_engine()
    snap = engine.mark_to_market(as_of, prices)
    return jsonify({"snapshot": snap.__dict__, "status": engine.status()})


@bp.route("/execute", methods=["POST"])
def execute():
    body = request.get_json(force=True)
    required = [
        "ticker", "side", "quantity", "reference_price", "submitted_at",
        "target_exposure", "post_trade_gross_exposure", "post_trade_net_exposure",
    ]
    missing = [k for k in required if k not in body]
    if missing:
        return jsonify({"error": f"missing fields: {', '.join(missing)}"}), 400

    risk_manager = _risk_manager()
    adapter_name = body.get("adapter", "paper")
    if adapter_name == "shadow":
        adapter = ShadowExecutionAdapter()
    else:
        adapter = PaperExecutionAdapter(_paper_engine())
    service = GuardedExecutionService(adapter=adapter, risk_manager=risk_manager)
    resp = service.submit(
        ExecutionRequest(
            ticker=body["ticker"],
            side=body["side"],
            quantity=float(body["quantity"]),
            reference_price=float(body["reference_price"]),
            submitted_at=str(body["submitted_at"]),
            target_exposure=float(body["target_exposure"]),
            post_trade_gross_exposure=float(body["post_trade_gross_exposure"]),
            post_trade_net_exposure=float(body["post_trade_net_exposure"]),
            reason=str(body.get("reason", "")),
        )
    )
    code = 200 if resp.accepted or resp.shadow_only else 409
    return jsonify(resp.to_dict()), code
