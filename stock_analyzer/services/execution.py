"""
Execution adapter interface with mandatory live-risk prechecks.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from .live_risk import LiveRiskManager, OrderIntent
from .paper_trading import PaperOrder, PaperTradingEngine


@dataclass
class ExecutionRequest:
    ticker: str
    side: str
    quantity: float
    reference_price: float
    submitted_at: str
    target_exposure: float
    post_trade_gross_exposure: float
    post_trade_net_exposure: float
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionResponse:
    accepted: bool
    adapter: str
    shadow_only: bool
    risk: Dict[str, Any]
    fill: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExecutionAdapter(ABC):
    name = "abstract"

    @abstractmethod
    def submit(self, request: ExecutionRequest) -> Dict[str, Any]:
        raise NotImplementedError


class PaperExecutionAdapter(ExecutionAdapter):
    name = "paper"

    def __init__(self, engine: Optional[PaperTradingEngine] = None):
        self.engine = engine or PaperTradingEngine()

    def submit(self, request: ExecutionRequest) -> Dict[str, Any]:
        fill = self.engine.submit_order(
            PaperOrder(
                ticker=request.ticker,
                side=request.side,
                quantity=request.quantity,
                reference_price=request.reference_price,
                submitted_at=request.submitted_at,
                reason=request.reason,
            )
        )
        return asdict(fill)


class ShadowExecutionAdapter(ExecutionAdapter):
    name = "shadow"

    def submit(self, request: ExecutionRequest) -> Dict[str, Any]:
        return {
            "ticker": request.ticker,
            "side": request.side,
            "quantity": request.quantity,
            "reference_price": request.reference_price,
            "submitted_at": request.submitted_at,
            "reason": request.reason,
            "status": "shadow_logged",
        }


class GuardedExecutionService:
    def __init__(
        self,
        adapter: Optional[ExecutionAdapter] = None,
        risk_manager: Optional[LiveRiskManager] = None,
    ):
        self.risk_manager = risk_manager or LiveRiskManager()
        self.adapter = adapter or PaperExecutionAdapter(PaperTradingEngine(risk_manager=self.risk_manager))

    def submit(self, request: ExecutionRequest) -> ExecutionResponse:
        risk = self.risk_manager.evaluate_order(
            OrderIntent(
                ticker=request.ticker,
                target_exposure=request.target_exposure,
                post_trade_gross_exposure=request.post_trade_gross_exposure,
                post_trade_net_exposure=request.post_trade_net_exposure,
            )
        )
        if not risk.approved:
            fill = None
            adapter_name = self.adapter.name
            if risk.shadow_only:
                adapter_name = ShadowExecutionAdapter.name
                fill = ShadowExecutionAdapter().submit(request)
            return ExecutionResponse(
                accepted=False,
                adapter=adapter_name,
                shadow_only=risk.shadow_only,
                risk=risk.to_dict(),
                fill=fill,
            )
        fill = self.adapter.submit(request)
        return ExecutionResponse(
            accepted=True,
            adapter=self.adapter.name,
            shadow_only=False,
            risk=risk.to_dict(),
            fill=fill,
        )
