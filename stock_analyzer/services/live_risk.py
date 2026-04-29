"""
Live-trading preflight risk controls.

This repo does not yet contain a broker integration. Any future live broker
adapter should call `LiveRiskManager.evaluate_order()` before placing capital.
Until the soak-period and kill-switch checks pass, orders are restricted to
shadow mode or rejected outright.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


def _state_path() -> Path:
    root = Path(__file__).resolve().parents[2] / "paper_trading"
    root.mkdir(parents=True, exist_ok=True)
    return root / "risk_state.json"


@dataclass
class LiveRiskControls:
    enabled: bool = True
    shadow_mode: bool = True
    kill_switch: bool = True
    max_daily_loss_pct: float = 0.02
    max_per_name_exposure: float = 0.10
    max_gross_exposure: float = 1.00
    max_net_exposure: float = 0.50
    min_paper_days: int = 14
    min_paper_trades: int = 20

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PaperTradingStats:
    days_completed: int = 0
    trades_completed: int = 0
    observed_days: List[str] = field(default_factory=list)
    trade_days: List[str] = field(default_factory=list)
    last_daily_pnl_pct: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PortfolioSnapshot:
    daily_pnl_pct: float = 0.0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    per_name_exposure: Dict[str, float] = field(default_factory=dict)
    paper: PaperTradingStats = field(default_factory=PaperTradingStats)

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["paper"] = self.paper.to_dict()
        return data


@dataclass
class OrderIntent:
    ticker: str
    target_exposure: float
    post_trade_gross_exposure: float
    post_trade_net_exposure: float

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RiskCheckResult:
    approved: bool
    shadow_only: bool
    reasons: List[str]

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class LiveRiskState:
    paper: PaperTradingStats = field(default_factory=PaperTradingStats)
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    per_name_exposure: Dict[str, float] = field(default_factory=dict)
    daily_pnl_history: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["paper"] = self.paper.to_dict()
        return data


class LiveRiskManager:
    """
    Stateless evaluator for live-order preflight checks.

    `shadow_only=True` means the signal may be logged/paper-traded but must not
    reach a broker.
    """

    def __init__(self, controls: Optional[LiveRiskControls] = None, state_path: Optional[Path] = None):
        self.controls = controls or LiveRiskControls()
        self._state_path = state_path or _state_path()
        self._state = self._load_state()

    def _load_state(self) -> LiveRiskState:
        path = self._state_path
        if not path.exists():
            return LiveRiskState()
        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            paper_raw = raw.get("paper", {})
            state = LiveRiskState(
                paper=PaperTradingStats(**paper_raw),
                gross_exposure=float(raw.get("gross_exposure", 0.0)),
                net_exposure=float(raw.get("net_exposure", 0.0)),
                per_name_exposure=dict(raw.get("per_name_exposure", {})),
                daily_pnl_history=dict(raw.get("daily_pnl_history", {})),
            )
            state.paper.days_completed = len(set(state.paper.observed_days))
            state.paper.trades_completed = max(
                state.paper.trades_completed,
                len(state.paper.trade_days),
            )
            return state
        except Exception:
            return LiveRiskState()

    def _save_state(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._state_path, "w", encoding="utf-8") as f:
            json.dump(self._state.to_dict(), f, indent=2)

    def record_paper_trade(self, as_of: str, ticker: str, pnl_pct: float) -> None:
        day = str(as_of)[:10]
        if day not in self._state.paper.trade_days:
            self._state.paper.trade_days.append(day)
        self._state.paper.trades_completed += 1
        self._state.per_name_exposure.setdefault(ticker, 0.0)
        self._save_state()

    def record_daily_pnl(
        self,
        as_of: str,
        daily_pnl_pct: float,
        gross_exposure: float,
        net_exposure: float,
        per_name_exposure: Optional[Dict[str, float]] = None,
    ) -> None:
        day = str(as_of)[:10]
        if day not in self._state.paper.observed_days:
            self._state.paper.observed_days.append(day)
        self._state.paper.days_completed = len(set(self._state.paper.observed_days))
        self._state.paper.last_daily_pnl_pct = float(daily_pnl_pct)
        self._state.daily_pnl_history[day] = float(daily_pnl_pct)
        self._state.gross_exposure = float(gross_exposure)
        self._state.net_exposure = float(net_exposure)
        if per_name_exposure is not None:
            self._state.per_name_exposure = dict(per_name_exposure)
        self._save_state()

    def snapshot(self) -> PortfolioSnapshot:
        return PortfolioSnapshot(
            daily_pnl_pct=self._state.paper.last_daily_pnl_pct,
            gross_exposure=self._state.gross_exposure,
            net_exposure=self._state.net_exposure,
            per_name_exposure=dict(self._state.per_name_exposure),
            paper=self._state.paper,
        )

    def evaluate_order(
        self,
        intent: OrderIntent,
        portfolio: Optional[PortfolioSnapshot] = None,
    ) -> RiskCheckResult:
        portfolio = portfolio or self.snapshot()
        reasons: List[str] = []
        shadow_only = False

        if not self.controls.enabled:
            return RiskCheckResult(approved=True, shadow_only=True, reasons=["live controls disabled"])

        if self.controls.kill_switch:
            reasons.append("kill switch is active")

        if portfolio.daily_pnl_pct <= -abs(self.controls.max_daily_loss_pct):
            reasons.append(
                f"daily loss {portfolio.daily_pnl_pct:.2%} exceeds "
                f"limit {-abs(self.controls.max_daily_loss_pct):.2%}"
            )

        current_name = portfolio.per_name_exposure.get(intent.ticker, 0.0)
        post_name = max(current_name, abs(intent.target_exposure))
        if post_name > self.controls.max_per_name_exposure:
            reasons.append(
                f"{intent.ticker} exposure {post_name:.2%} exceeds "
                f"limit {self.controls.max_per_name_exposure:.2%}"
            )

        if abs(intent.post_trade_gross_exposure) > self.controls.max_gross_exposure:
            reasons.append(
                f"gross exposure {intent.post_trade_gross_exposure:.2%} exceeds "
                f"limit {self.controls.max_gross_exposure:.2%}"
            )

        if abs(intent.post_trade_net_exposure) > self.controls.max_net_exposure:
            reasons.append(
                f"net exposure {intent.post_trade_net_exposure:.2%} exceeds "
                f"limit {self.controls.max_net_exposure:.2%}"
            )

        if (
            portfolio.paper.days_completed < self.controls.min_paper_days
            or portfolio.paper.trades_completed < self.controls.min_paper_trades
        ):
            shadow_only = True
            reasons.append(
                "paper-trading soak requirement not met "
                f"({portfolio.paper.days_completed}/{self.controls.min_paper_days} days, "
                f"{portfolio.paper.trades_completed}/{self.controls.min_paper_trades} trades)"
            )

        approved = not reasons or (shadow_only and len(reasons) == 1)
        if self.controls.shadow_mode:
            shadow_only = True
            if "shadow mode is active" not in reasons:
                reasons.append("shadow mode is active")
            approved = approved and not self.controls.kill_switch

        if self.controls.kill_switch:
            approved = False

        return RiskCheckResult(
            approved=approved and not shadow_only,
            shadow_only=shadow_only,
            reasons=reasons or ["order approved"],
        )
