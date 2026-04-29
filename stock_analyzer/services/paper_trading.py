"""
Persistent paper-trading ledger with broker-like fills and daily MTM snapshots.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .live_risk import LiveRiskManager


def _ledger_path() -> Path:
    root = Path(__file__).resolve().parents[2] / "paper_trading"
    root.mkdir(parents=True, exist_ok=True)
    return root / "ledger.json"


@dataclass
class PaperTradingConfig:
    initial_cash: float = 100_000.0
    commission_bps: float = 10.0
    slippage_bps: float = 5.0
    spread_bps: float = 2.0


@dataclass
class PaperOrder:
    ticker: str
    side: str
    quantity: float
    reference_price: float
    submitted_at: str
    reason: str = ""
    order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])


@dataclass
class PaperFill:
    order_id: str
    ticker: str
    side: str
    quantity: float
    reference_price: float
    fill_price: float
    notional: float
    commission: float
    slippage_cost: float
    filled_at: str
    reason: str = ""


@dataclass
class PaperPosition:
    ticker: str
    quantity: float
    avg_price: float


@dataclass
class DailyPnlSnapshot:
    as_of: str
    cash: float
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    daily_pnl: float
    gross_exposure: float
    net_exposure: float


@dataclass
class PaperLedger:
    config: PaperTradingConfig
    cash: float
    realized_pnl: float = 0.0
    positions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    fills: List[Dict[str, Any]] = field(default_factory=list)
    daily: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": asdict(self.config),
            "cash": self.cash,
            "realized_pnl": self.realized_pnl,
            "positions": self.positions,
            "fills": self.fills,
            "daily": self.daily,
        }


class PaperTradingEngine:
    def __init__(
        self,
        config: Optional[PaperTradingConfig] = None,
        risk_manager: Optional[LiveRiskManager] = None,
        ledger_path: Optional[Path] = None,
    ):
        self.config = config or PaperTradingConfig()
        self.risk_manager = risk_manager or LiveRiskManager()
        self._ledger_path = ledger_path or _ledger_path()
        self.ledger = self._load()

    def _load(self) -> PaperLedger:
        path = self._ledger_path
        if not path.exists():
            return PaperLedger(config=self.config, cash=self.config.initial_cash)
        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            cfg = PaperTradingConfig(**raw.get("config", {}))
            return PaperLedger(
                config=cfg,
                cash=float(raw.get("cash", cfg.initial_cash)),
                realized_pnl=float(raw.get("realized_pnl", 0.0)),
                positions=dict(raw.get("positions", {})),
                fills=list(raw.get("fills", [])),
                daily=list(raw.get("daily", [])),
            )
        except Exception:
            return PaperLedger(config=self.config, cash=self.config.initial_cash)

    def _save(self) -> None:
        self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._ledger_path, "w", encoding="utf-8") as f:
            json.dump(self.ledger.to_dict(), f, indent=2)

    def reset(self) -> PaperLedger:
        self.ledger = PaperLedger(config=self.config, cash=self.config.initial_cash)
        self._save()
        return self.ledger

    def submit_order(self, order: PaperOrder) -> PaperFill:
        bps = (self.config.spread_bps + self.config.slippage_bps) / 10_000.0
        fill_price = order.reference_price * (1 + bps if order.side.upper() == "BUY" else 1 - bps)

        side = order.side.upper()
        pos = self.ledger.positions.get(order.ticker, {"quantity": 0.0, "avg_price": 0.0})
        prev_qty = float(pos["quantity"])
        prev_avg = float(pos["avg_price"])
        exec_qty = float(order.quantity if side == "BUY" else min(order.quantity, prev_qty))
        notional = fill_price * exec_qty
        commission = notional * (self.config.commission_bps / 10_000.0)
        slippage_cost = abs(fill_price - order.reference_price) * exec_qty

        if side == "BUY" and exec_qty > 0:
            new_qty = prev_qty + exec_qty
            total_cost = (prev_qty * prev_avg) + notional + commission
            pos["quantity"] = new_qty
            pos["avg_price"] = total_cost / new_qty if new_qty > 0 else 0.0
            self.ledger.cash -= (notional + commission)
        else:
            proceeds = fill_price * exec_qty - commission
            realized = (fill_price - prev_avg) * exec_qty - commission
            pos["quantity"] = prev_qty - exec_qty
            if pos["quantity"] <= 1e-9:
                pos["quantity"] = 0.0
                pos["avg_price"] = 0.0
            self.ledger.cash += proceeds
            self.ledger.realized_pnl += realized

        if pos["quantity"] > 0:
            self.ledger.positions[order.ticker] = pos
        elif order.ticker in self.ledger.positions:
            del self.ledger.positions[order.ticker]

        fill = PaperFill(
            order_id=order.order_id,
            ticker=order.ticker,
            side=side,
            quantity=exec_qty,
            reference_price=order.reference_price,
            fill_price=fill_price,
            notional=notional,
            commission=commission,
            slippage_cost=slippage_cost,
            filled_at=order.submitted_at,
            reason=order.reason,
        )
        self.ledger.fills.append(asdict(fill))
        self.risk_manager.record_paper_trade(order.submitted_at, order.ticker, 0.0)
        self._save()
        return fill

    def mark_to_market(self, as_of: str, prices: Dict[str, float]) -> DailyPnlSnapshot:
        unrealized = 0.0
        market_values: Dict[str, float] = {}
        cost_bases: Dict[str, float] = {}
        equity = self.ledger.cash
        for ticker, pos in self.ledger.positions.items():
            qty = float(pos["quantity"])
            avg = float(pos["avg_price"])
            px = float(prices.get(ticker, avg))
            market_value = qty * px
            cost_basis = qty * avg
            market_values[ticker] = market_value
            cost_bases[ticker] = cost_basis
            equity += market_value
        gross = 0.0
        net = 0.0
        per_name_exposure: Dict[str, float] = {}
        for ticker, market_value in market_values.items():
            unrealized += market_value - cost_bases[ticker]
            exposure = market_value / equity if equity > 0 else 0.0
            per_name_exposure[ticker] = exposure
            gross += abs(exposure)
            net += exposure

        prev_equity = float(self.ledger.daily[-1]["equity"]) if self.ledger.daily else self.ledger.config.initial_cash
        daily_pnl = equity - prev_equity
        snap = DailyPnlSnapshot(
            as_of=str(as_of)[:10],
            cash=self.ledger.cash,
            equity=equity,
            realized_pnl=self.ledger.realized_pnl,
            unrealized_pnl=unrealized,
            daily_pnl=daily_pnl,
            gross_exposure=gross,
            net_exposure=net,
        )
        self.ledger.daily.append(asdict(snap))
        daily_pnl_pct = (daily_pnl / prev_equity) if prev_equity else 0.0
        self.risk_manager.record_daily_pnl(
            snap.as_of,
            daily_pnl_pct=daily_pnl_pct,
            gross_exposure=gross,
            net_exposure=net,
            per_name_exposure=per_name_exposure,
        )
        self._save()
        return snap

    def status(self) -> Dict[str, Any]:
        latest = self.ledger.daily[-1] if self.ledger.daily else None
        return {
            "cash": round(self.ledger.cash, 2),
            "realized_pnl": round(self.ledger.realized_pnl, 2),
            "positions": self.ledger.positions,
            "fills": self.ledger.fills[-50:],
            "latest_daily": latest,
            "risk_snapshot": self.risk_manager.snapshot().to_dict(),
        }
