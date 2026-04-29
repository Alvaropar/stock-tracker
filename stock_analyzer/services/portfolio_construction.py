"""
Cross-sectional portfolio construction from ML predictions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

_ENTRY_SIGNALS = {"BUY", "ENTRY", "STRONG ENTRY", "SPECULATIVE"}


@dataclass
class PortfolioConfig:
    max_positions: int = 8
    gross_target: float = 1.0
    max_weight: float = 0.20
    min_weight: float = 0.04
    max_sector_weight: float = 0.35
    min_score: float = 0.02


@dataclass
class PortfolioCandidate:
    ticker: str
    sector: str
    score: float
    signal: str
    conviction: str
    position_size: float
    regime: str


@dataclass
class PortfolioAllocation:
    ticker: str
    sector: str
    weight: float
    score: float
    regime: str
    conviction: str


@dataclass
class PortfolioPlan:
    allocations: List[PortfolioAllocation]
    gross_exposure: float
    cash_buffer: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allocations": [asdict(a) for a in self.allocations],
            "gross_exposure": self.gross_exposure,
            "cash_buffer": self.cash_buffer,
        }


def build_portfolio_plan(
    candidates: List[PortfolioCandidate],
    config: Optional[PortfolioConfig] = None,
) -> PortfolioPlan:
    cfg = config or PortfolioConfig()
    eligible = [
        c for c in candidates
        if c.signal in _ENTRY_SIGNALS
        and c.position_size > 0
        and c.score >= cfg.min_score
    ]
    eligible.sort(key=lambda c: (-c.score, -c.position_size, c.ticker))
    eligible = eligible[: cfg.max_positions]

    if not eligible:
        return PortfolioPlan(allocations=[], gross_exposure=0.0, cash_buffer=1.0)

    total_score = sum(c.score for c in eligible) or 1.0
    sector_used: Dict[str, float] = {}
    allocations: List[PortfolioAllocation] = []
    gross_used = 0.0

    for cand in eligible:
        raw_weight = cfg.gross_target * (cand.score / total_score)
        capped_weight = min(raw_weight, cfg.max_weight, cand.position_size)
        sector = cand.sector or "Unknown"
        sector_room = max(0.0, cfg.max_sector_weight - sector_used.get(sector, 0.0))
        final_weight = min(capped_weight, sector_room)
        if final_weight < cfg.min_weight:
            continue
        allocations.append(
            PortfolioAllocation(
                ticker=cand.ticker,
                sector=sector,
                weight=round(final_weight, 4),
                score=round(cand.score, 4),
                regime=cand.regime,
                conviction=cand.conviction,
            )
        )
        gross_used += final_weight
        sector_used[sector] = sector_used.get(sector, 0.0) + final_weight

    cash_buffer = max(0.0, round(1.0 - gross_used, 4))
    return PortfolioPlan(
        allocations=allocations,
        gross_exposure=round(gross_used, 4),
        cash_buffer=cash_buffer,
    )
