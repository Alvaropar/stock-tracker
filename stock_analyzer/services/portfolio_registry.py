"""
Portfolio-aware registry admission checks for batches of candidate models.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd


@dataclass
class PortfolioAdmissionCriteria:
    max_avg_correlation: float = 0.85
    max_position_overlap: float = 0.75
    max_avg_gross_exposure: float = 1.00
    max_avg_net_exposure: float = 0.75
    max_avg_daily_turnover: float = 0.35
    min_median_capacity_dollars: float = 5_000_000.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PortfolioAdmissionReport:
    ready: bool
    reasons: List[str]
    metrics: Dict[str, Any]
    criteria: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _pairwise_avg(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    mask = ~np.eye(values.shape[0], dtype=bool)
    pairwise = values[mask]
    return float(pairwise.mean()) if pairwise.size else 0.0


def evaluate_portfolio_candidates(
    candidates: List[Dict[str, Any]],
    criteria: PortfolioAdmissionCriteria | None = None,
) -> PortfolioAdmissionReport:
    crit = criteria or PortfolioAdmissionCriteria()
    reasons: List[str] = []
    if not candidates:
        return PortfolioAdmissionReport(
            ready=False,
            reasons=["no candidate models"],
            metrics={},
            criteria=crit.to_dict(),
        )

    rets = {}
    pos = {}
    capacities = []
    turnover = []
    for cand in candidates:
        ticker = cand["ticker"]
        result = cand["result"]
        wf = result.walk_forward
        if wf is None:
            continue
        rets[ticker] = pd.Series(wf.daily_returns, dtype=float)
        pos[ticker] = pd.Series(wf.position_exposure, dtype=float)
        turnover.append(float(getattr(wf, "avg_daily_turnover", 0.0)))
        df = cand.get("df")
        if df is not None and {"Close", "Volume"}.issubset(df.columns):
            capacities.append(float((df["Close"] * df["Volume"]).tail(60).mean()))

    if not rets:
        return PortfolioAdmissionReport(
            ready=False,
            reasons=["candidates missing walk-forward traces"],
            metrics={},
            criteria=crit.to_dict(),
        )

    ret_df = pd.concat(rets, axis=1).dropna(how="any")
    pos_df = pd.concat(pos, axis=1).fillna(0.0)

    corr_matrix = ret_df.corr().fillna(0.0).values if ret_df.shape[1] > 1 else np.array([[0.0]])
    avg_corr = _pairwise_avg(corr_matrix)

    pos_bin = (pos_df > 0).astype(float)
    overlap_vals = []
    cols = list(pos_bin.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            active = ((pos_bin[cols[i]] + pos_bin[cols[j]]) > 0).sum()
            both = ((pos_bin[cols[i]] == 1) & (pos_bin[cols[j]] == 1)).sum()
            overlap_vals.append(float(both / active) if active else 0.0)
    avg_overlap = float(np.mean(overlap_vals)) if overlap_vals else 0.0

    n_models = max(len(pos_df.columns), 1)
    gross = pos_df.abs().sum(axis=1) / n_models
    net = pos_df.sum(axis=1).abs() / n_models
    avg_gross = float(gross.mean()) if len(gross) else 0.0
    avg_net = float(net.mean()) if len(net) else 0.0
    avg_turnover = float(np.mean(turnover)) if turnover else 0.0
    median_capacity = float(np.median(capacities)) if capacities else 0.0

    metrics = {
        "n_models": len(rets),
        "avg_correlation": round(avg_corr, 4),
        "avg_position_overlap": round(avg_overlap, 4),
        "avg_gross_exposure": round(avg_gross, 4),
        "avg_net_exposure": round(avg_net, 4),
        "avg_daily_turnover": round(avg_turnover, 6),
        "median_capacity_dollars": round(median_capacity, 2),
    }

    if avg_corr > crit.max_avg_correlation:
        reasons.append(f"avg_correlation {avg_corr:.3f} > allowed {crit.max_avg_correlation:.3f}")
    if avg_overlap > crit.max_position_overlap:
        reasons.append(f"avg_position_overlap {avg_overlap:.3f} > allowed {crit.max_position_overlap:.3f}")
    if avg_gross > crit.max_avg_gross_exposure:
        reasons.append(f"avg_gross_exposure {avg_gross:.3f} > allowed {crit.max_avg_gross_exposure:.3f}")
    if avg_net > crit.max_avg_net_exposure:
        reasons.append(f"avg_net_exposure {avg_net:.3f} > allowed {crit.max_avg_net_exposure:.3f}")
    if avg_turnover > crit.max_avg_daily_turnover:
        reasons.append(f"avg_daily_turnover {avg_turnover:.3f} > allowed {crit.max_avg_daily_turnover:.3f}")
    if median_capacity < crit.min_median_capacity_dollars:
        reasons.append(
            f"median_capacity_dollars {median_capacity:,.0f} < required "
            f"{crit.min_median_capacity_dollars:,.0f}"
        )

    return PortfolioAdmissionReport(
        ready=not reasons,
        reasons=reasons or ["portfolio candidate set is admissible"],
        metrics=metrics,
        criteria=crit.to_dict(),
    )
