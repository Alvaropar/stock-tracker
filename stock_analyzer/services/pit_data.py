"""
Local point-in-time data store for market context, fundamentals, and sentiment.

Snapshots are append-only JSONL records keyed by namespace + symbol + as-of date.
Training code can align these snapshots onto a price-index without leaking future
values, and live fetchers can persist newly observed snapshots for later reuse.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def _store_root() -> Path:
    return Path(__file__).resolve().parents[2] / "pit_store"


def _iso_day(value: Any) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.strftime("%Y-%m-%d")


def _fundamentals_to_feature_row(snapshot: Dict[str, Any]) -> Dict[str, float]:
    def _score(v, good, okay, weak):
        if v is None:
            return None
        if v >= good:
            return 1.0
        if v >= okay:
            return 0.5
        if v >= weak:
            return -0.5
        return -1.0

    def _score_low(v, good, okay, weak):
        if v is None:
            return None
        if v <= good:
            return 1.0
        if v <= okay:
            return 0.5
        if v <= weak:
            return -0.5
        return -1.0

    pe = snapshot.get("pe_trail")
    pb = snapshot.get("pb")
    peg = snapshot.get("peg")
    roe = snapshot.get("roe")
    net_mgn = snapshot.get("net_mgn")
    rev_g = snapshot.get("rev_growth")
    eps_g = snapshot.get("eps_growth")
    debt_eq = snapshot.get("debt_eq")
    curr_ratio = snapshot.get("curr_ratio")

    val = [v for v in (_score_low(pe, 15, 25, 40), _score_low(pb, 2, 4, 8), _score_low(peg, 1, 2, 3)) if v is not None]
    qual = [v for v in (_score(roe, 20, 10, 0), _score(net_mgn, 20, 10, 0)) if v is not None]
    growth = [v for v in (_score(rev_g, 20, 5, -5), _score(eps_g, 20, 5, -5)) if v is not None]
    safety = [
        v for v in (
            _score_low(debt_eq, 30, 80, 150),
            _score(curr_ratio, 2, 1.2, 0.8),
        ) if v is not None
    ]
    return {
        "fund_value": float(np.mean(val)) if val else 0.0,
        "fund_quality": float(np.mean(qual)) if qual else 0.0,
        "fund_growth": float(np.mean(growth)) if growth else 0.0,
        "fund_safety": float(np.mean(safety)) if safety else 0.0,
    }


@dataclass
class PointInTimeStore:
    base_dir: Path = _store_root()

    def __post_init__(self):
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, namespace: str, symbol: str) -> Path:
        safe_symbol = symbol.replace("/", "_")
        path = self.base_dir / namespace
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{safe_symbol}.jsonl"

    def record_snapshot(
        self,
        namespace: str,
        symbol: str,
        as_of: Any,
        payload: Dict[str, Any],
    ) -> None:
        path = self._path(namespace, symbol)
        record = {"as_of": _iso_day(as_of), "payload": payload}

        existing = self.read_snapshots(namespace, symbol)
        merged = [r for r in existing if r.get("as_of") != record["as_of"]]
        merged.append(record)
        merged.sort(key=lambda r: r.get("as_of", ""))
        with open(path, "w", encoding="utf-8") as f:
            for row in merged:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

    def read_snapshots(self, namespace: str, symbol: str) -> List[Dict[str, Any]]:
        path = self._path(namespace, symbol)
        if not path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        rows.sort(key=lambda r: r.get("as_of", ""))
        return rows

    def snapshot_asof(self, namespace: str, symbol: str, as_of: Any) -> Optional[Dict[str, Any]]:
        as_of_day = _iso_day(as_of)
        chosen = None
        for row in self.read_snapshots(namespace, symbol):
            if row.get("as_of", "") <= as_of_day:
                chosen = row.get("payload", {})
            else:
                break
        return chosen

    def align_snapshots(
        self,
        namespace: str,
        symbol: str,
        index: Iterable[Any],
        transform=None,
        fields: Optional[List[str]] = None,
    ) -> Optional[pd.DataFrame]:
        rows = self.read_snapshots(namespace, symbol)
        if not rows:
            return None

        transformed = []
        for row in rows:
            payload = row.get("payload", {})
            payload = transform(payload) if transform else payload
            transformed.append({"as_of": pd.Timestamp(row["as_of"]), **payload})
        snap_df = pd.DataFrame(transformed).set_index("as_of").sort_index()
        target_idx = pd.DatetimeIndex(pd.to_datetime(list(index))).tz_localize(None)
        aligned = snap_df.reindex(target_idx, method="ffill")
        aligned.index = pd.Index(index)
        if fields:
            for col in fields:
                if col not in aligned.columns:
                    aligned[col] = np.nan
            aligned = aligned[fields]
        return aligned

    def record_market_context(self, as_of: Any, payload: Dict[str, Any]) -> None:
        persistable = {
            k: v for k, v in payload.items()
            if k in {"vix", "spy_trend_bull", "spy_ret_1m", "spy_ret_3m", "breadth_safe"}
        }
        if persistable:
            self.record_snapshot("market", "SPY_CTX", as_of, persistable)

    def align_market_context(self, index: Iterable[Any]) -> Optional[pd.DataFrame]:
        return self.align_snapshots(
            "market",
            "SPY_CTX",
            index,
            fields=["vix", "spy_trend_bull", "spy_ret_1m", "spy_ret_3m", "breadth_safe"],
        )

    def record_fundamentals(self, ticker: str, as_of: Any, payload: Dict[str, Any]) -> None:
        self.record_snapshot("fundamentals", ticker, as_of, payload)

    def align_fundamental_features(self, ticker: str, index: Iterable[Any]) -> Optional[pd.DataFrame]:
        return self.align_snapshots(
            "fundamentals",
            ticker,
            index,
            transform=_fundamentals_to_feature_row,
            fields=["fund_value", "fund_quality", "fund_growth", "fund_safety"],
        )

    def record_sentiment(self, ticker: str, as_of: Any, payload: Dict[str, Any]) -> None:
        self.record_snapshot("sentiment", ticker, as_of, payload)

    def align_sentiment_features(self, ticker: str, index: Iterable[Any]) -> Optional[pd.DataFrame]:
        def _transform(payload: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "sent_score": float(payload.get("score", 0.0)),
                "sent_momentum": float(payload.get("momentum", 0.0)),
                "sent_dispersion": float(payload.get("dispersion", 0.0)),
            }

        return self.align_snapshots(
            "sentiment",
            ticker,
            index,
            transform=_transform,
            fields=["sent_score", "sent_momentum", "sent_dispersion"],
        )
