from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from app.backend.api.backtest import _compute_signal_quality_metrics
from app.backend.server import create_app


def _make_history(seed: int = 0, base: float = 100.0, slope: float = 0.08) -> pd.DataFrame:
    steps = np.arange(280, dtype=float)
    close = (
        base
        + slope * steps
        + 4.0 * np.sin(steps / 8.0 + seed)
        + 1.5 * np.cos(steps / 17.0 + seed / 2.0)
    )
    open_ = close * (1.0 + 0.0015 * np.sin(steps / 5.0 + seed))
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    volume = 1_200_000 + 60_000 * np.sin(steps / 7.0 + seed)
    idx = pd.date_range("2024-01-01", periods=len(steps), freq="B")
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=idx,
    )


class SignalBacktestApiTests(unittest.TestCase):
    def test_signal_quality_metric_rewards_aligned_scores(self):
        n = 260
        raw_scores = np.sin(np.linspace(0, 10 * np.pi, n))
        prices = [100.0]
        for s in raw_scores[:-1]:
            prices.append(prices[-1] * (1.0 + 0.003 * s))

        aligned = _compute_signal_quality_metrics(prices, raw_scores.tolist())
        reversed_ = _compute_signal_quality_metrics(prices, (-raw_scores).tolist())

        self.assertIsNotNone(aligned["signal_quality_score"])
        self.assertIsNotNone(reversed_["signal_quality_score"])
        self.assertGreater(aligned["composite_ic"], 0.0)
        self.assertLess(reversed_["composite_ic"], 0.0)
        self.assertGreater(aligned["signal_quality_score"], 55.0)
        self.assertLess(reversed_["signal_quality_score"], 45.0)

    def test_backtest_route_defaults_to_technical_only(self):
        app = create_app()
        client = app.test_client()

        histories = {
            "SPY": _make_history(seed=10, base=450.0, slope=0.12),
            "AAA": _make_history(seed=1, base=100.0, slope=0.09),
            "BBB": _make_history(seed=2, base=75.0, slope=0.05),
        }

        def _fake_history(ticker: str, period: str = "2y", auto_adjust: bool = True):
            return histories.get(ticker)

        payload = {
            "tickers": ["AAA", "BBB"],
            "period": "2y",
            "technical": ["ma20", "ma50", "ma200", "cross", "rsi", "macd", "bb"],
            "weights": {"technical": 60, "fundamental": 40},
            "fundamentals_map": {
                "AAA": {"pe_trail": 14.0, "net_mgn": 24.0, "roe": 19.0, "rev_growth": 18.0, "target_px": 145.0},
                "BBB": {"pe_trail": 28.0, "net_mgn": 12.0, "roe": 9.0, "rev_growth": 7.0, "target_px": 88.0},
            },
        }

        with patch("app.backend.services.market_data.fetch_price_history", side_effect=_fake_history):
            res = client.post("/api/backtest/run", json=payload)

        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["ok"])
        self.assertEqual(sorted(data["results"].keys()), ["AAA", "BBB"])
        self.assertEqual(data["summary"]["n_tickers"], 2)
        self.assertIn("avg_signal_quality_score", data["summary"])
        self.assertGreater(data["results"]["AAA"]["n_points"], 0)
        self.assertEqual(data["results"]["AAA"]["fundamentals_source"], "disabled")
        self.assertEqual(data["results"]["BBB"]["fundamentals_source"], "disabled")
        self.assertEqual(data["results"]["AAA"]["fund_score"], 0.0)
        self.assertEqual(data["results"]["BBB"]["fund_score"], 0.0)

    def test_backtest_route_uses_supplied_fundamentals_when_enabled(self):
        app = create_app()
        client = app.test_client()

        histories = {
            "SPY": _make_history(seed=10, base=450.0, slope=0.12),
            "AAA": _make_history(seed=1, base=100.0, slope=0.09),
        }

        def _fake_history(ticker: str, period: str = "2y", auto_adjust: bool = True):
            return histories.get(ticker)

        payload = {
            "tickers": ["AAA"],
            "period": "2y",
            "technical": ["ma20", "ma50", "ma200", "cross", "rsi", "macd", "bb"],
            "fundamental": ["pe", "margins", "roe", "growth", "analyst"],
            "weights": {"technical": 60, "fundamental": 40},
            "use_fundamentals": True,
            "fundamentals_map": {
                "AAA": {"pe_trail": 14.0, "net_mgn": 24.0, "roe": 19.0, "rev_growth": 18.0, "target_px": 145.0},
            },
        }

        with patch("app.backend.services.market_data.fetch_price_history", side_effect=_fake_history):
            res = client.post("/api/backtest/run", json=payload)

        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["results"]["AAA"]["fundamentals_source"], "analysis")
        self.assertNotEqual(data["results"]["AAA"]["fund_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
