from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from app.backend.services.backtest import BacktestConfig, Backtester
from app.backend.services.live_risk import (
    LiveRiskControls,
    LiveRiskManager,
    OrderIntent,
    PaperTradingStats,
    PortfolioSnapshot,
)
from app.backend.services.market_data import compute_indicators
from app.backend.services.ml_engine import (
    MLEngine,
    REGIME_CLASSES,
    _purged_train_cutoff,
    _universe_holdout_masks,
)


class _StubModels:
    def __init__(
        self,
        probs: np.ndarray,
        entry_scores: np.ndarray,
        exit_scores: np.ndarray,
        uncertainty_rows=None,
    ):
        self.feature_names = ["dummy_feature"]
        self._probs = probs
        self._entry_scores = entry_scores
        self._exit_scores = exit_scores
        self._mc_uncertainty_rows = uncertainty_rows

    def predict(self, X, mc_passes=1):
        n = len(X)
        return (
            self._probs[:n],
            self._entry_scores[:n],
            self._exit_scores[:n],
            REGIME_CLASSES,
        )

    def feature_importance_dict(self):
        return {}


def _stub_engine(
    probs: np.ndarray,
    entry_scores: np.ndarray,
    exit_scores: np.ndarray,
    *,
    range_sharpe=None,
    uncertainty_rows=None,
    model_type: str = "lightgbm",
):
    return SimpleNamespace(
        _ticker="TEST",
        _models=_StubModels(probs, entry_scores, exit_scores, uncertainty_rows=uncertainty_rows),
        _range_regime_sharpe=range_sharpe,
        config=SimpleNamespace(model_type=model_type, mc_dropout_passes=20),
    )


def _ohlcv_frame() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "Open": [100.0, 110.0, 120.0, 130.0, 140.0],
            "High": [101.0, 111.0, 121.0, 131.0, 141.0],
            "Low": [99.0, 109.0, 119.0, 129.0, 139.0],
            "Close": [100.0, 111.0, 121.0, 131.0, 141.0],
            "Volume": [1_000_000] * 5,
        },
        index=idx,
    )


class MoneyPathTests(unittest.TestCase):
    def test_purged_train_cutoff_uses_forward_horizon(self):
        self.assertEqual(_purged_train_cutoff(100, gap=5, forward_horizon=21), 79)

    def test_universe_date_split_applies_calendar_purge(self):
        unique_dates = pd.date_range("2024-01-01", periods=20, freq="D").values
        dates = np.repeat(unique_dates, 2)
        train_mask, test_mask = _universe_holdout_masks(dates, forward_horizon=2, holdout_frac=0.2)

        train_dates = np.unique(dates[train_mask])
        test_dates = np.unique(dates[test_mask])

        self.assertEqual(str(train_dates[-1])[:10], "2024-01-14")
        self.assertEqual(str(test_dates[0])[:10], "2024-01-17")

    def test_next_bar_execution_uses_next_open(self):
        df = _ohlcv_frame()
        probs = np.array(
            [
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
            ],
            dtype=float,
        )
        entry = np.array([0.90, 0.10, 0.10, 0.10, 0.10], dtype=float)
        exit_ = np.array([0.10, 0.90, 0.10, 0.10, 0.10], dtype=float)
        engine = _stub_engine(probs, entry, exit_)

        result = Backtester(
            BacktestConfig(
                benchmark=None,
                commission_pct=0.0,
                slippage_pct=0.0,
                market_impact_coeff=0.0,
            )
        ).run(engine, df)

        self.assertEqual(len(result.trades), 1)
        trade = result.trades[0]
        self.assertEqual(trade.entry_date, "2024-01-02")
        self.assertEqual(trade.exit_date, "2024-01-03")
        self.assertAlmostEqual(trade.entry_price, 110.0, places=6)
        self.assertAlmostEqual(trade.exit_price, 120.0, places=6)

    def test_partial_reduce_sells_fraction_of_position(self):
        df = _ohlcv_frame()
        probs = np.array([[1, 0, 0, 0, 0]] * 5, dtype=float)
        entry = np.array([0.90, 0.10, 0.10, 0.10, 0.10], dtype=float)
        exit_ = np.array([0.10, 0.65, 0.90, 0.10, 0.10], dtype=float)
        engine = _stub_engine(probs, entry, exit_)

        result = Backtester(
            BacktestConfig(
                benchmark=None,
                commission_pct=0.0,
                slippage_pct=0.0,
                market_impact_coeff=0.0,
            )
        ).run(engine, df)

        self.assertEqual(len(result.trades), 2)
        self.assertGreater(result.trades[0].shares, result.trades[1].shares)
        self.assertGreater(result.trades[1].shares, 0.0)

    def test_predict_blocks_high_uncertainty_entry(self):
        probs = np.array([[1, 0, 0, 0, 0]], dtype=float)
        entry = np.array([0.95], dtype=float)
        exit_ = np.array([0.05], dtype=float)
        engine = MLEngine()
        engine._models = _StubModels(
            probs,
            entry,
            exit_,
            uncertainty_rows=[{"entry_std": 0.25, "regime_std": 0.20, "exit_std": 0.0}],
        )

        pred = engine.predict_from_df(_ohlcv_frame().iloc[:1])

        self.assertNotEqual(pred.decision["action"], "BUY")
        self.assertIsNotNone(pred.uncertainty)

    def test_predict_blocks_range_entries_when_range_sharpe_negative(self):
        probs = np.array([[0, 0, 0, 0, 1]], dtype=float)
        entry = np.array([0.95], dtype=float)
        exit_ = np.array([0.05], dtype=float)
        engine = MLEngine()
        engine._models = _StubModels(probs, entry, exit_)
        engine._range_regime_sharpe = -0.25

        pred = engine.predict_from_df(_ohlcv_frame().iloc[:1])

        self.assertNotEqual(pred.decision["action"], "BUY")
        self.assertTrue(
            any("RANGE entry blocked" in reason for reason in pred.decision["reasons"])
        )

    def test_predict_uses_optimized_policy_thresholds(self):
        probs = np.array([[1, 0, 0, 0, 0]], dtype=float)
        entry = np.array([0.90], dtype=float)
        exit_ = np.array([0.10], dtype=float)
        engine = MLEngine()
        engine._models = _StubModels(probs, entry, exit_)
        engine._policy_opt = {
            "best_entry_threshold": 0.95,
            "best_exit_threshold": 0.60,
            "best_min_regime_confidence": 0.35,
            "best_min_score_spread": 0.05,
        }

        pred = engine.predict_from_df(_ohlcv_frame().iloc[:1])

        self.assertNotEqual(pred.decision["action"], "BUY")
        self.assertEqual(pred.policy["entry_threshold"], 0.95)

    def test_predict_blocks_low_score_spread_entry(self):
        probs = np.array([[1, 0, 0, 0, 0]], dtype=float)
        entry = np.array([0.64], dtype=float)
        exit_ = np.array([0.57], dtype=float)
        engine = MLEngine()
        engine._models = _StubModels(probs, entry, exit_)

        pred = engine.predict_from_df(_ohlcv_frame().iloc[:1])

        self.assertNotEqual(pred.decision["action"], "BUY")
        self.assertTrue(
            any("score_spread" in reason for reason in pred.decision["reasons"])
        )

    def test_compute_indicators_handles_short_history(self):
        idx = pd.date_range("2024-01-01", periods=15, freq="D")
        short_df = pd.DataFrame(
            {
                "Open": np.linspace(100, 114, 15),
                "High": np.linspace(101, 115, 15),
                "Low": np.linspace(99, 113, 15),
                "Close": np.linspace(100, 114, 15),
                "Volume": np.full(15, 1_000_000),
            },
            index=idx,
        )

        out = compute_indicators(short_df)

        self.assertIn("Vol_Pctl", out.columns)
        self.assertFalse(out["Vol_Pctl"].dropna().empty)

    def test_live_risk_manager_requires_soak_period_and_kill_switch(self):
        controls = LiveRiskControls(kill_switch=True, shadow_mode=False)
        mgr = LiveRiskManager(controls)
        snapshot = PortfolioSnapshot(
            daily_pnl_pct=0.0,
            gross_exposure=0.2,
            net_exposure=0.1,
            per_name_exposure={"TEST": 0.05},
            paper=PaperTradingStats(days_completed=2, trades_completed=3),
        )
        intent = OrderIntent(
            ticker="TEST",
            target_exposure=0.08,
            post_trade_gross_exposure=0.28,
            post_trade_net_exposure=0.18,
        )

        result = mgr.evaluate_order(intent, snapshot)

        self.assertFalse(result.approved)
        self.assertTrue(any("kill switch" in reason for reason in result.reasons))
        self.assertTrue(any("paper-trading soak requirement" in reason for reason in result.reasons))


if __name__ == "__main__":
    unittest.main()
