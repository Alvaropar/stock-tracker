from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.backend.services.execution import (
    ExecutionRequest,
    GuardedExecutionService,
    PaperExecutionAdapter,
)
from app.backend.services.live_risk import LiveRiskControls, LiveRiskManager
from app.backend.services.ml_engine import MLConfig, _create_model
from app.backend.services.paper_trading import PaperOrder, PaperTradingEngine
from app.backend.services.pit_data import PointInTimeStore
from app.backend.services.portfolio_construction import (
    PortfolioCandidate,
    PortfolioConfig,
    build_portfolio_plan,
)


class TradingStackTests(unittest.TestCase):
    def test_pit_store_aligns_fundamental_and_sentiment_snapshots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = PointInTimeStore(base_dir=Path(tmpdir))
            idx = ["2024-01-10", "2024-01-11", "2024-01-12"]
            store.record_fundamentals(
                "AAPL",
                "2024-01-10",
                {"pe_trail": 12, "pb": 2, "peg": 1.2, "roe": 25, "net_mgn": 22, "rev_growth": 15, "eps_growth": 18, "debt_eq": 20, "curr_ratio": 2.1},
            )
            store.record_sentiment(
                "AAPL",
                "2024-01-11",
                {"score": 0.4, "momentum": 0.1, "dispersion": 0.2},
            )

            fund = store.align_fundamental_features("AAPL", idx)
            sent = store.align_sentiment_features("AAPL", idx)

            self.assertIsNotNone(fund)
            self.assertIsNotNone(sent)
            self.assertGreater(fund.loc["2024-01-12", "fund_quality"], 0.0)
            self.assertAlmostEqual(sent.loc["2024-01-12", "sent_score"], 0.4, places=6)

    def test_paper_engine_persists_daily_pnl_and_soak_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / "ledger.json"
            risk_path = Path(tmpdir) / "risk_state.json"
            risk = LiveRiskManager(
                LiveRiskControls(kill_switch=False, shadow_mode=False, min_paper_days=0, min_paper_trades=0),
                state_path=risk_path,
            )
            engine = PaperTradingEngine(risk_manager=risk, ledger_path=ledger_path)

            fill = engine.submit_order(
                order=PaperOrder(
                    ticker="AAPL",
                    side="BUY",
                    quantity=10,
                    reference_price=100.0,
                    submitted_at="2024-01-10",
                    reason="unit_test",
                )
            )
            snap = engine.mark_to_market("2024-01-10", {"AAPL": 105.0})

            self.assertTrue(ledger_path.exists())
            self.assertGreater(fill.fill_price, 100.0)
            self.assertGreater(snap.equity, 0.0)
            self.assertEqual(risk.snapshot().paper.days_completed, 1)

    def test_guarded_execution_rejects_until_soak_requirements_met(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / "ledger.json"
            risk_path = Path(tmpdir) / "risk_state.json"
            risk = LiveRiskManager(
                LiveRiskControls(kill_switch=False, shadow_mode=False, min_paper_days=5, min_paper_trades=5),
                state_path=risk_path,
            )
            adapter = PaperExecutionAdapter(PaperTradingEngine(risk_manager=risk, ledger_path=ledger_path))
            service = GuardedExecutionService(adapter=adapter, risk_manager=risk)

            resp = service.submit(
                ExecutionRequest(
                    ticker="AAPL",
                    side="BUY",
                    quantity=5,
                    reference_price=100.0,
                    submitted_at="2024-01-10",
                    target_exposure=0.05,
                    post_trade_gross_exposure=0.05,
                    post_trade_net_exposure=0.05,
                    reason="unit_test",
                )
            )

            self.assertFalse(resp.accepted)
            self.assertTrue(resp.risk["shadow_only"])
            self.assertEqual(resp.adapter, "shadow")
            self.assertEqual(resp.fill["status"], "shadow_logged")

    def test_portfolio_construction_ranks_and_caps_candidates(self):
        plan = build_portfolio_plan(
            [
                PortfolioCandidate("AAA", "Tech", 0.30, "STRONG ENTRY", "HIGH", 0.20, "TREND_UP"),
                PortfolioCandidate("BBB", "Tech", 0.25, "ENTRY", "HIGH", 0.20, "TREND_UP"),
                PortfolioCandidate("CCC", "Energy", 0.10, "BUY", "MEDIUM", 0.15, "REVERSAL_UP"),
                PortfolioCandidate("DDD", "Tech", 0.50, "HOLD", "NONE", 0.0, "RANGE"),
            ],
            PortfolioConfig(max_positions=3, max_weight=0.20, max_sector_weight=0.35),
        )

        self.assertGreater(plan.gross_exposure, 0.0)
        self.assertLessEqual(plan.gross_exposure, 1.0)
        self.assertTrue(all(a.weight <= 0.20 for a in plan.allocations))
        self.assertTrue(all(a.ticker != "DDD" for a in plan.allocations))

    def test_model_factory_supports_ensemble(self):
        model = _create_model(MLConfig(model_type="ensemble"))
        self.assertEqual(model.__class__.__name__, "_EnsembleModels")


if __name__ == "__main__":
    unittest.main()
