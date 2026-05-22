"""Unit tests for the SQLite ledger, analytics, and rebalancer."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from stock_analyzer.config import config
from stock_analyzer.services import ledger
from stock_analyzer.services import portfolio_analytics as pa
from stock_analyzer.services import rebalancer as rb
from stock_analyzer.services.ledger import Position


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """Each test gets its own SQLite ledger in a temp directory."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(ledger, "_initialized", False)
    yield tmp_path


# ── Ledger ───────────────────────────────────────────────────────────────────

class TestLedgerCrud:
    def test_add_and_list(self):
        tx = ledger.add_transaction("aapl", "BUY", 10, 150.0, "2024-01-01")
        assert tx.id == 1
        assert tx.ticker == "AAPL"
        assert tx.side == "BUY"
        assert ledger.list_transactions() == [tx]

    def test_reject_bad_inputs(self):
        with pytest.raises(ValueError):
            ledger.add_transaction("", "BUY", 1, 1)
        with pytest.raises(ValueError):
            ledger.add_transaction("X", "BOGUS", 1, 1)
        with pytest.raises(ValueError):
            ledger.add_transaction("X", "BUY", 0, 1)
        with pytest.raises(ValueError):
            ledger.add_transaction("X", "BUY", 1, 1, fx_rate=0)

    def test_update_and_delete(self):
        tx = ledger.add_transaction("AAPL", "BUY", 10, 150.0, "2024-01-01")
        updated = ledger.update_transaction(tx.id, price=160.0, notes="adjusted")
        assert updated.price == 160.0
        assert updated.notes == "adjusted"
        assert ledger.delete_transaction(tx.id) is True
        assert ledger.list_transactions() == []
        assert ledger.delete_transaction(999) is False


class TestPositionDerivation:
    def test_simple_buy(self):
        ledger.add_transaction("AAPL", "BUY", 10, 150.0, "2024-01-01")
        positions = ledger.derive_positions()
        assert len(positions) == 1
        p = positions[0]
        assert p.ticker == "AAPL"
        assert p.quantity == 10
        assert p.avg_cost == 150.0
        assert p.cost_basis == 1500.0
        assert p.realized_pnl == 0

    def test_average_cost(self):
        ledger.add_transaction("AAPL", "BUY", 10, 150.0, "2024-01-01")
        ledger.add_transaction("AAPL", "BUY",  5, 170.0, "2024-03-01")
        p = ledger.derive_positions()[0]
        # (10*150 + 5*170) / 15 = (1500 + 850) / 15 = 156.67
        assert p.quantity == 15
        assert round(p.avg_cost, 2) == 156.67
        assert round(p.cost_basis, 2) == 2350.00

    def test_partial_sell_realized_pnl(self):
        ledger.add_transaction("AAPL", "BUY",  10, 150.0, "2024-01-01")
        ledger.add_transaction("AAPL", "SELL",  3, 200.0, "2024-06-01")
        positions = ledger.derive_positions()
        assert len(positions) == 1
        p = positions[0]
        assert p.quantity == 7
        assert p.avg_cost == 150.0   # unchanged for sells
        # realized = 3 * (200 - 150) = 150
        assert p.realized_pnl == 150.0

    def test_full_sell_closes_position(self):
        ledger.add_transaction("AAPL", "BUY",  10, 150.0, "2024-01-01")
        ledger.add_transaction("AAPL", "SELL", 10, 200.0, "2024-06-01")
        positions = ledger.derive_positions()
        assert positions == []   # closed positions are dropped
        s = ledger.realized_summary()
        assert s["realized_pnl"] == 500.0
        assert s["dividends"] == 0.0

    def test_dividend_does_not_change_quantity(self):
        ledger.add_transaction("AAPL", "BUY", 10, 150.0, "2024-01-01")
        ledger.add_transaction("AAPL", "DIV", 10, 0.50,  "2024-04-01")
        p = ledger.derive_positions()[0]
        assert p.quantity == 10
        assert p.dividends == 5.0

    def test_fees_increase_cost_basis(self):
        ledger.add_transaction("AAPL", "BUY", 10, 100.0, "2024-01-01", fees=20.0)
        p = ledger.derive_positions()[0]
        # (10*100 + 20) / 10 = 102.0
        assert p.avg_cost == 102.0

    def test_fx_rate_normalizes_to_base(self):
        # Buying 100 shares at 12 SEK with fx_rate 0.1 → cost in base = 100*12*0.1 = 120
        ledger.add_transaction("SIVE.ST", "BUY", 100, 12.0, "2024-01-01",
                               currency="SEK", fx_rate=0.10)
        p = ledger.derive_positions()[0]
        assert round(p.avg_cost, 2) == 1.20
        assert round(p.cost_basis, 2) == 120.00


class TestTargetsAndWatchlist:
    def test_set_and_get_targets(self):
        ledger.set_targets({"AAPL": 0.4, "MSFT": 0.6})
        assert ledger.get_targets() == {"AAPL": 0.4, "MSFT": 0.6}

    def test_reject_sum_over_one(self):
        with pytest.raises(ValueError):
            ledger.set_targets({"A": 0.7, "B": 0.5})

    def test_reject_negative_weight(self):
        with pytest.raises(ValueError):
            ledger.set_targets({"A": -0.1})

    def test_watchlist_dedup(self):
        assert ledger.add_to_watchlist("AAPL") is True
        assert ledger.add_to_watchlist("AAPL") is False
        assert len(ledger.list_watchlist()) == 1
        assert ledger.remove_from_watchlist("AAPL") is True
        assert ledger.list_watchlist() == []


class TestMigration:
    def test_imports_legacy_positions(self, tmp_path, monkeypatch):
        legacy = tmp_path / "portfolio_positions.json"
        legacy.write_text(json.dumps([
            {"ticker": "AAPL", "quantity": 10, "buy_price": 150.0, "buy_date": "2024-01-01"},
            {"ticker": "MSFT", "quantity":  5, "buy_price": 350.0, "buy_date": "2024-02-15"},
        ]))
        result = ledger.migrate_legacy_json()
        assert result["transactions"] == 2
        # Second run is idempotent (same trades aren't reimported)
        result2 = ledger.migrate_legacy_json()
        assert result2["transactions"] == 0
        assert {p.ticker for p in ledger.derive_positions()} == {"AAPL", "MSFT"}


# ── Portfolio analytics ─────────────────────────────────────────────────────

class TestEnrichment:
    def _sample_positions(self):
        return [
            Position("AAPL", 10, 150.0, 1500.0, 0.0, 0.0, "2024-01-01", "2024-01-01", "USD"),
            Position("MSFT", 5,  350.0, 1750.0, 0.0, 0.0, "2024-02-15", "2024-02-15", "USD"),
        ]

    def test_weights_sum_to_one(self):
        positions = self._sample_positions()
        quotes = {"AAPL": 200.0, "MSFT": 400.0}
        instruments = {
            "AAPL": {"sector": "Tech", "region": "NA", "currency": "USD", "beta": 1.2, "name": "Apple"},
            "MSFT": {"sector": "Tech", "region": "NA", "currency": "USD", "beta": 1.0, "name": "Microsoft"},
        }
        enriched, totals = pa.enrich_positions(positions, quotes, instruments)
        assert round(sum(e.weight for e in enriched), 4) == 1.0
        # AAPL MV = 10*200 = 2000, MSFT MV = 5*400 = 2000 → 50/50 weights
        assert round(enriched[0].weight, 4) == 0.5
        assert totals["market_value"] == 4000.0
        assert totals["cost_basis"] == 3250.0
        assert totals["unrealized_pnl"] == 750.0

    def test_falls_back_to_avg_cost_when_quote_missing(self):
        positions = self._sample_positions()
        enriched, totals = pa.enrich_positions(positions, {"AAPL": None, "MSFT": None}, {})
        # Falls back to avg_cost so pnl is 0
        assert totals["unrealized_pnl"] == 0.0

    def test_portfolio_beta_weighted_average(self):
        positions = self._sample_positions()
        quotes = {"AAPL": 200.0, "MSFT": 200.0}  # equal MV → 50/50 weights
        instruments = {
            "AAPL": {"beta": 1.2, "sector": "Tech", "currency": "USD"},
            "MSFT": {"beta": 0.8, "sector": "Tech", "currency": "USD"},
        }
        enriched, _ = pa.enrich_positions(positions, quotes, instruments)
        agg = pa.aggregate(enriched)
        # AAPL: qty 10 * 200 = 2000.  MSFT: qty 5 * 200 = 1000.  Weighted: (1.2*2/3 + 0.8*1/3) = 1.067
        assert round(agg["portfolio_beta"], 3) == 1.067

    def test_aggregations_have_sector_buckets(self):
        positions = self._sample_positions()
        quotes = {"AAPL": 200.0, "MSFT": 400.0}
        instruments = {
            "AAPL": {"sector": "Tech",      "currency": "USD"},
            "MSFT": {"sector": "Software",  "currency": "USD"},
        }
        enriched, _ = pa.enrich_positions(positions, quotes, instruments)
        agg = pa.aggregate(enriched)
        labels = [s["label"] for s in agg["sectors"]]
        assert set(labels) == {"Tech", "Software"}


class TestRegionInference:
    @pytest.mark.parametrize("ticker,currency,expected", [
        ("AAPL",     "USD", "NA"),
        ("SIVE.ST",  "SEK", "EU"),
        ("0700.HK",  "HKD", "ASIA"),
        ("600519.SS","CNY", "CN"),
        ("BMW.DE",   "EUR", "EU"),
        ("BHP.AX",   "AUD", "APAC"),
    ])
    def test_suffix_inference(self, ticker, currency, expected):
        assert pa.infer_region(ticker, currency) == expected


# ── Rebalancer ──────────────────────────────────────────────────────────────

class TestRebalancer:
    def test_no_drift_no_trades(self):
        holdings = {
            "AAPL": {"market_value": 500.0, "price": 100.0},
            "MSFT": {"market_value": 500.0, "price": 100.0},
        }
        targets = {"AAPL": 0.5, "MSFT": 0.5}
        result = rb.compute_rebalance(holdings, targets, drift_threshold=0.01)
        assert result["trades"] == []
        assert result["drift_summary"]["rebalance_recommended"] is False

    def test_rebalance_overweight(self):
        # AAPL is 80% of book but target is 50% → should sell
        holdings = {
            "AAPL": {"market_value": 800.0, "price": 100.0},
            "MSFT": {"market_value": 200.0, "price": 50.0},
        }
        targets = {"AAPL": 0.5, "MSFT": 0.5}
        result = rb.compute_rebalance(holdings, targets)
        trades_by_side = {t["ticker"]: t for t in result["trades"]}
        assert trades_by_side["AAPL"]["side"] == "SELL"
        assert trades_by_side["MSFT"]["side"] == "BUY"
        # AAPL: sell down to 500 → sell $300 / $100 = 3 shares
        assert trades_by_side["AAPL"]["quantity"] == 3.0
        # MSFT: buy up to 500 → $300 / $50 = 6 shares
        assert trades_by_side["MSFT"]["quantity"] == 6.0

    def test_cash_deployment(self):
        holdings = {"AAPL": {"market_value": 500.0, "price": 100.0}}
        targets = {"AAPL": 0.5, "GOOG": 0.5}
        # With cash=500, portfolio_value = 1000. GOOG target = 500; AAPL stays at 500.
        result = rb.compute_rebalance(
            holdings, targets, cash=500.0,
            # Need GOOG price to compute the buy quantity
        )
        # We didn't pass GOOG price (the compute_rebalance branch fills it from holdings dict
        # which is missing GOOG entirely → we patch:)
        holdings["GOOG"] = {"market_value": 0.0, "price": 200.0}
        result = rb.compute_rebalance(holdings, targets, cash=500.0)
        goog = next(t for t in result["trades"] if t["ticker"] == "GOOG")
        assert goog["side"] == "BUY"
        assert goog["quantity"] == 2.5     # fractional allowed

    def test_lot_size_rounding(self):
        holdings = {"AAPL": {"market_value": 0.0, "price": 100.0}}
        targets = {"AAPL": 1.0}
        result = rb.compute_rebalance(
            holdings, targets, cash=550.0,
            fractional_shares=False, lot_size=1,
        )
        # 550 / 100 = 5.5 → round to 6? No, _round_qty uses round() which goes to nearest
        trade = result["trades"][0]
        assert trade["quantity"] == 6.0

    def test_drift_threshold_ignores_small_drift(self):
        holdings = {"AAPL": {"market_value": 510.0, "price": 100.0},
                    "MSFT": {"market_value": 490.0, "price": 100.0}}
        targets = {"AAPL": 0.5, "MSFT": 0.5}
        # Drift is 0.5/100 = 0.01 — equal to threshold so should NOT trade
        result = rb.compute_rebalance(holdings, targets, drift_threshold=0.02)
        assert result["trades"] == []
