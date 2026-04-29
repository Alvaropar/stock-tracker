"""
Comprehensive pytest test suite for the commodity-trading pipeline package.

Run from project root:
    pytest tests/ -v
"""
from __future__ import annotations

import json
from datetime import datetime

import pytest


# ===========================================================================
# 1. Test base_scraper.py (parse_date, Article, BaseScraper._deduplicate)
# ===========================================================================

from pipeline.scrapers.base_scraper import parse_date, Article, BaseScraper


class TestParseDate:
    def test_rfc2822(self):
        result = parse_date("Mon, 10 Feb 2026 12:00:00 GMT")
        assert result == "2026-02-10"

    def test_iso8601(self):
        result = parse_date("2026-02-10T15:30:00Z")
        assert result == "2026-02-10"

    def test_iso8601_with_offset(self):
        result = parse_date("2026-02-10T15:30:00+05:00")
        assert result == "2026-02-10"

    def test_empty_string_returns_today(self):
        result = parse_date("")
        expected = datetime.now().strftime("%Y-%m-%d")
        assert result == expected

    def test_american_format(self):
        result = parse_date("February 10, 2026")
        assert result == "2026-02-10"

    def test_british_format(self):
        result = parse_date("10 Feb 2026")
        assert result == "2026-02-10"

    def test_garbage_returns_today(self):
        result = parse_date("not a date")
        expected = datetime.now().strftime("%Y-%m-%d")
        assert result == expected


class TestArticle:
    def test_to_dict_includes_headline_alias(self):
        a = Article(title="Gold rises", date="2026-02-10")
        d = a.to_dict()
        assert "headline" in d
        assert d["headline"] == "Gold rises"
        assert d["title"] == "Gold rises"

    def test_default_fields(self):
        a = Article(title="Test", date="2026-01-01")
        assert a.source == ""
        assert a.url == ""
        assert a.summary == ""
        assert a.ticker == ""
        assert a.datetime == ""

    def test_sentiment_initially_none(self):
        a = Article(title="Test", date="2026-01-01")
        assert a.sentiment is None
        assert a.relevant is None


class TestDeduplication:
    def test_removes_case_insensitive_duplicates(self):
        articles = [
            Article(title="Gold Rises", date="2026-02-10"),
            Article(title="gold rises", date="2026-02-10"),
            Article(title="GOLD RISES", date="2026-02-10"),
            Article(title="Silver Falls", date="2026-02-10"),
        ]
        result = BaseScraper._deduplicate(articles)
        assert len(result) == 2
        assert result[0].title == "Gold Rises"
        assert result[1].title == "Silver Falls"

    def test_empty_list(self):
        assert BaseScraper._deduplicate([]) == []

    def test_no_duplicates_unchanged(self):
        articles = [
            Article(title="A", date="2026-01-01"),
            Article(title="B", date="2026-01-01"),
        ]
        result = BaseScraper._deduplicate(articles)
        assert len(result) == 2

    def test_whitespace_handling(self):
        articles = [
            Article(title="  Gold Rises  ", date="2026-02-10"),
            Article(title="gold rises", date="2026-02-10"),
        ]
        result = BaseScraper._deduplicate(articles)
        assert len(result) == 1


# ===========================================================================
# 2. Test relevance_filter.py (_parse_results)
# ===========================================================================

from pipeline.filters.relevance_filter import _parse_results


class TestParseResults:
    def test_valid_json_array(self):
        result = _parse_results('["yes", "no", "yes"]', expected=3)
        assert result == ["yes", "no", "yes"]

    def test_boolean_values(self):
        result = _parse_results('[true, false, true]', expected=3)
        assert result == ["yes", "no", "yes"]

    def test_with_think_block(self):
        text = '<think>reasoning about relevance</think>["yes","no"]'
        result = _parse_results(text, expected=2)
        assert result == ["yes", "no"]

    def test_with_code_fences(self):
        text = '```json\n["yes","no"]\n```'
        result = _parse_results(text, expected=2)
        assert result == ["yes", "no"]

    def test_pads_short_results(self):
        result = _parse_results('["yes"]', expected=3)
        assert result == ["yes", "no", "no"]

    def test_truncates_long_results(self):
        result = _parse_results('["yes","no","yes","no","yes"]', expected=2)
        assert len(result) == 2
        assert result == ["yes", "no"]

    def test_fallback_regex(self):
        text = "The answer is yes for the first and no for the second"
        result = _parse_results(text, expected=2)
        assert result == ["yes", "no"]

    def test_empty_text(self):
        result = _parse_results("", expected=3)
        assert result == ["no", "no", "no"]

    def test_nested_think_block(self):
        text = '<think>I think headline 1 is relevant because...</think>\n["yes", "yes", "no"]'
        result = _parse_results(text, expected=3)
        assert result == ["yes", "yes", "no"]

    def test_mixed_case_values(self):
        result = _parse_results('["Yes", "NO", "yes"]', expected=3)
        assert result == ["yes", "no", "yes"]


# ===========================================================================
# 3. Test lora_llm_sentiment.py (_parse method)
# ===========================================================================

from pipeline.sentiment.lora_llm_sentiment import LoRASentimentModel


class TestSentimentParse:
    def test_positive(self):
        assert LoRASentimentModel._parse("positive") == "positive"

    def test_negative(self):
        assert LoRASentimentModel._parse("negative") == "negative"

    def test_neutral(self):
        assert LoRASentimentModel._parse("neutral") == "neutral"

    def test_with_extra_text(self):
        assert LoRASentimentModel._parse("The sentiment is positive overall") == "positive"

    def test_with_special_tokens(self):
        assert LoRASentimentModel._parse("positive<|im_end|>") == "positive"

    def test_unknown_defaults_neutral(self):
        assert LoRASentimentModel._parse("unclear") == "neutral"

    def test_empty_string_defaults_neutral(self):
        assert LoRASentimentModel._parse("") == "neutral"

    def test_negative_with_explanation(self):
        assert LoRASentimentModel._parse("negative because prices dropped") == "negative"

    def test_multiline_takes_first_line(self):
        assert LoRASentimentModel._parse("positive\nsome extra text") == "positive"


# ===========================================================================
# 4. Test assets.py (AssetRegistry)
# ===========================================================================

from pipeline.config.assets import AssetRegistry, AssetType


class TestAssetRegistry:
    def test_known_commodity(self):
        info = AssetRegistry.get_commodity("gold")
        assert info.name == "gold"
        assert "GC=F" == info.yf_ticker

    def test_unknown_commodity_fallback(self):
        info = AssetRegistry.get_commodity("palladium")
        assert info.name == "palladium"
        assert "palladium" in info.filter_description

    def test_filter_description_commodity(self):
        desc = AssetRegistry.get_filter_description(AssetType.COMMODITY, "gold")
        assert "gold" in desc

    def test_filter_description_stock(self):
        desc = AssetRegistry.get_filter_description(AssetType.STOCK, "AAPL", "Apple")
        assert "Apple" in desc and "AAPL" in desc

    def test_known_commodity_silver(self):
        info = AssetRegistry.get_commodity("silver")
        assert info.name == "silver"
        assert info.yf_ticker == "SI=F"

    def test_commodity_case_insensitive(self):
        info = AssetRegistry.get_commodity("GOLD")
        assert info.name == "gold"
        assert info.yf_ticker == "GC=F"

    def test_stock_filter_description_without_display_name(self):
        desc = AssetRegistry.get_filter_description(AssetType.STOCK, "AAPL")
        assert "AAPL" in desc

    def test_unknown_commodity_has_generic_description(self):
        info = AssetRegistry.get_commodity("unobtanium")
        assert "price" in info.filter_description
        assert "trade" in info.filter_description


# ===========================================================================
# 5. Test Flask API endpoints (using Flask test client)
# ===========================================================================


class TestAPI:
    @pytest.fixture
    def client(self):
        from pipeline.client.local_app import _create_app

        app = _create_app()
        app.config["TESTING"] = True
        return app.test_client()

    def test_get_config(self, client):
        resp = client.get("/api/config")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "markets" in data
        assert "us_commodities" in data

    def test_set_market_valid(self, client):
        resp = client.post("/api/market", json={"market": "US"})
        assert resp.status_code == 200

    def test_set_market_invalid(self, client):
        resp = client.post("/api/market", json={"market": "INVALID"})
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["ok"] is False

    def test_set_market_missing_field(self, client):
        resp = client.post("/api/market", json={})
        assert resp.status_code == 400

    def test_set_asset(self, client):
        client.post("/api/market", json={"market": "US"})
        resp = client.post(
            "/api/asset", json={"asset_type": "commodity", "asset_id": "gold"}
        )
        assert resp.status_code == 200

    def test_browse_default(self, client):
        resp = client.get("/api/browse")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "current" in data
        assert "entries" in data

    def test_price_missing_params(self, client):
        resp = client.get("/api/price")
        assert resp.status_code == 400

    def test_status(self, client):
        resp = client.get("/api/status")
        assert resp.status_code == 200

    def test_task_idle(self, client):
        resp = client.get("/api/task")
        data = resp.get_json()
        assert data["status"] == "idle"

    def test_set_market_then_status(self, client):
        client.post("/api/market", json={"market": "US"})
        resp = client.get("/api/status")
        data = resp.get_json()
        assert data["market"] == "US"


# ===========================================================================
# 6. Test database.py (PipelineDB)
# ===========================================================================

from pipeline.database import PipelineDB


class TestDatabase:
    @pytest.fixture
    def db(self, tmp_path):
        return PipelineDB(db_path=tmp_path / "test.db")

    def test_create_session(self, db):
        sid = db.create_session("US", "commodity", "gold")
        assert isinstance(sid, int)

    def test_save_and_get_articles(self, db):
        sid = db.create_session("US", "commodity", "gold")
        articles = [
            {
                "title": "Gold rises",
                "date": "2026-02-10",
                "source": "Reuters",
                "sentiment": "positive",
                "relevant": True,
            },
            {
                "title": "Gold falls",
                "date": "2026-02-10",
                "source": "Reuters",
                "sentiment": "negative",
                "relevant": True,
            },
        ]
        db.save_articles(sid, articles)
        result = db.get_session_articles(sid)
        assert len(result) == 2

    def test_settings(self, db):
        db.set_setting("last_market", "US")
        assert db.get_setting("last_market") == "US"
        assert db.get_setting("nonexistent", "default") == "default"

    def test_export_csv(self, db):
        sid = db.create_session("US", "commodity", "gold")
        db.save_articles(
            sid, [{"title": "Test", "date": "2026-02-10", "sentiment": "positive"}]
        )
        csv_str = db.export_session_csv(sid)
        assert "Test" in csv_str
        assert "positive" in csv_str

    def test_sentiment_history(self, db):
        sid = db.create_session("US", "commodity", "gold")
        db.save_articles(
            sid,
            [
                {"title": "A", "date": "2026-02-10", "sentiment": "positive"},
                {"title": "B", "date": "2026-02-10", "sentiment": "negative"},
                {"title": "C", "date": "2026-02-10", "sentiment": "positive"},
            ],
        )
        history = db.get_sentiment_history("US", "commodity", "gold", days=30)
        assert len(history) >= 1
        day = history[0]
        assert day["positive"] == 2
        assert day["negative"] == 1

    def test_recent_sessions(self, db):
        db.create_session("US", "commodity", "gold")
        db.create_session("CHINA", "stock", "600547")
        sessions = db.get_recent_sessions()
        assert len(sessions) == 2

    def test_create_multiple_sessions(self, db):
        sid1 = db.create_session("US", "commodity", "gold")
        sid2 = db.create_session("US", "commodity", "silver")
        assert sid1 != sid2

    def test_empty_export(self, db):
        sid = db.create_session("US", "commodity", "gold")
        csv_str = db.export_session_csv(sid)
        assert csv_str == ""

    def test_setting_overwrite(self, db):
        db.set_setting("key", "value1")
        db.set_setting("key", "value2")
        assert db.get_setting("key") == "value2"

    def test_update_session_counts(self, db):
        sid = db.create_session("US", "commodity", "gold")
        db.update_session_counts(
            sid, article_count=10, filtered_count=5, positive=3, negative=1, neutral=1
        )
        sessions = db.get_recent_sessions()
        session = [s for s in sessions if s["id"] == sid][0]
        assert session["article_count"] == 10
        assert session["filtered_count"] == 5
        assert session["positive_count"] == 3


# ===========================================================================
# 7. Test pipeline/utils.py — safe_col helper
# ===========================================================================

from pipeline.utils import safe_col


class TestSafeCol:
    """Tests for the shared safe_col() column accessor."""

    def test_first_name_match(self):
        row = {"a": 1, "b": 2}
        assert safe_col(row, "a", "b") == 1

    def test_second_name_match(self):
        row = {"b": 2}
        assert safe_col(row, "a", "b") == 2

    def test_default_when_no_match(self):
        row = {"x": 10}
        assert safe_col(row, "a", "b", default="missing") == "missing"

    def test_default_none(self):
        row = {"x": 10}
        assert safe_col(row, "a") is None

    def test_none_value_skipped(self):
        """If first column exists but has None value, fall through to next."""
        row = {"a": None, "b": 42}
        assert safe_col(row, "a", "b") == 42

    def test_zero_value_returned(self):
        """Zero is a valid non-None value — should be returned."""
        row = {"a": 0}
        assert safe_col(row, "a", default=99) == 0

    def test_empty_string_returned(self):
        """Empty string is non-None — should be returned."""
        row = {"a": ""}
        assert safe_col(row, "a", default="fallback") == ""

    def test_chinese_column_name(self):
        row = {"内容": "新闻内容"}
        assert safe_col(row, "内容", "content") == "新闻内容"

    def test_non_dict_row_returns_default(self):
        """If row doesn't have .get() method, return default."""
        assert safe_col(42, "a", default="x") == "x"


# ===========================================================================
# 8. Test pipeline/config/models.py — discover_models, ModelConfig
# ===========================================================================

from pipeline.config.models import (
    discover_models,
    ModelConfig,
    FilterModelConfig,
    SentimentModelConfig,
)


class TestDiscoverModels:
    """Tests for the model discovery function."""

    def test_nonexistent_directory(self, tmp_path):
        result = discover_models(tmp_path / "nonexistent")
        assert result == {"base_models": [], "adapters": []}

    def test_empty_directory(self, tmp_path):
        result = discover_models(tmp_path)
        assert result == {"base_models": [], "adapters": []}

    def test_discovers_base_model(self, tmp_path):
        model_dir = tmp_path / "my-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        result = discover_models(tmp_path)
        assert len(result["base_models"]) == 1
        assert result["base_models"][0]["name"] == "my-model"

    def test_discovers_adapter(self, tmp_path):
        adapter_dir = tmp_path / "my-adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text("{}")
        result = discover_models(tmp_path)
        assert len(result["adapters"]) == 1
        assert "my-adapter" in result["adapters"][0]["name"]

    def test_dir_with_both_configs_is_adapter(self, tmp_path):
        """A dir with both config.json and adapter_config.json is adapter only."""
        both_dir = tmp_path / "both"
        both_dir.mkdir()
        (both_dir / "config.json").write_text("{}")
        (both_dir / "adapter_config.json").write_text("{}")
        result = discover_models(tmp_path)
        assert len(result["base_models"]) == 0
        assert len(result["adapters"]) == 1

    def test_files_not_dirs_ignored(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        result = discover_models(tmp_path)
        assert result == {"base_models": [], "adapters": []}


class TestModelConfig:
    """Tests for model configuration dataclasses."""

    def test_default_max_memory(self):
        cfg = ModelConfig()
        assert cfg.max_memory == {0: "6GiB", "cpu": "24GiB"}

    def test_custom_max_memory(self):
        cfg = ModelConfig(max_memory={0: "4GiB"})
        assert cfg.max_memory == {0: "4GiB"}

    def test_filter_model_config_defaults(self):
        cfg = FilterModelConfig()
        assert cfg.backend == "transformers"
        assert cfg.batch_size == 5
        assert cfg.temperature == 0.1

    def test_filter_model_transformers_path_alias(self):
        from pathlib import Path
        cfg = FilterModelConfig(base_model_path=Path("/tmp/model"))
        assert cfg.transformers_model_path == Path("/tmp/model")

    def test_sentiment_format_prompt(self):
        cfg = SentimentModelConfig()
        prompt = cfg.format_prompt("Gold prices rise")
        assert "<|im_start|>user" in prompt
        assert "Gold prices rise" in prompt
        assert "/no_think" in prompt


# ===========================================================================
# 9. Test pipeline/config/markets.py
# ===========================================================================

from pipeline.config.markets import MARKETS, Market


class TestMarkets:

    def test_us_market_exists(self):
        assert "US" in MARKETS

    def test_china_market_exists(self):
        assert "CHINA" in MARKETS

    def test_market_is_frozen(self):
        """Market is frozen dataclass — immutable."""
        m = MARKETS["US"]
        with pytest.raises(AttributeError):
            m.name = "CHANGED"

    def test_us_has_commodity_sources(self):
        assert len(MARKETS["US"].commodity_sources) > 0

    def test_china_has_stock_sources(self):
        assert len(MARKETS["CHINA"].stock_sources) > 0

    def test_default_lookback(self):
        assert MARKETS["US"].default_lookback_days == 30


# ===========================================================================
# 10. Test pipeline/exceptions.py
# ===========================================================================

from pipeline.exceptions import (
    PipelineError,
    ConfigError,
    ScraperError,
    FilterError,
    SentimentError,
    ModelLoadError,
    PriceDataError,
)


class TestExceptions:
    """Verify exception hierarchy and attributes."""

    def test_config_error_is_pipeline_error(self):
        assert issubclass(ConfigError, PipelineError)

    def test_scraper_error_attributes(self):
        err = ScraperError("reuters", "timeout", original=ValueError("inner"))
        assert err.source == "reuters"
        assert err.original is not None
        assert "[reuters]" in str(err)

    def test_model_load_error_attributes(self):
        err = ModelLoadError("/path/model", "OOM")
        assert err.model_path == "/path/model"
        assert "OOM" in str(err)

    def test_all_subclass_pipeline_error(self):
        for cls in [ConfigError, ScraperError, FilterError,
                    SentimentError, ModelLoadError, PriceDataError]:
            assert issubclass(cls, PipelineError)


# ===========================================================================
# 11. Test pipeline/client/_state.py — validation functions
# ===========================================================================

from pipeline.client._state import (
    _validate_market,
    _validate_asset_type,
    _validate_asset_id,
    _validate_path,
)


class TestValidateMarket:

    def test_valid_us(self):
        assert _validate_market("US") == "US"

    def test_valid_lowercase(self):
        assert _validate_market("us") == "US"

    def test_valid_china(self):
        assert _validate_market("china") == "CHINA"

    def test_whitespace_stripped(self):
        assert _validate_market("  US  ") == "US"

    def test_invalid_market(self):
        with pytest.raises(ConfigError, match="Unknown market"):
            _validate_market("JAPAN")


class TestValidateAssetType:

    def test_commodity(self):
        assert _validate_asset_type("commodity") == "commodity"

    def test_stock(self):
        assert _validate_asset_type("stock") == "stock"

    def test_case_insensitive(self):
        assert _validate_asset_type("COMMODITY") == "commodity"

    def test_invalid(self):
        with pytest.raises(ConfigError, match="Invalid asset type"):
            _validate_asset_type("crypto")


class TestValidateAssetId:

    def test_valid_ticker(self):
        assert _validate_asset_id("AAPL") == "AAPL"

    def test_valid_stock_code(self):
        assert _validate_asset_id("600547") == "600547"

    def test_empty_string(self):
        with pytest.raises(ConfigError, match="cannot be empty"):
            _validate_asset_id("")

    def test_whitespace_only(self):
        with pytest.raises(ConfigError, match="cannot be empty"):
            _validate_asset_id("   ")

    def test_too_long(self):
        with pytest.raises(ConfigError, match="too long"):
            _validate_asset_id("A" * 51)

    def test_special_chars_rejected(self):
        with pytest.raises(ConfigError, match="invalid characters"):
            _validate_asset_id("AAPL; DROP TABLE")

    def test_dots_and_hyphens_allowed(self):
        assert _validate_asset_id("BRK.B") == "BRK.B"
        assert _validate_asset_id("my-asset") == "my-asset"

    def test_whitespace_stripped(self):
        assert _validate_asset_id("  AAPL  ") == "AAPL"


class TestValidatePath:

    def test_valid_path(self, tmp_path):
        result = _validate_path(str(tmp_path))
        assert result == tmp_path

    def test_nonexistent_path(self):
        with pytest.raises(ConfigError, match="does not exist"):
            _validate_path("/nonexistent/path/12345")

    def test_file_not_dir(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("hello")
        with pytest.raises(ConfigError, match="not a directory"):
            _validate_path(str(f))


# ===========================================================================
# 12. Test pipeline/prices/price_provider.py — pure functions
# ===========================================================================

from pipeline.prices.price_provider import _change_stats, _resolve_us_ticker


class TestChangeStats:

    def test_positive_change(self):
        result = _change_stats(105.0, 100.0)
        assert result["change"] == 5.0
        assert result["change_pct"] == 5.0

    def test_negative_change(self):
        result = _change_stats(95.0, 100.0)
        assert result["change"] == -5.0
        assert result["change_pct"] == -5.0

    def test_no_change(self):
        result = _change_stats(100.0, 100.0)
        assert result["change"] == 0.0
        assert result["change_pct"] == 0.0

    def test_zero_prev_close(self):
        """Avoid division by zero."""
        result = _change_stats(10.0, 0.0)
        assert result["change"] == 10.0
        assert result["change_pct"] == 0.0

    def test_rounding(self):
        result = _change_stats(100.123456, 99.876543)
        assert result["change"] == 0.25  # rounded to 2 places


class TestResolveUsTicker:

    def test_gold_commodity(self):
        result = _resolve_us_ticker("commodity", "gold")
        assert result == "GC=F"

    def test_unknown_commodity(self):
        """Unknown commodity returns None (no yf_ticker)."""
        result = _resolve_us_ticker("commodity", "unobtanium")
        # Graceful fallback: CommodityInfo has empty yf_ticker
        assert result is None

    def test_stock_uppercased(self):
        result = _resolve_us_ticker("stock", "aapl")
        assert result == "AAPL"


# ===========================================================================
# 13. Test pipeline/client/_state.py — PipelineState (unit-level)
# ===========================================================================


class TestPipelineStateBrowse:
    """Test the file browser functionality (no mocking needed)."""

    def test_browse_default(self):
        from pipeline.client._state import PipelineState
        result = PipelineState.browse_directory(None)
        assert "current" in result
        assert "entries" in result
        assert isinstance(result["entries"], list)

    def test_browse_tmp(self, tmp_path):
        from pipeline.client._state import PipelineState
        (tmp_path / "subdir").mkdir()
        (tmp_path / "file.txt").write_text("hi")
        result = PipelineState.browse_directory(str(tmp_path))
        assert result["current"] == str(tmp_path.resolve())
        names = {e["name"] for e in result["entries"]}
        assert "subdir" in names
        assert "file.txt" in names

    def test_browse_detects_model(self, tmp_path):
        from pipeline.client._state import PipelineState
        model_dir = tmp_path / "my-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        result = PipelineState.browse_directory(str(tmp_path))
        model_entry = [e for e in result["entries"] if e["name"] == "my-model"][0]
        assert model_entry["is_model"] is True
        assert model_entry["is_adapter"] is False

    def test_browse_detects_adapter(self, tmp_path):
        from pipeline.client._state import PipelineState
        adapter_dir = tmp_path / "my-adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text("{}")
        result = PipelineState.browse_directory(str(tmp_path))
        adapter_entry = [e for e in result["entries"] if e["name"] == "my-adapter"][0]
        assert adapter_entry["is_adapter"] is True

    def test_browse_hides_dotfiles(self, tmp_path):
        from pipeline.client._state import PipelineState
        (tmp_path / ".hidden").mkdir()
        (tmp_path / "visible").mkdir()
        result = PipelineState.browse_directory(str(tmp_path))
        names = {e["name"] for e in result["entries"]}
        assert ".hidden" not in names
        assert "visible" in names

    def test_browse_has_parent(self, tmp_path):
        from pipeline.client._state import PipelineState
        sub = tmp_path / "sub"
        sub.mkdir()
        result = PipelineState.browse_directory(str(sub))
        assert result["parent"] is not None

    def test_browse_nonexistent_falls_back(self, tmp_path):
        from pipeline.client._state import PipelineState
        result = PipelineState.browse_directory(str(tmp_path / "nonexistent"))
        # Should fall back to parent or home, not crash
        assert "current" in result


class TestPipelineStateTask:
    """Test background task state machine."""

    def test_task_idle_initial(self):
        from pipeline.client._state import PipelineState
        state = PipelineState()
        status = state.get_task_status()
        assert status["status"] == "idle"

    def test_require_asset_raises(self):
        from pipeline.client._state import PipelineState
        state = PipelineState()
        # Clear any restored config
        state._market = None
        state._asset_type = None
        state._asset_id = None
        with pytest.raises(RuntimeError, match="Market not set"):
            state.start_fetch()

    def test_start_filter_no_articles(self):
        from pipeline.client._state import PipelineState
        state = PipelineState()
        with pytest.raises(RuntimeError, match="No articles"):
            state.start_filter()

    def test_start_sentiment_no_articles(self):
        from pipeline.client._state import PipelineState
        state = PipelineState()
        with pytest.raises(RuntimeError, match="No articles"):
            state.start_sentiment()


# ===========================================================================
# 14. Test Article.to_dict edge cases
# ===========================================================================


class TestArticleEdgeCases:

    def test_article_with_all_fields(self):
        a = Article(
            title="Test",
            url="http://example.com",
            source="test",
            date="2026-01-01",
            summary="Summary",
            sentiment="positive",
            relevant=True,
        )
        d = a.to_dict()
        assert d["title"] == "Test"
        assert d["headline"] == "Test"  # alias
        assert d["sentiment"] == "positive"
        assert d["url"] == "http://example.com"

    def test_article_empty_title(self):
        a = Article(title="", url="", source="", date="")
        d = a.to_dict()
        assert d["title"] == ""

    def test_article_sentiment_assignment(self):
        a = Article(title="Test", url="", source="", date="")
        assert a.sentiment is None
        a.sentiment = "negative"
        assert a.sentiment == "negative"


# ===========================================================================
# 15. Test version string
# ===========================================================================


class TestVersion:

    def test_version_exists(self):
        from pipeline import __version__
        assert __version__
        assert isinstance(__version__, str)
        # Should be semver-like
        parts = __version__.split(".")
        assert len(parts) >= 2
