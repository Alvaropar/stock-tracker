"""
Pipeline orchestrator.

Binds together scrapers, filters, and sentiment models.  Manages the
lifecycle of heavy resources (LLM loading / unloading) and coordinates
the scrape -> filter -> sentiment pipeline.

The orchestrator is *stateless* with respect to session data -- it does
not store articles or results.  That responsibility lives in the client
layer, which calls the orchestrator's methods and manages its own state.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Type

from ..config.assets import AssetRegistry, AssetType, CommodityInfo
from ..config.markets import MARKETS, Market
from ..config.models import FilterModelConfig, SentimentModelConfig
from ..filters.relevance_filter import RelevanceFilter
from ..scrapers.base_scraper import Article, BaseScraper
from ..scrapers.china_scraper import ChinaCommodityScraper, ChinaStockScraper
from ..scrapers.us_scraper import USCommodityScraper, USStockScraper
from ..sentiment.base_sentiment import BaseSentimentModel
from ..sentiment.lora_llm_sentiment import LoRASentimentModel


# Maps (market, asset_type) -> concrete scraper class
_SCRAPER_REGISTRY: Dict[tuple[str, AssetType], Type[BaseScraper]] = {
    ("US", AssetType.COMMODITY): USCommodityScraper,
    ("US", AssetType.STOCK): USStockScraper,
    ("CHINA", AssetType.COMMODITY): ChinaCommodityScraper,
    ("CHINA", AssetType.STOCK): ChinaStockScraper,
}


class PipelineOrchestrator:
    """
    Coordinates scraping, filtering, and sentiment analysis.

    Designed for repeated use within a single session: heavy models
    are loaded lazily and can be explicitly unloaded to reclaim GPU
    memory between pipeline stages.

    Model configs can be swapped at runtime via ``set_filter_config``
    and ``set_sentiment_config`` — the previously loaded model is
    unloaded automatically.
    """

    def __init__(
        self,
        filter_config: Optional[FilterModelConfig] = None,
        sentiment_config: Optional[SentimentModelConfig] = None,
    ):
        self._filter_cfg = filter_config or FilterModelConfig()
        self._sentiment_cfg = sentiment_config or SentimentModelConfig()

        # Lazy-initialised components
        self._filter: Optional[RelevanceFilter] = None
        self._sentiment: Optional[BaseSentimentModel] = None
        self._scrapers: Dict[tuple[str, AssetType], BaseScraper] = {}

    # ── config hot-swap ──────────────────────────────────────────────

    def set_filter_config(self, cfg: FilterModelConfig) -> None:
        """Replace filter config.  Unloads old model if loaded."""
        self.unload_filter()
        self._filter_cfg = cfg

    def set_sentiment_config(self, cfg: SentimentModelConfig) -> None:
        """Replace sentiment config.  Unloads old model if loaded."""
        self.unload_sentiment()
        self._sentiment_cfg = cfg

    # ── scraping ─────────────────────────────────────────────────────

    def fetch_news(
        self,
        market: str,
        asset_type: AssetType,
        asset_id: str,
        **scraper_kwargs,
    ) -> List[Article]:
        key = (market.upper(), asset_type)
        if key not in self._scrapers:
            cls = _SCRAPER_REGISTRY.get(key)
            if cls is None:
                raise ValueError(f"No scraper registered for {key}")
            self._scrapers[key] = cls()

        scraper = self._scrapers[key]

        if market.upper() == "US" and asset_type is AssetType.STOCK:
            if "company_name" not in scraper_kwargs:
                scraper_kwargs["company_name"] = (
                    AssetRegistry.get_us_stock_display_name(asset_id)
                )

        return scraper.scrape(asset_id, **scraper_kwargs)

    # ── filtering ────────────────────────────────────────────────────

    def filter_news(
        self,
        articles: List[Article],
        asset_type: AssetType,
        asset_id: str,
        display_name: Optional[str] = None,
    ) -> List[Article]:
        if self._filter is None:
            self._filter = RelevanceFilter(self._filter_cfg)

        description = AssetRegistry.get_filter_description(
            asset_type, asset_id, display_name)
        self._filter.set_asset_description(description)

        headlines = [a.title for a in articles]
        mask = self._filter.filter(headlines)

        relevant: List[Article] = []
        for article, is_relevant in zip(articles, mask):
            article.relevant = is_relevant
            if is_relevant:
                relevant.append(article)

        return relevant

    def unload_filter(self) -> None:
        if self._filter is not None:
            self._filter.unload()
            self._filter = None

    # ── sentiment ────────────────────────────────────────────────────

    def analyze_sentiment(
        self,
        articles: List[Article],
        progress_cb=None,
    ) -> List[Article]:
        if self._sentiment is None:
            self._sentiment = LoRASentimentModel(self._sentiment_cfg)

        headlines = [a.title for a in articles]
        labels = self._sentiment.analyze(headlines, progress_cb=progress_cb)

        for article, label in zip(articles, labels):
            article.sentiment = label

        return articles

    def unload_sentiment(self) -> None:
        if self._sentiment is not None:
            self._sentiment.unload()
            self._sentiment = None

    # ── convenience ──────────────────────────────────────────────────

    def run_full(
        self,
        market: str,
        asset_type: AssetType,
        asset_id: str,
        **scraper_kwargs,
    ) -> List[Article]:
        articles = self.fetch_news(market, asset_type, asset_id, **scraper_kwargs)
        relevant = self.filter_news(articles, asset_type, asset_id)
        self.unload_filter()
        self.analyze_sentiment(relevant)
        return articles

    def cleanup(self) -> None:
        self.unload_filter()
        self.unload_sentiment()
