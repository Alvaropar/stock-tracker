"""
Market definitions and source configuration.

Each market declares which scraper sources are available and how
news is fetched for commodities vs. stocks within that market.
"""
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class Market:
    """Immutable descriptor for a supported market."""
    name: str                           # e.g. "US", "CHINA"
    display_name: str                   # e.g. "US Markets"
    commodity_sources: List[str]        # scraper keys used for commodities
    stock_sources: List[str]            # scraper keys used for stocks
    default_lookback_days: int = 30


MARKETS: Dict[str, Market] = {
    "US": Market(
        name="US",
        display_name="US Markets (Reuters, Yahoo Finance, Google News)",
        commodity_sources=["reuters_rss"],
        stock_sources=["yahoo_finance", "google_news_rss"],
        default_lookback_days=30,
    ),
    "CHINA": Market(
        name="CHINA",
        display_name="China Markets (SHMET, Eastmoney, Sina Finance)",
        commodity_sources=["shmet", "eastmoney", "caixin", "sina"],
        stock_sources=["eastmoney_stock", "sina_stock"],
        default_lookback_days=30,
    ),
}
