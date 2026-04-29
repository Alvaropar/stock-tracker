"""
US market scrapers: Reuters commodity RSS and Yahoo Finance / Google News
for stocks.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional
from urllib.parse import quote

import feedparser
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .base_scraper import Article, BaseScraper, parse_date

log = logging.getLogger("pipeline.scrapers.us")

try:
    import yfinance as yf
    _YFINANCE = True
except ImportError:
    _YFINANCE = False


class USCommodityScraper(BaseScraper):
    """Fetch US commodity news via Google News RSS filtered to reuters.com."""

    def scrape(self, asset_id: str, **kwargs) -> List[Article]:
        query = asset_id.lower()
        rss_url = (
            f"https://news.google.com/rss/search?"
            f"q=site:reuters.com+{quote(query)}&hl=en-US&gl=US&ceid=US:en"
        )

        articles: List[Article] = []
        log.info("Fetching Reuters RSS for '%s'", query)

        try:
            feed = feedparser.parse(rss_url)
        except Exception as e:
            log.error("Failed to parse RSS for '%s': %s", query, e)
            return []

        for entry in feed.entries:
            link = entry.get("link", "")
            source_dict = entry.get("source", {})
            source_title = (
                source_dict.get("title", "")
                if isinstance(source_dict, dict) else str(source_dict)
            )
            if "reuters.com" not in link.lower() and "reuters" not in source_title.lower():
                continue

            headline = entry.get("title", "")
            headline = re.sub(r"\s*[-–]\s*Reuters.*$", "", headline, flags=re.IGNORECASE)

            published = entry.get("published", "")
            date = parse_date(published)

            articles.append(Article(
                title=headline,
                date=date,
                datetime=date,
                source="Reuters",
                url=link,
            ))

        log.info("Reuters RSS: %d articles for '%s'", len(articles), query)
        articles = self._deduplicate(articles)
        return self._sort_newest_first(articles)


class USStockScraper(BaseScraper):
    """Fetch US stock news from Yahoo Finance + Google News RSS."""

    def scrape(self, asset_id: str, **kwargs) -> List[Article]:
        ticker = asset_id.upper()
        company_name: Optional[str] = kwargs.get("company_name")

        articles: List[Article] = []
        log.info("Fetching news for ticker %s", ticker)

        # Yahoo Finance
        if _YFINANCE:
            articles.extend(self._yahoo(ticker))

        # Google News RSS
        articles.extend(self._google_rss(ticker, company_name))

        log.info("Total: %d articles for %s", len(articles), ticker)
        articles = self._deduplicate(articles)
        return self._sort_newest_first(articles)

    # ── private helpers ──────────────────────────────────────────────

    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        reraise=True,
    )
    def _yahoo(ticker: str) -> List[Article]:
        articles: List[Article] = []
        try:
            news = yf.Ticker(ticker).news
            # Defensive: handle case where news is not a list
            if not isinstance(news, list):
                log.warning("Yahoo Finance returned non-list for %s: %s", ticker, type(news))
                return articles
            for item in news:
                # Defensive: skip non-dict items
                if not isinstance(item, dict):
                    continue
                content = item.get("content", {}) if isinstance(item.get("content"), dict) else {}
                headline = content.get("title", "")
                if not headline:
                    continue

                # Defensive: handle case where URL fields might be lists
                def _get_url(field):
                    val = content.get(field, {})
                    if isinstance(val, dict):
                        return val.get("url", "")
                    return ""

                url = _get_url("canonicalUrl") or _get_url("clickThroughUrl")
                pub = content.get("pubDate", "")
                provider = content.get("provider", {})
                if isinstance(provider, dict):
                    publisher = provider.get("displayName", "Yahoo Finance")
                else:
                    publisher = "Yahoo Finance"
                summary = content.get("summary", "")
                articles.append(Article(
                    title=headline,
                    date=parse_date(pub),
                    datetime=parse_date(pub),
                    source=publisher,
                    url=url,
                    summary=summary,
                    ticker=ticker,
                ))
        except Exception as e:
            log.warning("Yahoo Finance failed for %s: %s", ticker, e)
        return articles

    @staticmethod
    def _google_rss(ticker: str, company_name: Optional[str]) -> List[Article]:
        articles: List[Article] = []
        queries = [f"{ticker} stock"]
        if company_name:
            queries.append(f"{company_name} stock")

        seen: set[str] = set()
        for q in queries:
            rss_url = (
                f"https://news.google.com/rss/search?"
                f"q={quote(q)}&hl=en-US&gl=US&ceid=US:en"
            )
            try:
                feed = feedparser.parse(rss_url)
                for entry in feed.entries:
                    headline = entry.get("title", "")
                    key = headline.lower().strip()
                    if key in seen:
                        continue
                    seen.add(key)

                    source = "Google News"
                    if " - " in headline:
                        parts = headline.rsplit(" - ", 1)
                        if len(parts) == 2:
                            headline, source = parts

                    url = entry.get("link", "")
                    published = entry.get("published", "")
                    summary = re.sub(r"<[^>]+>", "", entry.get("summary", ""))

                    articles.append(Article(
                        title=headline,
                        date=parse_date(published),
                        datetime=parse_date(published),
                        source=source,
                        url=url,
                        summary=summary[:200],
                        ticker=ticker,
                    ))
            except Exception as e:
                log.warning("Google RSS failed for query '%s': %s", q, e)
                continue
        return articles
