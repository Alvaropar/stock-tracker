"""
Base scraper interface and unified Article dataclass.

Every scraper in the pipeline returns ``List[Article]``.  The normalised
dataclass ensures downstream layers (filters, sentiment, visualisation)
never have to deal with field-name mismatches.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import List, Optional


@dataclass
class Article:
    """
    Canonical article representation used throughout the pipeline.

    Fields are intentionally simple strings so the struct serialises
    cleanly to JSON.  ``title`` is always populated; ``headline`` is
    an alias kept for backward compatibility with existing classified-data
    JSON files.
    """
    title: str
    date: str            # YYYY-MM-DD
    datetime: str = ""   # YYYY-MM-DD HH:MM:SS  (best-effort)
    source: str = ""
    url: str = ""
    summary: str = ""
    ticker: str = ""     # stock ticker / code (empty for commodities)

    # Fields populated later in the pipeline:
    relevant: Optional[bool] = None
    sentiment: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # Keep 'headline' alias for backward compat with existing JSON
        d["headline"] = d["title"]
        return d


def parse_date(date_str: str) -> str:
    """
    Best-effort date normaliser.  Accepts RSS RFC-2822, ISO-8601,
    and a handful of common fallback formats.  Returns ``YYYY-MM-DD``.
    """
    if not date_str:
        return datetime.now().strftime("%Y-%m-%d")

    date_str = date_str.strip()

    # RFC 2822 (standard RSS)
    try:
        return parsedate_to_datetime(date_str).strftime("%Y-%m-%d")
    except Exception:
        pass

    # ISO 8601 variants
    try:
        return datetime.fromisoformat(
            date_str.replace("Z", "+00:00")
        ).strftime("%Y-%m-%d")
    except Exception:
        pass

    for fmt in (
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S %z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y",
        "%d %b %Y",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    # Last resort: look for a dd Mon yyyy substring
    m = re.search(r"(\d{1,2})\s+(\w{3})\s+(\d{4})", date_str)
    if m:
        try:
            return datetime.strptime(
                f"{m.group(1)} {m.group(2)} {m.group(3)}", "%d %b %Y"
            ).strftime("%Y-%m-%d")
        except ValueError:
            pass

    return datetime.now().strftime("%Y-%m-%d")


class BaseScraper(ABC):
    """Interface that every market-specific scraper must implement."""

    @abstractmethod
    def scrape(self, asset_id: str, **kwargs) -> List[Article]:
        """
        Fetch news articles for *asset_id*.

        Parameters
        ----------
        asset_id : str
            Commodity name (e.g. ``"gold"``) or stock ticker / code.
        **kwargs :
            Scraper-specific options (``start_date``, ``end_date``, etc.)

        Returns
        -------
        list[Article]
            De-duplicated, date-sorted (newest first) articles.
        """

    @staticmethod
    def _deduplicate(articles: List[Article]) -> List[Article]:
        """Remove duplicates by title (case-insensitive)."""
        seen: set[str] = set()
        unique: List[Article] = []
        for a in articles:
            key = a.title.lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(a)
        return unique

    @staticmethod
    def _sort_newest_first(articles: List[Article]) -> List[Article]:
        return sorted(articles, key=lambda a: a.datetime or a.date, reverse=True)
