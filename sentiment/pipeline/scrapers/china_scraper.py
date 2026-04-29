"""
China market scrapers: SHMET, Eastmoney, Sina Finance.

Separated into ChinaCommodityScraper and ChinaStockScraper to keep
each class focused.  Both share a private _ChinaBase mixin for common
date parsing and keyword filtering.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import List, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

log = logging.getLogger("pipeline.scrapers.china")

try:
    import akshare as ak
    import pandas as pd
    _AKSHARE = True
except ImportError:
    _AKSHARE = False

from ..config.assets import CommodityInfo, AssetRegistry
from ..utils import safe_col as _col
from .base_scraper import Article, BaseScraper

_DATE_FMT = "%Y-%m-%d"
_DATETIME_FMT = "%Y-%m-%d %H:%M:%S"


class _ChinaBase:
    """Shared utilities for China scrapers."""

    _session: Optional[requests.Session] = None

    @classmethod
    def _get_session(cls) -> requests.Session:
        if cls._session is None:
            cls._session = requests.Session()
            cls._session.headers.update({
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            })
        return cls._session

    @staticmethod
    def _parse_dt(dt_str) -> tuple[str, str]:
        """Return (date, datetime) strings from various AkShare formats."""
        try:
            if hasattr(dt_str, "strftime"):
                return dt_str.strftime(_DATE_FMT), dt_str.strftime(_DATETIME_FMT)
        except Exception:
            pass

        raw = str(dt_str) if dt_str is not None else ""
        raw = re.sub(r"\+\d{2}:\d{2}$", "", raw)
        for fmt in (_DATETIME_FMT, _DATE_FMT, "%Y%m%d"):
            try:
                dt = datetime.strptime(raw.strip(), fmt)
                return dt.strftime(_DATE_FMT), dt.strftime(_DATETIME_FMT)
            except ValueError:
                continue

        now = datetime.now()
        return now.strftime(_DATE_FMT), now.strftime(_DATETIME_FMT)

    @staticmethod
    def _in_range(date_str: str,
                  start: Optional[datetime],
                  end: Optional[datetime]) -> bool:
        try:
            d = datetime.strptime(date_str, _DATE_FMT)
        except ValueError:
            return True  # pass through if unparseable
        if start and d < start:
            return False
        if end and d > end:
            return False
        return True

    @staticmethod
    def _matches_keywords(text: str, keywords: List[str]) -> bool:
        if not keywords:
            return True
        low = text.lower()
        return any(k.lower() in low for k in keywords)


class ChinaCommodityScraper(BaseScraper, _ChinaBase):
    """
    Aggregate commodity news from SHMET, Eastmoney, Caixin, and
    Sina Finance.
    """

    def scrape(self, asset_id: str, **kwargs) -> List[Article]:
        info = AssetRegistry.get_commodity(asset_id)
        start = kwargs.get("start_date")
        end = kwargs.get("end_date")
        if start is None:
            start = datetime.now() - timedelta(days=30)
        if end is None:
            end = datetime.now()

        is_gold = asset_id.lower() == "gold"
        articles: List[Article] = []

        if info.shmet_symbol:
            articles.extend(self._shmet(info, start, end, is_gold))
        if is_gold and info.cn_stock_codes:
            articles.extend(self._eastmoney(info, start, end))
        articles.extend(self._sina_feed(info, start, end, is_gold))

        articles = self._deduplicate(articles)
        return self._sort_newest_first(articles)

    # ── source methods ──────────────────────────────────────────────

    def _shmet(self, info: CommodityInfo,
               start: datetime, end: datetime,
               filter_kw: bool) -> List[Article]:
        if not _AKSHARE:
            return []
        out: List[Article] = []
        log.info("Fetching SHMET data for '%s'", info.shmet_symbol)
        try:
            df = ak.futures_news_shmet(symbol=info.shmet_symbol)
            for _, row in df.iterrows():
                content = str(_col(row, "内容", "content"))
                if not content:
                    continue
                date_s, dt_s = self._parse_dt(_col(row, "发布时间", "publish_time", "date"))
                if not self._in_range(date_s, start, end):
                    continue
                if filter_kw and not self._matches_keywords(content, info.cn_keywords):
                    continue
                headline = self._extract_headline(content)
                if headline:
                    out.append(Article(
                        title=headline, date=date_s, datetime=dt_s,
                        source="Shanghai Metals Market", summary=content,
                    ))
        except Exception as e:
            log.warning("SHMET scrape failed for '%s': %s", info.shmet_symbol, e)
        return out

    def _eastmoney(self, info: CommodityInfo,
                   start: datetime, end: datetime) -> List[Article]:
        if not _AKSHARE:
            return []
        out: List[Article] = []
        for code in info.cn_stock_codes:
            try:
                df = ak.stock_news_em(symbol=code)
                for _, row in df.iterrows():
                    headline = str(_col(row, "新闻标题", "news_title", "title"))
                    content = str(_col(row, "新闻内容", "news_content", "content"))
                    if not headline:
                        continue
                    date_s, dt_s = self._parse_dt(_col(row, "发布时间", "publish_time", "date"))
                    if not self._in_range(date_s, start, end):
                        continue
                    if not self._matches_keywords(headline + " " + content,
                                                  info.cn_keywords):
                        continue
                    source_name = str(_col(row, "文章来源", "source", default="东方财富"))
                    out.append(Article(
                        title=headline, date=date_s, datetime=dt_s,
                        source=f"Eastmoney ({source_name})", summary=content,
                    ))
            except Exception as e:
                log.warning("Eastmoney scrape failed for stock %s: %s", code, e)
                continue
        return out

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, requests.RequestException)),
        reraise=True,
    )
    def _sina_feed(self, info: CommodityInfo,
                   start: datetime, end: datetime,
                   filter_kw: bool, max_pages: int = 10) -> List[Article]:
        session = self._get_session()
        out: List[Article] = []
        base_url = "https://feed.mix.sina.com.cn/api/roll/get"
        for page in range(1, max_pages + 1):
            try:
                resp = session.get(base_url, params={
                    "pageid": "153", "lid": "2516",
                    "num": "50", "page": str(page),
                }, timeout=15)
                resp.raise_for_status()
                items = resp.json().get("result", {}).get("data", [])
                if not items:
                    break
                for item in items:
                    title = item.get("title", "")
                    if not title:
                        continue
                    ctime = item.get("ctime", "")
                    date_s, dt_s = self._ts_to_date(ctime)
                    if not self._in_range(date_s, start, end):
                        continue
                    summary = item.get("summary", "") or item.get("intro", "") or ""
                    if filter_kw and not self._matches_keywords(
                            title + " " + summary, info.cn_keywords):
                        continue
                    media = item.get("media_name", "新浪财经") or "新浪财经"
                    out.append(Article(
                        title=title, date=date_s, datetime=dt_s,
                        source=f"Sina Finance ({media})", summary=summary,
                    ))
            except Exception as e:
                log.warning("Sina feed page %d failed: %s", page, e)
                break
        return out

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_headline(content: str) -> str:
        m = re.match(r"【([^】]+)】", content)
        if m:
            return m.group(1)
        parts = re.split(r"[。\n]", content)
        h = parts[0].strip() if parts else content
        return (h[:97] + "...") if len(h) > 100 else h

    @staticmethod
    def _ts_to_date(ctime: str) -> tuple[str, str]:
        if ctime:
            try:
                dt = datetime.fromtimestamp(int(ctime))
                return dt.strftime(_DATE_FMT), dt.strftime(_DATETIME_FMT)
            except (ValueError, OSError):
                pass
        now = datetime.now()
        return now.strftime(_DATE_FMT), now.strftime(_DATETIME_FMT)


class ChinaStockScraper(BaseScraper, _ChinaBase):
    """Fetch stock-specific news from Eastmoney and Sina Finance."""

    def scrape(self, asset_id: str, **kwargs) -> List[Article]:
        stock_code = asset_id
        start = kwargs.get("start_date", datetime.now() - timedelta(days=30))
        end = kwargs.get("end_date", datetime.now())

        articles: List[Article] = []
        articles.extend(self._eastmoney_stock(stock_code, start, end))
        articles.extend(self._sina_stock(stock_code, start, end))

        articles = self._deduplicate(articles)
        return self._sort_newest_first(articles)

    def _eastmoney_stock(self, code: str,
                         start: datetime, end: datetime) -> List[Article]:
        if not _AKSHARE:
            return []
        out: List[Article] = []
        try:
            df = ak.stock_news_em(symbol=code)
            for _, row in df.iterrows():
                headline = str(_col(row, "新闻标题", "news_title", "title"))
                if not headline:
                    continue
                date_s, dt_s = self._parse_dt(_col(row, "发布时间", "publish_time", "date"))
                if not self._in_range(date_s, start, end):
                    continue
                content = str(_col(row, "新闻内容", "news_content", "content"))
                source_name = str(_col(row, "文章来源", "source", default="东方财富"))
                out.append(Article(
                    title=headline, date=date_s, datetime=dt_s,
                    source=f"Eastmoney ({source_name})",
                    summary=content, ticker=code,
                ))
        except Exception as e:
            log.warning("Eastmoney stock news failed for %s: %s", code, e)
        return out

    def _sina_stock(self, code: str,
                    start: datetime, end: datetime,
                    max_pages: int = 5) -> List[Article]:
        """Search Sina feed API filtered by stock name."""
        stock_name = AssetRegistry.get_china_stock_name(code)
        if not stock_name:
            return []

        session = self._get_session()
        out: List[Article] = []
        base_url = "https://feed.mix.sina.com.cn/api/roll/get"
        for page in range(1, max_pages + 1):
            try:
                resp = session.get(base_url, params={
                    "pageid": "153", "lid": "2516",
                    "num": "50", "page": str(page),
                }, timeout=15)
                resp.raise_for_status()
                items = resp.json().get("result", {}).get("data", [])
                if not items:
                    break
                for item in items:
                    title = item.get("title", "")
                    summary = item.get("summary", "") or item.get("intro", "") or ""
                    if not title:
                        continue
                    if stock_name not in (title + " " + summary):
                        continue
                    ctime = item.get("ctime", "")
                    date_s, dt_s = ChinaCommodityScraper._ts_to_date(ctime)
                    if not self._in_range(date_s, start, end):
                        continue
                    media = item.get("media_name", "新浪财经") or "新浪财经"
                    out.append(Article(
                        title=title, date=date_s, datetime=dt_s,
                        source=f"Sina Finance ({media})",
                        summary=summary, ticker=code,
                    ))
            except Exception as e:
                log.warning("Sina stock feed page %d failed for %s: %s", page, code, e)
                break
        return out
