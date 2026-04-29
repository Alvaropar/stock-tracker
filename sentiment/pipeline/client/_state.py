"""
Internal pipeline state management.

This module is the backend that ``local_app.py`` calls.  It maintains
session state (current market, asset, articles, results), manages
background tasks for long-running operations (filter, sentiment), and
coordinates the pipeline orchestrator.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config.assets import AssetRegistry, AssetType, COMMODITIES, US_STOCK_NAMES
from ..config.markets import MARKETS, Market
from ..config.models import (
    FilterModelConfig,
    SentimentModelConfig,
    discover_models,
)
from ..core.orchestrator import PipelineOrchestrator
from ..database import PipelineDB
from ..exceptions import ConfigError, PipelineError
from ..scrapers.base_scraper import Article

log = logging.getLogger("pipeline.state")

_PROJECT_ROOT = Path(
    os.environ.get("PIPELINE_PROJECT_ROOT", "")
) if os.environ.get("PIPELINE_PROJECT_ROOT") else Path(__file__).resolve().parent.parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"
_CLASSIFIED_DIR = _DATA_DIR / "classified"
_CLASSIFIED_DIR.mkdir(parents=True, exist_ok=True)

# ── Input validation ─────────────────────────────────────────────────────────

_MAX_ASSET_ID_LEN = 50
_SAFE_ASSET_RE = re.compile(r"^[A-Za-z0-9._\-]+$")
_VALID_MARKETS = set(MARKETS.keys())
_VALID_ASSET_TYPES = {"commodity", "stock"}


def _validate_market(market: str) -> str:
    """Validate and normalise a market name."""
    key = market.strip().upper()
    if key not in _VALID_MARKETS:
        raise ConfigError(f"Unknown market '{market}'. Choose from: {sorted(_VALID_MARKETS)}")
    return key


def _validate_asset_type(asset_type: str) -> str:
    val = asset_type.strip().lower()
    if val not in _VALID_ASSET_TYPES:
        raise ConfigError(f"Invalid asset type '{asset_type}'. Choose from: {sorted(_VALID_ASSET_TYPES)}")
    return val


def _validate_asset_id(asset_id: str) -> str:
    val = asset_id.strip()
    if not val:
        raise ConfigError("Asset ID cannot be empty")
    if len(val) > _MAX_ASSET_ID_LEN:
        raise ConfigError(f"Asset ID too long (max {_MAX_ASSET_ID_LEN} chars)")
    if not _SAFE_ASSET_RE.match(val):
        raise ConfigError("Asset ID contains invalid characters (use alphanumeric, dots, hyphens, underscores)")
    return val


def _validate_path(path_str: str, label: str = "path") -> Path:
    """Validate a model path string — must exist and be a directory."""
    p = Path(path_str.strip())
    if not p.exists():
        raise ConfigError(f"{label}: path does not exist: {p}")
    if not p.is_dir():
        raise ConfigError(f"{label}: not a directory: {p}")
    return p

_US_COMMODITIES = ["gold", "silver", "copper", "aluminum", "oil", "platinum"]
_CHINA_COMMODITIES = ["gold", "copper", "aluminum", "lead", "zinc", "nickel", "tin"]
_US_STOCKS = list(US_STOCK_NAMES.keys())[:15]


class PipelineState:
    """
    Holds session state and delegates to the orchestrator.

    Long-running operations (filter, sentiment) are dispatched to a
    background thread.  The frontend polls ``/api/task`` to check
    progress and retrieve results when done.
    """

    def __init__(
        self,
        filter_config: Optional[FilterModelConfig] = None,
        sentiment_config: Optional[SentimentModelConfig] = None,
    ):
        self._orchestrator = PipelineOrchestrator(
            filter_config=filter_config,
            sentiment_config=sentiment_config,
        )
        self._data_dir = _CLASSIFIED_DIR
        self._db = PipelineDB()
        self._session_id: Optional[int] = None

        self._market: Optional[Market] = None
        self._asset_type: Optional[AssetType] = None
        self._asset_id: Optional[str] = None
        self._display_name: Optional[str] = None
        self._articles: List[Article] = []
        self._filtered: List[Article] = []
        self._classified: List[Article] = []
        self._data_lock = threading.Lock()  # protects _articles/_filtered/_classified

        # ── background task state ────────────────────────────────────
        self._task_lock = threading.Lock()
        self._task_status: str = "idle"       # idle | running | done | error
        self._task_type: str = ""             # "filter" | "sentiment"
        self._task_message: str = ""
        self._task_result: Optional[List[Dict]] = None
        self._task_error: Optional[str] = None

        # ── Restore last session config from DB ──
        self._restore_last_config()

    # ── read-only properties ─────────────────────────────────────────

    @property
    def market(self) -> Optional[str]:
        return self._market.name if self._market else None

    @property
    def asset_type(self) -> Optional[str]:
        return self._asset_type.value if self._asset_type else None

    @property
    def asset_id(self) -> Optional[str]:
        return self._asset_id

    # ── config / metadata endpoints ──────────────────────────────────

    def get_config(self) -> Dict:
        from .. import __version__
        models = discover_models()
        return {
            "version": __version__,
            "markets": [
                {"name": m.name, "display_name": m.display_name}
                for m in MARKETS.values()
            ],
            "us_commodities": _US_COMMODITIES,
            "china_commodities": _CHINA_COMMODITIES,
            "us_stocks": _US_STOCKS,
            "base_models": models["base_models"],
            "adapters": models["adapters"],
        }

    def get_status(self) -> Dict:
        return {
            "market": self.market,
            "asset_type": self.asset_type,
            "asset_id": self.asset_id,
            "article_count": len(self._articles),
            "filtered_count": len(self._filtered),
        }

    def get_task_status(self) -> Dict:
        with self._task_lock:
            resp: Dict[str, Any] = {
                "status": self._task_status,
                "type": self._task_type,
                "message": self._task_message,
            }
            if self._task_status == "done":
                resp["result"] = self._task_result
                # Auto-reset to idle after result is consumed
                self._task_status = "idle"
            elif self._task_status == "error":
                resp["error"] = self._task_error
                self._task_status = "idle"
            return resp

    # ── session restore ─────────────────────────────────────────────

    def _restore_last_config(self) -> None:
        """Restore last market/asset/model config from the database."""
        try:
            last_market = self._db.get_setting("last_market")
            last_asset_type = self._db.get_setting("last_asset_type")
            last_asset_id = self._db.get_setting("last_asset_id")
            if last_market and last_market.upper() in MARKETS:
                self._market = MARKETS[last_market.upper()]
            if last_asset_type:
                try:
                    self._asset_type = AssetType(last_asset_type)
                except ValueError:
                    pass
            if last_asset_id:
                self._asset_id = last_asset_id
            # Restore model paths
            for key, setter in [
                ("filter_base", "filter"), ("sentiment_base", "sentiment")
            ]:
                base = self._db.get_setting(f"last_{key}")
                adapter = self._db.get_setting(f"last_{key.replace('base', 'adapter')}")
                if base and Path(base).exists():
                    if key.startswith("filter"):
                        self._orchestrator.set_filter_config(FilterModelConfig(
                            base_model_path=Path(base),
                            adapter_path=Path(adapter) if adapter and Path(adapter).exists() else None,
                        ))
                    else:
                        self._orchestrator.set_sentiment_config(SentimentModelConfig(
                            base_model_path=Path(base),
                            adapter_path=Path(adapter) if adapter and Path(adapter).exists() else None,
                        ))
            log.info("Restored session config from database")
        except Exception as e:
            log.debug("Could not restore last config: %s", e)

    def get_last_config(self) -> Dict:
        """Return the last saved market/asset config for frontend pre-fill."""
        return {
            "last_market": self._db.get_setting("last_market"),
            "last_asset_type": self._db.get_setting("last_asset_type"),
            "last_asset_id": self._db.get_setting("last_asset_id"),
            "last_filter_base": self._db.get_setting("last_filter_base"),
            "last_filter_adapter": self._db.get_setting("last_filter_adapter"),
            "last_sentiment_base": self._db.get_setting("last_sentiment_base"),
            "last_sentiment_adapter": self._db.get_setting("last_sentiment_adapter"),
        }

    # ── model config ─────────────────────────────────────────────────

    def set_filter_model(self, base_model_path: str,
                         adapter_path: Optional[str]) -> None:
        """Hot-swap the filter model config.  Unloads any loaded model."""
        bp = _validate_path(base_model_path, "Filter base model")
        ap = _validate_path(adapter_path, "Filter adapter") if adapter_path else None
        cfg = FilterModelConfig(
            base_model_path=bp,
            adapter_path=ap,
        )
        self._orchestrator.set_filter_config(cfg)
        # Persist for session restore
        self._db.set_setting("last_filter_base", str(bp))
        self._db.set_setting("last_filter_adapter", str(ap) if ap else "")
        log.info("Filter model set: base=%s adapter=%s", bp, ap)

    def set_sentiment_model(self, base_model_path: str,
                            adapter_path: Optional[str]) -> None:
        """Hot-swap the sentiment model config.  Unloads any loaded model."""
        bp = _validate_path(base_model_path, "Sentiment base model")
        ap = _validate_path(adapter_path, "Sentiment adapter") if adapter_path else None
        cfg = SentimentModelConfig(
            base_model_path=bp,
            adapter_path=ap,
        )
        self._orchestrator.set_sentiment_config(cfg)
        self._db.set_setting("last_sentiment_base", str(bp))
        self._db.set_setting("last_sentiment_adapter", str(ap) if ap else "")
        log.info("Sentiment model set: base=%s adapter=%s", bp, ap)

    # ── configuration ────────────────────────────────────────────────

    def set_market(self, market: str) -> None:
        key = _validate_market(market)
        self._market = MARKETS[key]
        with self._data_lock:
            self._articles.clear()
            self._filtered.clear()
            self._classified.clear()
        self._db.set_setting("last_market", key)
        log.info("Market set: %s", key)

    def set_asset(self, asset_type: str, asset_id: str) -> None:
        atype = _validate_asset_type(asset_type)
        aid = _validate_asset_id(asset_id)
        self._asset_type = AssetType(atype)
        self._asset_id = aid

        if self._asset_type is AssetType.STOCK:
            if self._market and self._market.name == "US":
                self._display_name = AssetRegistry.get_us_stock_display_name(aid)
            else:
                self._display_name = AssetRegistry.get_china_stock_name(aid)
        else:
            self._display_name = aid.capitalize()

        with self._data_lock:
            self._articles.clear()
            self._filtered.clear()
            self._classified.clear()
        self._db.set_setting("last_asset_type", atype)
        self._db.set_setting("last_asset_id", aid)
        log.info("Asset set: %s / %s", atype, aid)

    # ── pipeline steps (all async via background thread) ─────────────

    def start_fetch(self, **kwargs) -> bool:
        """Start news fetch in background thread.  Returns False if busy."""
        self._require_asset()
        return self._start_task("fetch", lambda: self._run_fetch(**kwargs))

    def start_filter(self) -> bool:
        """Start filter in background thread.  Returns False if busy."""
        with self._data_lock:
            if not self._articles:
                raise RuntimeError("No articles to filter. Fetch news first.")
        return self._start_task("filter", self._run_filter)

    def start_sentiment(self) -> bool:
        """Start sentiment in background thread.  Returns False if busy."""
        with self._data_lock:
            target = self._filtered if self._filtered else self._articles
            if not target:
                raise RuntimeError("No articles to analyze. Fetch news first.")
        return self._start_task("sentiment", self._run_sentiment)

    # ── background task internals ────────────────────────────────────

    def _start_task(self, task_type: str, fn) -> bool:
        with self._task_lock:
            if self._task_status == "running":
                return False
            self._task_status = "running"
            self._task_type = task_type
            self._task_message = f"Starting {task_type}..."
            self._task_result = None
            self._task_error = None

        t = threading.Thread(target=fn, daemon=True)
        t.start()
        return True

    def _run_fetch(self, **kwargs) -> None:
        try:
            self._set_msg("Fetching news articles...")
            articles = self._orchestrator.fetch_news(
                market=self._market.name,
                asset_type=self._asset_type,
                asset_id=self._asset_id,
                **kwargs,
            )
            with self._data_lock:
                self._articles = articles
            result = [a.to_dict() for a in articles]
            with self._task_lock:
                self._task_result = result
                self._task_status = "done"
                self._task_message = f"Fetched {len(articles)} articles"
        except Exception as e:
            log.error("Fetch failed: %s", e, exc_info=True)
            with self._task_lock:
                self._task_status = "error"
                self._task_error = str(e)
                self._task_message = f"Fetch error: {e}"

    def _run_filter(self) -> None:
        try:
            self._set_msg("Loading filter model...")
            with self._data_lock:
                articles_copy = list(self._articles)
            filtered = self._orchestrator.filter_news(
                articles=articles_copy,
                asset_type=self._asset_type,
                asset_id=self._asset_id,
                display_name=self._display_name,
            )
            with self._data_lock:
                self._filtered = filtered
            self._set_msg("Unloading filter model...")
            self._orchestrator.unload_filter()
            result = [a.to_dict() for a in filtered]
            with self._task_lock:
                self._task_result = result
                self._task_status = "done"
                self._task_message = f"Filtered: {len(filtered)} relevant articles"
        except Exception as e:
            log.error("Filter failed: %s", e, exc_info=True)
            with self._task_lock:
                self._task_status = "error"
                self._task_error = str(e)
                self._task_message = f"Filter error: {e}"

    def _run_sentiment(self) -> None:
        try:
            with self._data_lock:
                target = list(self._filtered if self._filtered else self._articles)
                all_articles = list(self._articles)
            n = len(target)
            self._set_msg(f"Loading sentiment model... (0/{n})")

            def _progress(done, total):
                self._set_msg(f"Analyzing sentiment... ({done}/{total})")

            self._orchestrator.analyze_sentiment(target, progress_cb=_progress)

            for a in all_articles:
                if a.relevant is False and a.sentiment is None:
                    a.sentiment = "neutral"

            with self._data_lock:
                self._classified = list(all_articles)
            self._save_classified()

            # ── Persist to database ──
            try:
                sid = self._db.create_session(
                    market=self._market.name if self._market else "unknown",
                    asset_type=self._asset_type.value if self._asset_type else "unknown",
                    asset_id=self._asset_id or "unknown",
                )
                self._session_id = sid
                article_dicts = [a.to_dict() for a in self._classified]
                self._db.save_articles(sid, article_dicts)
                pos = sum(1 for a in self._classified if a.sentiment == "positive")
                neg = sum(1 for a in self._classified if a.sentiment == "negative")
                neu = sum(1 for a in self._classified if a.sentiment == "neutral")
                self._db.update_session_counts(
                    sid,
                    article_count=len(self._articles),
                    filtered_count=len(self._filtered),
                    positive=pos, negative=neg, neutral=neu,
                )
                log.info("Session %d saved: %d articles (%d+/%d-/%dn)",
                         sid, len(article_dicts), pos, neg, neu)
            except Exception as db_err:
                log.warning("Failed to save session to database: %s", db_err)

            result = [a.to_dict() for a in self._classified if a.sentiment]

            with self._task_lock:
                self._task_result = result
                self._task_status = "done"
                self._task_message = f"Classified {len(result)} articles"
        except Exception as e:
            log.error("Sentiment analysis failed: %s", e, exc_info=True)
            with self._task_lock:
                self._task_status = "error"
                self._task_error = str(e)
                self._task_message = f"Sentiment error: {e}"

    def _set_msg(self, msg: str) -> None:
        with self._task_lock:
            self._task_message = msg

    # ── results / persistence ────────────────────────────────────────

    def get_results(self) -> List[Dict]:
        if self._classified:
            return [a.to_dict() for a in self._classified]
        loaded = self._load_classified()
        return loaded if loaded else []

    def _classified_path(self) -> Path:
        market = self._market.name.lower() if self._market else "unknown"
        atype = self._asset_type.value if self._asset_type else "unknown"
        return self._data_dir / f"{market}_{atype}_{self._asset_id}_classified.json"

    def _save_classified(self) -> None:
        path = self._classified_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "market": self._market.name if self._market else None,
            "asset_type": self._asset_type.value if self._asset_type else None,
            "asset": self._asset_id,
            "last_updated": datetime.now().isoformat(),
            "total_articles": len(self._classified),
            "articles": [a.to_dict() for a in self._classified],
        }
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _load_classified(self) -> List[Dict]:
        path = self._classified_path()
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("articles", [])
        except Exception:
            return []

    def _require_asset(self) -> None:
        if self._market is None:
            raise RuntimeError("Market not set.")
        if self._asset_type is None or self._asset_id is None:
            raise RuntimeError("Asset not set.")

    # ── file browser ────────────────────────────────────────────────

    # Allowed root directories for the file browser (sandboxing)
    _BROWSE_ROOTS: List[Path] = []

    @classmethod
    def _init_browse_roots(cls) -> None:
        """Build the allowed root list once."""
        if cls._BROWSE_ROOTS:
            return
        candidates = [
            _PROJECT_ROOT / "models",
            _PROJECT_ROOT,
            Path.home(),
        ]
        # On Windows, add common drive roots
        import sys
        if sys.platform == "win32":
            for letter in "CDEFGH":
                candidates.append(Path(f"{letter}:\\"))
        cls._BROWSE_ROOTS = [p for p in candidates if p.exists()]

    @staticmethod
    def _is_path_allowed(p: Path) -> bool:
        """Check whether *p* falls under an allowed browse root."""
        PipelineState._init_browse_roots()
        resolved = p.resolve()
        for root in PipelineState._BROWSE_ROOTS:
            try:
                resolved.relative_to(root.resolve())
                return True
            except ValueError:
                continue
        return False

    @staticmethod
    def browse_directory(path: Optional[str] = None) -> Dict:
        """List contents of a directory for the model file browser.

        Returns ``{current: str, parent: str|None, entries: [...]}``.
        Each entry: ``{name, path, is_dir, is_model, is_adapter}``.

        Paths are sandboxed to allowed roots (project dir, home, drives).
        """
        if not path:
            # Try the project's models/ dir first, fall back to user home
            models_dir = _PROJECT_ROOT / "models"
            if models_dir.exists():
                path = str(models_dir)
            else:
                path = str(Path.home())

        p = Path(path).resolve()

        # Sandbox check — block paths outside allowed roots
        if not PipelineState._is_path_allowed(p):
            log.warning("Browse blocked for path outside allowed roots: %s", p)
            p = _PROJECT_ROOT / "models" if (_PROJECT_ROOT / "models").exists() else Path.home()

        if not p.exists() or not p.is_dir():
            # Fall back to parent, then home, then drives
            for fallback in [p.parent, Path.home(), Path("C:\\")]:
                if fallback.exists():
                    p = fallback
                    break

        entries: List[Dict[str, Any]] = []
        try:
            for item in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                if item.name.startswith("."):
                    continue
                # Skip system/hidden directories
                if item.name in ("$Recycle.Bin", "System Volume Information",
                                 "Windows", "ProgramData"):
                    continue
                entry: Dict[str, Any] = {
                    "name": item.name,
                    "path": str(item),
                    "is_dir": item.is_dir(),
                    "is_model": False,
                    "is_adapter": False,
                }
                if item.is_dir():
                    entry["is_model"] = (item / "config.json").exists()
                    entry["is_adapter"] = (item / "adapter_config.json").exists()
                entries.append(entry)
        except PermissionError:
            return {"current": str(p), "parent": str(p.parent), "entries": [],
                    "error": "Permission denied"}

        return {
            "current": str(p),
            "parent": str(p.parent) if p.parent != p else None,
            "entries": entries,
        }

    # ── price data (delegated to pipeline.prices) ──────────────────

    @staticmethod
    def get_price_data(market: str, asset_type: str, asset_id: str) -> Dict:
        from ..prices import PriceProvider
        return PriceProvider.get_price_data(market, asset_type, asset_id)

    @staticmethod
    def get_live_price(market: str, asset_type: str, asset_id: str) -> Dict:
        from ..prices import PriceProvider
        return PriceProvider.get_live_price(market, asset_type, asset_id)

    # ── database queries (for UI) ────────────────────────────────────

    def export_csv(self) -> str:
        """Export the current (or last) session as CSV."""
        if self._session_id:
            return self._db.export_session_csv(self._session_id)
        # Fall back to in-memory classified data
        if not self._classified:
            return ""
        import csv, io
        buf = io.StringIO()
        fields = ["title", "date", "source", "sentiment", "url", "summary"]
        w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for a in self._classified:
            w.writerow(a.to_dict())
        return buf.getvalue()

    def get_history(self) -> List[Dict]:
        """Return recent analysis sessions."""
        return self._db.get_recent_sessions(limit=25)

    def get_sentiment_history(self, days: int = 30) -> List[Dict]:
        """Return per-day sentiment aggregation for the current asset."""
        if not self._market or not self._asset_type or not self._asset_id:
            return []
        return self._db.get_sentiment_history(
            market=self._market.name,
            asset_type=self._asset_type.value,
            asset_id=self._asset_id,
            days=days,
        )

    def cleanup(self) -> None:
        self._orchestrator.cleanup()
        try:
            self._db.close()
        except Exception:
            pass
