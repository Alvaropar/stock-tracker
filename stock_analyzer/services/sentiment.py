"""
Sentiment analysis service for the app.

Fetches news headlines via scrapers, then classifies sentiment using either:
  - Cloud LLM APIs (Claude, GPT, Gemini, or Grok)
  - A local model (base LLM + optional LoRA adapter via transformers/PEFT)
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import feedparser
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

log = logging.getLogger("app.sentiment")


# ── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class SentimentResult:
    score:        float = 0.0
    signal:       str   = "NEUTRAL"
    n_articles:   int   = 0
    n_positive:   int   = 0
    n_negative:   int   = 0
    n_neutral:    int   = 0
    momentum:     float = 0.0
    dispersion:   float = 0.0
    volume_trend: str   = "stable"
    weekly_score: float = 0.0
    monthly_score: float = 0.0
    headlines:    List[Tuple[str, str]] = field(default_factory=list)
    all_articles: List[dict]            = field(default_factory=list)


def _map_signal(score: float) -> str:
    if score >=  0.5: return "STRONG BULLISH"
    if score >=  0.2: return "BULLISH"
    if score >= -0.2: return "NEUTRAL"
    if score >= -0.5: return "BEARISH"
    return "STRONG BEARISH"


# ── News scraper (built-in, no external pipeline dependency) ─────────────────

try:
    import yfinance as yf
    _YFINANCE = True
except ImportError:
    _YFINANCE = False


def _parse_date(date_str: str) -> str:
    """Best-effort date normaliser → YYYY-MM-DD."""
    if not date_str:
        return datetime.now().strftime("%Y-%m-%d")
    date_str = date_str.strip()
    from email.utils import parsedate_to_datetime
    try:
        return parsedate_to_datetime(date_str).strftime("%Y-%m-%d")
    except Exception:
        pass
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00")).strftime("%Y-%m-%d")
    except Exception:
        pass
    return datetime.now().strftime("%Y-%m-%d")


def _scrape_news(ticker: str, company_name: str = "",
                  max_articles: int = 50, days: int = 15) -> List[dict]:
    """Fetch news headlines from Yahoo Finance + Google News RSS.

    Args:
        days: Only keep articles published within the last N days (default 15).
    """
    articles: List[dict] = []
    seen: set = set()
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    def _add(title, date, source, url="", summary=""):
        key = title.lower().strip()
        if key not in seen and title:
            # Filter by time window
            if date < cutoff:
                return
            seen.add(key)
            articles.append({
                "title": title, "headline": title, "date": date,
                "source": source, "url": url, "summary": summary[:200],
                "ticker": ticker,
            })

    # Yahoo Finance
    if _YFINANCE:
        try:
            import concurrent.futures as _cf
            def _yf_news():
                return yf.Ticker(ticker).news
            with _cf.ThreadPoolExecutor() as _pool:
                news = _pool.submit(_yf_news).result(timeout=15)
            if isinstance(news, list):
                for item in news:
                    if not isinstance(item, dict):
                        continue
                    content = item.get("content", {}) if isinstance(item.get("content"), dict) else {}
                    title = content.get("title", "")
                    pub = content.get("pubDate", "")
                    provider = content.get("provider", {})
                    src = provider.get("displayName", "Yahoo Finance") if isinstance(provider, dict) else "Yahoo Finance"
                    summary = content.get("summary", "")
                    canon = content.get("canonicalUrl", {})
                    url = canon.get("url", "") if isinstance(canon, dict) else ""
                    _add(title, _parse_date(pub), src, url, summary)
        except Exception as e:
            log.warning("Yahoo Finance news failed for %s: %s", ticker, e)

    # Google News RSS
    queries = [f"{ticker} stock"]
    if company_name:
        queries.append(f"{company_name} stock")
    for q in queries:
        try:
            # 'when:Nd' tells Google News to only return articles from the last N days
            rss_url = f"https://news.google.com/rss/search?q={quote(q)}+when:{days}d&hl=en-US&gl=US&ceid=US:en"
            rss_resp = requests.get(rss_url, timeout=10)
            feed = feedparser.parse(rss_resp.text)
            for entry in feed.entries:
                headline = entry.get("title", "")
                source = "Google News"
                if " - " in headline:
                    parts = headline.rsplit(" - ", 1)
                    if len(parts) == 2:
                        headline, source = parts
                url = entry.get("link", "")
                pub = entry.get("published", "")
                summary = re.sub(r"<[^>]+>", "", entry.get("summary", ""))
                _add(headline, _parse_date(pub), source, url, summary)
        except Exception as e:
            log.warning("Google RSS failed for '%s': %s", q, e)

    # Sort newest first, limit
    articles.sort(key=lambda a: a.get("date", ""), reverse=True)
    return articles[:max_articles]


# ── LLM API providers ───────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a financial sentiment classifier. For each news headline about a stock, "
    "respond with exactly one word: positive, negative, or neutral.\n\n"
    "Rules:\n"
    "- positive: the headline suggests the stock price will go UP (good earnings, upgrades, partnerships, growth)\n"
    "- negative: the headline suggests the stock price will go DOWN (losses, downgrades, lawsuits, layoffs)\n"
    "- neutral: the headline is factual/ambiguous with no clear directional impact\n\n"
    "Respond ONLY with a JSON array of strings. Example:\n"
    '[\"positive\", \"negative\", \"neutral\"]'
)


def _build_user_prompt(ticker: str, headlines: List[str]) -> str:
    numbered = "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines))
    return (
        f"Classify the sentiment of each headline for {ticker}.\n"
        f"Return a JSON array with exactly {len(headlines)} entries "
        f"(one per headline, in order).\n\n{numbered}"
    )


def _parse_sentiments(text: str, expected: int) -> List[str]:
    """Extract sentiment labels from LLM response text."""
    # Try to find a JSON array
    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if match:
        try:
            arr = json.loads(match.group())
            result = []
            for item in arr:
                s = str(item).lower().strip()
                if s in ("positive", "negative", "neutral"):
                    result.append(s)
                else:
                    result.append("neutral")
            if len(result) == expected:
                return result
            # Pad or trim
            while len(result) < expected:
                result.append("neutral")
            return result[:expected]
        except json.JSONDecodeError:
            pass

    # Fallback: look for line-by-line labels
    labels = re.findall(r'\b(positive|negative|neutral)\b', text.lower())
    while len(labels) < expected:
        labels.append("neutral")
    return labels[:expected]


class _LLMProvider:
    """Base class for LLM API calls."""

    def classify(self, ticker: str, headlines: List[str]) -> List[str]:
        raise NotImplementedError


class _ClaudeProvider(_LLMProvider):
    """Anthropic Claude API."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key
        self.model = model

    def classify(self, ticker: str, headlines: List[str]) -> List[str]:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 4096,
                "system": _SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": _build_user_prompt(ticker, headlines)}],
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["content"][0]["text"]
        return _parse_sentiments(text, len(headlines))


class _OpenAIProvider(_LLMProvider):
    """OpenAI GPT API (also compatible with Grok via base_url)."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini",
                 base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    def classify(self, ticker: str, headlines: List[str]) -> List[str]:
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_prompt(ticker, headlines)},
                ],
                "temperature": 0.0,
                "max_tokens": 4096,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return _parse_sentiments(text, len(headlines))


class _GeminiProvider(_LLMProvider):
    """Google Gemini API."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.model = model

    def classify(self, ticker: str, headlines: List[str]) -> List[str]:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        resp = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json={
                "systemInstruction": {"parts": [{"text": _SYSTEM_PROMPT}]},
                "contents": [{"parts": [{"text": _build_user_prompt(ticker, headlines)}]}],
                "generationConfig": {"temperature": 0.0, "maxOutputTokens": 4096},
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return _parse_sentiments(text, len(headlines))


class _LocalModelProvider(_LLMProvider):
    """Local transformer model with optional LoRA adapter (runs on GPU/CPU)."""

    def __init__(self, model_path: str, adapter_path: str = ""):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return

        log.info("Loading local model from '%s'%s",
                 self.model_path,
                 f" + adapter '{self.adapter_path}'" if self.adapter_path else " (base only)")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from transformers import PreTrainedTokenizerBase

        base_path = self.model_path
        adapter_path = self.adapter_path if self.adapter_path else None
        cuda = torch.cuda.is_available()

        # Quantisation config for GPU
        bnb = None
        if cuda:
            try:
                bnb = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    llm_int8_enable_fp32_cpu_offload=True,
                )
            except Exception:
                bnb = None

        # Tokenizer — prefer adapter dir (may have fine-tuned tokens)
        tok_path = adapter_path if (adapter_path and Path(adapter_path).exists()) else base_path

        # Workaround for transformers 4.57+ / Qwen compatibility
        orig_set_special_tokens = PreTrainedTokenizerBase._set_model_specific_special_tokens
        def patched_set_special_tokens(self_tok, special_tokens):
            if isinstance(special_tokens, list):
                special_tokens = {}
            return orig_set_special_tokens(self_tok, special_tokens)
        PreTrainedTokenizerBase._set_model_specific_special_tokens = patched_set_special_tokens

        self._tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Base model
        load_kw = dict(trust_remote_code=True)
        if cuda and bnb:
            load_kw.update(
                quantization_config=bnb,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
        elif cuda:
            load_kw.update(device_map="auto", torch_dtype=torch.float16)
        else:
            load_kw.update(torch_dtype=torch.float32, low_cpu_mem_usage=True)

        base = AutoModelForCausalLM.from_pretrained(base_path, **load_kw)

        # Optionally merge LoRA adapter
        if adapter_path:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
        else:
            self._model = base

        self._model.eval()
        log.info("Local model loaded successfully (%s)", "GPU" if cuda else "CPU")

    def _classify_one(self, headline: str) -> str:
        import torch

        prompt = (
            f"<|im_start|>user\n"
            f"Classify the sentiment of this financial news headline as exactly "
            f"one word: positive, negative, or neutral.\n{headline}<|im_end|>\n"
            f"<|im_start|>assistant\n/no_think"
        )
        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self._model.device)

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        response = self._tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip().lower()

        # Parse first label from response
        response = response.split("<|")[0].split("\n")[0].strip()
        if "positive" in response:
            return "positive"
        if "negative" in response:
            return "negative"
        return "neutral"

    def classify(self, ticker: str, headlines: List[str]) -> List[str]:
        self._load()
        return [self._classify_one(h) for h in headlines]

    def unload(self):
        if self._model is not None:
            log.info("Unloading local sentiment model")
            del self._model, self._tokenizer
            self._model = self._tokenizer = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass


def sentiment_available(provider: str = "", cfg: Optional[Dict[str, Any]] = None) -> bool:
    """
    True when the requested sentiment provider can run with the given config.

    - "local": needs a model_path pointing to a directory with `config.json`.
    - cloud providers (claude / gpt / gemini / grok): need an api_key in
      `cfg` or `API_KEY` in the project `.env`.
    - "compactifai": needs `API_KEY` + `API_URL` in `.env`.
    - "" (no provider): returns True if EITHER cloud or local is available
      anywhere on the install — used by callers that haven't picked a provider
      yet (e.g. /api/settings/llm-status).
    """
    from ..config import config as _cfg
    cfg = cfg or {}
    p = (provider or "").lower().strip()

    api_key = cfg.get("api_key") or _cfg.API_KEY
    api_url = _cfg.API_URL

    if p == "local":
        mp = cfg.get("model_path") or ""
        return bool(mp and (Path(mp) / "config.json").exists())
    if p == "compactifai":
        return bool(_cfg.API_KEY and _cfg.API_URL)
    if p in ("claude", "gpt", "gemini", "grok"):
        return bool(api_key)
    # Unknown / empty provider: report install-wide capability
    cloud_ok = bool(api_key and api_url)
    mp = cfg.get("model_path") or ""
    local_ok = bool(mp and (Path(mp) / "config.json").exists())
    return cloud_ok or local_ok


def _load_compactifai_key() -> tuple[str, str]:
    """Load CompactifAI API key and base URL from the project .env file."""
    import sys as _sys
    from pathlib import Path as _Path
    if getattr(_sys, "frozen", False):
        env_path = _Path(_sys.executable).resolve().parent / ".env"
    else:
        env_path = _Path(__file__).resolve().parents[2] / ".env"
    key, url = "", "https://api.compactif.ai/v1"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            v = v.strip().strip('"').strip("'")
            if k.strip() == "API_KEY":
                key = v
            elif k.strip() == "API_URL":
                # strip /chat/completions suffix if present so base_url works for _OpenAIProvider
                url = v.removesuffix("/chat/completions").rstrip("/")
    return key, url


def _make_provider(provider: str, api_key: str = "", model: str = "",
                   model_path: str = "", adapter_path: str = "") -> _LLMProvider:
    """Factory to create the right LLM provider."""
    p = provider.lower().strip()
    if p == "local":
        if not model_path:
            raise ValueError("Model path is required for local model provider.")
        return _LocalModelProvider(model_path, adapter_path)
    if p == "claude":
        return _ClaudeProvider(api_key, model or "claude-sonnet-4-20250514")
    elif p == "gpt":
        return _OpenAIProvider(api_key, model or "gpt-4o-mini")
    elif p == "gemini":
        return _GeminiProvider(api_key, model or "gemini-2.0-flash")
    elif p == "grok":
        return _OpenAIProvider(api_key, model or "grok-3-mini-fast",
                               base_url="https://api.x.ai/v1")
    elif p == "compactifai":
        env_key, env_url = _load_compactifai_key()
        return _OpenAIProvider(env_key, model or "gpt-oss-120b", base_url=env_url)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use local, claude, gpt, gemini, grok, or compactifai.")


# ── News sentiment cache ────────────────────────────────────────────────────

class _SentimentCache:
    """
    SQLite cache for classified headlines.

    Keyed by (ticker, headline_hash). Avoids re-classifying the same headline
    across repeated analysis runs. Entries expire after `ttl_days`.
    """

    def __init__(self, db_path: Optional[Path] = None, ttl_days: int = 7):
        if db_path is None:
            if getattr(sys, "frozen", False):
                db_path = Path(sys.executable).resolve().parent / "news_cache.db"
            else:
                db_path = Path(__file__).resolve().parents[2] / "news_cache.db"
        self._path = db_path
        self._ttl_days = ttl_days
        self._conn: Optional[sqlite3.Connection] = None

    def _connect(self):
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._path), timeout=5)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    ticker    TEXT    NOT NULL,
                    h_hash    TEXT    NOT NULL,
                    headline  TEXT    NOT NULL,
                    sentiment TEXT    NOT NULL,
                    cached_at TEXT    NOT NULL,
                    PRIMARY KEY (ticker, h_hash)
                )
            """)
            self._conn.commit()

    @staticmethod
    def _hash(headline: str) -> str:
        return hashlib.md5(headline.strip().lower().encode("utf-8")).hexdigest()

    def lookup(self, ticker: str, headlines: List[str]) -> Dict[str, str]:
        """
        Return {headline: sentiment} for all headlines that are already cached
        and not expired.
        """
        self._connect()
        cutoff = (datetime.utcnow() - timedelta(days=self._ttl_days)).isoformat()
        result: Dict[str, str] = {}
        hashes = {self._hash(h): h for h in headlines}

        if not hashes:
            return result

        # Query in batches of 500 to stay within SQLite variable limits
        hash_list = list(hashes.keys())
        for i in range(0, len(hash_list), 500):
            batch = hash_list[i:i + 500]
            placeholders = ",".join("?" * len(batch))
            rows = self._conn.execute(
                f"SELECT h_hash, sentiment FROM cache "
                f"WHERE ticker = ? AND h_hash IN ({placeholders}) AND cached_at > ?",
                [ticker] + batch + [cutoff],
            ).fetchall()
            for h_hash, sentiment in rows:
                if h_hash in hashes:
                    result[hashes[h_hash]] = sentiment

        return result

    def store(self, ticker: str, classified: List[Tuple[str, str]]):
        """Store [(headline, sentiment), ...] into the cache."""
        if not classified:
            return
        self._connect()
        now = datetime.utcnow().isoformat()
        rows = [(ticker, self._hash(h), h, s, now) for h, s in classified]
        self._conn.executemany(
            "INSERT OR REPLACE INTO cache (ticker, h_hash, headline, sentiment, cached_at) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def purge_expired(self):
        """Delete entries older than ttl_days."""
        self._connect()
        cutoff = (datetime.utcnow() - timedelta(days=self._ttl_days)).isoformat()
        self._conn.execute("DELETE FROM cache WHERE cached_at <= ?", [cutoff])
        self._conn.commit()

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        self._connect()
        total = self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        cutoff = (datetime.utcnow() - timedelta(days=self._ttl_days)).isoformat()
        valid = self._conn.execute(
            "SELECT COUNT(*) FROM cache WHERE cached_at > ?", [cutoff]
        ).fetchone()[0]
        return {"total": total, "valid": valid, "expired": total - valid}


# ── Main analyser class ─────────────────────────────────────────────────────

# Process headlines in batches to stay within token limits
_BATCH_SIZE = 30


class SentimentAnalyzer:
    """
    News sentiment analyser using cloud LLM APIs or a local model.

    Fetches headlines from Yahoo Finance + Google News, then classifies them
    via Claude/GPT/Gemini/Grok API or a local transformer model with optional
    LoRA adapter.
    """

    def __init__(
        self,
        provider: str = "local",
        api_key: str = "",
        model: str = "",
        model_path: str = "",
        adapter_path: str = "",
        max_articles: int = 50,
        cache_ttl_days: int = 7,
        **kwargs,
    ):
        _p = provider.lower().strip()
        if _p not in ("local", "compactifai") and not api_key:
            raise ValueError("API key is required for cloud sentiment analysis.")
        self._provider = _make_provider(
            provider, api_key, model,
            model_path=model_path, adapter_path=adapter_path,
        )
        self._max_articles = max_articles
        self._days = kwargs.get("days", 15)
        self._provider_name = provider
        self._cache = _SentimentCache(ttl_days=cache_ttl_days)

    def analyze_asset(
        self,
        ticker: str,
        company_name: str = "",
        progress_cb=None,
    ) -> SentimentResult:
        """Fetch news and classify sentiment for one asset."""
        try:
            # 1. Scrape news
            articles = _scrape_news(ticker, company_name, self._max_articles, self._days)
            if not articles:
                log.info("No articles found for %s", ticker)
                return SentimentResult()

            headlines = [a["title"] for a in articles]

            # 2. Check cache for already-classified headlines
            cached = self._cache.lookup(ticker, headlines)
            uncached_headlines = [h for h in headlines if h not in cached]

            if cached:
                log.info("%s: %d/%d headlines from cache, %d new to classify",
                         ticker, len(cached), len(headlines), len(uncached_headlines))

            # 3. Classify only NEW headlines via LLM
            new_sentiments: Dict[str, str] = {}
            if uncached_headlines:
                for i in range(0, len(uncached_headlines), _BATCH_SIZE):
                    batch = uncached_headlines[i:i + _BATCH_SIZE]
                    try:
                        batch_results = self._provider.classify(ticker, batch)
                        for h, s in zip(batch, batch_results):
                            new_sentiments[h] = s
                    except Exception as e:
                        log.warning("LLM batch failed for %s (batch %d): %s",
                                    ticker, i // _BATCH_SIZE, e)
                        # Do NOT cache failures — they fall back to neutral here
                        # and will be retried on the next run.
                    if i + _BATCH_SIZE < len(uncached_headlines):
                        time.sleep(0.5)

                # Only persist successfully classified headlines
                if new_sentiments:
                    self._cache.store(ticker, list(new_sentiments.items()))

            # 4. Merge cached + new sentiments, preserving original article order
            all_sentiments: List[str] = []
            for h in headlines:
                if h in cached:
                    all_sentiments.append(cached[h])
                elif h in new_sentiments:
                    all_sentiments.append(new_sentiments[h])
                else:
                    all_sentiments.append("neutral")

            # 5. Attach sentiments to articles
            for art, sent in zip(articles, all_sentiments):
                art["sentiment"] = sent

            # 6. Compute scores
            pos = all_sentiments.count("positive")
            neg = all_sentiments.count("negative")
            neu = all_sentiments.count("neutral")
            total = len(all_sentiments)
            raw_score = (pos - neg) / total if total else 0.0

            # Simple momentum: compare recent vs older sentiment
            mid = total // 2
            if mid > 0:
                recent = all_sentiments[:mid]
                older = all_sentiments[mid:]
                r_score = (recent.count("positive") - recent.count("negative")) / len(recent)
                o_score = (older.count("positive") - older.count("negative")) / len(older)
                momentum = round(r_score - o_score, 3)
            else:
                momentum = 0.0

            # Volume trend (based on date spread)
            dates = [a.get("date", "") for a in articles]
            unique_dates = len(set(dates))
            volume_trend = "rising" if unique_dates > 5 else "stable" if unique_dates > 2 else "falling"

            # Top headlines with sentiment
            top_headlines = [(a["title"], a["sentiment"]) for a in articles[:10]]

            if progress_cb:
                progress_cb(ticker, total, pos, neg, neu)

            result = SentimentResult(
                score=round(raw_score, 3),
                signal=_map_signal(raw_score),
                n_articles=total,
                n_positive=pos,
                n_negative=neg,
                n_neutral=neu,
                momentum=momentum,
                dispersion=round(neu / total, 3) if total else 0.0,
                volume_trend=volume_trend,
                weekly_score=round(raw_score, 3),   # simplified
                monthly_score=round(raw_score, 3),
                headlines=top_headlines,
                all_articles=articles,
            )
            try:
                from .pit_data import PointInTimeStore
                as_of = articles[0].get("date") if articles and articles[0].get("date") else pd.Timestamp.utcnow()
                PointInTimeStore().record_sentiment(
                    ticker,
                    as_of,
                    {
                        "score": result.score,
                        "momentum": result.momentum,
                        "dispersion": result.dispersion,
                        "n_articles": result.n_articles,
                    },
                )
            except Exception:
                pass
            return result

        except Exception as e:
            import traceback
            log.error("Sentiment analysis failed for %s: %s", ticker, e)
            log.error("Full traceback:\n%s", traceback.format_exc())
            return SentimentResult()

    def unload(self):
        """Release model resources and close cache."""
        if hasattr(self._provider, 'unload'):
            self._provider.unload()
        try:
            self._cache.purge_expired()
            self._cache.close()
        except Exception:
            pass
