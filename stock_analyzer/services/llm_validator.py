"""
LLM + RAG validation layer — two-turn debate pipeline.

Architecture
============

Turn 1 — Independent analysis
    The LLM receives raw market indicators + fundamentals (NO computed scores,
    NO regime classification, NO sentiment score) together with live web
    search results (news, macro, analyst upgrades/downgrades) and SEC EDGAR
    8-K/10-Q filings.  It reasons step-by-step inside a <thinking> block,
    then independently estimates:
      • trend situation  • market regime  • signal direction + score estimate
      • technical score  • fundamental score  • sentiment score
      • risk score       • dip score

Turn 2 — Debate
    The model's actual computed scores are revealed.  The LLM compares its
    own estimates against the quant model, produces a score delta table,
    argues where they agree/disagree and why, then issues a final verdict
    and recommendation.

Retrieval
=========
• 8 sequential DuckDuckGo queries covering: recent news, earnings/guidance,
  analyst upgrades/downgrades, short-interest/bearish thesis, macro
  environment, sector/industry tailwinds, regulatory/geopolitical risks,
  and insider activity.
• SEC EDGAR: recent 8-K and 10-Q filings via the EDGAR full-text search API.
• yfinance: structured analyst consensus, price targets, upcoming earnings
  date, recent recommendation changes, and last-quarter earnings surprise.

Indicator / score split
=======================
_INDICATOR_KEYS  — raw values shown in Turn 1 (RSI, BB%, MA levels, ADX,
                   returns, fundamentals, relative strength, checklist items)
_SCORE_KEYS      — withheld until Turn 2 (all composite scores, computed
                   regime/trend labels, sentiment data, signal labels)
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

log = logging.getLogger("app.llm_validator")


# ── Config from .env ──────────────────────────────────────────────────────────

def _load_env() -> Dict[str, str]:
    if getattr(sys, "frozen", False):
        env_path = Path(sys.executable).resolve().parent / ".env"
    else:
        env_path = Path(__file__).resolve().parents[2] / ".env"
    env: Dict[str, str] = {}
    if not env_path.exists():
        return env
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


_ENV = _load_env()
COMPACTIFAI_API_KEY = _ENV.get("API_KEY", "")
COMPACTIFAI_API_URL = _ENV.get("API_URL", "https://api.compactif.ai/v1/chat/completions")
COMPACTIFAI_MODEL   = _ENV.get("MODEL", "gpt-oss-120b")


# ── API client ────────────────────────────────────────────────────────────────

def api_status() -> Dict[str, Any]:
    if not COMPACTIFAI_API_KEY:
        return {"available": False, "model": COMPACTIFAI_MODEL,
                "api_url": COMPACTIFAI_API_URL, "error": "API_KEY not found in .env"}
    try:
        r = requests.post(
            COMPACTIFAI_API_URL,
            headers={"Authorization": f"Bearer {COMPACTIFAI_API_KEY}",
                     "Content-Type": "application/json"},
            json={"model": COMPACTIFAI_MODEL, "max_tokens": 1,
                  "messages": [{"role": "user", "content": "hi"}]},
            timeout=8,
        )
        r.raise_for_status()
        return {"available": True, "model": COMPACTIFAI_MODEL,
                "api_url": COMPACTIFAI_API_URL, "error": ""}
    except Exception as e:
        return {"available": False, "model": COMPACTIFAI_MODEL,
                "api_url": COMPACTIFAI_API_URL, "error": str(e)}


_ALLOWED_MODELS = {"gpt-oss-120b", "glm-5-1"}


def _chat(messages: List[Dict], *, temperature: float = 0.3, timeout: int = 240,
          model: Optional[str] = None) -> str:
    m = model if model in _ALLOWED_MODELS else COMPACTIFAI_MODEL
    r = requests.post(
        COMPACTIFAI_API_URL,
        headers={"Authorization": f"Bearer {COMPACTIFAI_API_KEY}",
                 "Content-Type": "application/json"},
        json={"model": m, "temperature": temperature,
              "messages": messages},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# ── Indicator / score split ───────────────────────────────────────────────────

# Raw indicators shown to LLM in Turn 1
_INDICATOR_KEYS = frozenset({
    "ticker", "name", "sector", "currency",
    # Price & returns
    "price", "ret_1d", "ret_1w", "ret_1m", "ret_3m",
    "w52_pct", "w52_hi", "w52_lo",
    # Technical (raw values)
    "rsi", "bb_pct", "vol_ratio", "atr", "atr_pct", "adx",
    "ma_cross", "macd_bull", "macd", "macd_sig",
    "ma20", "ma50", "ma200",
    "vol_pctl", "rs_1m", "rs_55d", "rs_3m",
    "elder_d", "elder_w", "trend_ext",
    # Entry checklist (individual checks, not composite)
    "checklist", "chk_passed", "chk_total", "confidence",
    # Fundamentals
    "pe_trail", "pe_fwd", "peg", "pb", "gross_mgn", "op_mgn", "net_mgn",
    "roe", "roa", "rev_growth", "eps_growth", "debt_eq", "curr_ratio",
    "quick_ratio", "fcf", "mkt_cap", "beta", "div_yield",
    "target_px", "rec_mean", "n_analysts", "short_float",
    "ev_ebitda", "ps",
})

# Computed scores / classifications withheld in Turn 1, revealed in Turn 2
_SCORE_KEYS = frozenset({
    "regime", "mkt_regime", "regime_chg", "trend_stage", "vol_regime",
    "tech_score", "technical_score", "fund_score", "fundamental_score",
    "sent_score", "sentiment_score", "overall_score", "score", "raw_score",
    "momentum_score", "risk_score", "dip_score", "adj_confidence",
    "signal", "signal_css", "ctx_signal", "ctx_hint", "raw_sig",
    "sent_signal", "n_articles", "n_positive", "n_negative", "n_neutral",
    "sent_momentum", "sent_weekly", "sent_monthly", "sent_vol_trend",
    "headlines", "articles",
    "ml_regime", "ml_regime_conf", "ml_regime_probs",
    "ml_entry", "ml_exit", "ml_signal", "ml_decision", "ml_uncertainty",
    "_data",
})

_SCORE_LABELS = {
    "overall_score":  ("Overall signal score",  "-1 (strong sell) → +1 (strong buy)"),
    "tech_score":     ("Technical score",        "-1 → +1"),
    "fund_score":     ("Fundamental score",      "-1 → +1"),
    "sent_score":     ("News sentiment score",   "-1 → +1"),
    "momentum_score": ("Momentum score",         "0 (weak) → 1 (strong)"),
    "risk_score":     ("Risk score",             "0 (low risk) → 1 (high risk)"),
    "dip_score":      ("Dip opportunity score",  "-1 (overbought) → +1 (strong dip)"),
    "adj_confidence": ("Signal confidence",      "0 → 1"),
    "trend_stage":    ("Trend stage label",      "EARLY_UP / UPTREND / LATE_UP / TOPPING / EARLY_DOWN / DOWNTREND / LATE_DOWN / BOTTOMING / SIDEWAYS"),
    "mkt_regime":     ("Market regime",          "TREND / MEAN_REVERSION / NEUTRAL"),
    "regime_chg":     ("Regime change flag",     "UP / DOWN / None"),
    "ctx_signal":     ("Contextual signal label","STRONG BUY / BUY / NEUTRAL / SELL / STRONG SELL / AVOID etc."),
}


def _split_analysis(analysis: Dict[str, Any]) -> Tuple[Dict, Dict]:
    """Return (indicators_dict, scores_dict)."""
    indicators = {k: v for k, v in analysis.items()
                  if k in _INDICATOR_KEYS and v not in (None, "", [], {})}
    scores = {k: v for k, v in analysis.items()
              if k in _SCORE_KEYS and v not in (None, "", [], {}, "NONE")}
    return indicators, scores


# ── DDG web search ────────────────────────────────────────────────────────────

try:
    from ddgs import DDGS
    _DDG_AVAILABLE = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        _DDG_AVAILABLE = True
    except ImportError:
        _DDG_AVAILABLE = False
        log.info("ddgs not installed; news retrieval will be limited to RSS fallback")


def _ddg_search(query: str, max_results: int = 8, timelimit: str = "m") -> List[dict]:
    if not _DDG_AVAILABLE:
        return []
    try:
        with DDGS() as ddgs:
            raw = list(ddgs.news(query, max_results=max_results, timelimit=timelimit))
        return [{"title": r.get("title",""), "url": r.get("url",""),
                 "date": r.get("date",""), "body": r.get("body",""),
                 "source": r.get("source",""), "category": ""} for r in raw]
    except Exception as e:
        log.warning("DDG '%s': %s", query[:60], e)
        return []


def _multi_search(ticker: str, company_name: str, sector: str = "") -> List[dict]:
    """
    8 sequential queries covering all relevant information angles.
    Sequential to avoid DDG rate-limit (403).
    """
    name_q = company_name or ticker
    sec_q  = f"{sector} sector" if sector else "technology sector"
    queries = [
        # angle                          query                                     timelimit  category
        (f"{name_q} {ticker} stock news earnings results 2025",                   "m",  "news"),
        (f"{name_q} {ticker} earnings revenue guidance beats misses",             "m",  "earnings"),
        (f"{name_q} {ticker} analyst upgrade downgrade price target rating",      "m",  "analysts"),
        (f"{name_q} {ticker} short interest bears risks lawsuit regulation",      "m",  "bearish"),
        (f"US economy macro interest rates inflation GDP outlook 2025",           "w",  "macro"),
        (f"{sec_q} outlook trends disruption competition 2025",                   "m",  "sector"),
        (f"{name_q} {ticker} insider trading SEC filing executive",               "m",  "insider"),
        (f"{name_q} {ticker} merger acquisition partnership product launch",      "m",  "catalysts"),
    ]

    seen_urls: set = set()
    results: List[dict] = []
    for i, (q, tl, cat) in enumerate(queries):
        if i > 0:
            time.sleep(1.5)
        for item in _ddg_search(q, max_results=8, timelimit=tl):
            url = item.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                item["category"] = cat
                results.append(item)
    return results


def _rss_fallback(ticker: str, company_name: str, days: int = 21) -> List[dict]:
    try:
        from .sentiment import _scrape_news
        arts = _scrape_news(ticker, company_name, max_articles=50, days=days)
        return [{"title": a.get("title",""), "url": a.get("url",""),
                 "date": a.get("date",""), "body": a.get("summary",""),
                 "source": a.get("source",""), "category": "news"} for a in arts]
    except Exception as e:
        log.warning("RSS fallback: %s", e)
        return []


# ── SEC EDGAR ─────────────────────────────────────────────────────────────────

_EDGAR_HEADERS = {"User-Agent": "stock-analyzer/1.0 research@example.com"}
_EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_EDGAR_TICKERS_CACHE: Optional[Dict] = None

# 8-K item codes → human-readable descriptions
_8K_ITEMS = {
    "1.01": "Entry into Material Agreement",
    "1.02": "Termination of Material Agreement",
    "1.05": "Material Cybersecurity Incident",
    "2.01": "Completion of Acquisition/Disposition",
    "2.02": "Results of Operations (Earnings)",
    "2.04": "Triggering Events / Accelerated Repayment",
    "2.06": "Material Impairment",
    "3.01": "Delisting Notice",
    "4.01": "Auditor Change",
    "5.02": "Director/Officer Departure or Appointment",
    "5.03": "Amendment to Articles of Incorporation",
    "5.07": "Shareholder Voting Results",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Material Events",
    "9.01": "Financial Statements / Exhibits",
}
_RELEVANT_FORMS = {"8-K", "10-Q", "10-K", "6-K"}


def _get_cik(ticker: str) -> Optional[str]:
    global _EDGAR_TICKERS_CACHE
    try:
        if _EDGAR_TICKERS_CACHE is None:
            r = requests.get(_EDGAR_TICKERS_URL, headers=_EDGAR_HEADERS, timeout=8)
            r.raise_for_status()
            _EDGAR_TICKERS_CACHE = r.json()
        for entry in _EDGAR_TICKERS_CACHE.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                return str(entry["cik_str"])
    except Exception as e:
        log.warning("EDGAR ticker lookup failed: %s", e)
    return None


def _fetch_edgar(company_name: str, ticker: str, days: int = 90) -> List[dict]:
    """
    Fetch recent material SEC filings via the EDGAR Submissions API.
    Returns list of {date, form, title, entity, url, items}.
    """
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    cik = _get_cik(ticker)
    if not cik:
        return []

    try:
        cik_padded = cik.zfill(10)
        url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
        r   = requests.get(url, headers=_EDGAR_HEADERS, timeout=8)
        r.raise_for_status()
        data     = r.json()
        entity   = data.get("name", company_name)
        recent   = data.get("filings", {}).get("recent", {})
        forms    = recent.get("form", [])
        dates    = recent.get("filingDate", [])
        accns    = recent.get("accessionNumber", [])
        items_l  = recent.get("items", [])
        descs    = recent.get("primaryDocument", [])

        results = []
        for form, date, accn, items_raw, doc in zip(forms, dates, accns, items_l, descs):
            if date < cutoff:
                continue
            if form not in _RELEVANT_FORMS:
                continue
            item_codes = [c.strip() for c in items_raw.split(",") if c.strip()] \
                         if items_raw else []
            item_labels = [_8K_ITEMS.get(c, c) for c in item_codes]
            title = " | ".join(item_labels) if item_labels else f"{form} filing"
            filing_url = (f"https://www.sec.gov/Archives/edgar/data/{cik}/"
                          f"{accn.replace('-','')}/{doc}")
            results.append({
                "date":   date,
                "form":   form,
                "title":  title,
                "entity": entity,
                "items":  item_codes,
                "url":    filing_url,
            })

        return results[:12]
    except Exception as e:
        log.warning("EDGAR submissions for %s: %s", ticker, e)
        return []


# ── yfinance structured data ──────────────────────────────────────────────────

def _fetch_yf_analyst(ticker: str) -> Dict[str, Any]:
    """
    Pull structured analyst data via yfinance:
    price targets, recent recommendations, upcoming earnings, earnings surprise.
    """
    result: Dict[str, Any] = {}
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)

        # Analyst price targets
        try:
            apt = t.analyst_price_targets
            if isinstance(apt, dict) and apt.get("mean"):
                result["analyst_targets"] = {
                    "mean":    round(float(apt["mean"]), 2),
                    "low":     round(float(apt.get("low", 0)), 2),
                    "high":    round(float(apt.get("high", 0)), 2),
                    "n":       int(apt.get("numberOfAnalysts", 0)),
                }
        except Exception:
            pass

        # Recent recommendations (last 8 rows)
        try:
            recs = t.recommendations
            if recs is not None and not recs.empty:
                recs = recs.tail(8).reset_index()
                result["recent_recommendations"] = [
                    {
                        "date":      str(row.get("Datetime", row.get("Date", "")))[:10],
                        "firm":      str(row.get("Firm", "")),
                        "action":    str(row.get("Action", "")),
                        "to_grade":  str(row.get("To Grade", "")),
                        "from_grade":str(row.get("From Grade", "")),
                    }
                    for _, row in recs.iterrows()
                ]
        except Exception:
            pass

        # Recommendations summary (buy/hold/sell counts)
        try:
            rs = t.recommendations_summary
            if rs is not None and not rs.empty:
                latest_row = rs.iloc[0]
                result["consensus"] = {
                    "strong_buy":  int(latest_row.get("strongBuy",  0)),
                    "buy":         int(latest_row.get("buy",        0)),
                    "hold":        int(latest_row.get("hold",       0)),
                    "sell":        int(latest_row.get("sell",       0)),
                    "strong_sell": int(latest_row.get("strongSell", 0)),
                }
        except Exception:
            pass

        # Upcoming earnings date
        try:
            cal = t.calendar
            if isinstance(cal, dict):
                earnings_date = cal.get("Earnings Date", [])
                if earnings_date:
                    ed = earnings_date[0] if isinstance(earnings_date, list) else earnings_date
                    result["next_earnings"] = str(ed)[:10]
                    result["earnings_rev_est"]  = cal.get("Revenue Estimate")
                    result["earnings_eps_est"]  = cal.get("EPS Estimate")
            elif hasattr(cal, "to_dict"):
                cal_d = cal.to_dict()
                for k, v in cal_d.items():
                    if "earnings" in k.lower() and v:
                        result["next_earnings"] = str(list(v.values())[0])[:10] if isinstance(v, dict) else str(v)[:10]
                        break
        except Exception:
            pass

        # Last earnings surprise
        try:
            ed = t.earnings_dates
            if ed is not None and not ed.empty:
                ed = ed.dropna(subset=["Surprise(%)"] if "Surprise(%)" in ed.columns else [])
                if not ed.empty:
                    last = ed.iloc[0]
                    result["last_earnings_surprise"] = {
                        "date":        str(last.name)[:10] if hasattr(last, "name") else "",
                        "surprise_pct":round(float(last.get("Surprise(%)", 0)), 2),
                        "eps_actual":  round(float(last.get("Reported EPS", 0) or 0), 3),
                        "eps_est":     round(float(last.get("EPS Estimate", 0) or 0), 3),
                    }
        except Exception:
            pass

    except Exception as e:
        log.warning("yfinance analyst data for %s: %s", ticker, e)

    return result


# ── Reranking ─────────────────────────────────────────────────────────────────

_STOP = set(
    "the a an of and or to for in on at by with from is are was were be been "
    "this that as it its their stock share shares price market inc corp ltd llc".split()
)


def _tokenize(text: str) -> List[str]:
    return [w for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", (text or "").lower())
            if w not in _STOP]


def _rerank(items: List[dict], query_terms: List[str], k: int = 20) -> List[dict]:
    if not items:
        return []
    today = datetime.now().date()
    qset  = set(query_terms)
    # Category priority boost
    cat_boost = {"earnings": 1.5, "analysts": 1.2, "catalysts": 1.2,
                 "bearish": 1.0, "macro": 0.8, "sector": 0.8,
                 "insider": 1.0, "news": 1.0}
    scored = []
    for a in items:
        text    = (a.get("title","") + " " + a.get("body","")).strip()
        terms   = set(_tokenize(text))
        overlap = len(terms & qset)
        raw_date = a.get("date","") or ""
        try:
            age = max((today - datetime.fromisoformat(raw_date[:10]).date()).days, 0) \
                  if raw_date and "ago" not in raw_date else 0
        except Exception:
            age = 14
        recency = max(0.0, 1.0 - age / 30.0)
        boost   = cat_boost.get(a.get("category",""), 1.0)
        scored.append((overlap * 1.5 * boost + recency * 2.0, a))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [a for _, a in scored[:k]]


# ── Prompt construction ───────────────────────────────────────────────────────

_SYSTEM_TURN1 = """You are a senior quantitative analyst and portfolio manager conducting an independent equity assessment.

You will receive:
  1. Raw market indicators and fundamentals for a stock (NO composite scores have been computed yet)
  2. Live web search results: recent news, earnings, analyst calls, macro context, sector dynamics
  3. SEC EDGAR recent filings
  4. Structured analyst consensus data

Your process:
  STEP 1 — Write exhaustive reasoning inside <thinking>...</thinking> tags covering:
    a) Technical picture: trend direction, momentum, mean-reversion signals, volatility regime
    b) Fundamental picture: valuation vs peers, growth quality, balance sheet, analyst consensus
    c) Macroeconomic context: rates, inflation, sector rotation, broad market regime
    d) Industry/sector dynamics: competitive landscape, tailwinds, headwinds, disruption risks
    e) News & catalysts: material events, earnings surprises, regulatory risk, management signals
    f) Sentiment picture: tone and volume of recent coverage, insider activity, short interest
    g) Risk assessment: downside scenarios, volatility, liquidity, event risk
    h) Dip assessment: whether current weakness is a buying opportunity or a genuine breakdown
    i) Time horizon alignment: match your assessment to the implied regime/holding period

  STEP 2 — Output ONLY valid JSON (no markdown fences) conforming to the schema below.
           Do NOT reference or infer scores — derive everything from the raw indicators and sources."""

_TURN1_SCHEMA = """{
  "trend_assessment": "STRONG_UPTREND|UPTREND|EARLY_UPTREND|SIDEWAYS|EARLY_DOWNTREND|DOWNTREND|STRONG_DOWNTREND|BOTTOMING|TOPPING",
  "regime_assessment": "TREND|MEAN_REVERSION|NEUTRAL",
  "regime_change": "ACCELERATING|STABLE|DECELERATING|REVERSING|null",
  "signal_direction": "STRONG_BUY|BUY|NEUTRAL|SELL|STRONG_SELL",
  "signal_score_estimate": <-1.0 to 1.0>,
  "technical_score_estimate": <-1.0 to 1.0>,
  "fundamental_score_estimate": <-1.0 to 1.0>,
  "sentiment_score_estimate": <-1.0 to 1.0>,
  "sentiment_assessment": "VERY_BULLISH|BULLISH|NEUTRAL|BEARISH|VERY_BEARISH",
  "risk_level": "VERY_LOW|LOW|MODERATE|HIGH|VERY_HIGH",
  "risk_score_estimate": <0.0 to 1.0>,
  "dip_opportunity": "STRONG|MODERATE|WEAK|NONE|TRAP",
  "dip_score_estimate": <-1.0 to 1.0>,
  "macro_context": "<2-3 sentences on macro backdrop and how it affects this stock>",
  "industry_context": "<2-3 sentences on sector/competitive dynamics>",
  "time_horizon_rationale": "<1-2 sentences on which time horizon this assessment applies to>",
  "bull_case": ["<point, cite [N]>"],
  "bear_case": ["<point, cite [N]>"],
  "key_catalysts": ["<upcoming or recent catalyst>"],
  "key_risks": ["<risk 1>", "<risk 2>"],
  "missed_by_quant_model": ["<information quant model likely did not capture>"],
  "supporting_sources": [<source numbers>],
  "contradicting_sources": [<source numbers>]
}"""

_SYSTEM_TURN2 = """You are the same senior quantitative analyst from Turn 1.

You have now been shown the actual scores computed by the quantitative model.
Your job is to compare them against your independent estimates, argue explicitly
where you agree and disagree, and issue a final verdict.

Be specific: if the model says risk_score = 0.72 but you estimated 0.40,
explain what the model is capturing that you may have missed, or vice versa.
Cite news sources by [N] when they support your argument.
Output ONLY valid JSON — no markdown, no prose outside the JSON."""

_TURN2_SCHEMA = """{
  "score_comparison": {
    "overall_signal":  {"llm": <float>, "model": <float>, "delta": <float>, "verdict": "ALIGNED|LLM_HIGHER|LLM_LOWER", "comment": "..."},
    "technical":       {"llm": <float>, "model": <float>, "delta": <float>, "verdict": "...", "comment": "..."},
    "fundamental":     {"llm": <float>, "model": <float>, "delta": <float>, "verdict": "...", "comment": "..."},
    "sentiment":       {"llm": <float>, "model": <float>, "delta": <float>, "verdict": "...", "comment": "..."},
    "risk":            {"llm": <float>, "model": <float>, "delta": <float>, "verdict": "...", "comment": "..."},
    "dip":             {"llm": <float>, "model": <float>, "delta": <float>, "verdict": "...", "comment": "..."}
  },
  "trend_comparison":   {"llm": "<label>", "model": "<label>", "verdict": "ALIGNED|DIVERGED", "comment": "..."},
  "regime_comparison":  {"llm": "<label>", "model": "<label>", "verdict": "ALIGNED|DIVERGED", "comment": "..."},
  "agreements": ["<specific point of agreement with citation>"],
  "disagreements": ["<specific point of disagreement with reasoning>"],
  "model_blind_spots": ["<what the model captured that I missed, or what I caught that the model missed>"],
  "final_verdict": "AGREE|DISAGREE|MIXED",
  "final_confidence": <0.0 to 1.0>,
  "final_recommendation": "HOLD_SIGNAL|UPGRADE|DOWNGRADE|REVIEW_MANUALLY",
  "final_summary": "<3-4 sentence synthesis of the debate and final recommendation>",
  "supporting_sources": [<source numbers that support the model's signal>],
  "contradicting_sources": [<source numbers that contradict the model's signal>]
}"""


def _horizon_context(indicators: Dict[str, Any]) -> str:
    regime    = indicators.get("mkt_regime") or indicators.get("regime", "")
    adx       = indicators.get("adx")
    atr_pct   = indicators.get("atr_pct")
    rs_1m     = indicators.get("rs_1m")

    if regime == "TREND":
        horizon = "medium-term trend-following signal (4–8 week typical holding period)"
        style   = "trend continuation, momentum, and breakout validity"
    elif regime == "MEAN_REVERSION":
        horizon = "short-term mean-reversion signal (3–10 day typical holding period)"
        style   = "overbought/oversold extremes, reversion speed, and support/resistance"
    else:
        horizon = "neutral/mixed regime — treat as 2–4 week tactical signal"
        style   = "both trend and reversion signals with equal weight"

    parts = [f"This is a {horizon}. Focus your assessment on {style}."]
    if adx is not None:
        parts.append(f"ADX={adx:.1f} ({'strong' if adx > 25 else 'weak'} trend strength).")
    if atr_pct is not None:
        parts.append(f"ATR%={atr_pct:.2f}% ({'high' if atr_pct > 3 else 'normal'} volatility).")
    if rs_1m is not None:
        parts.append(f"1-month relative strength vs SPY: {rs_1m:+.1f}% "
                     f"({'outperforming' if rs_1m > 0 else 'underperforming'}).")
    return " ".join(parts)


def _format_indicators(indicators: Dict[str, Any]) -> str:
    return json.dumps(
        {k: (round(v, 4) if isinstance(v, float) else v)
         for k, v in sorted(indicators.items())},
        indent=2
    )


def _format_news(items: List[dict]) -> str:
    if not items:
        return "(no recent web results available)"
    lines = []
    for i, a in enumerate(items):
        title  = (a.get("title","") or "").strip()
        body   = (a.get("body","") or "").strip()[:300]
        date   = (a.get("date","") or "")[:10] or "?"
        source = a.get("source","")
        cat    = a.get("category","")
        line   = f"[{i+1}] ({date}) [{cat.upper()}] {title}"
        if body:
            line += f"\n     {body}"
        if source:
            line += f"\n     — {source}"
        lines.append(line)
    return "\n\n".join(lines)


def _format_edgar(filings: List[dict]) -> str:
    if not filings:
        return "(no recent SEC filings found)"
    lines = []
    for f in filings:
        lines.append(f"• [{f.get('date','')}] {f.get('form','')} — {f.get('entity','')} "
                     f"| {f.get('title','')}")
    return "\n".join(lines)


def _format_analyst(analyst: Dict[str, Any]) -> str:
    if not analyst:
        return "(no structured analyst data available)"
    parts = []
    if t := analyst.get("analyst_targets"):
        parts.append(f"Price targets: mean=${t['mean']}, low=${t['low']}, "
                     f"high=${t['high']} ({t['n']} analysts)")
    if c := analyst.get("consensus"):
        total = sum(c.values())
        parts.append(f"Consensus ({total} analysts): "
                     f"Strong Buy={c['strong_buy']}, Buy={c['buy']}, "
                     f"Hold={c['hold']}, Sell={c['sell']}, Strong Sell={c['strong_sell']}")
    if e := analyst.get("next_earnings"):
        parts.append(f"Next earnings: {e}")
        if analyst.get("earnings_eps_est"):
            parts.append(f"  EPS estimate: {analyst['earnings_eps_est']}")
    if s := analyst.get("last_earnings_surprise"):
        parts.append(f"Last earnings surprise: {s.get('surprise_pct',0):+.1f}% "
                     f"(actual EPS {s.get('eps_actual','?')} vs est {s.get('eps_est','?')}) "
                     f"on {s.get('date','?')}")
    if recs := analyst.get("recent_recommendations"):
        parts.append(f"Recent rating changes ({len(recs)} actions):")
        for r in recs[-5:]:
            parts.append(f"  • {r['date']} {r['firm']}: {r['action']} "
                         f"{r['from_grade']} → {r['to_grade']}")
    return "\n".join(parts) if parts else "(no structured analyst data available)"


def _build_turn1_prompt(
    ticker: str, company_name: str,
    indicators: Dict, news: List[dict],
    edgar: List[dict], analyst: Dict,
) -> str:
    horizon = _horizon_context(indicators)
    return (
        f"# Independent equity assessment: {ticker} ({company_name})\n\n"
        f"## Time horizon & regime context\n{horizon}\n\n"
        f"## Raw market indicators & fundamentals\n"
        f"```json\n{_format_indicators(indicators)}\n```\n\n"
        f"## Structured analyst data\n{_format_analyst(analyst)}\n\n"
        f"## SEC EDGAR recent filings\n{_format_edgar(edgar)}\n\n"
        f"## Live web search results ({len(news)} sources)\n"
        f"{_format_news(news)}\n\n"
        f"---\n"
        f"Reason exhaustively in <thinking>...</thinking>, then output STRICT JSON "
        f"matching this schema:\n{_TURN1_SCHEMA}"
    )


def _build_turn2_user(scores: Dict, score_labels: Dict = _SCORE_LABELS) -> str:
    lines = ["Here are the quantitative model's actual computed scores and classifications:\n"]
    for k, v in scores.items():
        label_info = score_labels.get(k)
        if label_info:
            label, scale = label_info
            lines.append(f"  {label:35s}: {v}  ({scale})")
        else:
            lines.append(f"  {k:35s}: {v}")
    lines.append(
        "\nCompare these against your independent estimates from Turn 1. "
        "For each dimension, argue explicitly whether the model is right or wrong and why. "
        "Then issue your final verdict.\n"
        f"Output STRICT JSON matching this schema:\n{_TURN2_SCHEMA}"
    )
    return "\n".join(lines)


def _parse_turn1(raw: str) -> Tuple[str, Dict]:
    """Extract <thinking>...</thinking> block and JSON from Turn 1 response."""
    thinking = ""
    m = re.search(r"<thinking>(.*?)</thinking>", raw, re.DOTALL | re.IGNORECASE)
    if m:
        thinking = m.group(1).strip()
        raw_json = raw[m.end():]
    else:
        raw_json = raw

    m2 = re.search(r"\{.*\}", raw_json, re.DOTALL)
    if not m2:
        return thinking, {"verdict": "PARSE_ERROR", "raw": raw}
    try:
        return thinking, json.loads(m2.group())
    except json.JSONDecodeError:
        try:
            return thinking, json.loads(m2.group().rsplit("}", 1)[0] + "}")
        except Exception:
            return thinking, {"verdict": "PARSE_ERROR", "raw": raw}


def _parse_turn2(raw: str) -> Dict:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {"final_verdict": "PARSE_ERROR", "raw": raw}
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        try:
            return json.loads(m.group().rsplit("}", 1)[0] + "}")
        except Exception:
            return {"final_verdict": "PARSE_ERROR", "raw": raw}


# ── Public API ────────────────────────────────────────────────────────────────

def validate_analysis(
    ticker: str,
    analysis: Dict[str, Any],
    *,
    company_name: str = "",
    top_k: int = 20,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the two-turn RAG validation pipeline.

    Returns:
        ok              bool
        turn1           dict  — LLM independent assessment
        thinking        str   — LLM chain-of-thought
        turn2           dict  — debate & final verdict
        sources         list  — ranked web sources shown to LLM
        edgar_filings   list  — SEC filings shown to LLM
        analyst_data    dict  — structured analyst data shown to LLM
        n_sources       int
        model           str
    """
    if not COMPACTIFAI_API_KEY:
        return {"ok": False, "error": "API_KEY not configured in .env"}

    active_model = model if model in _ALLOWED_MODELS else COMPACTIFAI_MODEL

    # 1. Split indicators from scores
    indicators, scores = _split_analysis(analysis)
    sector       = analysis.get("sector", "")
    company_name = company_name or analysis.get("name", ticker)

    # 2. Parallel-ish retrieval (sequential due to DDG rate limits)
    log.info("LLM validator: fetching news, EDGAR, analyst data for %s", ticker)
    news_items  = (_multi_search(ticker, company_name, sector)
                   if _DDG_AVAILABLE
                   else _rss_fallback(ticker, company_name))
    edgar_items = _fetch_edgar(company_name, ticker)
    analyst     = _fetch_yf_analyst(ticker)

    # 3. Rerank news
    rationale   = " ".join(str(v) for v in indicators.values() if isinstance(v, str))
    query_terms = list(set(_tokenize(rationale) + [ticker.lower()] + _tokenize(company_name)))
    ranked      = _rerank(news_items, query_terms, k=top_k)

    # 4. Turn 1 — independent analysis
    log.info("LLM validator: Turn 1 for %s (%d sources, %d filings)",
             ticker, len(ranked), len(edgar_items))
    t1_prompt = _build_turn1_prompt(ticker, company_name, indicators, ranked, edgar_items, analyst)
    try:
        t1_messages = [
            {"role": "system",  "content": _SYSTEM_TURN1},
            {"role": "user",    "content": t1_prompt},
        ]
        t1_raw = _chat(t1_messages, temperature=0.35, model=active_model)
    except Exception as e:
        return {"ok": False, "error": f"Turn 1 LLM call failed: {e}"}

    thinking, turn1 = _parse_turn1(t1_raw)
    turn1_ok = turn1.get("verdict") != "PARSE_ERROR" and "signal_direction" in turn1

    # 5. Turn 2 — debate with model scores
    log.info("LLM validator: Turn 2 (debate) for %s", ticker)
    t2_user = _build_turn2_user(scores)
    try:
        t2_messages = [
            {"role": "system",    "content": _SYSTEM_TURN2},
            {"role": "user",      "content": t1_prompt},
            {"role": "assistant", "content": t1_raw},      # full Turn 1 in context
            {"role": "user",      "content": t2_user},
        ]
        t2_raw = _chat(t2_messages, temperature=0.25, model=active_model)
    except Exception as e:
        return {
            "ok": turn1_ok, "partial": True,
            "error": f"Turn 2 failed: {e}",
            "thinking": thinking, "turn1": turn1,
            "sources": _serialise_sources(ranked),
            "edgar_filings": edgar_items,
            "analyst_data": analyst,
            "n_sources": len(ranked), "model": active_model,
        }

    turn2 = _parse_turn2(t2_raw)

    return {
        "ok":            True,
        "thinking":      thinking,
        "turn1":         turn1,
        "turn2":         turn2,
        "sources":       _serialise_sources(ranked),
        "edgar_filings": edgar_items,
        "analyst_data":  analyst,
        "n_sources":     len(ranked),
        "model":         active_model,
    }


def _serialise_sources(items: List[dict]) -> List[dict]:
    return [
        {
            "n":       i + 1,
            "title":   a.get("title", ""),
            "url":     a.get("url", ""),
            "date":    (a.get("date","") or "")[:10],
            "source":  a.get("source",""),
            "category":a.get("category",""),
        }
        for i, a in enumerate(items)
    ]


# ════════════════════════════════════════════════════════════════════════════
#  AI QUANT TRADER — cross-sectional ranking
# ════════════════════════════════════════════════════════════════════════════

# Fields extracted per stock for cross-sectional ranking prompt
_RANK_FIELDS = {
    "ticker", "name", "sector", "price", "ret_1d", "ret_1w", "ret_1m", "ret_3m", "w52_pct",
    "rsi", "adx", "vol_ratio", "ma_cross", "macd_bull",
    "tech_score", "fund_score", "sent_score", "overall_score",
    "signal", "trend_stage", "regime", "mkt_regime", "regime_chg",
    "confidence", "chk_passed", "chk_total",
    "pe_trail", "pe_fwd", "peg", "pb", "roe", "rev_growth", "eps_growth",
    "gross_mgn", "op_mgn", "mkt_cap", "beta", "short_float",
    "target_px", "rec_mean", "n_analysts",
}


def _extract_json(text: str) -> Any:
    """Extract a JSON object from text that may contain markdown code fences."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError("No JSON found in LLM response")


def quant_rank(results: List[Dict[str, Any]], *, model: Optional[str] = None) -> Dict[str, Any]:
    """
    Cross-sectional ranking of analyzed stocks.

    Sends a curated subset of each stock's scores and indicators to the LLM
    and returns a ranked conviction table with rationale and portfolio notes.
    """
    if not COMPACTIFAI_API_KEY:
        return {"ok": False, "error": "LLM API key not configured — check .env"}

    slim = [
        {k: v for k, v in r.items() if k in _RANK_FIELDS and v not in (None, "", {}, [])}
        for r in results
    ]

    system_prompt = (
        "You are a quantitative equity analyst performing cross-sectional stock ranking. "
        "Be specific and data-driven. Reference actual scores and indicators. "
        "Scores range from -1 (worst) to +1 (best). "
        "Respond ONLY with a valid JSON object — no markdown, no prose outside the JSON."
    )

    user_msg = (
        f"Rank the following {len(slim)} stocks from strongest buy to strongest sell "
        f"based on the cross-sectional data below.\n\n"
        f"Stock data:\n{json.dumps(slim, indent=2)}\n\n"
        "Respond with this exact JSON structure:\n"
        "{\n"
        '  "market_context": "<2-3 sentence observation about this set of stocks>",\n'
        '  "rankings": [\n'
        '    {\n'
        '      "ticker": "<ticker>",\n'
        '      "rank": <1 = top buy>,\n'
        '      "conviction": "<STRONG BUY | BUY | HOLD | REDUCE | SELL | STRONG SELL>",\n'
        '      "rationale": "<1-2 sentence reasoning citing specific scores/indicators>",\n'
        '      "key_factors": ["<factor1>", "<factor2>"]\n'
        "    }\n"
        "  ],\n"
        '  "top_picks": ["<ticker>"],\n'
        '  "avoid": ["<ticker>"],\n'
        '  "portfolio_suggestion": "<1-2 sentence construction advice>"\n'
        "}"
    )

    active_model = model if model in _ALLOWED_MODELS else COMPACTIFAI_MODEL
    try:
        raw = _chat(
            [{"role": "system", "content": system_prompt},
             {"role": "user",   "content": user_msg}],
            temperature=0.2,
            timeout=180,
            model=active_model,
        )
        data = _extract_json(raw)
        return {"ok": True, "data": data}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ════════════════════════════════════════════════════════════════════════════
#  AI PORTFOLIO MANAGER — portfolio review
# ════════════════════════════════════════════════════════════════════════════

def portfolio_review(positions: List[Dict[str, Any]], *, model: Optional[str] = None) -> Dict[str, Any]:
    """
    AI portfolio manager review of current holdings.

    Receives enriched position data (returns, holding period, live signal)
    and produces actionable per-position recommendations plus portfolio-level advice.
    """
    if not COMPACTIFAI_API_KEY:
        return {"ok": False, "error": "LLM API key not configured — check .env"}

    # Slim positions to fields relevant for the review
    slim_fields = {
        "ticker", "name", "sector", "quantity", "buy_price", "buy_date",
        "days_held", "current_price", "market_value", "cost_basis",
        "unrealized_pnl", "unrealized_pct", "annualized_return_pct", "signal", "notes",
    }
    slim = [
        {k: v for k, v in p.items() if k in slim_fields and v not in (None, "", {}, [])}
        for p in positions
    ]

    system_prompt = (
        "You are an AI portfolio manager performing a review of client equity holdings. "
        "Be concise, specific, and data-driven — reference actual return figures, "
        "holding periods, and signal values. "
        "Respond ONLY with a valid JSON object — no markdown, no prose outside the JSON."
    )

    user_msg = (
        f"Review the following {len(slim)} portfolio positions and provide actionable advice.\n\n"
        f"Positions:\n{json.dumps(slim, indent=2)}\n\n"
        "Respond with this exact JSON structure:\n"
        "{\n"
        '  "overview": "<2-3 sentence portfolio health summary>",\n'
        '  "grade": "<A | B | C | D>",\n'
        '  "grade_rationale": "<1 sentence>",\n'
        '  "positions": [\n'
        '    {\n'
        '      "ticker": "<ticker>",\n'
        '      "action": "<HOLD | ADD | REDUCE | EXIT | TAKE_PROFIT | CUT_LOSS>",\n'
        '      "urgency": "<low | medium | high>",\n'
        '      "rationale": "<1-2 sentence reasoning>"\n'
        '    }\n'
        '  ],\n'
        '  "risk_alerts": ["<alert1>", "<alert2>"],\n'
        '  "rebalancing_suggestion": "<1-2 sentence advice>",\n'
        '  "key_opportunity": "<1 sentence best near-term action>"\n'
        "}"
    )

    active_model = model if model in _ALLOWED_MODELS else COMPACTIFAI_MODEL
    try:
        raw = _chat(
            [{"role": "system", "content": system_prompt},
             {"role": "user",   "content": user_msg}],
            temperature=0.2,
            timeout=120,
            model=active_model,
        )
        data = _extract_json(raw)
        return {"ok": True, "data": data}
    except Exception as e:
        return {"ok": False, "error": str(e)}
