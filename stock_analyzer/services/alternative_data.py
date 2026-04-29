"""
Alternative data service.

Provides four independent data sources:

  Social Sentiment     StockTwits (free public stream) with Reddit WSB/stocks
                       fallback when StockTwits is blocked (Cloudflare).

  EDGAR Form 4         Insider buy/sell transactions parsed from SEC filings.
                       Uses the submissions API for filing discovery, then
                       fetches the raw XML (stripping the XSL-rendered path).

  FINRA Daily Short    Daily consolidated short-volume data from FINRA CDN
                       (https://cdn.finra.org/equity/regsho/daily/).
                       Returns short % of volume for the last 2 trading days.

  Options Data         yfinance option chain for nearest expiries.
                       Computes ATM IV, IV rank, put/call ratio, skew, implied
                       move, and flags upcoming earnings.
"""
from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests

log = logging.getLogger("app.alt_data")

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; StockAnalyzer/1.0; research@example.com)",
    "Accept": "application/json, text/plain, */*",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Social Sentiment  (StockTwits → Reddit fallback)
# ─────────────────────────────────────────────────────────────────────────────

_ST_URL = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"

_BULLISH_WORDS = {
    "bull", "bullish", "buy", "long", "calls", "moon", "rally", "breakout",
    "upside", "strong", "growth", "boom", "surge", "gain", "profit",
}
_BEARISH_WORDS = {
    "bear", "bearish", "sell", "short", "puts", "crash", "dump", "downside",
    "weak", "overvalued", "bubble", "drop", "fall", "risk", "loss",
}


def fetch_stocktwits(ticker: str, limit: int = 30) -> Dict[str, Any]:
    """
    Fetch social sentiment for *ticker*.

    Tries StockTwits first.  If blocked (403/5xx), falls back to Reddit
    (r/wallstreetbets + r/stocks search).

    Returns a unified dict regardless of source:
        {ok, source, ticker, n_messages, n_bullish, n_bearish, n_untagged,
         bull_pct, bear_pct, bull_bear_ratio, messages, error}
    """
    limit = max(1, min(limit, 50))
    st_result = _try_stocktwits(ticker, limit)
    if st_result["ok"]:
        return st_result
    # Fall back to Reddit
    reddit = _fetch_reddit_sentiment(ticker, limit)
    if not reddit["ok"]:
        reddit["error"] = f"StockTwits: {st_result['error']}; Reddit: {reddit['error']}"
    return reddit


def _try_stocktwits(ticker: str, limit: int) -> Dict[str, Any]:
    url = _ST_URL.format(ticker=ticker.upper())
    try:
        r = requests.get(url, params={"limit": limit}, timeout=8, headers=_HEADERS)
        if r.status_code in (403, 429, 503):
            return _st_error(ticker, f"StockTwits {r.status_code} (blocked)", "stocktwits")
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return _st_error(ticker, str(e), "stocktwits")

    msgs_raw = data.get("messages", [])
    messages, n_bull, n_bear, n_untag = [], 0, 0, 0
    for m in msgs_raw:
        entities = m.get("entities") or {}
        sent_obj = entities.get("sentiment")  # None or {"basic": "Bullish"}
        sent = (sent_obj or {}).get("basic") if sent_obj else None
        if sent == "Bullish":
            n_bull += 1
        elif sent == "Bearish":
            n_bear += 1
        else:
            n_untag += 1
        messages.append({
            "created_at": m.get("created_at", ""),
            "body":       (m.get("body") or "")[:280],
            "sentiment":  sent,
            "username":   (m.get("user") or {}).get("username", ""),
            "source":     "stocktwits",
        })

    tagged   = n_bull + n_bear
    bull_pct = n_bull / tagged if tagged else 0.0
    bear_pct = n_bear / tagged if tagged else 0.0
    bbr      = n_bull / tagged if tagged else None

    return {
        "ok":             True,
        "source":         "stocktwits",
        "ticker":         ticker.upper(),
        "n_messages":     len(msgs_raw),
        "n_total":        len(msgs_raw),   # alias kept for compatibility
        "n_bullish":      n_bull,
        "n_bearish":      n_bear,
        "n_untagged":     n_untag,
        "bull_pct":       round(bull_pct, 4),
        "bear_pct":       round(bear_pct, 4),
        "bull_bear_ratio":round(bbr, 4) if bbr is not None else None,
        "messages":       messages,
        "error":          "",
    }


def _fetch_reddit_sentiment(ticker: str, limit: int = 25) -> Dict[str, Any]:
    """
    Search Reddit for recent posts mentioning the ticker and infer sentiment
    from title keywords + upvote ratio.
    """
    url = "https://www.reddit.com/search.json"
    params = {
        "q":      f"{ticker} stock",
        "sort":   "new",
        "limit":  min(limit, 25),
        "t":      "week",
        "type":   "link",
    }
    hdr = {
        "User-Agent": "StockAnalyzer/1.0 (research bot)",
        "Accept": "application/json",
    }
    try:
        r = requests.get(url, params=params, headers=hdr, timeout=10)
        if r.status_code != 200:
            return _st_error(ticker, f"Reddit {r.status_code}", "reddit")
        posts = r.json().get("data", {}).get("children", [])
    except Exception as e:
        return _st_error(ticker, str(e), "reddit")

    messages, n_bull, n_bear, n_untag = [], 0, 0, 0
    ticker_u = ticker.upper()
    ticker_l = ticker.lower()

    for post in posts:
        d = post.get("data", {})
        title = d.get("title", "")
        title_l = title.lower()

        # Only include posts that actually mention the ticker
        if ticker_u not in title and f"${ticker_l}" not in title_l and ticker_l not in title_l:
            continue

        words    = set(re.split(r"\W+", title_l))
        bull_hit = bool(words & _BULLISH_WORDS)
        bear_hit = bool(words & _BEARISH_WORDS)
        upvr     = float(d.get("upvote_ratio") or 0.5)

        if bull_hit and not bear_hit:
            sent = "Bullish"; n_bull += 1
        elif bear_hit and not bull_hit:
            sent = "Bearish"; n_bear += 1
        elif upvr >= 0.72:
            sent = "Bullish"; n_bull += 1
        elif upvr <= 0.38:
            sent = "Bearish"; n_bear += 1
        else:
            sent = None; n_untag += 1

        messages.append({
            "created_at": str(d.get("created_utc", "")),
            "body":       title[:280],
            "sentiment":  sent,
            "username":   d.get("author", ""),
            "subreddit":  d.get("subreddit", ""),
            "score":      int(d.get("score") or 0),
            "source":     "reddit",
        })

    tagged   = n_bull + n_bear
    bull_pct = n_bull / tagged if tagged else 0.0
    bear_pct = n_bear / tagged if tagged else 0.0

    return {
        "ok":             True,
        "source":         "reddit",
        "ticker":         ticker.upper(),
        "n_messages":     len(messages),
        "n_total":        len(messages),
        "n_bullish":      n_bull,
        "n_bearish":      n_bear,
        "n_untagged":     n_untag,
        "bull_pct":       round(bull_pct, 4),
        "bear_pct":       round(bear_pct, 4),
        "bull_bear_ratio":round(n_bull / tagged, 4) if tagged else None,
        "messages":       messages,
        "error":          "",
    }


def _st_error(ticker: str, msg: str, source: str = "stocktwits") -> Dict[str, Any]:
    return {
        "ok": False, "source": source, "ticker": ticker,
        "n_messages": 0, "n_total": 0,
        "n_bullish": 0, "n_bearish": 0, "n_untagged": 0,
        "bull_pct": 0.0, "bear_pct": 0.0, "bull_bear_ratio": None,
        "messages": [], "error": msg,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2.  EDGAR Form 4 — Insider Transactions
# ─────────────────────────────────────────────────────────────────────────────

_BUY_CODES   = {"P"}              # open-market purchase
_SELL_CODES  = {"S"}              # open-market sale
_AWARD_CODES = {"A", "M", "G", "D", "F"}  # awards / option exercises — ignore

_EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
_EDGAR_ARCHIVE     = "https://www.sec.gov/Archives/edgar/data/{cik}/{accn}/"

_CIK_CACHE: Dict[str, str] = {}


def _resolve_cik(ticker: str) -> Optional[str]:
    t = ticker.upper()
    if t in _CIK_CACHE:
        return _CIK_CACHE[t]
    try:
        r = requests.get("https://www.sec.gov/files/company_tickers.json",
                         headers=_HEADERS, timeout=8)
        r.raise_for_status()
        for entry in r.json().values():
            if entry.get("ticker", "").upper() == t:
                cik = str(entry["cik_str"])
                _CIK_CACHE[t] = cik
                return cik
    except Exception as e:
        log.warning("CIK lookup %s: %s", ticker, e)
    return None


def _find_form4_xml_url(cik: str, accn_clean: str, primary_doc: str) -> Optional[str]:
    """
    Resolve the raw Form 4 XML URL.

    EDGAR's primaryDocument for Form 4 is often stored as
    'xslF345X06/wk-form4_XXXXXX.xml' — the XSL-rendered HTML version.
    The actual XML lives at the same filename without the subdirectory prefix.
    If the primary doc is already a plain XML name, use it directly.
    """
    base = _EDGAR_ARCHIVE.format(cik=cik, accn=accn_clean)

    # Strip any XSL subdirectory prefix (e.g. "xslF345X06/")
    doc_file = primary_doc.split("/")[-1]

    if doc_file.endswith(".xml"):
        return base + doc_file

    # Fallback: fetch the filing index HTML and find the first non-schema XML
    try:
        idx_url = base.rstrip("/") + f"/{accn_clean}-index.htm"
        r = requests.get(idx_url, headers=_HEADERS, timeout=6)
        if r.status_code == 200:
            xml_links = re.findall(
                r'/Archives/edgar/data/\d+/\d+/([^"\'<>\s]+\.xml)', r.text
            )
            for name in xml_links:
                if not name.endswith(".xsd") and "xslF345" not in name:
                    return base + name
            # Accept any XML if nothing else found
            for name in xml_links:
                if not name.endswith(".xsd"):
                    return base + name
    except Exception:
        pass

    return None


def _parse_form4_xml(xml_text: str) -> List[Dict[str, Any]]:
    """Parse SEC Form 4 XML, return list of non-derivative transactions."""
    txns = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return txns

    for txn in root.iter("nonDerivativeTransaction"):
        code_el   = txn.find("transactionCoding/transactionCode")
        shares_el = txn.find("transactionAmounts/transactionShares/value")
        price_el  = txn.find("transactionAmounts/transactionPricePerShare/value")
        date_el   = txn.find("transactionDate/value")
        title_el  = txn.find("securityTitle/value")

        code = (code_el.text or "").strip().upper() if code_el is not None else ""
        if not code or code in _AWARD_CODES:
            continue

        try:
            shares = float(shares_el.text or 0) if shares_el is not None else 0.0
            price  = float(price_el.text  or 0) if price_el  is not None else 0.0
        except ValueError:
            continue

        txns.append({
            "code":    code,
            "type":    "buy" if code in _BUY_CODES else "sell",
            "shares":  shares,
            "price":   price,
            "value":   round(shares * price, 2),
            "date":    (date_el.text  or "").strip() if date_el  is not None else "",
            "title":   (title_el.text or "").strip() if title_el is not None else "",
            "is_buy":  code in _BUY_CODES,
            "is_sell": code in _SELL_CODES,
        })
    return txns


def fetch_insider_transactions(ticker: str, days: int = 90) -> Dict[str, Any]:
    """Fetch and parse recent insider Form 4 transactions for *ticker*."""
    cik = _resolve_cik(ticker)
    if not cik:
        return _ins_error(ticker, f"CIK not found for {ticker}")

    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        url  = _EDGAR_SUBMISSIONS.format(cik=cik.zfill(10))
        r    = requests.get(url, headers=_HEADERS, timeout=10)
        r.raise_for_status()
        data   = r.json()
        recent = data.get("filings", {}).get("recent", {})
        forms  = recent.get("form", [])
        dates  = recent.get("filingDate", [])
        accns  = recent.get("accessionNumber", [])
        docs   = recent.get("primaryDocument", [])
    except Exception as e:
        return _ins_error(ticker, f"EDGAR submissions: {e}")

    # Form 4 / 4A within cutoff, capped at 20 to limit HTTP calls
    form4_filings = [
        (date, accn.replace("-", ""), doc)
        for form, date, accn, doc in zip(forms, dates, accns, docs)
        if form in ("4", "4/A") and date >= cutoff
    ][:20]

    all_txns: List[Dict] = []
    for date, accn_clean, primary_doc in form4_filings:
        xml_url = _find_form4_xml_url(cik, accn_clean, primary_doc)
        if not xml_url:
            log.debug("Could not resolve XML for %s/%s", cik, accn_clean)
            continue
        try:
            xr = requests.get(xml_url, headers=_HEADERS, timeout=8)
            if xr.status_code != 200:
                continue
            txns = _parse_form4_xml(xr.text)
            for t in txns:
                t["filing_date"] = date
            all_txns.extend(txns)
        except Exception as e:
            log.debug("Form 4 parse %s: %s", accn_clean, e)
        time.sleep(0.12)   # polite rate limit

    buys      = [t for t in all_txns if t["is_buy"]]
    sells     = [t for t in all_txns if t["is_sell"]]
    buy_val   = sum(t["value"] for t in buys)
    sell_val  = sum(t["value"] for t in sells)
    total_dir = len(buys) + len(sells)
    bsr       = len(buys) / total_dir if total_dir else None

    return {
        "ok":               True,
        "ticker":           ticker.upper(),
        "n_filings":        len(form4_filings),
        "n_buys":           len(buys),
        "n_sells":          len(sells),
        "total_buy_value":  round(buy_val,  2),
        "total_sell_value": round(sell_val, 2),
        "buy_sell_ratio":   round(bsr, 4) if bsr is not None else None,
        "transactions":     sorted(all_txns, key=lambda x: x.get("date", ""), reverse=True)[:30],
        "error":            "",
    }


def _ins_error(ticker: str, msg: str) -> Dict[str, Any]:
    return {
        "ok": False, "ticker": ticker, "n_filings": 0,
        "n_buys": 0, "n_sells": 0, "total_buy_value": 0.0,
        "total_sell_value": 0.0, "buy_sell_ratio": None,
        "transactions": [], "error": msg,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FINRA Daily Short Volume
# ─────────────────────────────────────────────────────────────────────────────
#
# FINRA publishes consolidated NMS short-sale data every trading day at:
#   https://cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt
#
# File format (pipe-delimited):
#   Date | Symbol | ShortVolume | ShortExemptVolume | TotalVolume | Market
#
# We fetch the last two available trading-day files and compare.

_FINRA_DAILY = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt"
_finra_cache: Dict[str, Dict] = {}


def _recent_trading_dates(n: int = 10) -> List[str]:
    """Return the last *n* weekday dates as YYYYMMDD strings."""
    today = datetime.now().date()
    dates: List[str] = []
    d = today
    while len(dates) < n:
        if d.weekday() < 5:   # Mon–Fri
            dates.append(d.strftime("%Y%m%d"))
        d -= timedelta(days=1)
    return dates


def _fetch_finra_day(date_str: str) -> Optional[Dict[str, Any]]:
    """Fetch and parse one FINRA daily short-volume file. Returns {symbol: data}."""
    if date_str in _finra_cache:
        return _finra_cache[date_str]
    url = _FINRA_DAILY.format(date=date_str)
    try:
        r = requests.get(url, timeout=15, headers=_HEADERS)
        if r.status_code != 200:
            return None
        lines   = r.text.strip().splitlines()
        parsed: Dict[str, Dict] = {}
        for line in lines[1:]:   # skip header row
            parts = line.strip().split("|")
            if len(parts) < 5:
                continue
            # Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market
            symbol = parts[1].upper()
            try:
                short_vol = int(parts[2])
                total_vol = int(parts[4])
                parsed[symbol] = {
                    "short_volume": short_vol,
                    "total_volume": total_vol,
                    "short_pct":   round(short_vol / total_vol, 4) if total_vol else 0.0,
                    "date":        date_str,
                }
            except (ValueError, IndexError):
                pass
        _finra_cache[date_str] = parsed
        return parsed
    except Exception as e:
        log.debug("FINRA daily %s: %s", date_str, e)
        return None


def fetch_finra_short_interest(ticker: str) -> Dict[str, Any]:
    """
    Return short-volume data from the last two available FINRA daily files.

    Tries up to 10 recent trading days to find two files that contain the
    given ticker (not all tickers appear every day).
    """
    symbol   = ticker.upper()
    readings: List[Tuple[str, Dict]] = []

    for date_str in _recent_trading_dates(10):
        parsed = _fetch_finra_day(date_str)
        if parsed and symbol in parsed:
            readings.append((date_str, parsed[symbol]))
        if len(readings) >= 2:
            break

    if not readings:
        return _finra_error(ticker, "No FINRA daily data found for this ticker")

    latest_date, latest = readings[0]
    prev = readings[1][1] if len(readings) >= 2 else None

    return {
        "ok":             True,
        "ticker":         symbol,
        "latest_date":    latest_date,
        "short_pct":      latest["short_pct"],
        "short_pct_prev": prev["short_pct"] if prev else None,
        "short_pct_chg":  round(latest["short_pct"] - prev["short_pct"], 4) if prev else None,
        "short_volume":   latest["short_volume"],
        "total_volume":   latest["total_volume"],
        "error":          "",
    }


def _finra_error(ticker: str, msg: str) -> Dict[str, Any]:
    return {
        "ok": False, "ticker": ticker, "latest_date": "",
        "short_pct": None, "short_pct_prev": None, "short_pct_chg": None,
        "short_volume": None, "total_volume": None, "error": msg,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Options Data + Earnings Proximity
# ─────────────────────────────────────────────────────────────────────────────

def fetch_options_data(ticker: str) -> Dict[str, Any]:
    """
    Fetch options market data for *ticker* using yfinance.

    Returns ATM IV, IV rank (vs 30d realized vol), put/call ratio, 25-delta
    skew proxy, implied move, and earnings proximity flag.
    """
    try:
        import yfinance as yf
        import numpy as np

        tk = yf.Ticker(ticker)

        # ── Earnings proximity ────────────────────────────────────────────
        next_earnings      = None
        earnings_proximity = None
        earnings_risk_flag = False
        try:
            cal = tk.calendar
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date", [])
                ne = ed[0] if isinstance(ed, list) and ed else (ed if ed else None)
                if ne:
                    ne_date = (ne.date() if hasattr(ne, "date")
                               else datetime.strptime(str(ne)[:10], "%Y-%m-%d").date())
                    next_earnings      = str(ne_date)
                    earnings_proximity = (ne_date - datetime.now().date()).days
                    earnings_risk_flag = 0 <= earnings_proximity <= 7
        except Exception:
            pass

        # ── Option chain ──────────────────────────────────────────────────
        exps = tk.options
        if not exps:
            return _opt_base(ticker, next_earnings, earnings_proximity,
                             earnings_risk_flag, "No option expirations")

        today      = datetime.now().date()
        near_exps  = []
        for e in exps:
            dte = (datetime.strptime(e, "%Y-%m-%d").date() - today).days
            if dte >= 5:
                near_exps.append((dte, e))
            if len(near_exps) >= 2:
                break

        if not near_exps:
            return _opt_base(ticker, next_earnings, earnings_proximity,
                             earnings_risk_flag, "No near-term expirations")

        dte, exp  = near_exps[0]
        chain     = tk.option_chain(exp)
        calls     = chain.calls
        puts      = chain.puts

        info = tk.fast_info
        spot = (getattr(info, "last_price", None)
                or getattr(info, "regularMarketPrice", None))
        if not spot and not calls.empty:
            spot = float(calls["strike"].median())
        if not spot:
            return _opt_base(ticker, next_earnings, earnings_proximity,
                             earnings_risk_flag, "Cannot determine spot price")

        def _atm_iv(df):
            if df.empty:
                return None
            df2 = df[df["impliedVolatility"] > 0.001].copy()
            if df2.empty:
                return None
            df2["dist"] = (df2["strike"] - spot).abs()
            return float(df2.nsmallest(2, "dist")["impliedVolatility"].mean())

        call_iv = _atm_iv(calls)
        put_iv  = _atm_iv(puts)
        atm_iv  = None
        if call_iv and put_iv:
            atm_iv = round((call_iv + put_iv) / 2, 4)
        elif call_iv:
            atm_iv = round(call_iv, 4)
        elif put_iv:
            atm_iv = round(put_iv, 4)

        implied_move_pct = (
            round(atm_iv * (dte / 252) ** 0.5 * 100, 2)
            if atm_iv and dte > 0 else None
        )

        put_vol  = float(puts["volume"].sum())  if not puts.empty  else 0.0
        call_vol = float(calls["volume"].sum()) if not calls.empty else 0.0
        pc_ratio = round(put_vol / call_vol, 3) if call_vol > 0 else None

        skew = None
        try:
            otm_p = puts[(puts["strike"]  < spot * 0.97) & (puts["impliedVolatility"]  > 0.001)]
            otm_c = calls[(calls["strike"] > spot * 1.03) & (calls["impliedVolatility"] > 0.001)]
            if not otm_p.empty and not otm_c.empty:
                p25 = float(otm_p.nlargest(3,  "strike")["impliedVolatility"].mean())
                c25 = float(otm_c.nsmallest(3, "strike")["impliedVolatility"].mean())
                skew = round(p25 - c25, 4)
        except Exception:
            pass

        iv_rank = None
        try:
            hist  = tk.history(period="1y", auto_adjust=True)
            if not hist.empty and len(hist) >= 30:
                ret  = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
                hv30 = float(ret.rolling(30).std().dropna().iloc[-1] * (252 ** 0.5))
                iv_rank = round(atm_iv / hv30, 3) if (atm_iv and hv30 > 0) else None
        except Exception:
            pass

        return {
            "ok":                 True,
            "ticker":             ticker.upper(),
            "expiry":             exp,
            "dte":                dte,
            "spot":               round(spot, 4),
            "atm_iv":             atm_iv,
            "atm_iv_pct":         round(atm_iv * 100, 2) if atm_iv else None,
            "iv_rank":            iv_rank,
            "put_call_ratio":     pc_ratio,
            "skew_25d":           skew,
            "implied_move_pct":   implied_move_pct,
            "call_volume":        int(call_vol),
            "put_volume":         int(put_vol),
            "next_earnings":      next_earnings,
            "earnings_proximity": earnings_proximity,
            "earnings_risk_flag": earnings_risk_flag,
            "error":              "",
        }

    except Exception as e:
        return _opt_base(ticker, None, None, False, str(e))


def _opt_base(ticker, next_earnings, prox, risk_flag, error=""):
    return {
        "ok": not bool(error), "ticker": ticker.upper(),
        "expiry": None, "dte": None, "spot": None,
        "atm_iv": None, "atm_iv_pct": None, "iv_rank": None,
        "put_call_ratio": None, "skew_25d": None, "implied_move_pct": None,
        "call_volume": None, "put_volume": None,
        "next_earnings": next_earnings, "earnings_proximity": prox,
        "earnings_risk_flag": risk_flag, "error": error,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5.  All sources combined
# ─────────────────────────────────────────────────────────────────────────────

def fetch_all_alt_data(ticker: str, stocktwits_limit: int = 30) -> Dict[str, Any]:
    """Fetch all alternative data sources for a single ticker."""
    return {
        "stocktwits":     fetch_stocktwits(ticker, limit=stocktwits_limit),
        "insiders":       fetch_insider_transactions(ticker),
        "short_interest": fetch_finra_short_interest(ticker),
        "options":        fetch_options_data(ticker),
    }
