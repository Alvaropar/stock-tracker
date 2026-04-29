"""
Unified price data provider for US and China markets.

Provides both full historical candle data and lightweight live price
fetches.  All network calls use tenacity retry with exponential backoff.

Usage::

    data = PriceProvider.get_price_data("US", "commodity", "gold")
    live = PriceProvider.get_live_price("US", "commodity", "gold")
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from ..config.assets import COMMODITIES
from ..utils import safe_col as _col

log = logging.getLogger("pipeline.prices")


def _change_stats(current: float, prev_close: float) -> Dict[str, float]:
    """Compute change and change_pct from current and previous close."""
    change = round(current - prev_close, 2)
    change_pct = round((change / prev_close) * 100, 2) if prev_close else 0.0
    return {"change": change, "change_pct": change_pct}


# ── Ticker resolution ────────────────────────────────────────────────────────

def _resolve_us_ticker(asset_type: str, asset_id: str) -> Optional[str]:
    """Map asset to a yfinance ticker string.  Returns None if not found."""
    if asset_type == "commodity":
        info = COMMODITIES.get(asset_id.lower())
        return info.yf_ticker if info and info.yf_ticker else None
    return asset_id.upper()


# ═══════════════════════════════════════════════════════════════════════════════
# PriceProvider
# ═══════════════════════════════════════════════════════════════════════════════

class PriceProvider:
    """Static facade for fetching price data across markets."""

    # ── Public API ────────────────────────────────────────────────────────

    @staticmethod
    def get_price_data(market: str, asset_type: str, asset_id: str) -> Dict:
        """Fetch current price + 6-month OHLCV candles.

        Returns ``{current_price, prev_close, change, change_pct,
        currency, candles: [{time, open, high, low, close}]}``.
        """
        try:
            if market.upper() == "US":
                return PriceProvider._price_us(asset_type, asset_id)
            else:
                return PriceProvider._price_china(asset_type, asset_id)
        except Exception as e:
            log.warning("Price data fetch failed: %s", e)
            return {"error": str(e)}

    @staticmethod
    def get_live_price(market: str, asset_type: str, asset_id: str) -> Dict:
        """Fetch only the current price (lightweight, no history).

        Returns ``{current_price, prev_close, change, change_pct, currency}``.
        """
        try:
            if market.upper() == "US":
                return PriceProvider._live_price_us(asset_type, asset_id)
            else:
                return PriceProvider._live_price_china(asset_type, asset_id)
        except Exception as e:
            log.warning("Live price fetch failed: %s", e)
            return {"error": str(e)}

    # ── US: live price ────────────────────────────────────────────────────

    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _live_price_us(asset_type: str, asset_id: str) -> Dict:
        import yfinance as yf

        ticker_str = _resolve_us_ticker(asset_type, asset_id)
        if not ticker_str:
            return {"error": f"No yfinance ticker for commodity '{asset_id}'"}

        ticker = yf.Ticker(ticker_str)
        try:
            fi = ticker.fast_info
            current = round(float(fi.last_price), 2)
            prev_close = round(float(fi.previous_close), 2)
        except Exception:
            hist = ticker.history(period="2d")
            if hist.empty:
                return {"error": f"No price data for {ticker_str}"}
            current = round(float(hist.iloc[-1]["Close"]), 2)
            prev_close = round(float(hist.iloc[-2]["Close"]), 2) if len(hist) > 1 else current

        return {
            "current_price": current,
            "prev_close": prev_close,
            **_change_stats(current, prev_close),
            "currency": "USD",
        }

    # ── US: full candle history ───────────────────────────────────────────

    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _price_us(asset_type: str, asset_id: str) -> Dict:
        import yfinance as yf

        ticker_str = _resolve_us_ticker(asset_type, asset_id)
        if not ticker_str:
            return {"error": f"No yfinance ticker for commodity '{asset_id}'"}

        ticker = yf.Ticker(ticker_str)
        hist = ticker.history(period="6mo")

        if hist.empty:
            return {"error": f"No price data for {ticker_str}"}

        candles: List[Dict[str, Any]] = []
        for dt, row in hist.iterrows():
            candles.append({
                "time": dt.strftime("%Y-%m-%d"),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
            })

        # Live price for display (not just last candle close)
        try:
            fi = ticker.fast_info
            current = round(float(fi.last_price), 2)
            prev_close = round(float(fi.previous_close), 2)
        except Exception:
            last = hist.iloc[-1]
            prev = hist.iloc[-2] if len(hist) > 1 else hist.iloc[-1]
            current = round(float(last["Close"]), 2)
            prev_close = round(float(prev["Close"]), 2)

        return {
            "current_price": current,
            "prev_close": prev_close,
            **_change_stats(current, prev_close),
            "currency": "USD",
            "candles": candles,
        }

    # ── China: live price ─────────────────────────────────────────────────

    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _live_price_china(asset_type: str, asset_id: str) -> Dict:
        import akshare as ak

        try:
            if asset_type == "commodity":
                df = ak.futures_main_sina(symbol=asset_id)
                if df is None or df.empty:
                    return {"error": f"No data for '{asset_id}'"}
                last_row = df.iloc[-1]
                last = round(float(_col(last_row, "close", "收盘价")), 2)
                prev_row = df.iloc[-2] if len(df) > 1 else last_row
                prev = round(float(_col(prev_row, "close", "收盘价")), 2)
            else:
                df = ak.stock_zh_a_spot_em()
                row = df[df["代码"] == asset_id]
                if row.empty:
                    return {"error": f"No spot data for stock {asset_id}"}
                last = round(float(row.iloc[0]["最新价"]), 2)
                prev = round(float(row.iloc[0]["昨收"]), 2)
        except Exception as e:
            return {"error": str(e)}

        return {
            "current_price": last,
            "prev_close": prev,
            **_change_stats(last, prev),
            "currency": "CNY",
        }

    # ── China: full candle history ────────────────────────────────────────

    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _price_china(asset_type: str, asset_id: str) -> Dict:
        import akshare as ak

        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=180)).strftime("%Y%m%d")

        if asset_type == "commodity":
            info = COMMODITIES.get(asset_id.lower())
            try:
                df = ak.futures_main_sina(symbol=asset_id)
                if df is None or df.empty:
                    return {"error": f"No China futures data for '{asset_id}'"}
            except Exception:
                return {"error": f"Could not fetch China commodity data for '{asset_id}'"}

            candles: List[Dict[str, Any]] = []
            for _, row in df.iterrows():
                candles.append({
                    "time": str(_col(row, "date", "日期", default=""))[:10],
                    "open": round(float(_col(row, "open", "开盘价")), 2),
                    "high": round(float(_col(row, "high", "最高价")), 2),
                    "low": round(float(_col(row, "low", "最低价")), 2),
                    "close": round(float(_col(row, "close", "收盘价")), 2),
                })
        else:
            # A-share stock
            try:
                df = ak.stock_zh_a_hist(
                    symbol=asset_id, period="daily",
                    start_date=start_date, end_date=end_date, adjust="qfq",
                )
            except Exception as e:
                return {"error": f"Could not fetch stock data: {e}"}

            if df is None or df.empty:
                return {"error": f"No data for stock {asset_id}"}

            candles = []
            for _, row in df.iterrows():
                candles.append({
                    "time": str(_col(row, "日期", default=""))[:10],
                    "open": round(float(_col(row, "开盘", default=0)), 2),
                    "high": round(float(_col(row, "最高", default=0)), 2),
                    "low": round(float(_col(row, "最低", default=0)), 2),
                    "close": round(float(_col(row, "收盘", default=0)), 2),
                })

        if not candles:
            return {"error": "No candle data"}

        last_close = candles[-1]["close"]
        prev_close = candles[-2]["close"] if len(candles) > 1 else last_close

        return {
            "current_price": last_close,
            "prev_close": prev_close,
            **_change_stats(last_close, prev_close),
            "currency": "CNY",
            "candles": candles,
        }
