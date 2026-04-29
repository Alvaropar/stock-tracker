"""
Asset registry and description templates.

Centralises the mapping between user-facing asset names and the
metadata needed by scrapers and filters (SHMET symbols, keywords,
stock codes, filter descriptions, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class AssetType(Enum):
    COMMODITY = "commodity"
    STOCK = "stock"


# ── Commodity metadata ───────────────────────────────────────────────────────

@dataclass
class CommodityInfo:
    """Everything the pipeline needs to know about one commodity."""
    name: str
    # yfinance futures ticker for price data (e.g. "GC=F" for gold)
    yf_ticker: str = ""
    # Chinese SHMET category symbol (e.g. "贵金属")
    shmet_symbol: str = ""
    # Chinese keywords for keyword-based filtering (gold only, etc.)
    cn_keywords: List[str] = field(default_factory=list)
    # Related stock codes for Eastmoney scraping (gold-mining stocks etc.)
    cn_stock_codes: List[str] = field(default_factory=list)
    # Human-readable description used in relevance-filter prompts
    filter_description: str = ""


COMMODITIES: Dict[str, CommodityInfo] = {
    "gold": CommodityInfo(
        name="gold",
        yf_ticker="GC=F",
        shmet_symbol="贵金属",
        cn_keywords=[
            "黄金", "金价", "金矿", "黄金期货", "黄金ETF",
            "金条", "金币", "贵金属", "避险", "金银",
            "沪金", "伦敦金", "纽约金", "COMEX黄金",
        ],
        cn_stock_codes=[
            "600547", "600489", "600988", "002155",
            "000975", "600311", "002737",
        ],
        filter_description="gold (price, production, mining, trade, reserves, demand, supply)",
    ),
    "silver": CommodityInfo(
        name="silver",
        yf_ticker="SI=F",
        shmet_symbol="贵金属",
        cn_keywords=["白银", "银价"],
        filter_description="silver (price, production, mining, trade, industrial use)",
    ),
    "copper": CommodityInfo(
        name="copper",
        yf_ticker="HG=F",
        shmet_symbol="铜",
        filter_description="copper (price, production, mining, trade, demand, supply)",
    ),
    "aluminum": CommodityInfo(
        name="aluminum",
        yf_ticker="ALI=F",
        shmet_symbol="铝",
        filter_description="aluminum/aluminium (price, production, trade, industry)",
    ),
    "oil": CommodityInfo(
        name="oil",
        yf_ticker="CL=F",
        shmet_symbol="",
        filter_description="crude oil / petroleum (price, production, OPEC, drilling, refining)",
    ),
    "platinum": CommodityInfo(
        name="platinum",
        yf_ticker="PL=F",
        shmet_symbol="贵金属",
        filter_description="platinum (price, production, mining, catalytic converters)",
    ),
    "lead": CommodityInfo(
        name="lead",
        yf_ticker="",
        shmet_symbol="铅",
        filter_description="lead metal (price, production, batteries, trade)",
    ),
    "zinc": CommodityInfo(
        name="zinc",
        yf_ticker="",
        shmet_symbol="锌",
        filter_description="zinc (price, production, galvanizing, trade)",
    ),
    "nickel": CommodityInfo(
        name="nickel",
        yf_ticker="",
        shmet_symbol="镍",
        filter_description="nickel (price, production, stainless steel, batteries)",
    ),
    "tin": CommodityInfo(
        name="tin",
        yf_ticker="",
        shmet_symbol="锡",
        filter_description="tin (price, production, soldering, trade)",
    ),
}

# Well-known US stock ticker → display name mapping (for convenience only)
US_STOCK_NAMES: Dict[str, str] = {
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google Alphabet",
    "AMZN": "Amazon", "NVDA": "Nvidia", "TSLA": "Tesla",
    "META": "Meta Facebook", "JPM": "JPMorgan Chase", "V": "Visa",
    "JNJ": "Johnson & Johnson", "WMT": "Walmart", "PG": "Procter & Gamble",
    "MA": "Mastercard", "UNH": "UnitedHealth", "HD": "Home Depot",
    "DIS": "Disney", "BAC": "Bank of America", "XOM": "Exxon Mobil",
    "PFE": "Pfizer", "KO": "Coca-Cola", "NKE": "Nike",
    "INTC": "Intel", "AMD": "AMD", "NFLX": "Netflix",
    "CRM": "Salesforce", "ORCL": "Oracle", "CSCO": "Cisco",
    "IBM": "IBM", "GS": "Goldman Sachs", "BA": "Boeing",
}


class AssetRegistry:
    """
    Resolve user input into structured asset metadata.

    Designed as a class so it can be extended (e.g. dynamic stock-name
    lookups via yfinance or akshare) without touching the rest of the
    pipeline.
    """

    @staticmethod
    def get_commodity(name: str) -> CommodityInfo:
        key = name.lower()
        if key in COMMODITIES:
            return COMMODITIES[key]
        # Graceful fallback for unknown commodities
        return CommodityInfo(
            name=key,
            filter_description=f"{key} (price, production, trade, market news)",
        )

    @staticmethod
    def get_filter_description(asset_type: AssetType, asset_id: str,
                               display_name: Optional[str] = None) -> str:
        """Build the one-line description fed into the relevance filter prompt."""
        if asset_type is AssetType.COMMODITY:
            return AssetRegistry.get_commodity(asset_id).filter_description

        # Stock
        label = display_name or asset_id
        return f"{label} ({asset_id}) - company news, stock price, earnings, business operations"

    @staticmethod
    def get_us_stock_display_name(ticker: str) -> Optional[str]:
        """Best-effort lookup; returns None if unknown."""
        upper = ticker.upper()
        if upper in US_STOCK_NAMES:
            return US_STOCK_NAMES[upper]
        try:
            import yfinance as yf
            info = yf.Ticker(upper).info
            return info.get("shortName") or info.get("longName")
        except Exception:
            return None

    @staticmethod
    def get_china_stock_name(stock_code: str) -> Optional[str]:
        """Lookup Chinese A-share stock name via akshare."""
        try:
            import akshare as ak
            df = ak.stock_info_a_code_name()
            match = df[df["code"] == stock_code]
            if not match.empty:
                return match.iloc[0]["name"]
        except Exception:
            pass
        return None
