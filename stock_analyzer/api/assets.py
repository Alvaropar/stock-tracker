"""
Asset search and commodity list endpoints.
"""
from flask import Blueprint, jsonify, request

bp = Blueprint("assets", __name__, url_prefix="/api/assets")

# Predefined commodities (from sentiment-wrapper pipeline)
COMMODITIES = [
    {"ticker": "GC=F",  "name": "Gold",     "sector": "Commodity", "type": "commodity", "currency": "USD"},
    {"ticker": "SI=F",  "name": "Silver",   "sector": "Commodity", "type": "commodity", "currency": "USD"},
    {"ticker": "HG=F",  "name": "Copper",   "sector": "Commodity", "type": "commodity", "currency": "USD"},
    {"ticker": "CL=F",  "name": "Crude Oil","sector": "Commodity", "type": "commodity", "currency": "USD"},
    {"ticker": "NG=F",  "name": "Natural Gas","sector":"Commodity", "type": "commodity", "currency": "USD"},
    {"ticker": "PL=F",  "name": "Platinum", "sector": "Commodity", "type": "commodity", "currency": "USD"},
    {"ticker": "ALI=F", "name": "Aluminum", "sector": "Commodity", "type": "commodity", "currency": "USD"},
    {"ticker": "ZC=F",  "name": "Corn",     "sector": "Commodity", "type": "commodity", "currency": "USD"},
    {"ticker": "ZW=F",  "name": "Wheat",    "sector": "Commodity", "type": "commodity", "currency": "USD"},
    {"ticker": "BTC-USD","name":"Bitcoin",  "sector": "Crypto",    "type": "crypto",    "currency": "USD"},
    {"ticker": "ETH-USD","name":"Ethereum", "sector": "Crypto",    "type": "crypto",    "currency": "USD"},
]

# Default portfolio (from stock-track-sheet)
DEFAULT_PORTFOLIO = [
    {"ticker": "HY9H.F",  "name": "SK Hynix",                "sector": "Semiconductors",  "type": "stock", "currency": "EUR"},
    {"ticker": "IREN",    "name": "IREN Ltd",                 "sector": "Bitcoin Mining",  "type": "stock", "currency": "USD"},
    {"ticker": "MU",      "name": "Micron Technology",        "sector": "Semiconductors",  "type": "stock", "currency": "USD"},
    {"ticker": "BABA",    "name": "Alibaba Group",            "sector": "E-Commerce",      "type": "stock", "currency": "USD"},
    {"ticker": "BIDU",    "name": "Baidu Inc",                "sector": "Technology",      "type": "stock", "currency": "USD"},
    {"ticker": "GOOG",    "name": "Alphabet (Google)",        "sector": "Technology",      "type": "stock", "currency": "USD"},
    {"ticker": "AXTI",    "name": "AXT Inc",                  "sector": "Semiconductors",  "type": "stock", "currency": "USD"},
    {"ticker": "AAOI",    "name": "Applied Optoelectronics",  "sector": "Networking",      "type": "stock", "currency": "USD"},
    {"ticker": "LITE",    "name": "Lumentum Holdings",        "sector": "Photonics",       "type": "stock", "currency": "USD"},
    {"ticker": "NBIS",    "name": "Nebius Group",             "sector": "AI / Cloud",      "type": "stock", "currency": "USD"},
    {"ticker": "BTC3.L",  "name": "Bitcoin ETC (LSEETF)",    "sector": "Crypto ETF",      "type": "stock", "currency": "USD"},
    {"ticker": "3KOR.L",  "name": "WisdomTree Korea 3x",     "sector": "Leveraged ETF",   "type": "stock", "currency": "USD"},
    {"ticker": "RTX",     "name": "RTX Corporation",          "sector": "Defense",         "type": "stock", "currency": "USD"},
    {"ticker": "IDR.MC",  "name": "Indra Sistemas",           "sector": "Defense / IT",    "type": "stock", "currency": "EUR"},
    {"ticker": "SNDK",    "name": "SanDisk Corporation",      "sector": "Storage",         "type": "stock", "currency": "USD"},
]


TAM_BASKETS = {
    "ai_infrastructure": {
        "label": "AI Infrastructure",
        "description": "Chips, data centers, AI compute",
        "stocks": [
            {"ticker": "NVDA",  "name": "NVIDIA",               "sector": "Semiconductors", "type": "stock", "currency": "USD"},
            {"ticker": "AMD",   "name": "Advanced Micro Devices","sector": "Semiconductors", "type": "stock", "currency": "USD"},
            {"ticker": "AVGO",  "name": "Broadcom",              "sector": "Semiconductors", "type": "stock", "currency": "USD"},
            {"ticker": "ARM",   "name": "Arm Holdings",          "sector": "Semiconductors", "type": "stock", "currency": "USD"},
            {"ticker": "SMCI",  "name": "Super Micro Computer",  "sector": "Technology",     "type": "stock", "currency": "USD"},
            {"ticker": "MRVL",  "name": "Marvell Technology",    "sector": "Semiconductors", "type": "stock", "currency": "USD"},
            {"ticker": "INTC",  "name": "Intel",                 "sector": "Semiconductors", "type": "stock", "currency": "USD"},
        ],
    },
    "cloud_saas": {
        "label": "Cloud & SaaS",
        "description": "Cloud platforms and enterprise software",
        "stocks": [
            {"ticker": "AMZN",  "name": "Amazon",                "sector": "Technology", "type": "stock", "currency": "USD"},
            {"ticker": "MSFT",  "name": "Microsoft",             "sector": "Technology", "type": "stock", "currency": "USD"},
            {"ticker": "GOOG",  "name": "Alphabet (Google)",     "sector": "Technology", "type": "stock", "currency": "USD"},
            {"ticker": "CRM",   "name": "Salesforce",            "sector": "Technology", "type": "stock", "currency": "USD"},
            {"ticker": "SNOW",  "name": "Snowflake",             "sector": "Technology", "type": "stock", "currency": "USD"},
            {"ticker": "DDOG",  "name": "Datadog",               "sector": "Technology", "type": "stock", "currency": "USD"},
            {"ticker": "NET",   "name": "Cloudflare",            "sector": "Technology", "type": "stock", "currency": "USD"},
        ],
    },
    "semiconductors": {
        "label": "Semiconductors",
        "description": "Chip makers and equipment",
        "stocks": [
            {"ticker": "TSM",   "name": "TSMC",                  "sector": "Semiconductors", "type": "stock", "currency": "USD"},
            {"ticker": "ASML",  "name": "ASML Holding",          "sector": "Semiconductors", "type": "stock", "currency": "USD"},
            {"ticker": "MU",    "name": "Micron Technology",     "sector": "Semiconductors", "type": "stock", "currency": "USD"},
            {"ticker": "LRCX",  "name": "Lam Research",          "sector": "Semiconductors", "type": "stock", "currency": "USD"},
            {"ticker": "AMAT",  "name": "Applied Materials",     "sector": "Semiconductors", "type": "stock", "currency": "USD"},
            {"ticker": "KLAC",  "name": "KLA Corporation",       "sector": "Semiconductors", "type": "stock", "currency": "USD"},
            {"ticker": "QCOM",  "name": "Qualcomm",              "sector": "Semiconductors", "type": "stock", "currency": "USD"},
        ],
    },
    "cybersecurity": {
        "label": "Cybersecurity",
        "description": "Security platforms and threat detection",
        "stocks": [
            {"ticker": "PANW",  "name": "Palo Alto Networks",    "sector": "Cybersecurity", "type": "stock", "currency": "USD"},
            {"ticker": "CRWD",  "name": "CrowdStrike",           "sector": "Cybersecurity", "type": "stock", "currency": "USD"},
            {"ticker": "ZS",    "name": "Zscaler",               "sector": "Cybersecurity", "type": "stock", "currency": "USD"},
            {"ticker": "FTNT",  "name": "Fortinet",              "sector": "Cybersecurity", "type": "stock", "currency": "USD"},
            {"ticker": "S",     "name": "SentinelOne",           "sector": "Cybersecurity", "type": "stock", "currency": "USD"},
            {"ticker": "OKTA",  "name": "Okta",                  "sector": "Cybersecurity", "type": "stock", "currency": "USD"},
        ],
    },
    "fintech": {
        "label": "Fintech",
        "description": "Digital payments and financial technology",
        "stocks": [
            {"ticker": "V",     "name": "Visa",                  "sector": "Fintech", "type": "stock", "currency": "USD"},
            {"ticker": "MA",    "name": "Mastercard",            "sector": "Fintech", "type": "stock", "currency": "USD"},
            {"ticker": "PYPL",  "name": "PayPal",                "sector": "Fintech", "type": "stock", "currency": "USD"},
            {"ticker": "SQ",    "name": "Block (Square)",        "sector": "Fintech", "type": "stock", "currency": "USD"},
            {"ticker": "COIN",  "name": "Coinbase",              "sector": "Crypto / Fintech", "type": "stock", "currency": "USD"},
            {"ticker": "AFRM",  "name": "Affirm",                "sector": "Fintech", "type": "stock", "currency": "USD"},
        ],
    },
    "ev_clean_energy": {
        "label": "EV & Clean Energy",
        "description": "Electric vehicles and renewable energy",
        "stocks": [
            {"ticker": "TSLA",  "name": "Tesla",                 "sector": "EV / Automotive",  "type": "stock", "currency": "USD"},
            {"ticker": "RIVN",  "name": "Rivian",                "sector": "EV / Automotive",  "type": "stock", "currency": "USD"},
            {"ticker": "NIO",   "name": "NIO",                   "sector": "EV / Automotive",  "type": "stock", "currency": "USD"},
            {"ticker": "ENPH",  "name": "Enphase Energy",        "sector": "Clean Energy",      "type": "stock", "currency": "USD"},
            {"ticker": "FSLR",  "name": "First Solar",           "sector": "Clean Energy",      "type": "stock", "currency": "USD"},
            {"ticker": "NEE",   "name": "NextEra Energy",        "sector": "Utilities",         "type": "stock", "currency": "USD"},
        ],
    },
    "biotech": {
        "label": "Biotech / Genomics",
        "description": "Biotech innovation and genomics",
        "stocks": [
            {"ticker": "MRNA",  "name": "Moderna",               "sector": "Biotech", "type": "stock", "currency": "USD"},
            {"ticker": "BNTX",  "name": "BioNTech",              "sector": "Biotech", "type": "stock", "currency": "USD"},
            {"ticker": "REGN",  "name": "Regeneron",             "sector": "Biotech", "type": "stock", "currency": "USD"},
            {"ticker": "VRTX",  "name": "Vertex Pharmaceuticals","sector": "Biotech", "type": "stock", "currency": "USD"},
            {"ticker": "ILMN",  "name": "Illumina",              "sector": "Genomics","type": "stock", "currency": "USD"},
            {"ticker": "AMGN",  "name": "Amgen",                 "sector": "Biotech", "type": "stock", "currency": "USD"},
        ],
    },
    "consumer_tech": {
        "label": "Consumer Tech",
        "description": "Consumer platforms, media, and apps",
        "stocks": [
            {"ticker": "AAPL",  "name": "Apple",                 "sector": "Technology",   "type": "stock", "currency": "USD"},
            {"ticker": "META",  "name": "Meta Platforms",        "sector": "Social Media", "type": "stock", "currency": "USD"},
            {"ticker": "NFLX",  "name": "Netflix",               "sector": "Streaming",    "type": "stock", "currency": "USD"},
            {"ticker": "SPOT",  "name": "Spotify",               "sector": "Streaming",    "type": "stock", "currency": "USD"},
            {"ticker": "SNAP",  "name": "Snap",                  "sector": "Social Media", "type": "stock", "currency": "USD"},
            {"ticker": "PINS",  "name": "Pinterest",             "sector": "Social Media", "type": "stock", "currency": "USD"},
        ],
    },
}


@bp.route("/search")
def search():
    q = request.args.get("q", "").strip()
    limit = min(int(request.args.get("limit", 10)), 20)
    if not q:
        return jsonify([])
    from ..services.market_data import search_assets
    results = search_assets(q, limit=limit)
    return jsonify({"results": results})


@bp.route("/commodities")
def commodities():
    return jsonify(COMMODITIES)


@bp.route("/preset")
def preset():
    name = request.args.get("name", "portfolio")
    if name == "portfolio":
        return jsonify({"assets": DEFAULT_PORTFOLIO})
    if name == "commodities":
        return jsonify({"assets": COMMODITIES})
    if name in TAM_BASKETS:
        return jsonify({"assets": TAM_BASKETS[name]["stocks"]})
    return jsonify({"assets": []})


@bp.route("/tam-baskets")
def tam_baskets():
    return jsonify([
        {"key": k, "label": v["label"], "description": v["description"], "count": len(v["stocks"])}
        for k, v in TAM_BASKETS.items()
    ])
