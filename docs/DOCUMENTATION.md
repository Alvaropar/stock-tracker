# Stock Analysis Platform — Full Documentation

> **Version:** 4.3
> **Last updated:** 2026-04-06
> **Author:** Álvaro (project owner)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Installation & Setup](#3-installation--setup)
4. [Running the Application](#4-running-the-application)
5. [User Interface Guide](#5-user-interface-guide)
6. [Technical Indicators](#6-technical-indicators)
7. [Fundamental Analysis](#7-fundamental-analysis)
8. [Sentiment Analysis](#8-sentiment-analysis)
9. [Scoring System](#9-scoring-system)
10. [Signal Backtest](#10-signal-backtest)
10.1. [IC Weight Calibration](docs/IC_CALIBRATION.md) — *See separate detailed guide*
11. [Excel Export](#11-excel-export)
12. [Buying Checklist & Elder Impulse](#12-buying-checklist--elder-impulse)
13. [API Reference](#13-api-reference)
14. [Building the Desktop Application](#14-building-the-desktop-application)
15. [Settings & Configuration](#15-settings--configuration)
16. [Project Structure](#16-project-structure)
17. [Standalone Components](#17-standalone-components)
18. [Dependencies](#18-dependencies)
19. [Troubleshooting](#19-troubleshooting)

---

## 1. Overview

The **Stock Analysis Platform** is a comprehensive financial analysis tool that combines:

- **Technical analysis** — Moving averages, RSI, MACD, Bollinger Bands, ADX, Elder Impulse System, and a buying confidence checklist
- **Relative Strength vs SPY** — RS 1M (21-day), RS 55D (55-day IBD-inspired), RS 3M (63-day); all three surfaced in the UI, Excel export, and backtest
- **Quantitative scoring (v4.2)** — Momentum score, risk score, dip/oversold detection, analyst adjustment, MA200 soft band, consolidation exemption, generalization-hardened 6-step contextual signal decision tree with 12+ verbal signal types
- **Fundamental analysis** — P/E ratios, margins, ROE, debt metrics, analyst consensus, target price gap
- **AI-powered sentiment analysis** — News scraping from Yahoo Finance and Google News, classified by cloud LLMs (Claude, ChatGPT, Gemini, or Grok)
- **Signal Backtest (Step 6)** — Replay the full tech + fund scoring pipeline on up to 5 years of historical price data per ticker; interactive dual-axis Chart.js chart with hover tooltips showing exact verbal signals, signal distribution stats, and fund score summary
- **Point-in-time ML data capture** — Market context, fundamentals, and sentiment snapshots can be persisted and aligned as-of each bar for safer historical ML training
- **Guarded paper trading** — Persistent paper ledger, broker-like fills, soak-period tracking, and pre-trade live-risk checks before any execution adapter is allowed to submit
- **Weighted scoring** — Combines all three pillars into a single composite score per asset
- **Professional Excel export** — 43-column dashboard (including RS 1M / RS 55D / RS 3M), fundamentals sheet, sentiment sheet, and per-stock sheets with embedded matplotlib charts (candlestick + volume + RSI + MACD)

The platform runs as a native desktop application (via PyWebView) or in any web browser. It can be built into a standalone `.exe` distributable via PyInstaller.

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────┐
│                    Frontend (Vanilla JS)              │
│          index.html  ·  app.js  ·  style.css         │
│              5-step wizard  ·  SSE listener           │
├──────────────────────────────────────────────────────┤
│                 Flask Backend (Python)                │
│                                                      │
│  API Blueprints:                                     │
│    /api/assets/*      Asset search & presets          │
│    /api/analysis/*    SSE analysis pipeline           │
│    /api/export/*      Excel generation                │
│    /api/browse/*      File browser (desktop mode)     │
│    /api/settings/*    User settings persistence       │
│    /api/trading/*     Paper trading + guarded exec    │
│                                                      │
│  Services:                                           │
│    market_data.py     yfinance + indicator engine     │
│    scoring.py         Weighted scoring [-1, +1]       │
│    sentiment.py       News scraping + LLM classify    │
│    pit_data.py        Point-in-time snapshot store    │
│    ml_engine.py       ML trading engine + registry    │
│    backtest.py        Strategy backtester             │
│    live_risk.py       Kill switch + exposure checks   │
│    paper_trading.py   Persistent paper ledger         │
│    execution.py       Execution adapter interface     │
├──────────────────────────────────────────────────────┤
│                 Entry Point: run.py                   │
│    Desktop mode: PyWebView native window              │
│    Web mode: Flask server + browser auto-open         │
└──────────────────────────────────────────────────────┘
```

### Data flow for a single analysis run

1. **User configures** assets, indicators, sentiment provider, and weights via the 6-step UI
2. **Frontend** sends `POST /api/analysis/start` with the full configuration JSON
3. **Backend** spawns a background thread that:
   - Phase 0: Fetches market context (VIX, NYSE breadth)
   - Phase 1: Fetches OHLCV data and fundamentals per asset via yfinance
   - Phase 2: Computes technical indicators, weekly indicators, Elder Impulse, and buying checklist
   - Phase 3: Scrapes news headlines and classifies sentiment via the chosen LLM API
   - Phase 4: Persists point-in-time market/fundamental/sentiment snapshots for later ML alignment
   - Phase 5: Computes weighted scores, sorts results, streams completion event
4. **Frontend** receives real-time progress via `GET /api/analysis/stream/<task_id>` (Server-Sent Events)
5. **Results** displayed in a sortable table; user can export to Excel via `POST /api/export/excel`

### ML and trading safeguards

- ML training defaults to `per_ticker`; pooled `universe` mode remains research-oriented until a full portfolio construction/execution stack is in place.
- Fundamental and sentiment features stay out of historical training unless point-in-time snapshots exist and the corresponding training flags are explicitly enabled.
- Registry refresh is gated twice: first by single-name walk-forward readiness, then by portfolio-aware checks for correlation, overlap, exposure, turnover, and capacity.
- Every execution request must pass `LiveRiskManager` before submission. The built-in adapters are paper and shadow adapters; no live broker adapter ships in-tree.

---

## 3. Installation & Setup

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Internet connection (for market data and sentiment APIs)

### Install dependencies

```bash
cd C:\Projects\stock-analyzer\app
pip install -r requirements.txt
```

**Core dependencies** (always required):
- `flask>=3.0` — Web framework
- `yfinance>=0.2.36` — Market data
- `pandas>=2.0`, `numpy>=1.24` — Data manipulation
- `openpyxl>=3.1` — Excel generation
- `matplotlib` — Chart rendering
- `feedparser>=6.0` — RSS news parsing
- `tenacity>=8.0` — Retry logic with exponential backoff
- `requests` — HTTP client for LLM APIs

**Optional dependencies:**
- `pywebview>=5.0` — Native desktop window (falls back to browser if absent)

**Not required for production** (cloud API replaces local models):
- `torch`, `transformers`, `peft`, `bitsandbytes`, `accelerate` — Only needed if using the `sentiment/pipeline/` local LLM approach

### Sentiment API keys

To use sentiment analysis, you need an API key from one of:
- **Claude** (Anthropic): https://console.anthropic.com/
- **ChatGPT** (OpenAI): https://platform.openai.com/
- **Gemini** (Google): https://aistudio.google.com/
- **Grok** (xAI): https://console.x.ai/

The API key is entered in the UI at runtime and is **not** persisted to disk for security.

---

## 4. Running the Application

### Desktop mode (default)

```bash
cd C:\Projects\stock-analyzer\app
python run.py
```

Opens a native window (1360×880) with the full application. Requires `pywebview`.

### Browser mode

```bash
python run.py --web              # Opens default browser
python run.py --web --no-browser # Server only (manual navigation)
```

### Custom port

```bash
python run.py --port 8080
```

Default port is 5050. If occupied, a random free port is chosen automatically.

### From the built executable

```
dist\StockAnalyzer\StockAnalyzer.exe          # Native window
dist\StockAnalyzer\StockAnalyzer.exe --web     # Browser mode
```

---

## 5. User Interface Guide

The UI is a 6-step wizard with a dark professional theme.

### Step 1: Assets

- **Search** any stock by ticker or company name (queries yfinance)
- **Add/remove** individual assets to your analysis list
- **Load presets:**
  - *My Portfolio* — 15 pre-configured stocks (SK Hynix, Micron, Alibaba, Google, RTX, etc.)
  - *Commodities* — 11 assets (Gold, Silver, Copper, Crude Oil, Natural Gas, Platinum, Aluminum, Corn, Wheat, Bitcoin, Ethereum)
- Assets display with ticker, name, sector, and currency

### Step 2: Indicators

**Technical indicators** (toggle on/off):
| Indicator | Description |
|-----------|-------------|
| MA 20 | 20-day simple moving average |
| MA 50 | 50-day simple moving average |
| MA 200 | 200-day simple moving average |
| MA Cross | Golden Cross / Death Cross (MA50 vs MA200) |
| RSI 14 | Relative Strength Index (14-period) |
| MACD | Moving Average Convergence Divergence (12/26/9) |
| Bollinger Bands | 20-day, ±2 standard deviations |
| ATR | Average True Range (14-period) |
| Volume Ratio | Current volume vs 20-day average |

**Fundamental metrics** (toggle on/off):
| Metric | Description |
|--------|-------------|
| P/E Ratio | Trailing and forward price-to-earnings |
| PEG | Price/earnings to growth ratio |
| P/Book | Price-to-book value |
| Margins | Net profit margin (%) |
| ROE | Return on equity (%) |
| Revenue Growth | Year-over-year revenue growth (%) |
| Debt/Equity | Total debt-to-equity ratio |
| Current Ratio | Current assets / current liabilities |
| Analyst | Mean analyst recommendation (1–5 scale) |

**Data period:** 6 months, 1 year (default), 2 years, or 5 years

### Step 3: Sentiment

- **Enable/disable** sentiment analysis
- **Select AI provider:**
  - Claude (Anthropic) — default, model: `claude-sonnet-4-20250514`
  - ChatGPT (OpenAI) — model: `gpt-4o-mini`
  - Gemini (Google) — model: `gemini-2.0-flash`
  - Grok (xAI) — model: `grok-3-mini-fast`
- **Enter API key** (password field, not saved)
- **Model override** — optionally specify a different model name
- **Max articles** — number of news articles to analyze per asset (default: 50)

### Step 4: Weights

Three sliders (0–100%) controlling how much each pillar contributes:
- **Technical weight** — default 40%
- **Fundamental weight** — default 40%
- **Sentiment weight** — default 20%

Must sum to exactly 100%. If sentiment is disabled, its weight is redistributed proportionally between technical and fundamental.

### Step 5: Results

- **Sortable table** — click any column header to sort
- **Columns:** Ticker, Company, Sector, Price, Returns (1D/1W/1M/3M), 52W Pos, RSI, MA Cross, MACD, BB%, P/E, Fwd P/E, ADX, Regime, ATR%, Vol Ratio, **RS 1M**, **RS 55D**, **RS 3M**, Trend Stage, Vol Regime, Market Regime, Regime Change, Momentum, Risk, Dip, Tech Score, Fund Score, Sent Score, Confidence%, Elder, Overall Score, Contextual Signal, ML Signal
- **Color-coded RS columns:** Green (outperforming SPY), red (underperforming), bold when |RS| > 5 pp
- **Color-coded signals:** Strong Buy (dark green) → Buy (green) → Neutral (amber) → Sell (orange) → Strong Sell / Avoid (red)
- **Export to Excel** button — generates the 43-column dashboard workbook
- **📊 Backtest** button — jumps directly to Step 6 with current assets pre-loaded

### Step 6: Signal Backtest

Replays the tech + fund scoring pipeline on historical price data without re-running a full live analysis.

- **Period selector:** 1 Year / 2 Years / 5 Years
- **Stock chips** show which tickers have been backtested (highlighted in blue when done)
- **Run Backtest** sends `POST /api/backtest/run` with the same indicators and weights configured in Steps 2 and 4
- **Dual-axis chart** (Chart.js):
  - Left Y-axis: price line in blue
  - Right Y-axis: signal score line in [−1, +1], **each segment coloured by signal zone** (green = BUY, amber = NEUTRAL, red = SELL)
  - Green/red shaded bands mark the BUY (≥ 0.2) and SELL (≤ −0.2) zones
  - Dashed reference lines at 0, +0.2, −0.2
  - **Hover tooltip:** shows exact date, price, score, and verbal signal (e.g. `BUY (MOMENTUM CONTINUATION)`, `AVOID (EXTREME RISK)`)
- **Ticker tabs** for switching between stocks without re-running
- **Signal Summary panel:**
  - Current signal and score
  - Fundamental score (static — see limitation note)
  - Trading days analysed
  - Vertical bar chart of signal distribution (% of days in each signal zone)
- **Limitation note:** Fundamental score is fixed at current yfinance values across all historical dates (no daily historical P/E / ROE data). Technical, regime, and RS signals are computed from actual historical OHLCV and are fully accurate.

---

## 6. Technical Indicators

All indicators are computed in `app/backend/services/market_data.py` using pandas/numpy on OHLCV data fetched from yfinance.

### Moving Averages

- **MA20/MA50/MA200** — Simple moving averages (20, 50, 200 days)
- **EMA8/EMA13/EMA20** — Exponential moving averages (for Elder Impulse and checklist)
- **Golden Cross** — MA50 crosses above MA200 (bullish)
- **Death Cross** — MA50 crosses below MA200 (bearish)

### RSI (Relative Strength Index)

- **Period:** 14
- **Method:** Exponential weighted moving average of gains and losses
- **Interpretation:** <30 = oversold (bullish), >70 = overbought (bearish)

### MACD (Moving Average Convergence Divergence)

- **MACD Line:** 12-day EMA − 26-day EMA
- **Signal Line:** 9-day EMA of MACD
- **Histogram:** MACD − Signal (momentum indicator)
- **Bullish:** MACD above Signal; **Bearish:** MACD below Signal

### Bollinger Bands

- **Middle:** 20-day SMA
- **Upper:** Middle + 2σ
- **Lower:** Middle − 2σ
- **BB%:** Position within bands (0% = at lower band, 100% = at upper band)

### ATR (Average True Range)

- **Period:** 14
- **Measures** volatility as the average of max(High−Low, |High−PrevClose|, |Low−PrevClose|)

### Volume Ratio

- Current day volume divided by the 20-day average volume
- Values > 1.0 indicate above-average trading activity

### Performance Returns

- **1 Day, 5 Day, 21 Day, 63 Day** — Percentage returns over respective periods

### 52-Week Range

- **52W High/Low** — Rolling 252-day high and low
- **52W Position** — Where current price sits within the range (0–100%)

### Trend Stage

Classifies the stock's trend maturity based on distance from key moving averages:

| Stage | Condition | Interpretation |
|-------|-----------|----------------|
| EARLY | Price recently crossed above MA50 | New trend, low risk |
| ESTABLISHED | Sustained above MA50, moderate extension | Healthy trend |
| EXTENDED | Significantly above MAs, high extension | Trend aging, pullback likely |
| PARABOLIC | Extreme extension + high ATR | Unsustainable, high risk |

### Volume Regime

| Regime | Condition |
|--------|-----------|
| LOW | Volume < 80% of 20-day average |
| NORMAL | Volume between 80% and 150% of average |
| HIGH | Volume > 150% of average |
| CLIMACTIC | Volume > 250% of average (capitulation or blow-off) |

### Market Regime

Determined from the relationship between price, MA50, and MA200:

| Regime | Condition |
|--------|-----------|
| BULLISH | Price > MA50 > MA200 |
| BEARISH | Price < MA50 < MA200 |
| TRANSITION | Mixed signals (e.g., price above MA200 but below MA50) |

### Regime Change

Detects transitions between market regimes:

| Change | Meaning |
|--------|---------|
| BULLISH REVERSAL | Transitioning from bearish to bullish |
| BEARISH REVERSAL | Transitioning from bullish to bearish |
| WEAKENING | Bullish trend losing strength |
| POTENTIAL BOTTOM | Bearish trend showing reversal signs |
| NONE | No regime change detected |

### ADX (Average Directional Index)

- **Period:** 14
- Measures trend strength regardless of direction
- < 20: weak/no trend, 20–25: emerging trend, 25–40: strong trend, > 40: very strong trend

### Relative Strength vs SPY

Three periods are computed and surfaced in the UI table, Excel export, and Signal Backtest:

| Column | Period | Calculation |
|--------|--------|-------------|
| **RS 1M** | 21 trading days | Asset 21-day return − SPY 21-day return |
| **RS 55D** | 55 trading days | Asset 55-day return − SPY 55-day return (IBD-inspired) |
| **RS 3M** | 63 trading days | Asset 63-day return − SPY 63-day return |

Positive = outperforming the market; Negative = underperforming.

The 55-day window is IBD-inspired — it captures intermediate-term strength that is often more stable than 1-month windows while being more responsive than 3-month windows.

All three RS values feed into the composite RS used by the momentum score and RS adjustment step of the scoring pipeline with weights 30% / 40% / 30%.

---

## 7. Fundamental Analysis

Fundamental data is fetched from yfinance's `Ticker.info` dictionary. All metrics are optional — if data is unavailable for an asset, the field is `None`.

| Metric | Source field | Notes |
|--------|-------------|-------|
| Trailing P/E | `trailingPE` | Price / trailing 12-month earnings |
| Forward P/E | `forwardPE` | Price / estimated forward earnings |
| PEG Ratio | `pegRatio` | P/E divided by earnings growth rate |
| Price/Book | `priceToBook` | Market price / book value per share |
| Net Margin | `profitMargins` | Net income / revenue (converted to %) |
| ROE | `returnOnEquity` | Net income / shareholder equity (converted to %) |
| Revenue Growth | `revenueGrowth` | YoY revenue growth (converted to %) |
| Debt/Equity | `debtToEquity` | Total debt / total equity |
| Current Ratio | `currentRatio` | Current assets / current liabilities |
| Market Cap | `marketCap` | Total market capitalization |
| Beta | `beta` | Volatility relative to S&P 500 |
| Dividend Yield | `dividendYield` | Annual dividend / price (converted to %) |
| Target Price | `targetMeanPrice` | Mean analyst price target |
| Recommendation | `recommendationMean` | 1=Strong Buy, 2=Buy, 3=Hold, 4=Sell, 5=Strong Sell |
| Analyst Count | `numberOfAnalystOpinions` | Number of analysts covering |
| Short Float | `shortPercentOfFloat` | Short interest as % of float |

---

## 8. Sentiment Analysis

### Architecture

The sentiment system in the app (`app/backend/services/sentiment.py`) uses cloud LLM APIs — no local GPU or model weights required.

### News Sources

1. **Yahoo Finance** — via `yfinance.Ticker(ticker).news` API
2. **Google News RSS** — queries `news.google.com/rss/search?q=<ticker> stock`
   - Also searches by company name if provided
   - Extracts headline, source, date, summary, URL

Headlines are de-duplicated by title (case-insensitive) and sorted newest-first.

### LLM Classification

Headlines are sent in batches of 30 to the selected LLM API with this system prompt:

> "You are a financial sentiment classifier. For each news headline about a stock, respond with exactly one word: positive, negative, or neutral."

Each headline is classified as:
- **positive** — suggests stock price will go UP (earnings beats, upgrades, partnerships, growth)
- **negative** — suggests stock price will go DOWN (losses, downgrades, lawsuits, layoffs)
- **neutral** — factual/ambiguous with no clear directional impact

### Supported Providers

| Provider | API Endpoint | Default Model |
|----------|-------------|---------------|
| Claude | `api.anthropic.com/v1/messages` | `claude-sonnet-4-20250514` |
| ChatGPT | `api.openai.com/v1/chat/completions` | `gpt-4o-mini` |
| Gemini | `generativelanguage.googleapis.com/v1beta` | `gemini-2.0-flash` |
| Grok | `api.x.ai/v1/chat/completions` | `grok-3-mini-fast` |

### Sentiment Metrics

| Metric | Calculation |
|--------|-------------|
| **Score** | `(positive - negative) / total`, range [-1, +1] |
| **Signal** | ≥0.5 STRONG BULLISH, ≥0.2 BULLISH, ≥-0.2 NEUTRAL, ≥-0.5 BEARISH, <-0.5 STRONG BEARISH |
| **Momentum** | Difference between recent-half score and older-half score |
| **Dispersion** | Fraction of neutral articles (agreement measure) |
| **Volume Trend** | "rising" if >5 unique article dates, "stable" if 3–5, "falling" if ≤2 |
| **Headlines** | Top 10 headlines with their sentiment labels |

### Response Parsing

The `_parse_sentiments()` function handles various LLM response formats:
1. **Primary:** JSON array like `["positive", "negative", "neutral"]`
2. **Fallback:** Regex extraction of individual `positive`/`negative`/`neutral` keywords
3. **Padding:** If fewer labels than expected, pads with `"neutral"`

---

## 9. Scoring System

Defined in `app/backend/services/scoring.py` (v4.2).

### Technical Score [-1, +1]

Each enabled indicator contributes a weighted score. Weights are regime-conditioned (TREND / MEAN_REVERSION / NEUTRAL profiles):

| Indicator | Typical Weight | Bullish (+) | Bearish (−) |
|-----------|----------------|-------------|-------------|
| MA 20 | 0.5–1.0 | Price > MA20 | Price < MA20 |
| MA 50 | 0.5–1.5 | Price > MA50 | Price < MA50 |
| MA 200 | 0.5–2.0 | Price > MA200 | Price < MA200 |
| MA Cross | 0.5–1.5 | Golden Cross (MA50 > MA200) | Death Cross |
| RSI | 0.5–2.0 | RSI < 30 → full, RSI < 45 → ×0.25 | RSI > 70 → full, RSI > 55 → ×0.25 |
| MACD | 0.5–2.0 | Line above signal (×0.67) + histogram positive (×0.33) | Inverse |
| Bollinger | 0.5–2.0 | BB% < 15 (near lower band) | BB% > 85 (near upper band) |

**MA200 soft band (v4.2):** Instead of a binary above/below, a ±3% tolerance zone is applied. Stocks within 3% of MA200 score +20% of the full MA200 weight (mild signal), avoiding false penalties for stocks consolidating just below the line (e.g. −1.9% from MA200 no longer scores as "bearish").

Final score = sum of signals / sum of max weights for selected indicators.

### Fundamental Score [-1, +1]

Each selected metric maps to a signal value; all are equally weighted:

| Metric | Strong bullish (+1.0) | Bullish (+0.5) | Neutral (0.0) | Bearish (−0.5) | Strong bearish (−1.0) |
|--------|----------------------|----------------|---------------|----------------|----------------------|
| P/E | < 10 | 10–20 | 20–30 | 30–50 | > 50 |
| PEG | < 0.5 | 0.5–1.0 | 1.0–2.0 | 2.0–3.0 | > 3.0 |
| P/Book | < 1.0 | 1.0–2.5 | 2.5–5.0 | 5.0–10.0 | > 10.0 |
| Net Margin | ≥ 25% | 10–25% | 0–10% | −10–0% | < −10% |
| ROE | ≥ 25% | 15–25% | 5–15% | 0–5% (−0.25) | < 0% |
| Rev Growth | ≥ 20% | 10–20% | 0–10% | −5–0% | < −5% |
| Debt/Equity | < 0.3 | 0.3–0.7 | 0.7–1.5 | 1.5–3.0 | > 3.0 |
| Current Ratio | ≥ 2.5 | 1.5–2.5 | 1.0–1.5 | — | < 1.0 |
| Analyst | 1.0 (Strong Buy) ↔ 5.0 (Strong Sell), mapped linearly to [+1, −1] |

Final score = average of all signal values.

### Overall Score [-1, +1]

```
overall = (tech_score × tech_weight + fund_score × fund_weight + sent_score × sent_weight) / total_weight
```

Default weights: 40% technical, 40% fundamental, 20% sentiment.

If sentiment is disabled, weights are redistributed proportionally (e.g., 40/40 → 50/50).

### Signal Mapping

| Score range | Signal | Color |
|-------------|--------|-------|
| ≥ 0.5 | STRONG BUY | Dark green `#00873D` |
| ≥ 0.2 | BUY | Green `#70AD47` |
| ≥ −0.2 | NEUTRAL | Amber `#FFC000` |
| ≥ −0.5 | SELL | Orange `#FF6600` |
| < −0.5 | STRONG SELL | Red `#C00000` |

### Quantitative Scores (v4.1)

Three composite scores augment the base technical/fundamental/sentiment scoring:

#### Momentum Score [0, 1]

Measures trend continuation strength. Components averaged equally across available inputs:

| Component | Scoring |
|-----------|---------|
| ADX Strength | < 20 → 0, 20–25 → 0.25, 25–30 → 0.50, 30–40 → 0.75, > 40 → 1.0 |
| RS Composite (30% 1M / 40% 55D / 30% 3M) | ≤ 0% → 0, 0–5% → 0.15, 5–20% → 0.33, 20–50% → 0.67, ≥ 50% → 1.0 |
| Volume Ratio | < 0.8 → 0, 0.8–1.2 → 0.25, 1.2–1.5 → 0.50, 1.5–2.0 → 0.75, > 2.0 → 1.0 |
| MACD Confirmation | Bullish → 1.0, Bearish → 0.0 |

The composite RS uses all three available periods (RS 1M, RS 55D, RS 3M) with IBD-inspired weighting (40% to 55D as the primary intermediate window), normalising by available weight sum so the score remains valid when fewer periods are available.

#### Risk Score

Measures compound risk from trend extension and volatility:

```
RISK_SCORE = |Trend Extension| × ATR%
```

| Risk Level | Score | Impact |
|------------|-------|--------|
| Low | < 0.30 | No penalty |
| Moderate | 0.30–1.0 | Score × 0.94 |
| High | 1.0–2.0 | Score × 0.85 |
| Very High | 2.0–5.0 | Score × 0.72 |
| Extreme | > 5.0 | Score × 0.55 |

#### Dip Score [0, 1]

Identifies oversold stocks with strong fundamentals (buy-the-dip candidates). Gate: RSI must be < 45 (lowered from < 40 in v4.1 to catch early-stage dips). Components averaged equally:

| Component | Scoring |
|-----------|---------|
| RSI Oversold Depth | RSI < 20 → 1.0, < 25 → 0.8, < 30 → 0.6, < 35 → 0.35, < 40 → 0.15, < 45 → 0.05 |
| Fundamental Quality | fund_score ≥ 0.50 → 1.0, ≥ 0.30 → 0.7, ≥ 0.10 → 0.4, else 0.0 |
| Volume (Capitulation) | VR ≥ 2.0 → 1.0, ≥ 1.5 → 0.7, ≥ 1.0 → 0.4, else 0.2 |
| Bollinger Band Position | BB% < 5 → 1.0, < 15 → 0.6, < 25 → 0.3, else 0.0 |
| Analyst Target Gap | (target_px − price)/price ≥ 40% → 0.9, ≥ 25% → 0.6, ≥ 15% → 0.3 |

**Target gap guard (v4.2):** The analyst target gap component requires `n_analysts ≥ 3`. Pre-revenue biotech/speculative stocks with 1–2 DCF-model targets 200–500% above price would otherwise produce spurious dip scores on broken charts.

**Anti-falling-knife filters:**
- BEARISH REVERSAL regime change → dip_score forced to 0
- BEARISH market regime → dip_score × 0.40
- TRANSITION market regime → dip_score × 0.70

### 9-Step Score Pipeline

The overall score goes through 9 sequential adjustments:

1. **Base Score** — Weighted average of tech, fund, and sentiment scores
2. **Volume Adjustment** — Direction-aware: low volume penalizes bullish scores more harshly
3. **Relative Strength** — Multi-period composite RS adjustment (30% RS 1M / 40% RS 55D / 30% RS 3M)
3.5. **Analyst Adjustment (v4.2)** — Additive ±0.10 from target price gap + rec_mean; requires ≥ 3 analyst opinions
4. **Dip Boost** — Additive +0.05 to +0.25 for quality dip candidates (dip_score ≥ 0.35, score < 0.35)
5. **Trend Maturity** — Penalty for EXTENDED/PARABOLIC stages (softened if momentum ≥ 0.65)
6. **Volatility** — ATR% penalty for extreme volatility
7. **Risk Penalty** — Multiplicative penalty based on risk_score (only for bullish scores)
8. **Regime Transition** — BEARISH REVERSAL × 0.40, BULLISH REVERSAL × 1.20, WEAKENING × 0.75
9. **SPY Filter** — Dampens bullish scores by 20% when SPY is below its MA200

#### Analyst Adjustment (v4.2)

Two sub-components, result capped at [−0.10, +0.10]:

| Target Gap (target − price)/price | Adjustment |
|-----------------------------------|-----------|
| ≥ +40% (large institutional upside) | +0.08 |
| ≥ +25% | +0.05 |
| ≥ +15% | +0.02 |
| ≤ −15% (trading above target) | −0.03 |
| ≤ −25% | −0.06 |

| Rec Mean (1=Strong Buy … 5=Strong Sell) | Adjustment |
|-----------------------------------------|-----------|
| < 1.5 | +0.04 |
| < 2.0 | +0.02 |
| > 3.5 | −0.02 |
| > 4.0 | −0.04 |

### 6-Step Signal Decision Tree (v4.2)

Contextual signals are determined by ordered evaluation (first match wins):

1. **AVOID** — PARABOLIC stage with ATR ≥ 12% or volume < 1.0; or risk_score ≥ 3.0
2. **Regime Transitions** — BEARISH REVERSAL → SELL/HOLD, BULLISH REVERSAL → BUY (EARLY REVERSAL), WEAKENING → HOLD, POTENTIAL BOTTOM → BUY (POTENTIAL BOTTOM)
   - **2b. Oversold Dip** — dip_score ≥ 0.55 + fund ≥ 0.30 → BUY (OVERSOLD DIP); dip ≥ 0.35 + fund ≥ 0.20 + RSI < 30 → BUY (MEAN REVERSION DIP)
   - **2c. Above Analyst Target (v4.2)** — target_gap < −25% AND score ≥ −0.3 AND momentum_score < 0.55 → HOLD/SELL (ABOVE ANALYST TARGET). The `momentum_score < 0.55` guard prevents penalising legitimate breakout stocks where analysts simply haven't updated stale targets yet.
3. **Momentum Continuation** — EXTENDED + momentum ≥ 0.65 + score ≥ 0.1 → BUY (MOMENTUM CONTINUATION)
4. **Trend Maturity HOLDs** — Strong trend pullback, parabolic reduce, overextended, extended wait
5. **Weak Momentum HOLD** — Triggered when ≥ 2 of: vol_ratio < 0.8, RS 1M < −2%, momentum < 0.30; exempt when dip_score ≥ 0.35
   - **Consolidation exemption (v4.2):** RS 55D > 10% AND RS 3M > 10% AND fund ≥ 0.50 AND RS 1M > −15% → exempt (stock is consolidating after a big run, not reversing). The `RS 1M > −15%` guard prevents exempting stocks that are actively breaking down with catastrophic recent performance.
   - **Quality Discount hint:** fund_score ≥ 0.30 AND target_gap > 20% → HOLD (QUALITY DISCOUNT) instead of HOLD (WEAK MOMENTUM)
6. **Default Labels** — Standard BUY/SELL/NEUTRAL with context hints (EARLY TREND, STRONG MOMENTUM, MEAN REVERSION, etc.)

### Signal Categories

| Signal | Type | Description |
|--------|------|-------------|
| BUY (OVERSOLD DIP) | Dip | Quality oversold stock with strong fundamentals |
| BUY (MEAN REVERSION DIP) | Dip | Moderate dip with RSI < 30 |
| BUY (MOMENTUM CONTINUATION) | Momentum | Extended trend with strong momentum |
| BUY (EARLY REVERSAL) | Regime | Bullish regime reversal detected |
| BUY (POTENTIAL BOTTOM) | Regime | Bearish regime showing bottom signs |
| BUY (EARLY TREND) | Trend | New trend, low risk entry |
| BUY (STRONG MOMENTUM) | Momentum | High momentum confirmation |
| HOLD (STRONG TREND – WAIT FOR PULLBACK) | Trend | Strong but extended, wait for entry |
| HOLD (WEAK MOMENTUM) | Filter | Insufficient momentum confirmation |
| HOLD (QUALITY DISCOUNT) | Filter | Weak technically but large analyst upside |
| HOLD (ABOVE ANALYST TARGET) | Valuation | Stock trading > 25% above consensus target |
| HOLD (EXTENDED – WAIT FOR PULLBACK) | Trend | Overextended, patience needed |
| AVOID (PARABOLIC / HIGH RISK) | Risk | Extreme risk, do not enter |
| AVOID (EXTREME RISK) | Risk | High risk score on non-parabolic name |
| SELL (BEARISH REVERSAL) | Regime | Regime turning bearish |
| SELL (BEAR CONFIRMED) | Regime | Bearish confirmation signal |

### Confidence Adjustment

Base confidence (from buying checklist) is adjusted by:
- **Volatility regime:** HIGH → ×0.85, CLIMACTIC → ×0.70
- **Trend stage:** PARABOLIC → ×0.70, EXTENDED → ×0.85
- **Market regime:** BEARISH → ×0.80
- **Risk score drag:** Interpolated reduction for risk > 0.30
- **Momentum boost:** momentum ≥ 0.65 → ×1.10, ≥ 0.50 → ×1.05
- **Dip boost:** dip_score ≥ 0.65 → ×1.20, ≥ 0.45 → ×1.12, ≥ 0.35 → ×1.06

---

## 10. Signal Backtest

Defined in `app/backend/api/backtest.py`, rendered by the `btRender*` functions in `app.js`.

### How it works

The backtest endpoint (`POST /api/backtest/run`) replays the full scoring pipeline on historical OHLCV data:

1. Fetches full asset history via `fetch_asset_data()` for the requested period
2. Downloads SPY independently with the same period to compute per-date RS 1M / RS 55D / RS 3M
3. Skips the first 200 bars (warmup) so that MA200 and all long-window indicators are valid
4. For every remaining trading day:
   - Reads historical indicator values from the pre-computed DataFrame (MA20/50/200, RSI, MACD, ADX, BB%, ATR%, Vol_Ratio, Trend_Stage, Mkt_Regime, etc.)
   - Looks up the SPY row for that exact date to compute per-date RS vs SPY
   - Computes `target_gap` from the static analyst target vs the historical price
   - Runs the full pipeline: `compute_technical_score` → `compute_momentum_score` → `compute_risk_score` → `compute_dip_score` → `compute_overall_score` → `contextual_signal`
5. Returns arrays of dates, prices, scores [−1, +1], verbal signals, and CSS classes

### Response structure

```json
{
  "ok": true,
  "results": {
    "NVDA": {
      "dates":          ["2024-04-08", "2024-04-09", ...],
      "prices":         [762.0, 853.4, ...],
      "scores":         [0.2341, 0.3102, ...],
      "verbal_signals": ["BUY (EARLY TREND)", "BUY (MOMENTUM CONTINUATION)", ...],
      "css_classes":    ["buy", "buy", ...],
      "fund_score":     0.4812,
      "n_points":       253
    }
  }
}
```

### Known limitations

| Limitation | Reason | Impact |
|-----------|--------|--------|
| Fundamental score is static | yfinance does not provide daily historical P/E, margins, ROE | Fund component is a constant bias; tech and regime signals are accurate |
| Analyst target is static | Historical consensus targets not available via yfinance | `target_gap` varies with historical price but target itself is fixed at current value |
| No P&L / return calculation | Backtest shows signal, not strategy return | Cannot directly measure profitability; use as signal calibration tool |
| No slippage / execution model | Signal is generated end-of-day | Forward returns from signal change would require a separate pass |

### Chart rendering

The frontend uses **Chart.js 4.4.7** with the `chartjs-plugin-annotation` plugin:

- `segment.borderColor` callback colours each line segment by its right-endpoint score (BUY zone = green, SELL zone = red)
- `annotation` plugin draws BUY/SELL bands and dashed reference lines
- Tooltip `callbacks.label` reads the `verbal_signals` array by data index to surface the exact verbal signal on hover
- `pointHoverBackgroundColor` callback uses `btScoreColor()` to match the hover dot colour to the signal zone

### IC Weight Calibration & Optimization

The backtest page includes an **Information Coefficient (IC) calibration feature** that learns which indicators are most predictive for a specific stock:

- **⚙ Calibrate IC Weights**: Measures correlation between each indicator and forward 21-day returns using rank correlation (Spearman). Positive-IC indicators are weighted; negative ones are suppressed.
- **💾 Save Weights Config**: Saves the calibrated weights under the stock's name in Step 4, so they persist across sessions.
- **↻ Rerun with IC Weights**: Reruns the same backtest period using the calibrated weights.

**Key insight**: Calibration uses data strictly *before* the backtest window to prevent overfitting. Example: backtest 2 years of recent data, calibrate on years 5–2 ago (zero overlap).

For a detailed explanation of the methodology, IC formula, weight normalization, and how to interpret results, see [**IC Weight Calibration Guide**](docs/IC_CALIBRATION.md).

---

## 11. Excel Export

The Excel workbook is generated in `app/backend/api/export.py` using openpyxl with embedded matplotlib charts.

### Sheet structure

1. **Dashboard** — Summary table with all assets, **43 columns**:
   - Row number, Ticker, Company, Sector, Price
   - Returns: 1 Day, 1 Week, 1 Month, 3 Months
   - 52-Week Position (with bar chart visual)
   - RSI 14 (color-gradient: green < 30, red > 70)
   - VS MA50, VS MA200 (distance from moving averages)
   - MA Cross (Golden/Death)
   - MACD (bullish/bearish)
   - Bollinger Band %
   - Trailing P/E, Forward P/E
   - ADX, Regime, ATR%, Vol Ratio
   - **RS 1M, RS 55D, RS 3M** — color-coded: green ≥ +2%, red ≤ −2%
   - Trend Stage, Volume Regime, Market Regime, Regime Change
   - Momentum Score, Risk Score, Dip Score
   - Tech Score, Fund Score, Sentiment Score/Signal, Articles count
   - Confidence %, Elder Impulse
   - Overall Score, ML Signal, Contextual Signal (color-coded background)

2. **Fundamentals** — Detailed fundamental data per asset:
   - P/E (trailing + forward), PEG, P/Book
   - Net Margin, ROE, Revenue Growth
   - Debt/Equity, Current Ratio
   - Market Cap, Beta, Dividend Yield
   - Target Price, Analyst Recommendation, Analyst Count

3. **Sentiment** — Sentiment analysis details:
   - Article counts (total, positive, negative, neutral)
   - Sentiment score and signal
   - Momentum, volume trend
   - Top 10 headlines with sentiment labels per asset

4. **Per-stock sheets** (one per asset) — Professional 4-panel chart:

### Chart panels

Each stock sheet contains a matplotlib-rendered PNG chart (16×10 inches, 150 DPI) with:

**Panel 1 — Price (Candlestick)**
- Green/red candlestick bars (OHLC)
- MA20 (orange), MA50 (green), MA200 (red)
- Bollinger Bands (upper + lower, blue shaded fill)
- Elder Impulse strip at the top edge (green/red/blue colored bar)
- Title with ticker, current price, daily change ($ and %)

**Panel 2 — Volume**
- Color-coded bars (green = up day, red = down day)
- Auto-formatted Y axis (M for millions, K for thousands)

**Panel 3 — RSI**
- RSI(14) line
- Overbought (70) and oversold (30) dashed reference lines
- Shaded overbought/oversold zones
- Labels for levels

**Panel 4 — MACD**
- MACD line (blue), Signal line (orange)
- Histogram bars with 4-tone coloring:
  - Bright green: positive and rising
  - Dark green: positive but falling
  - Bright red: negative and falling
  - Dark red: negative but rising
- Zero reference line

### Styling

- Dark theme: `#1a1a2e` background, `#16213e` panel backgrounds
- Professional color palette matching financial terminals
- Charts embedded at cell F1 of each stock sheet

---

## 12. Buying Checklist & Elder Impulse

### Elder Impulse System

A trend-momentum indicator combining two components:

**Daily Elder Impulse** (`Elder_D`):
- **Green** (bullish): 13-day EMA rising AND MACD histogram rising → strong buying pressure
- **Red** (bearish): 13-day EMA falling AND MACD histogram falling → strong selling pressure
- **Blue** (neutral): Mixed signals → transition period

**Weekly Elder Impulse** (`elder_w`):
- Same logic applied to weekly-resampled data
- Uses 13-week EMA and weekly MACD histogram

### Buying Checklist (11 items)

An objective, rules-based checklist inspired by top-down analysis (annualizethis.substack.com). Each item is `True` (passed), `False` (failed), or `None` (insufficient data).

| # | Check | Category |
|---|-------|----------|
| 1 | 13-Week MA Rising | Weekly trend |
| 2 | 34-Week MA Rising | Weekly trend |
| 3 | 13-Week EMA > 34-Week EMA | Weekly trend (weekly golden cross) |
| 4 | Weekly MACD Histogram Rising | Weekly momentum |
| 5 | Daily MACD Histogram Rising | Daily momentum |
| 6 | 8-Day EMA > 20-Day EMA | Daily momentum |
| 7 | Price > 13-Week EMA | Price position |
| 8 | Price > 50-Day SMA | Price position |
| 9 | Daily MACD Positive | Trend confirmation |
| 10 | Weekly Elder Not Red | Elder Impulse |
| 11 | Daily Elder Not Red | Elder Impulse |

**Confidence score** = (passed items / total valid items) × 100%

A confidence above 70% suggests the stock is in a favorable buying zone across multiple timeframes.

---

## 13. API Reference

All endpoints are served from `http://127.0.0.1:5050` (default).

### Assets

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/assets/search?q=AAPL&limit=10` | Search assets by ticker/name |
| `GET` | `/api/assets/commodities` | List predefined commodities |
| `GET` | `/api/assets/preset?name=portfolio` | Load preset asset list (`portfolio` or `commodities`) |

### Analysis

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/analysis/start` | Start analysis task; returns `{ok, task_id}` |
| `GET` | `/api/analysis/stream/<task_id>` | SSE stream of progress events |
| `GET` | `/api/analysis/results/<task_id>` | Final results JSON |

**POST body for `/api/analysis/start`:**
```json
{
  "assets": [
    {"ticker": "AAPL", "name": "Apple Inc", "sector": "Technology", "currency": "USD"}
  ],
  "indicators": {
    "period": "1y",
    "technical": ["ma20", "ma50", "ma200", "cross", "rsi", "macd", "bb"],
    "fundamental": ["pe", "margins", "roe", "growth", "analyst"]
  },
  "sentiment": {
    "enabled": true,
    "provider": "claude",
    "api_key": "sk-ant-...",
    "model": "",
    "max_articles": 50
  },
  "weights": {
    "technical": 40,
    "fundamental": 40,
    "sentiment": 20
  }
}
```

**SSE event types:**
| Event type | Fields | Description |
|------------|--------|-------------|
| `start` | `total` | Analysis started, total asset count |
| `progress` | `stage`, `pct`, `ticker`, `done`, `total`, `msg` | Progress update |
| `warn` | `ticker`, `msg` | Non-fatal warning |
| `error` | `message` | Fatal error |
| `complete` | — | Analysis finished |

### Trading

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/trading/status` | Current paper ledger, fills, positions, and live-risk snapshot |
| `POST` | `/api/trading/paper/reset` | Reset the persisted paper-trading ledger |
| `POST` | `/api/trading/paper/mark` | Mark positions to market for a given date and price map |
| `POST` | `/api/trading/execute` | Submit an order through the guarded execution service |

`/api/trading/execute` always evaluates the order against `LiveRiskManager` first. Depending on settings and soak progress, the response may be rejected, accepted in shadow mode, or routed to the paper adapter.

### Backtest

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/backtest/run` | Run historical signal backtest; returns per-ticker score time series |

**POST body for `/api/backtest/run`:**
```json
{
  "tickers":     ["NVDA", "MU"],
  "period":      "2y",
  "technical":   ["ma20", "ma50", "ma200", "cross", "rsi", "macd", "bb"],
  "fundamental": ["pe", "margins", "roe", "growth", "analyst"],
  "weights":     {"technical": 60, "fundamental": 40}
}
```

### Export

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/export/excel` | Generate and download Excel workbook |

**POST body:** `{task_id: "..."}` — references a completed analysis task.

### Settings

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/settings` | Load all saved settings |
| `POST` | `/api/settings` | Save settings (merge with existing) |

Stored in `settings.json` next to the executable (or in `app/` during development).

The trading stack also reads:
- `live_trading` — kill switch, shadow mode, daily loss and exposure limits, paper soak thresholds
- `paper_trading` — starting cash plus commission/slippage/spread assumptions for broker-like fills

### File Browser (desktop mode)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/browse?path=<dir>` | List directory contents (sandboxed) |

---

## 14. Building the Desktop Application

### Quick build

```bash
cd C:\Projects\stock-analyzer
python build.py
```

Output: `dist\StockAnalyzer\StockAnalyzer.exe` (~270 MB)

### Build options

```bash
python build.py --clean          # Clean previous build first
python build.py --onefile        # Single-file exe (slower startup)
```

### What the build does

1. **Installs build dependencies** (`pyinstaller`, `pillow`, `pywebview`)
2. **Runs PyInstaller** using `app/StockAnalyzer.spec`
3. **Excludes heavy ML libraries** — torch, transformers, peft, bitsandbytes, accelerate (cloud APIs are used instead)
4. **Bundles frontend** — HTML, CSS, JS files included via `--add-data`
5. **Copies extras** — README, LICENSE, sentiment pipeline source (without model weights)
6. **Prints summary** — executable location and size

### PyInstaller spec

The spec file (`app/StockAnalyzer.spec`) defines:
- Entry point: `app/run.py`
- Bundled data: `frontend/` (HTML, CSS, JS)
- Hidden imports: All Flask blueprints, services, matplotlib, yfinance, etc.
- Excluded modules: torch, transformers, peft, bitsandbytes, accelerate, safetensors, tokenizers
- Windowed mode: No console window
- Icon: `sentiment/assets/app.ico` (if present)

### Distribution

The `dist/StockAnalyzer/` folder is self-contained and can be zipped for distribution. Users only need:
- Windows 10/11
- Internet connection (for market data and sentiment API)
- An API key from Claude, OpenAI, Google, or xAI (for sentiment analysis)

---

## 15. Settings & Configuration

### Settings file

Location:
- **Development:** `C:\Projects\stock-analyzer\app\settings.json`
- **Built exe:** Same folder as `StockAnalyzer.exe`

Format: JSON, merged on save (not overwritten).

### Persisted settings

| Key | Type | Description |
|-----|------|-------------|
| `provider` | string | Local sentiment provider mode |
| `sent_model` | string | Optional local model override |
| `model_path` | string | Local sentiment model path |
| `adapter_path` | string | Local sentiment adapter path |
| `live_trading` | object | Execution safeguards: shadow mode, kill switch, daily loss and exposure caps, minimum paper soak |
| `paper_trading` | object | Paper account assumptions: cash, commission, slippage, spread |

### Not persisted (by design)

- **API keys** — Entered at runtime for security; never saved to disk
- **Asset list** — Re-configured each session
- **Weights** — Re-configured each session

### PyWebView JS↔Python bridge

In desktop mode, `window.pywebview.api` exposes:
- `save_file_dialog(default_name)` — Native "Save As" dialog for Excel export
- `choose_folder()` — Native folder picker

---

## 16. Project Structure

```
C:\Projects\stock-analyzer\
│
├── app/                                # Main application
│   ├── run.py                          # Entry point (desktop/web modes)
│   ├── requirements.txt                # Python dependencies
│   ├── StockAnalyzer.spec              # PyInstaller build spec
│   ├── settings.json                   # User settings (auto-created)
│   │
│   ├── backend/
│   │   ├── server.py                   # Flask app factory
│   │   ├── __init__.py
│   │   │
│   │   ├── api/                        # API blueprint modules
│   │   │   ├── __init__.py
│   │   │   ├── analysis.py             # SSE analysis pipeline
│   │   │   ├── backtest.py             # NEW — historical signal replay (POST /api/backtest/run)
│   │   │   ├── export.py               # Excel generation + charts (43-column dashboard)
│   │   │   ├── assets.py               # Asset search + presets
│   │   │   ├── browse.py               # File browser API
│   │   │   ├── settings.py             # Settings persistence
│   │   │   └── trading.py              # Paper trading + guarded execution API
│   │   │
│   │   └── services/                   # Business logic
│   │       ├── __init__.py
│   │       ├── market_data.py          # yfinance + indicators + RS 55D + Elder Impulse + checklist
│   │       ├── scoring.py              # v4.2 scoring engine — RS composite, analyst adj, MA200 soft band,
│   │       │                           #   consolidation exemption, generalization guards
│   │       ├── sentiment.py            # News scraping + LLM classification
│   │       ├── pit_data.py             # Point-in-time snapshot store for market/fund/sent data
│   │       ├── ml_engine.py            # ML trading engine: purged splits, PIT-aware training, registry readiness
│   │       ├── backtest.py             # Strategy backtester: next-bar execution, partial reduce, costs
│   │       ├── live_risk.py            # Live/paper pre-trade risk controls and soak tracking
│   │       ├── paper_trading.py        # Persistent paper ledger with daily PnL snapshots
│   │       ├── execution.py            # Guarded execution service and adapters
│   │       └── portfolio_registry.py   # Portfolio-level admission gate for registry refresh
│   │
│   ├── pit_store/                      # Append-only PIT snapshots (created at runtime)
│   ├── paper_trading/                  # Paper ledger + risk state (created at runtime)
│   ├── ml_models/                      # Cached models, registry, portfolio admission reports
│   │
│   └── frontend/
│       ├── index.html                  # Single-page app (6-step wizard: Assets→Indicators→Sentiment→Weights→Results→Backtest)
│       ├── css/
│       │   └── style.css               # Dark professional theme (includes backtest chart styles)
│       └── js/
│           └── app.js                  # Frontend logic — analysis, SSE, export, signal backtest (btRender*, Chart.js)
│
├── sentiment/                          # Standalone sentiment pipeline
│   ├── pipeline/
│   │   ├── __main__.py                 # Pipeline web UI entry point
│   │   ├── core/
│   │   │   └── orchestrator.py         # Pipeline coordinator
│   │   ├── scrapers/                   # News scrapers
│   │   │   ├── base_scraper.py         # Article dataclass + BaseScraper ABC
│   │   │   ├── us_scraper.py           # US: Yahoo Finance + Google News + Reuters
│   │   │   └── china_scraper.py        # China: Eastmoney, NetEase, Sina Finance
│   │   ├── sentiment/                  # Sentiment models
│   │   │   ├── base_sentiment.py       # Abstract interface
│   │   │   └── lora_llm_sentiment.py   # LoRA fine-tuned Qwen3-8B
│   │   ├── filters/
│   │   │   └── relevance_filter.py     # LLM-based relevance filtering
│   │   ├── prices/
│   │   │   └── price_provider.py       # Price data for correlation
│   │   ├── config/                     # Market/asset/model configs
│   │   ├── database.py                 # SQLite article storage
│   │   └── client/                     # Standalone pipeline web UI
│   ├── models/                         # LLM weights (not in git)
│   ├── tests/
│   └── requirements.txt
│
├── tracker/                            # Standalone Excel tracker
│   ├── stock_tracker.py                # Generates stock_tracker.xlsx
│   ├── sentiment_analyzer.py           # Local LLM sentiment
│   └── requirements.txt
│
├── build.py                            # PyInstaller build script
├── Makefile                            # Build shortcuts
├── README.md
├── CHANGELOG.md
├── LICENSE
├── DOCUMENTATION.md                    # This file
└── pyproject.toml
```

---

## 17. Standalone Components

### Sentiment Pipeline (`sentiment/`)

A fully modular sentiment analysis pipeline that can run independently of the main app.

**Components:**
- **Scrapers:** US (Reuters RSS, Yahoo Finance, Google News) and China (Eastmoney, NetEase, Sina Finance)
- **Filters:** LLM-based relevance filtering
- **Models:** LoRA fine-tuned Qwen3-8B with 4-bit quantization (optional, requires GPU)
- **Orchestrator:** Coordinates scraping → filtering → classification → storage
- **Database:** SQLite storage for classified articles
- **Web UI:** Standalone client for pipeline management

**Note:** The main app does NOT depend on this pipeline. It uses its own built-in scraper + cloud API approach in `services/sentiment.py`.

### Excel Tracker (`tracker/`)

A standalone script for generating formatted Excel dashboards.

```bash
cd C:\Projects\stock-analyzer\tracker
python stock_tracker.py
```

Generates `stock_tracker.xlsx` with:
- Dashboard sheet (same color scheme as the main app)
- Per-stock analysis sheets
- Hardcoded stock universe (configurable in source)

Can be scheduled via cron (Linux) or Task Scheduler (Windows) for automated daily reports.

---

## 18. Dependencies

### Core (required)

| Package | Version | Purpose |
|---------|---------|---------|
| flask | ≥ 3.0 | Web framework & API |
| yfinance | ≥ 0.2.36 | Market data (OHLCV, fundamentals) |
| pandas | ≥ 2.0 | Data manipulation |
| numpy | ≥ 1.24 | Numerical computation |
| openpyxl | ≥ 3.1 | Excel workbook generation |
| matplotlib | latest | Chart rendering (dark professional theme) |
| feedparser | ≥ 6.0 | RSS feed parsing (Google News) |
| tenacity | ≥ 8.0 | Retry logic with exponential backoff |
| requests | latest | HTTP client for LLM API calls |

### Optional

| Package | Version | Purpose |
|---------|---------|---------|
| pywebview | ≥ 5.0 | Native desktop window (fallback: browser) |

### Build-only

| Package | Version | Purpose |
|---------|---------|---------|
| pyinstaller | ≥ 6.0 | Executable packaging |
| pillow | ≥ 10.0 | Image handling (for icon) |

### Excluded from build

These are explicitly excluded from the built executable (cloud APIs replace local models):

| Package | Reason |
|---------|--------|
| torch | Local LLM no longer needed |
| transformers | Local LLM no longer needed |
| peft | LoRA adapter loading |
| bitsandbytes | 4-bit quantization |
| accelerate | Model parallelism |
| safetensors | Weight format |
| tokenizers | Tokenizer backend |

---

## 19. Troubleshooting

### "No articles found" for sentiment

- Yahoo Finance RSS may be rate-limited; wait a few minutes and retry
- Some tickers (especially non-US) may have limited news coverage
- Verify internet connectivity

### Sentiment API errors

- **401 Unauthorized** — Invalid API key; verify it's correct for the selected provider
- **429 Rate Limited** — Too many requests; reduce max articles or wait before retrying
- **Timeout** — API call exceeded 60s; try a smaller batch or faster model (e.g., `gpt-4o-mini`)

### "No data" for an asset

- yfinance may not have data for that ticker
- Check the ticker symbol is correct (e.g., `HY9H.F` for Frankfurt-listed SK Hynix)
- Some commodities require futures ticker format (e.g., `GC=F` for Gold)

### Excel charts not appearing

- Ensure `matplotlib` is installed
- Charts require at least 20 data points; very short periods may skip chart generation
- Check the log for matplotlib import errors

### PyWebView not available

If `pywebview` is not installed, the app automatically falls back to browser mode:
```
pywebview not installed — falling back to browser mode.
  pip install pywebview
```

### Build fails

- Ensure PyInstaller 6.0+ is installed: `pip install pyinstaller>=6.0`
- Run `python build.py --clean` to remove stale artifacts
- Check that `app/StockAnalyzer.spec` exists and is not corrupted

### Port already in use

The app automatically selects a free port if the default (5050) is occupied. To force a specific port:
```bash
python run.py --port 9090
```

### Settings not persisting

- In development, settings are saved to `app/settings.json`
- In the built exe, settings are saved next to `StockAnalyzer.exe`
- Check file permissions in the output directory

### Backtest chart not rendering

- Ensure Chart.js CDN scripts load (requires internet); check browser console for 404s
- If the app runs in a fully offline/air-gapped environment, download Chart.js 4.4.7 and `chartjs-plugin-annotation` 3.1.0 and serve them from `/js/` instead of CDN
- "No valid backtest data returned" — try a longer period (2Y or 5Y); a 6-month period may not provide enough bars after the 200-bar warmup
- Very low `n_points` (< 50) usually means the ticker has limited history — use a period that matches the asset's listing age

### Backtest shows unexpectedly high BUY signals for an old period

- This is the known fundamental score static bias — the fund component uses today's P/E, margins, ROE regardless of historical date. For a company that improved fundamentals dramatically over the backtest window (e.g. Micron from 2023 loss-making to 2025 supercycle), historical BUY signals will be inflated by current-day fundamentals. Use the backtest primarily to validate the technical and regime components.

---

*End of documentation.*
