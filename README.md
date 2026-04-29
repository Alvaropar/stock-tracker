# Stock Analyzer

A comprehensive financial analysis platform combining **technical analysis**, **fundamental metrics**, **LLM-based sentiment analysis**, and a **historical signal backtest** for stocks and commodities.

## Quick Start

```bash
# Install dependencies
pip install -e .

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys

# Run (desktop window by default)
python main.py

# Or run in browser mode
python main.py --web
```

### Optional dependency groups

```bash
pip install -e ".[ml]"       # Local LLM sentiment (torch, transformers, peft)
pip install -e ".[desktop]"  # Native window (pywebview)
pip install -e ".[dev]"      # Tests and linting (pytest, ruff)
```

### Sentiment Pipeline (standalone)

```bash
cd sentiment
pip install -r requirements.txt
python -m pipeline
```

## Features

- **Real-time analysis** via Server-Sent Events (SSE) — no polling
- **Technical indicators**: MA20/50/200, RSI, MACD, Bollinger Bands, ATR, ADX, volume ratio, Elder Impulse
- **Relative Strength vs SPY**: RS 1M (21-day), RS 55D (55-day IBD-inspired), RS 3M (63-day)
- **Fundamental metrics**: P/E, PEG, P/B, margins, ROE, growth, analyst consensus, debt ratios, target price gap
- **Sentiment analysis**: Cloud LLMs (Claude, ChatGPT, Gemini, Grok) or local Qwen3-8B with LoRA
- **Multi-region**: US markets (Yahoo Finance, Google News) and China markets (Eastmoney, NetEase, Sina)
- **Quantitative scoring engine**: Momentum score, risk score, dip/oversold detection, analyst adjustment, contextual signal decision tree
- **Signal Backtest**: Replay the full scoring pipeline on historical price data with per-indicator IC analysis and walk-forward validation
- **Excel export**: 43-column dashboard with candlestick charts, Fundamentals, and Sentiment sheets
- **ML Lab**: Per-ticker or universe-mode classifier (LightGBM / MLP), walk-forward training, regime detection
- **Paper trading**: Persistent paper ledger with broker-like fills, live-risk pre-checks, and guarded execution adapter

## Requirements

- Python ≥ 3.10
- Internet connection (for yfinance market data and cloud sentiment APIs)
- GPU recommended for local LLM sentiment (CPU fallback available)
- CUDA-compatible GPU with ≥ 8 GB VRAM for Qwen3-8B in 4-bit mode if want to use local LLM
- LLM api if want to use LLM agent through api

## Models

Place model files under `sentiment/models/`:

```
sentiment/models/
├── Qwen3-8B/       # Base 8-billion-parameter LLM (HuggingFace format)
└── FinQwen3-8B/    # Fine-tuned LoRA adapter
```

The models directory is excluded from version control (see `.gitignore`).

## Project Structure

```
stock-tracker/
├── stock_analyzer/         # Main Python package
│   ├── app.py              # Flask application factory
│   ├── config.py           # Central configuration (env vars)
│   ├── exceptions.py       # Custom exception hierarchy
│   ├── logging_config.py   # Structured logging setup
│   ├── api/                # REST endpoints + SSE streaming (12 blueprints)
│   └── services/           # Market data, scoring, ML, trading (16 modules)
├── frontend/               # SPA (HTML / CSS / JS)
├── sentiment/              # Standalone sentiment analysis pipeline
├── tests/
│   ├── conftest.py
│   ├── unit/
│   └── integration/
├── scripts/
│   └── build.py            # PyInstaller executable build
├── data/                   # Runtime data (gitignored)
├── docs/                   # Technical documentation
├── main.py                 # Entry point
├── pyproject.toml          # Project config & dependencies
├── Makefile                # Development commands
└── .env.example            # Environment variable template
```

## Development

```bash
make install-dev   # Install all optional groups
make run-dev       # Flask dev server with debug mode
make test          # Run all tests
make lint          # Ruff lint check
make build         # PyInstaller executable
```

## License

See [LICENSE](LICENSE).
