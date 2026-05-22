"""Central application configuration loaded from environment variables."""
from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent


class Config:
    # LLM / cloud sentiment API
    API_KEY: str = os.getenv("API_KEY", "")
    API_URL: str = os.getenv("API_URL", "")
    MODEL: str = os.getenv("MODEL", "")

    # Market data fallback
    POLYGON_API_KEY: str = os.getenv("POLYGON_API_KEY", "")

    # Portfolio
    BASE_CURRENCY: str = os.getenv("BASE_CURRENCY", "USD").upper()
    BENCHMARK_TICKER: str = os.getenv("BENCHMARK_TICKER", "SPY").upper()

    # Flask
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "9000"))

    # Paths (resolved at import time)
    DATA_DIR: Path = ROOT_DIR / "data"
    MODELS_DIR: Path = ROOT_DIR / "sentiment" / "models"
    FRONTEND_DIR: Path = ROOT_DIR / "frontend"

    @classmethod
    def llm_available(cls) -> bool:
        """True when a cloud LLM provider is configured. When false, callers
        should skip the sentiment component rather than fall back to a partial
        result."""
        return bool(cls.API_KEY and cls.API_URL)


config = Config()
