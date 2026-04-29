"""Centralized logging configuration."""
from __future__ import annotations

import logging
import os
import sys


def configure_logging(level: str | None = None) -> None:
    """Configure application-wide logging.

    Reads LOG_LEVEL from the environment if *level* is not provided.
    Noisy third-party libraries are quieted to WARNING.
    """
    level = level or os.getenv("LOG_LEVEL", "INFO")
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(handler)
    root.setLevel(numeric_level)

    for noisy in ("urllib3", "yfinance", "peewee", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
