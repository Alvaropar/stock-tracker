"""
Centralised logging configuration for the pipeline.

Call ``setup_logging()`` once at application startup.
All modules import their logger with::

    import logging
    log = logging.getLogger(__name__)
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path


# Default log directory — next to the project root
_PROJECT_ROOT = Path(
    os.environ.get("PIPELINE_PROJECT_ROOT", "")
) if os.environ.get("PIPELINE_PROJECT_ROOT") else Path(__file__).resolve().parent.parent
_LOG_DIR = _PROJECT_ROOT / "logs"


class _JsonishFormatter(logging.Formatter):
    """
    Compact structured formatter that outputs key=value pairs.

    Not full JSON (avoids import overhead on every log line), but
    trivially parseable by log aggregators.
    """

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        msg = record.getMessage()
        base = f"{ts} [{record.levelname:<7}] {record.name}: {msg}"
        if record.exc_info and record.exc_info[1]:
            base += f"\n{self.formatException(record.exc_info)}"
        return base


def setup_logging(
    level: str = "INFO",
    log_to_file: bool = True,
    log_dir: Path | None = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB per file
    backup_count: int = 5,
) -> None:
    """
    Configure logging for the entire pipeline.

    - Console handler: always enabled, INFO level
    - File handler: rotating file in logs/ directory (optional)

    Call once at startup (desktop_app.py or local_app.py).
    """
    root = logging.getLogger("pipeline")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Don't add handlers if already configured
    if root.handlers:
        return

    formatter = _JsonishFormatter()

    # Console handler
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    root.addHandler(console)

    # File handler (rotating)
    if log_to_file:
        dest = log_dir or _LOG_DIR
        dest.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            dest / "pipeline.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        root.addHandler(fh)

    # Suppress noisy third-party loggers
    for name in ("urllib3", "requests", "werkzeug", "feedparser"):
        logging.getLogger(name).setLevel(logging.WARNING)
