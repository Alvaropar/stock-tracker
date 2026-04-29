"""
Desktop application for the Commodity Trading Sentiment Analyzer.

Uses pywebview to create a native window with the Flask-based UI.
The Flask server runs on a background thread; pywebview renders the
frontend in a native webview (Edge/Chromium on Windows).

Launch directly::

    python -m pipeline.desktop_app

Or as an executable (after building with PyInstaller)::

    dist/SentimentAnalyzer.exe
"""
from __future__ import annotations

import atexit
import ctypes
import logging
import os
import signal
import socket
import sys
import threading
from pathlib import Path

# ── Windows console encoding fix ──────────────────────────────────────────────
if sys.platform == "win32":
    try:
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
    except Exception:
        pass

# ── Resolve frozen vs. development paths ──────────────────────────────────────
if getattr(sys, "frozen", False):
    # Running as PyInstaller bundle
    _BASE_DIR = Path(sys._MEIPASS)
    _PROJECT_ROOT = Path(sys.executable).parent
else:
    _BASE_DIR = Path(__file__).resolve().parent
    _PROJECT_ROOT = _BASE_DIR.parent

os.environ.setdefault("PIPELINE_PROJECT_ROOT", str(_PROJECT_ROOT))

# ── Structured logging ────────────────────────────────────────────────────────
from pipeline.logging_config import setup_logging
setup_logging(level="INFO", log_to_file=True)
log = logging.getLogger("pipeline.desktop")


def _find_free_port(start: int = 5000, end: int = 5100) -> int:
    """Find an available port in the given range."""
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in range {start}-{end}")


def _start_flask(port: int) -> None:
    """Start the Flask server in the current thread (blocking)."""
    from pipeline.client.local_app import _create_app

    app = _create_app()
    # Suppress Flask request logs in production
    flask_log = logging.getLogger("werkzeug")
    flask_log.setLevel(logging.WARNING)

    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)


def _wait_for_server(port: int, timeout: float = 30.0) -> bool:
    """Block until the Flask server is accepting connections."""
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.15)
    return False


def main():
    try:
        import webview
    except ImportError:
        print("pywebview is required. Install with: pip install pywebview")
        sys.exit(1)

    port = _find_free_port()
    url = f"http://127.0.0.1:{port}"

    log.info("Starting Flask server on port %d ...", port)

    # Start Flask in a daemon thread
    server_thread = threading.Thread(
        target=_start_flask, args=(port,), daemon=True
    )
    server_thread.start()

    # Wait for server to be ready
    if not _wait_for_server(port):
        log.error("Flask server failed to start within 30s")
        sys.exit(1)

    log.info("Server ready. Opening application window...")

    # ── Create native window ──────────────────────────────────────────────
    window = webview.create_window(
        title="Commodity Sentiment Analyzer",
        url=url,
        width=1120,
        height=820,
        min_size=(800, 600),
        resizable=True,
        text_select=True,
        confirm_close=False,
    )

    # Start the webview event loop (blocks until window is closed)
    webview.start(
        debug=("--debug" in sys.argv),
        gui="edgechromium",  # Best renderer on Windows
    )

    log.info("Window closed. Shutting down.")
    # Force-kill to ensure background threads (Flask, LLM) die
    os._exit(0)


if __name__ == "__main__":
    main()
