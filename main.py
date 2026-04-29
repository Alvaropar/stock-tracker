#!/usr/bin/env python3
"""
Stock Analysis Platform – entry point.

Modes:
    python main.py              →  native desktop window (default)
    python main.py --web        →  open in browser instead
    python main.py --web --no-browser  →  server only, no auto-open
    python main.py --port 8080  →  custom port
"""
import argparse
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

from stock_analyzer.logging_config import configure_logging

configure_logging()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _free_port(preferred: int = 9000) -> int:
    """Return *preferred* if available, otherwise pick a random free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]


def _port_in_use(port: int) -> bool:
    """Return True if something is already bound to *port*."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            return False
        except OSError:
            return True


def _pids_on_port(port: int):
    """Return list of PIDs listening on *port* using whatever is available."""
    import subprocess as _sp
    pids = []

    # 1. ss (iproute2, standard on modern Linux)
    try:
        out = _sp.run(
            ["ss", "-lptn", f"sport = :{port}"],
            capture_output=True, text=True, timeout=3,
        ).stdout
        # lines look like:  LISTEN  0  128  0.0.0.0:9000  ...  users:(("python",pid=12345,...))
        import re
        for m in re.finditer(r'pid=(\d+)', out):
            pids.append(int(m.group(1)))
        if pids:
            return pids
    except (FileNotFoundError, Exception):
        pass

    # 2. /proc/net/tcp — pure Python, always available on Linux
    try:
        hex_port = f"{port:04X}"
        with open("/proc/net/tcp") as f:
            for line in f:
                parts = line.split()
                # local_address field: "0100007F:2328" (hex ip:port)
                if len(parts) > 9 and parts[3] == "0A":   # 0A = LISTEN
                    if parts[1].split(":")[1].upper() == hex_port:
                        inode = int(parts[9])
                        # Match inode to a PID via /proc/<pid>/fd
                        import glob
                        for fdlink in glob.glob("/proc/*/fd/*"):
                            try:
                                if f"socket:[{inode}]" == os.readlink(fdlink):
                                    pids.append(int(fdlink.split("/")[2]))
                            except (OSError, ValueError):
                                pass
        if pids:
            return list(set(pids))
    except Exception:
        pass

    # 3. lsof (macOS / some Linux)
    try:
        out = _sp.run(
            ["lsof", "-ti", f"tcp:{port}"],
            capture_output=True, text=True, timeout=3,
        ).stdout
        for pid_str in out.strip().split():
            try:
                pids.append(int(pid_str))
            except ValueError:
                pass
        if pids:
            return pids
    except FileNotFoundError:
        pass

    # 4. fuser (procps / util-linux)
    try:
        out = _sp.run(
            ["fuser", f"{port}/tcp"],
            capture_output=True, text=True, timeout=3,
        ).stdout
        for pid_str in out.split():
            try:
                pids.append(int(pid_str))
            except ValueError:
                pass
    except FileNotFoundError:
        pass

    return pids


def _release_port(port: int) -> None:
    """Kill any process holding *port* and wait until it's actually free."""
    if not _port_in_use(port):
        return

    import signal as _signal

    pids = _pids_on_port(port)
    if not pids:
        print(f"Warning: port {port} in use but could not identify the PID", flush=True)
        return

    for pid in pids:
        try:
            os.kill(pid, _signal.SIGTERM)
            print(f"  Stopped existing server (PID {pid}) on port {port}", flush=True)
        except (ProcessLookupError, PermissionError):
            pass

    # Wait up to 5 s for the port to actually become free
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not _port_in_use(port):
            return
        time.sleep(0.2)

    print(f"Warning: port {port} may still be in use after releasing", flush=True)


def _wait_for_server(port: int, timeout: float = 15.0):
    """Block until Flask is accepting connections (or timeout)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.15)
    return False


# ── Loading screen shown while Flask boots ───────────────────────────────────

_LOADING_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    background: linear-gradient(135deg, #0a0e27 0%, #1a1a3e 40%, #0d1b2a 100%);
    color: #e0e0e0;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100vh;
    overflow: hidden;
  }
  .container {
    text-align: center;
    animation: fadeIn 0.6s ease;
  }
  @keyframes fadeIn { from { opacity:0; transform:translateY(20px); } to { opacity:1; transform:translateY(0); } }

  .icon {
    width: 80px; height: 80px; margin: 0 auto 28px;
    border-radius: 18px;
    background: linear-gradient(135deg, #4fc3f7 0%, #1976d2 100%);
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 8px 32px rgba(79,195,247,0.3);
  }
  .icon svg { width:44px; height:44px; fill: white; }

  h1 {
    font-size: 26px; font-weight: 600; letter-spacing: -0.5px;
    margin-bottom: 8px;
    background: linear-gradient(90deg, #4fc3f7, #81d4fa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .sub { color: #7a8ba6; font-size: 13px; margin-bottom: 36px; }

  .spinner-track {
    width: 220px; height: 3px; background: rgba(255,255,255,0.08);
    border-radius: 3px; margin: 0 auto; overflow: hidden;
  }
  .spinner-bar {
    width: 40%; height: 100%; border-radius: 3px;
    background: linear-gradient(90deg, #4fc3f7, #1976d2);
    animation: slide 1.4s ease-in-out infinite;
  }
  @keyframes slide { 0%{transform:translateX(-100%)} 50%{transform:translateX(150%)} 100%{transform:translateX(-100%)} }

  .status { color: #546e7a; font-size: 11px; margin-top: 18px; }
</style>
</head>
<body>
  <div class="container">
    <div class="icon">
      <svg viewBox="0 0 24 24"><path d="M3 13h2v8H3zm4-4h2v12H7zm4-4h2v16h-2zm4 8h2v8h-2zm4-6h2v14h-2z"/></svg>
    </div>
    <h1>Stock Analysis Platform</h1>
    <p class="sub">Initialising analysis engine&hellip;</p>
    <div class="spinner-track"><div class="spinner-bar"></div></div>
    <p class="status">Starting local server</p>
  </div>
</body>
</html>
"""


# ── Desktop mode (native window) ────────────────────────────────────────────

class _JsApi:
    """Python↔JS bridge exposed to the webview frontend via `window.pywebview.api`."""

    def __init__(self, wv_window):
        self._window = wv_window

    def save_file_dialog(self, default_name: str = "stock_analysis.xlsx"):
        """
        Open a native "Save As" dialog. Returns the chosen file path or None.
        Called from JS:  await window.pywebview.api.save_file_dialog("name.xlsx")
        """
        import webview
        result = self._window.create_file_dialog(
            webview.SAVE_DIALOG,
            save_filename=default_name,
            file_types=("Excel Files (*.xlsx)",),
        )
        if result:
            # pywebview returns a tuple/string depending on platform
            return result if isinstance(result, str) else result[0]
        return None

    def choose_folder(self):
        """
        Open a native folder picker. Returns the chosen folder path or None.
        Called from JS:  await window.pywebview.api.choose_folder()
        """
        import webview
        result = self._window.create_file_dialog(webview.FOLDER_DIALOG)
        if result:
            return result if isinstance(result, str) else result[0]
        return None


def _run_desktop(port: int):
    """Launch Flask in a background thread and display a native window."""
    import webview

    from stock_analyzer.app import create_app
    flask_app = create_app()

    url = f"http://127.0.0.1:{port}"

    # Placeholder — _JsApi needs the window ref, set after creation
    js_api = _JsApi.__new__(_JsApi)

    window = webview.create_window(
        "Stock Analysis Platform",
        html=_LOADING_HTML,
        js_api=js_api,
        width=1360,
        height=880,
        resizable=True,
        min_size=(900, 600),
        text_select=True,
    )
    # Now that the window exists, finish initialising the API
    js_api._window = window

    def _start_flask():
        flask_app.run(
            host="127.0.0.1",
            port=port,
            debug=False,
            use_reloader=False,
            threaded=True,
        )

    def _on_ready():
        """Called once the native window is up — start Flask, then navigate."""
        flask_thread = threading.Thread(target=_start_flask, daemon=True)
        flask_thread.start()

        if _wait_for_server(port):
            window.load_url(url)
        else:
            window.load_html(
                "<html><body style='background:#1a1a2e;color:#ef5350;"
                "display:flex;align-items:center;justify-content:center;"
                "height:100vh;font-family:sans-serif'>"
                "<h2>Failed to start server. Please restart the application.</h2>"
                "</body></html>"
            )

    webview.start(_on_ready, debug=False)


# ── Web / browser mode ──────────────────────────────────────────────────────

def _run_web(port: int, *, open_browser: bool = True, debug: bool = False):
    """Classic Flask server that optionally opens the default browser."""
    from stock_analyzer.app import create_app
    flask_app = create_app()

    url = f"http://127.0.0.1:{port}"

    if open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    print(f"\n  Stock Analysis Platform  ->  {url}\n")
    flask_app.run(
        host="0.0.0.0",
        port=port,
        debug=debug,
        use_reloader=False,
        threaded=True,
    )


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stock Analysis Platform")
    parser.add_argument("--port", type=int, default=9000,
                        help="HTTP port (default 9000)")
    parser.add_argument("--web", action="store_true",
                        help="Open in browser instead of native window")
    parser.add_argument("--no-browser", action="store_true",
                        help="(web mode) don't auto-open browser")
    parser.add_argument("--debug", action="store_true",
                        help="Enable Flask debug mode (web only)")
    args = parser.parse_args()

    port = args.port
    _release_port(port)

    if args.web:
        _run_web(port, open_browser=not args.no_browser, debug=args.debug)
    else:
        # Default: native desktop window
        try:
            import webview  # noqa: F401

            # Detect if a GUI toolkit is actually available
            from webview.guilib import initialize as _wv_init
            _wv_init(None)

        except Exception as e:
            print(f"Desktop mode unavailable ({e}) — falling back to browser mode.\n")
            _run_web(port, open_browser=False, debug=args.debug)
            return

        _run_desktop(port)


if __name__ == "__main__":
    main()
