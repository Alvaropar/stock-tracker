"""
Flask application factory.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from flask import Flask, send_from_directory


def _base_dir() -> Path:
    """Return the project root, works both in dev and PyInstaller."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).parent.parent


FRONTEND = _base_dir() / "frontend"


def create_app() -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["JSON_SORT_KEYS"] = False

    # ── Register API blueprints ───────────────────────────────────────────────
    from .api.assets     import bp as assets_bp
    from .api.analysis   import bp as analysis_bp
    from .api.export     import bp as export_bp
    from .api.browse     import bp as browse_bp
    from .api.settings   import bp as settings_bp
    from .api.ml         import bp as ml_bp
    from .api.trading    import bp as trading_bp
    from .api.backtest   import bp as backtest_bp
    from .api.thresholds import bp as thresholds_bp
    from .api.llm        import bp as llm_bp
    from .api.alt_data   import bp as alt_data_bp
    from .api.portfolio  import bp as portfolio_bp

    app.register_blueprint(assets_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(export_bp)
    app.register_blueprint(browse_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(ml_bp)
    app.register_blueprint(trading_bp)
    app.register_blueprint(backtest_bp)
    app.register_blueprint(thresholds_bp)
    app.register_blueprint(llm_bp)
    app.register_blueprint(alt_data_bp)
    app.register_blueprint(portfolio_bp)

    # ── Serve frontend ────────────────────────────────────────────────────────
    @app.route("/")
    def index():
        return send_from_directory(FRONTEND, "index.html")

    @app.route("/css/<path:filename>")
    def css(filename):
        return send_from_directory(FRONTEND / "css", filename)

    @app.route("/js/<path:filename>")
    def js(filename):
        return send_from_directory(FRONTEND / "js", filename)

    @app.route("/assets/<path:filename>")
    def static_assets(filename):
        return send_from_directory(FRONTEND / "assets", filename)

    @app.route("/ml-dashboard")
    def ml_dashboard():
        return send_from_directory(FRONTEND, "ml-dashboard.html")

    # ── Security headers ──────────────────────────────────────────────────────
    @app.after_request
    def _headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"]        = "SAMEORIGIN"
        return response

    return app
