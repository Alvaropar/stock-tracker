"""
User settings persistence API.

GET  /api/settings              → load all settings
POST /api/settings              → save settings (merge)
GET  /api/settings/llm-status   → whether sentiment / LLM features can run

Settings are stored in a JSON file next to the executable (or in the app/
directory during development). This is more reliable than localStorage because
it works across ports, webview resets, and browser/desktop mode switches.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

from flask import Blueprint, jsonify, request

from ..config import config

bp = Blueprint("settings", __name__, url_prefix="/api/settings")
log = logging.getLogger("app.settings")


def _settings_path() -> Path:
    """Return the path to the settings JSON file."""
    if getattr(sys, "frozen", False):
        # Next to the .exe
        return Path(sys.executable).resolve().parent / "settings.json"
    else:
        # Project root during development
        return Path(__file__).resolve().parents[2] / "settings.json"


def load_settings() -> Dict[str, Any]:
    p = _settings_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("Failed to read settings: %s", e)
    return {}


def save_settings(data: Dict[str, Any]):
    p = _settings_path()
    current = load_settings()
    current.update(data)
    try:
        p.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        log.error("Failed to write settings: %s", e)
        raise


def _detect_model_paths() -> Dict[str, str]:
    """Auto-detect local model and adapter paths in the project tree."""
    # Walk up from this file to find the repo root's sentiment/models/
    repo_root = Path(__file__).resolve().parents[2]  # api/ -> stock_analyzer/ -> repo root
    models_dir = repo_root / "sentiment" / "models"
    if not models_dir.is_dir():
        return {}

    result: Dict[str, str] = {}
    for child in sorted(models_dir.iterdir()):
        if not child.is_dir():
            continue
        # Check if it's a base model (has config.json + safetensors/bin)
        if (child / "config.json").exists():
            has_weights = any(child.glob("*.safetensors")) or any(child.glob("*.bin"))
            if has_weights and "model_path" not in result:
                result["model_path"] = str(child)
        # Check if it's a LoRA adapter (has adapter_config.json)
        if (child / "adapter_config.json").exists():
            if "adapter_path" not in result:
                result["adapter_path"] = str(child)

    return result


@bp.route("", methods=["GET"])
def get_settings():
    settings = load_settings()
    # Inject detected model paths as defaults if not already saved
    if not settings.get("model_path") or not settings.get("adapter_path"):
        defaults = _detect_model_paths()
        if defaults.get("model_path") and not settings.get("model_path"):
            settings.setdefault("model_path", defaults["model_path"])
        if defaults.get("adapter_path") and not settings.get("adapter_path"):
            settings.setdefault("adapter_path", defaults["adapter_path"])
    return jsonify(settings)


@bp.route("", methods=["POST"])
def post_settings():
    body = request.get_json(force=True, silent=True) or {}
    try:
        save_settings(body)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/llm-status", methods=["GET"])
def llm_status():
    """
    Report whether sentiment / LLM features can run on this install.

    A cloud LLM is available when `API_KEY` and `API_URL` are set in `.env`.
    A local LLM is available when a model directory under `sentiment/models/`
    holds a `config.json` plus safetensors/bin weights.

    The frontend uses this to disable the sentiment toggle (and skip the
    sentiment component in the scoring engine) when nothing is configured.
    """
    cloud = config.llm_available()
    local_paths = _detect_model_paths()
    local = bool(local_paths.get("model_path"))
    return jsonify({
        "available":       cloud or local,
        "cloud_available": cloud,
        "local_available": local,
        "model_path":      local_paths.get("model_path", ""),
        "adapter_path":    local_paths.get("adapter_path", ""),
    })


@bp.route("/clear-cache", methods=["POST"])
def clear_cache():
    """Delete the news sentiment cache (news_cache.db)."""
    import sys as _sys
    if getattr(_sys, "frozen", False):
        db = Path(_sys.executable).resolve().parent / "news_cache.db"
    else:
        db = Path(__file__).resolve().parents[2] / "news_cache.db"
    try:
        if db.exists():
            db.unlink()
            return jsonify({"ok": True, "msg": "Cache cleared"})
        return jsonify({"ok": True, "msg": "No cache to clear"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
