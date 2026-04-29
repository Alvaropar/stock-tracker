"""
File browser API for selecting model directories.

GET /api/browse?path=/some/dir  -> list directory contents
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Blueprint, jsonify, request

bp = Blueprint("browse", __name__, url_prefix="/api/browse")
log = logging.getLogger("app.browse")

# ── Configuration ────────────────────────────────────────────────────────────
import sys as _sys
if getattr(_sys, "frozen", False):
    PROJECT_ROOT = Path(_sys.executable).resolve().parent
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]  # api/ -> stock_analyzer/ -> repo root

# Allowed root directories for the file browser (sandboxing)
_BROWSE_ROOTS: List[Path] = []


def _init_browse_roots() -> None:
    """Build the allowed root list once."""
    global _BROWSE_ROOTS
    if _BROWSE_ROOTS:
        return
    candidates = [
        PROJECT_ROOT / "models",
        PROJECT_ROOT,
        Path.home(),
    ]
    # On Windows, add common drive roots
    import sys
    if sys.platform == "win32":
        for letter in "CDEFGH":
            candidates.append(Path(f"{letter}:\\"))
    _BROWSE_ROOTS = [p for p in candidates if p.exists()]


def _is_path_allowed(p: Path) -> bool:
    """Check whether *p* falls under an allowed browse root."""
    _init_browse_roots()
    resolved = p.resolve()
    for root in _BROWSE_ROOTS:
        try:
            resolved.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def browse_directory(path: Optional[str] = None) -> Dict:
    """List contents of a directory for the model file browser.

    Returns ``{current: str, parent: str|None, entries: [...]}``.
    Each entry: ``{name, path, is_dir, is_model, is_adapter}``.

    Paths are sandboxed to allowed roots (project dir, home, drives).
    """
    if not path:
        # Try the project's models/ dir first, fall back to user home
        models_dir = PROJECT_ROOT / "models"
        if models_dir.exists():
            path = str(models_dir)
        else:
            path = str(Path.home())

    p = Path(path).resolve()

    # Sandbox check — block paths outside allowed roots
    if not _is_path_allowed(p):
        log.warning("Browse blocked for path outside allowed roots: %s", p)
        p = PROJECT_ROOT / "models" if (PROJECT_ROOT / "models").exists() else Path.home()

    if not p.exists() or not p.is_dir():
        # Fall back to parent, then home, then drives
        for fallback in [p.parent, Path.home(), Path("C:\\")]:
            if fallback.exists():
                p = fallback
                break

    entries: List[Dict[str, Any]] = []
    try:
        for item in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            if item.name.startswith("."):
                continue
            # Skip system/hidden directories
            if item.name in ("$Recycle.Bin", "System Volume Information",
                             "Windows", "ProgramData"):
                continue
            entry: Dict[str, Any] = {
                "name": item.name,
                "path": str(item),
                "is_dir": item.is_dir(),
                "is_model": False,
                "is_adapter": False,
            }
            if item.is_dir():
                entry["is_model"] = (item / "config.json").exists()
                entry["is_adapter"] = (item / "adapter_config.json").exists()
            entries.append(entry)
    except PermissionError:
        return {"current": str(p), "parent": str(p.parent), "entries": [],
                "error": "Permission denied"}

    return {
        "current": str(p),
        "parent": str(p.parent) if p.parent != p else None,
        "entries": entries,
    }


@bp.route("", methods=["GET"])
def browse():
    """List directory contents for model file browser."""
    path = request.args.get("path", "")
    result = browse_directory(path if path else None)
    return jsonify(result)
