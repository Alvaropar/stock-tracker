"""
Build script for the Commodity Trading Sentiment Analyzer.

Produces a standalone Windows executable using PyInstaller.

Usage::

    python build.py              # Build the .exe
    python build.py --onefile    # Single-file executable (slower startup)
    python build.py --clean      # Clean build artifacts first

The resulting executable will be in ``dist/SentimentAnalyzer/``.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST_DIR = ROOT / "dist"
BUILD_DIR = ROOT / "build"
APP_NAME = "SentimentAnalyzer"
ENTRY_POINT = ROOT / "pipeline" / "desktop_app.py"
ICON_PATH = ROOT / "assets" / "app.ico"


def clean():
    """Remove previous build artifacts."""
    for d in [DIST_DIR, BUILD_DIR]:
        if d.exists():
            print(f"Removing {d}")
            shutil.rmtree(d)
    spec = ROOT / f"{APP_NAME}.spec"
    if spec.exists():
        spec.unlink()


def _add_package_data(cmd: list, package_name: str) -> None:
    """Add all non-Python data files from an installed package to PyInstaller."""
    import importlib
    try:
        mod = importlib.import_module(package_name)
    except ImportError:
        return
    pkg_dir = Path(mod.__file__).parent
    for root, _, files in os.walk(pkg_dir):
        for f in files:
            if f.endswith((".json", ".csv", ".txt", ".dat", ".proto", ".yaml", ".yml")):
                full = Path(root) / f
                # Destination preserves the package's internal structure
                dest = str(full.parent.relative_to(pkg_dir.parent))
                cmd.extend(["--add-data", f"{full}{os.pathsep}{dest}"])


def build(onefile: bool = False):
    """Run PyInstaller with the appropriate settings."""
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", APP_NAME,
        "--noconfirm",
        "--windowed",            # No console window
        "--log-level", "WARN",
        "--specpath", str(ROOT),
        "--distpath", str(DIST_DIR),
        "--workpath", str(BUILD_DIR),
    ]

    if onefile:
        cmd.append("--onefile")
    else:
        cmd.append("--onedir")

    # Icon
    if ICON_PATH.exists():
        cmd.extend(["--icon", str(ICON_PATH)])

    # ── Hidden imports that PyInstaller misses ──
    hidden = [
        "pipeline",
        "pipeline.client",
        "pipeline.client._state",
        "pipeline.client.local_app",
        "pipeline.config",
        "pipeline.config.assets",
        "pipeline.config.markets",
        "pipeline.config.models",
        "pipeline.core",
        "pipeline.core.orchestrator",
        "pipeline.filters",
        "pipeline.filters.relevance_filter",
        "pipeline.scrapers",
        "pipeline.scrapers.base_scraper",
        "pipeline.scrapers.us_scraper",
        "pipeline.scrapers.china_scraper",
        "pipeline.sentiment",
        "pipeline.sentiment.base_sentiment",
        "pipeline.sentiment.lora_llm_sentiment",
        "pipeline.desktop_app",
        "pipeline.prices",
        "pipeline.prices.price_provider",
        "flask",
        "jinja2",
        "markupsafe",
        "webview",
    ]
    for h in hidden:
        cmd.extend(["--hidden-import", h])

    # ── Data files to include ──
    # Include the pipeline package source (for templates, etc.)
    cmd.extend(["--add-data", f"{ROOT / 'pipeline'}{os.pathsep}pipeline"])

    # Include data files from third-party packages that have non-Python assets
    _add_package_data(cmd, "akshare")
    _add_package_data(cmd, "yfinance")

    # ── Exclude heavy packages that are loaded at runtime ──
    # These are imported dynamically and should be in the user's env
    excludes = [
        "matplotlib", "tkinter", "test", "unittest",
        "IPython", "notebook", "pytest", "scipy",
        "bitsandbytes",  # CUDA libs not available, causes warnings
        "triton",  # Not available on Windows
    ]
    for e in excludes:
        cmd.extend(["--exclude-module", e])

    cmd.append(str(ENTRY_POINT))

    print(f"\n{'='*60}")
    print(f"Building {APP_NAME}")
    print(f"Mode: {'single file' if onefile else 'directory'}")
    print(f"Entry: {ENTRY_POINT}")
    print(f"{'='*60}\n")

    subprocess.run(cmd, check=True)

    # Summary
    if onefile:
        exe = DIST_DIR / f"{APP_NAME}.exe"
    else:
        exe = DIST_DIR / APP_NAME / f"{APP_NAME}.exe"

    if exe.exists():
        size_mb = exe.stat().st_size / (1024 * 1024)
        print(f"\n{'='*60}")
        print(f"BUILD SUCCESSFUL")
        print(f"Executable: {exe}")
        print(f"Size: {size_mb:.1f} MB")
        print(f"{'='*60}")
    else:
        print(f"\nWARNING: Expected executable not found at {exe}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Sentiment Analyzer executable")
    parser.add_argument("--onefile", action="store_true",
                        help="Create a single-file executable")
    parser.add_argument("--clean", action="store_true",
                        help="Clean build artifacts before building")
    parser.add_argument("--clean-only", action="store_true",
                        help="Only clean, don't build")
    args = parser.parse_args()

    if args.clean or args.clean_only:
        clean()
    if not args.clean_only:
        build(onefile=args.onefile)
