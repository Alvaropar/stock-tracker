#!/usr/bin/env python3
"""
Build script for Stock Analysis Platform executable.

Creates a distributable folder at  dist/StockAnalyzer/  containing:
    StockAnalyzer.exe   – native desktop application (no browser needed)

Usage:
    python scripts/build.py                  # standard build (windowed app)
    python scripts/build.py --onefile        # single-file exe (slower startup)
    python scripts/build.py --include-ml     # also bundle torch/transformers (large!)
    python scripts/build.py --clean          # clean previous build artifacts first
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
DIST_DIR  = ROOT / "dist"
BUILD_DIR = ROOT / "build"
FRONTEND  = ROOT / "frontend"


def clean():
    """Remove previous build artifacts."""
    for d in [DIST_DIR, BUILD_DIR]:
        if d.exists():
            print(f"  Removing {d}")
            shutil.rmtree(d)
    print("  Clean done.\n")


def install_deps():
    """Ensure build dependencies are installed."""
    deps = ["pyinstaller>=6.0", "pillow>=10.0", "pywebview>=5.0"]
    for dep in deps:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", dep, "-q"],
            stdout=subprocess.DEVNULL,
        )


def build_exe(*, onefile: bool = False, include_ml: bool = False):
    """Run PyInstaller to create the executable."""
    if include_ml:
        os.environ["SA_INCLUDE_ML"] = "1"

    cmd = [sys.executable, "-m", "PyInstaller"]

    common_flags = [
        "--windowed",
        "--name", "StockAnalyzer",
        "--add-data", f"{FRONTEND / 'index.html'}{os.pathsep}frontend",
        "--add-data", f"{FRONTEND / 'css'}{os.pathsep}{str(Path('frontend') / 'css')}",
        "--add-data", f"{FRONTEND / 'js'}{os.pathsep}{str(Path('frontend') / 'js')}",
        "--hidden-import", "flask",
        "--hidden-import", "webview",
        "--hidden-import", "yfinance",
        "--hidden-import", "openpyxl",
        "--hidden-import", "feedparser",
        "--hidden-import", "matplotlib",
        "--hidden-import", "mplfinance",
        "--hidden-import", "stock_analyzer.app",
        "--hidden-import", "stock_analyzer.api.analysis",
        "--hidden-import", "stock_analyzer.api.assets",
        "--hidden-import", "stock_analyzer.api.export",
        "--hidden-import", "stock_analyzer.api.browse",
        "--hidden-import", "stock_analyzer.api.settings",
        "--hidden-import", "stock_analyzer.services.market_data",
        "--hidden-import", "stock_analyzer.services.scoring",
        "--hidden-import", "stock_analyzer.services.sentiment",
    ]

    for excl in [
        "scipy", "pyarrow", "statsmodels", "py_mini_racer",
        "sklearn", "scikit_learn", "sympy",
        "IPython", "notebook", "jupyter",
        "pytest", "setuptools", "pip", "wheel",
        "fontTools", "pythonwin", "pywin32",
    ]:
        common_flags += ["--exclude-module", excl]

    if not include_ml:
        for excl in ["torch", "transformers", "peft", "bitsandbytes", "accelerate", "safetensors", "tokenizers"]:
            common_flags += ["--exclude-module", excl]
    else:
        common_flags += [
            "--hidden-import", "torch",
            "--hidden-import", "transformers",
            "--hidden-import", "peft",
        ]

    ico = ROOT / "sentiment" / "assets" / "app.ico"
    if ico.exists():
        common_flags += ["--icon", str(ico)]

    if onefile:
        cmd += ["--onefile"]
    cmd += common_flags
    cmd += ["--paths", str(ROOT), str(ROOT / "main.py")]
    cmd += ["--distpath", str(DIST_DIR), "--workpath", str(BUILD_DIR), "--noconfirm"]

    print("  Running PyInstaller ...\n")
    subprocess.check_call(cmd, cwd=str(ROOT))


def copy_extras():
    """Copy non-Python extras to the dist folder."""
    out = DIST_DIR / "StockAnalyzer"
    if not out.exists():
        return

    for f in ["README.md", "LICENSE"]:
        src = ROOT / f
        if src.exists():
            shutil.copy2(src, out / f)

    sent_src = ROOT / "sentiment"
    sent_dst = out / "sentiment"
    if sent_src.exists() and not sent_dst.exists():
        print("  Copying sentiment pipeline source...")

        def _ignore_large(directory, contents):
            ignored = set()
            for item in contents:
                full = os.path.join(directory, item)
                if item.endswith(('.safetensors', '.bin', '.pt', '.pth', '.gguf', '.onnx')):
                    ignored.add(item)
                if item in ('__pycache__', '.git', 'build', 'dist', '.venv', 'venv',
                            'models', 'data', 'tests') and os.path.isdir(full):
                    ignored.add(item)
            return ignored

        shutil.copytree(sent_src, sent_dst, ignore=_ignore_large)
        (sent_dst / "models").mkdir(exist_ok=True)
        print(f"  Sentiment pipeline copied (model weights excluded)")
        print(f"  Place model folders in: {sent_dst / 'models'}")

    print(f"\n  Extras copied to {out}")


def print_summary():
    """Print build results."""
    exe = DIST_DIR / "StockAnalyzer" / "StockAnalyzer.exe"
    onefile_exe = DIST_DIR / "StockAnalyzer.exe"

    if exe.exists():
        size_mb = exe.stat().st_size / (1024 * 1024)
        folder_size = sum(
            f.stat().st_size for f in (DIST_DIR / "StockAnalyzer").rglob("*") if f.is_file()
        ) / (1024 * 1024)
        print(f"\n{'='*60}")
        print(f"  BUILD SUCCESSFUL")
        print(f"{'='*60}")
        print(f"  Output folder : {DIST_DIR / 'StockAnalyzer'}")
        print(f"  Executable    : {exe}")
        print(f"  Exe size      : {size_mb:.1f} MB")
        print(f"  Total size    : {folder_size:.1f} MB")
        print(f"\n  Usage:")
        print(f"    Double-click StockAnalyzer.exe   (native window)")
        print(f"    StockAnalyzer.exe --web          (open in browser)")
        print(f"    StockAnalyzer.exe --port 8080    (custom port)")
        print(f"{'='*60}\n")
    elif onefile_exe.exists():
        size_mb = onefile_exe.stat().st_size / (1024 * 1024)
        print(f"\n{'='*60}")
        print(f"  BUILD SUCCESSFUL (single-file)")
        print(f"{'='*60}")
        print(f"  Executable : {onefile_exe}")
        print(f"  Size       : {size_mb:.1f} MB")
        print(f"{'='*60}\n")
    else:
        print("\n  BUILD FAILED – no executable found.\n")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Stock Analysis Platform executable")
    parser.add_argument("--onefile", action="store_true", help="Single-file exe (slower startup)")
    parser.add_argument("--include-ml", action="store_true",
                        help="Bundle torch/transformers for local model support (large build!)")
    parser.add_argument("--clean", action="store_true", help="Clean build artifacts first")
    args = parser.parse_args()

    print("\n  Stock Analysis Platform - Build\n")
    if args.include_ml:
        print("  NOTE: Including ML libraries (torch/transformers). Build will be large.\n")

    if args.clean:
        clean()

    install_deps()
    build_exe(onefile=args.onefile, include_ml=args.include_ml)
    copy_extras()
    print_summary()
