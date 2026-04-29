.PHONY: install install-dev install-ml run run-web run-dev run-sentiment build build-onefile build-ml clean lint format test test-unit test-integration

# ── Installation ──────────────────────────────────────────────────────────────

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,ml,desktop,llm]"

install-ml:
	pip install -e ".[ml]"

# ── Running ───────────────────────────────────────────────────────────────────

run:
	python main.py

run-web:
	python main.py --web

run-dev:
	python main.py --web --debug

run-sentiment:
	cd sentiment && python -m pipeline

# ── Testing ───────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

# ── Code quality ──────────────────────────────────────────────────────────────

lint:
	ruff check stock_analyzer/ tests/ main.py

format:
	ruff format stock_analyzer/ tests/ main.py

# ── Build (PyInstaller) ───────────────────────────────────────────────────────

build:
	python scripts/build.py

build-onefile:
	python scripts/build.py --onefile

build-ml:
	python scripts/build.py --include-ml

# ── Clean ─────────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info/
