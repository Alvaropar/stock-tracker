"""Pytest configuration and shared fixtures."""
from __future__ import annotations

import pytest

from stock_analyzer.app import create_app


@pytest.fixture()
def app():
    flask_app = create_app()
    flask_app.config.update({"TESTING": True})
    yield flask_app


@pytest.fixture()
def client(app):
    return app.test_client()
