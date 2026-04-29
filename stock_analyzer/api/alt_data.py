"""
Alternative data API.

GET  /api/altdata/stocktwits/<ticker>?limit=30
GET  /api/altdata/insiders/<ticker>?days=90
GET  /api/altdata/short/<ticker>
GET  /api/altdata/options/<ticker>
GET  /api/altdata/all/<ticker>?st_limit=30
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request

from ..services.alternative_data import (
    fetch_stocktwits,
    fetch_insider_transactions,
    fetch_finra_short_interest,
    fetch_options_data,
    fetch_all_alt_data,
)

bp = Blueprint("alt_data", __name__, url_prefix="/api/altdata")


@bp.route("/stocktwits/<ticker>", methods=["GET"])
def stocktwits(ticker: str):
    limit = int(request.args.get("limit", 30))
    return jsonify(fetch_stocktwits(ticker.upper(), limit=limit))


@bp.route("/insiders/<ticker>", methods=["GET"])
def insiders(ticker: str):
    days = int(request.args.get("days", 90))
    return jsonify(fetch_insider_transactions(ticker.upper(), days=days))


@bp.route("/short/<ticker>", methods=["GET"])
def short_interest(ticker: str):
    return jsonify(fetch_finra_short_interest(ticker.upper()))


@bp.route("/options/<ticker>", methods=["GET"])
def options(ticker: str):
    return jsonify(fetch_options_data(ticker.upper()))


@bp.route("/all/<ticker>", methods=["GET"])
def all_data(ticker: str):
    st_limit = int(request.args.get("st_limit", 30))
    return jsonify(fetch_all_alt_data(ticker.upper(), stocktwits_limit=st_limit))
