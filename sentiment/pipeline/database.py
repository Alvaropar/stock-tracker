"""SQLite persistence layer for the commodity-trading pipeline.

Provides thread-safe database access for storing analysis sessions, articles,
application settings, and CSV export. Uses one connection per thread via
``threading.local()`` and parameterized queries throughout.

The database file is stored at ``{PROJECT_ROOT}/data/pipeline.db``.  Tables are
created automatically on first instantiation.
"""

from __future__ import annotations

import csv
import io
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Project root resolution (matches the rest of the pipeline package)
# ---------------------------------------------------------------------------

_PROJECT_ROOT: Path = (
    Path(os.environ.get("PIPELINE_PROJECT_ROOT", ""))
    if os.environ.get("PIPELINE_PROJECT_ROOT")
    else Path(__file__).resolve().parent.parent
)

_DEFAULT_DB_PATH: Path = _PROJECT_ROOT / "data" / "pipeline.db"


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS sessions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    market         TEXT    NOT NULL,
    asset_type     TEXT    NOT NULL,
    asset_id       TEXT    NOT NULL,
    created_at     TEXT    NOT NULL DEFAULT (datetime('now')),
    article_count  INTEGER DEFAULT 0,
    filtered_count INTEGER DEFAULT 0,
    positive_count INTEGER DEFAULT 0,
    negative_count INTEGER DEFAULT 0,
    neutral_count  INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS articles (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER REFERENCES sessions(id),
    title      TEXT    NOT NULL,
    date       TEXT,
    datetime   TEXT,
    source     TEXT,
    url        TEXT,
    summary    TEXT,
    ticker     TEXT,
    relevant   INTEGER,
    sentiment  TEXT,
    created_at TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_articles_session
    ON articles(session_id);

CREATE INDEX IF NOT EXISTS idx_articles_sentiment_date
    ON articles(sentiment, date);

CREATE TABLE IF NOT EXISTS app_settings (
    key        TEXT PRIMARY KEY,
    value      TEXT,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


# ---------------------------------------------------------------------------
# PipelineDB
# ---------------------------------------------------------------------------

class PipelineDB:
    """Thread-safe SQLite persistence for pipeline sessions and articles.

    Parameters
    ----------
    db_path : Path | None
        Override the default database location.  When *None* the database is
        placed at ``{PROJECT_ROOT}/data/pipeline.db``.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path: Path = db_path or _DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_tables()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        """Return a thread-local ``sqlite3.Connection``.

        Each thread gets its own connection so that the module is safe to use
        from multi-threaded Flask / orchestrator contexts.
        """
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    def _init_tables(self) -> None:
        """Create the schema if it does not already exist."""
        conn = self._conn()
        conn.executescript(_SCHEMA_SQL)
        conn.commit()

    @staticmethod
    def _rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
        """Convert a list of ``sqlite3.Row`` objects to plain dicts."""
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def create_session(self, market: str, asset_type: str, asset_id: str) -> int:
        """Create a new analysis session and return its ``id``.

        Parameters
        ----------
        market : str
            Market identifier (e.g. ``"US"``, ``"China"``).
        asset_type : str
            Asset category (e.g. ``"commodity"``, ``"equity"``).
        asset_id : str
            Specific asset key (e.g. ``"crude_oil"``).

        Returns
        -------
        int
            The auto-generated session id.
        """
        conn = self._conn()
        cursor = conn.execute(
            "INSERT INTO sessions (market, asset_type, asset_id) VALUES (?, ?, ?)",
            (market, asset_type, asset_id),
        )
        conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def update_session_counts(
        self,
        session_id: int,
        article_count: int = 0,
        filtered_count: int = 0,
        positive: int = 0,
        negative: int = 0,
        neutral: int = 0,
    ) -> None:
        """Update aggregate counts on an existing session.

        Parameters
        ----------
        session_id : int
            Session to update.
        article_count, filtered_count, positive, negative, neutral : int
            Count values to store.
        """
        conn = self._conn()
        conn.execute(
            """
            UPDATE sessions
               SET article_count  = ?,
                   filtered_count = ?,
                   positive_count = ?,
                   negative_count = ?,
                   neutral_count  = ?
             WHERE id = ?
            """,
            (article_count, filtered_count, positive, negative, neutral, session_id),
        )
        conn.commit()

    def get_recent_sessions(self, limit: int = 50) -> list[dict]:
        """Return the most recent sessions, newest first.

        Parameters
        ----------
        limit : int
            Maximum number of sessions to return (default 50).

        Returns
        -------
        list[dict]
            Each dict mirrors the ``sessions`` table columns.
        """
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return self._rows_to_dicts(rows)

    # ------------------------------------------------------------------
    # Articles
    # ------------------------------------------------------------------

    def save_articles(self, session_id: int, articles: list[dict]) -> None:
        """Bulk-insert articles linked to *session_id*.

        Parameters
        ----------
        session_id : int
            The owning session.
        articles : list[dict]
            Each dict should contain keys matching the ``articles`` table
            columns (typically produced by ``Article.to_dict()``).  Missing
            keys default to ``None``.
        """
        if not articles:
            return

        conn = self._conn()
        rows = [
            (
                session_id,
                a.get("title", ""),
                a.get("date"),
                a.get("datetime"),
                a.get("source"),
                a.get("url"),
                a.get("summary"),
                a.get("ticker"),
                a.get("relevant"),
                a.get("sentiment"),
            )
            for a in articles
        ]
        conn.executemany(
            """
            INSERT INTO articles
                (session_id, title, date, datetime, source, url,
                 summary, ticker, relevant, sentiment)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()

    def get_session_articles(self, session_id: int) -> list[dict]:
        """Return all articles belonging to *session_id*.

        Parameters
        ----------
        session_id : int
            The session whose articles to retrieve.

        Returns
        -------
        list[dict]
        """
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM articles WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        return self._rows_to_dicts(rows)

    def get_sentiment_history(
        self,
        market: str,
        asset_type: str,
        asset_id: str,
        days: int = 30,
    ) -> list[dict]:
        """Return per-day sentiment aggregation for a specific asset.

        Looks back *days* calendar days from the current UTC date.  Only
        articles with a non-null ``sentiment`` value are counted.

        Parameters
        ----------
        market, asset_type, asset_id : str
            Identify the asset.
        days : int
            Number of days of history to return (default 30).

        Returns
        -------
        list[dict]
            Sorted ascending by date.  Each dict contains:
            ``{date, positive, negative, neutral, total}``.
        """
        conn = self._conn()
        rows = conn.execute(
            """
            SELECT a.date,
                   SUM(CASE WHEN a.sentiment = 'positive' THEN 1 ELSE 0 END) AS positive,
                   SUM(CASE WHEN a.sentiment = 'negative' THEN 1 ELSE 0 END) AS negative,
                   SUM(CASE WHEN a.sentiment = 'neutral'  THEN 1 ELSE 0 END) AS neutral,
                   COUNT(*) AS total
              FROM articles a
              JOIN sessions s ON s.id = a.session_id
             WHERE s.market     = ?
               AND s.asset_type = ?
               AND s.asset_id   = ?
               AND a.sentiment IS NOT NULL
               AND a.date >= date('now', ? || ' days')
             GROUP BY a.date
             ORDER BY a.date ASC
            """,
            (market, asset_type, asset_id, f"-{days}"),
        ).fetchall()
        return self._rows_to_dicts(rows)

    # ------------------------------------------------------------------
    # Settings (key/value)
    # ------------------------------------------------------------------

    def get_setting(self, key: str, default: str | None = None) -> str | None:
        """Read a single setting value.

        Parameters
        ----------
        key : str
            Setting name.
        default : str | None
            Returned when the key does not exist.

        Returns
        -------
        str | None
        """
        conn = self._conn()
        row = conn.execute(
            "SELECT value FROM app_settings WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return default
        return row["value"]

    def set_setting(self, key: str, value: str) -> None:
        """Create or update a setting.

        Parameters
        ----------
        key : str
            Setting name.
        value : str
            Setting value (always stored as text).
        """
        conn = self._conn()
        conn.execute(
            """
            INSERT INTO app_settings (key, value, updated_at)
                 VALUES (?, ?, datetime('now'))
            ON CONFLICT(key) DO UPDATE
                SET value      = excluded.value,
                    updated_at = excluded.updated_at
            """,
            (key, value),
        )
        conn.commit()

    def get_all_settings(self) -> dict[str, str]:
        """Return every setting as a ``{key: value}`` dict.

        Returns
        -------
        dict[str, str]
        """
        conn = self._conn()
        rows = conn.execute("SELECT key, value FROM app_settings").fetchall()
        return {row["key"]: row["value"] for row in rows}

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_session_csv(self, session_id: int) -> str:
        """Export a session's articles as a CSV string.

        Parameters
        ----------
        session_id : int
            The session to export.

        Returns
        -------
        str
            Full CSV content including header row.
        """
        articles = self.get_session_articles(session_id)
        if not articles:
            return ""

        fieldnames = [
            "id",
            "session_id",
            "title",
            "date",
            "datetime",
            "source",
            "url",
            "summary",
            "ticker",
            "relevant",
            "sentiment",
            "created_at",
        ]

        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(articles)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the current thread's database connection, if open."""
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def __repr__(self) -> str:
        return f"PipelineDB(db_path={self.db_path!r})"
