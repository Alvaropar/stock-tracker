"""
Shared utility functions for the pipeline package.
"""
from __future__ import annotations

from typing import Any


def safe_col(row: Any, *names: str, default: Any = None) -> Any:
    """Get the first matching column value from a DataFrame *row*.

    Akshare may use Chinese or English column names depending on version.
    This helper avoids KeyError when columns are renamed.

    Usage::

        content = safe_col(row, "内容", "content", default="")
        price = safe_col(row, "收盘", "close", default=0)
    """
    for name in names:
        try:
            val = row.get(name)
            if val is not None:
                return val
        except Exception:
            continue
    return default
