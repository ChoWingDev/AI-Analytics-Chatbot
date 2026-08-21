"""
app/charts.py
─────────────
Decide how to draw a SQL result, if at all.

Kept out of the Streamlit file on purpose: this is a pure function over a
DataFrame, so it is testable without starting a browser or a server.

Guiding rule: never guess. A table is always a correct rendering of a result;
a wrong chart is not. When the shape is ambiguous, return None.
"""

import re
import warnings

import pandas as pd
from pandas.api import types as ptypes

# Column names that name a time axis. Checked before value parsing because
# an integer year column parses as a number, not a date.
_DATE_NAME = re.compile(r"date|month|day|year|week|quarter", re.IGNORECASE)

MAX_BAR_ROWS = 25
"""Above this a bar chart is unreadable — the table is the better rendering."""


def _is_date_like(df: pd.DataFrame, col: str) -> bool:
    """True if the column names a time axis or its values parse as dates."""
    if ptypes.is_datetime64_any_dtype(df[col]):
        return True
    if _DATE_NAME.search(str(col)):
        return True
    if ptypes.is_numeric_dtype(df[col]):
        return False  # plain numbers are values, not dates
    try:
        with warnings.catch_warnings():
            # Mixed / unparseable strings are the normal case here, not a problem.
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(df[col], errors="coerce")
    except Exception:
        return False
    return parsed.notna().all()


def choose_chart(df) -> tuple[str | None, dict]:
    """
    Pick a chart for a query result.

    Returns (kind, params) where kind is "metric", "bar", "line", or None:

      metric — 1 row, exactly 1 numeric column   params: {"label", "value"}
      line   — a date-like column + numeric, > 2 rows   params: {"x", "y"}
      bar    — a text column + numeric, 2-25 rows       params: {"x", "y"}
      None   — anything else; render the table alone    params: {}
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None, {}

    numeric = [c for c in df.columns if ptypes.is_numeric_dtype(df[c])]
    dates = [c for c in df.columns if _is_date_like(df, c)]
    # A date-like column is a time axis, never a value axis.
    numeric = [c for c in numeric if c not in dates]
    text = [c for c in df.columns if c not in numeric and c not in dates]

    if len(numeric) != 1:
        return None, {}  # zero or several value columns: ambiguous
    value = numeric[0]

    # Single number — e.g. return rate 10.01%
    if len(df) == 1 and not dates:
        label = str(df[text[0]].iloc[0]) if len(text) == 1 else str(value)
        return "metric", {"label": label, "value": df[value].iloc[0]}

    # Time series
    if len(dates) == 1 and not text and len(df) > 2:
        return "line", {"x": dates[0], "y": value}

    # Ranking / breakdown
    if len(text) == 1 and not dates and 2 <= len(df) <= MAX_BAR_ROWS:
        return "bar", {"x": text[0], "y": value}

    return None, {}
