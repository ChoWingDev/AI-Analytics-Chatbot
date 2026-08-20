"""
Consistency checks between config/glossary.json and the real database schema.

Why this exists: the glossary is the governance layer -- it tells the LLM which
table to query and which rules to follow. When an entry names columns that do
not exist in its own preferred_source, the model dutifully writes SQL against
them and the query dies at execution. The generated SQL is not wrong; the
governance data is.

These tests need the database. Build it first:
    python scripts/build_db.py
They skip (not fail) when it is absent, so a fresh checkout without data does
not report false failures.
"""

import json
import re
import sqlite3

import pytest

from src.sql_agent.config import DB_PATH, GLOSSARY_PATH


def _schema():
    conn = sqlite3.connect(DB_PATH)
    try:
        names = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
            )
        ]
        return {
            name: {r[1] for r in conn.execute(f"PRAGMA table_info({name})")}
            for name in names
        }
    finally:
        conn.close()


@pytest.fixture(scope="module")
def schema():
    import os

    if not os.path.exists(DB_PATH):
        pytest.skip(f"database not built at {DB_PATH} -- run scripts/build_db.py")
    return _schema()


@pytest.fixture(scope="module")
def glossary():
    with open(GLOSSARY_PATH, encoding="utf-8") as f:
        return json.load(f)


def _real_metrics(path=GLOSSARY_PATH):
    """Metric keys, skipping pure aliases. Used to parametrize at collection time."""
    with open(path, encoding="utf-8") as f:
        metrics = json.load(f)["metrics"]
    return [k for k, v in metrics.items() if "alias_of" not in v]


@pytest.mark.parametrize("metric_key", _real_metrics())
def test_preferred_source_exists(metric_key, glossary, schema):
    source = glossary["metrics"][metric_key].get("preferred_source")
    assert source, f"{metric_key} has no preferred_source"
    assert source in schema, (
        f"{metric_key}.preferred_source is '{source}', which is not a table "
        f"or view in the database"
    )


@pytest.mark.parametrize("metric_key", _real_metrics())
def test_declared_fields_exist_in_preferred_source(metric_key, glossary, schema):
    metric = glossary["metrics"][metric_key]
    source = metric.get("preferred_source")
    if source not in schema:
        pytest.skip("covered by test_preferred_source_exists")

    declared = set(metric.get("preferred_source_fields", []))
    missing = sorted(declared - schema[source])
    assert not missing, (
        f"{metric_key}.preferred_source_fields names {missing}, which "
        f"do not exist in {source}. Columns available: {sorted(schema[source])}"
    )


@pytest.mark.parametrize("metric_key", _real_metrics())
def test_rules_do_not_reference_foreign_columns(metric_key, glossary, schema):
    """
    business_logic and sql_formula are injected into the prompt verbatim and
    take precedence over everything else, so a column name mentioned there
    must exist in the source the metric points at. A name that exists
    somewhere else in the schema is the dangerous case: it looks valid, so
    the model uses it confidently.
    """
    metric = glossary["metrics"][metric_key]
    source = metric.get("preferred_source")
    if source not in schema:
        pytest.skip("covered by test_preferred_source_exists")

    known_columns = set().union(*schema.values())
    # Prohibitions ("do NOT use mart_product_sales.units_sold") must name the
    # column they rule out, so only positive guidance is checked.
    positive = [
        rule for rule in metric.get("business_logic", [])
        if not re.search(r"\b(do not|don't|never)\b", rule, re.IGNORECASE)
    ]
    prose = " ".join(positive) + " " + str(metric.get("sql_formula", ""))
    # Qualified references name their own table (products.retail_price,
    # order_items.sale_price) and are legitimate cross-table guidance. Only
    # bare column names are read as "a column of the preferred source".
    prose = re.sub(r"\b[a-z_]+\.[a-z_]+\b", " ", prose)
    mentioned = set(re.findall(r"\b[a-z_]{4,}\b", prose)) & known_columns

    foreign = sorted(mentioned - schema[source])
    assert not foreign, (
        f"{metric_key} rules reference {foreign}, which exist elsewhere in the "
        f"schema but not in its preferred_source '{source}'"
    )
