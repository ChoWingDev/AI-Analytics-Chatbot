"""
scripts/lint_glossary.py
------------------------
Static checks that the business glossary agrees with the physical database.

The glossary is the semantic layer that constrains Text-to-SQL. When it drifts
from the real schema it stops preventing hallucination and starts CAUSING it:
the model faithfully obeys a rule naming a column that does not exist, and the
query dies. That is exactly what happened to `return_rate`, which pointed at
`mart_product_sales` while its rules referenced `status` / `order_returned_at`
(columns that mart does not have) and whose sql_formula had unbalanced parens.

For every metric this asserts:
  1. preferred_source (and any fallback_sources) exist in the database
  2. every preferred_source_field exists on that source
  3. the sql_formula actually parses and executes against that source

Run it after ANY glossary edit, and in CI before a semantic-layer change ships:
    python scripts/lint_glossary.py
Exit code 0 = clean, 1 = at least one metric is broken.
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.sql_agent.config import GLOSSARY_PATH, DB_PATH  # noqa: E402


def table_columns(conn: sqlite3.Connection, source: str) -> list[str]:
    """Real column list for a table OR view. Empty list if it doesn't exist."""
    rows = conn.execute(f"PRAGMA table_info({source})").fetchall()
    return [r[1] for r in rows]


def lint(glossary_path: str = GLOSSARY_PATH, db_path: str = DB_PATH) -> int:
    glossary = json.loads(Path(glossary_path).read_text(encoding="utf-8"))
    conn = sqlite3.connect(db_path)

    existing = {
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
        )
    }

    failures = 0
    print(f"Linting {len(glossary['metrics'])} metrics against {Path(db_path).name}\n")

    for name, m in glossary["metrics"].items():
        problems: list[str] = []

        source = m.get("preferred_source")
        sources = [source] + list(m.get("fallback_sources", []))

        # 1. sources exist
        for s in sources:
            if s and s not in existing:
                problems.append(f"source '{s}' does not exist in the database")

        if source in existing:
            cols = table_columns(conn, source)

            # 2. declared fields exist on the source
            for field in m.get("preferred_source_fields", []):
                if field not in cols:
                    problems.append(
                        f"preferred_source_field '{field}' is not a column on '{source}' "
                        f"(has: {', '.join(cols)})"
                    )

            # 3. the formula actually runs
            formula = m.get("sql_formula")
            if formula:
                try:
                    conn.execute(f"SELECT {formula} FROM {source}").fetchone()
                except Exception as e:
                    problems.append(f"sql_formula failed to execute: {e}")

        if problems:
            failures += 1
            print(f"  [FAIL] {name}")
            for p in problems:
                print(f"         - {p}")
        else:
            print(f"  [OK]   {name}")

    conn.close()

    print()
    if failures:
        print(f"{failures} metric(s) are inconsistent with the database.")
        return 1
    print("Glossary is consistent with the database.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check that the business glossary agrees with the physical database."
    )
    parser.add_argument("--glossary", default=GLOSSARY_PATH,
                        help="Glossary JSON to lint (default: config/glossary.json).")
    parser.add_argument("--db", default=DB_PATH,
                        help="SQLite database to lint against (default: the app config DB).")
    args = parser.parse_args()
    sys.exit(lint(args.glossary, args.db))
