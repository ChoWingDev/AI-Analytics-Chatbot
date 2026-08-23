"""
src/sql_agent/config.py
───────────────────────
Canonical paths for the Text-to-SQL agent.

Paths are resolved from this file's location, not the working directory, so
`python -m app.copilot`, `python scripts/build_db.py`, and notebook runs all
point at the same files.

Reconstructed 2026-08-19: the original was never committed — a bare `config.py`
line in .gitignore (intended for IPython's ipython_config.py) matched at every
directory level and silently excluded it. That rule is now narrowed.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Business metric glossary — metric definitions, formulas, business rules.
GLOSSARY_PATH = str(PROJECT_ROOT / "config" / "glossary.json")

# SQLite analytics database built by scripts/build_db.py (TheLook + marts).
DB_PATH = str(PROJECT_ROOT / "data" / "database" / "thelook_ecommerce.db")

# Question / expected-SQL pairs used by accuracy_report.py.
TEST_CASE_PATH = str(PROJECT_ROOT / "data" / "evaluation" / "test_cases.json")
