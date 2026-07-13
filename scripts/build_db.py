"""
scripts/build_db.py
───────────────────
One-command, reproducible builder for the analytics database the SQL agent
queries. Replaces the stale data/database/setup_db.py and the hardcoded-path
notebooks 02/03.

What it does:
  1. Obtains the 7 TheLook base-table CSVs (downloads from Kaggle, or uses a
     local --csv-dir you already have).
  2. Loads them into SQLite at the path the app config points to
     (data/database/thelook_ecommerce.db).
  3. Builds the 5 analytics marts the glossary expects — 4 SQL views plus
     mart_user_segment (a table built with the pandas RFM logic ported
     verbatim from notebook 03_create_marts).

Usage:
    # Download from Kaggle (needs Kaggle credentials — see README / --help)
    python scripts/build_db.py

    # Use CSVs you already downloaded/extracted somewhere
    python scripts/build_db.py --csv-dir "C:/path/to/thelook_csvs"

    # Override dataset or output db
    python scripts/build_db.py --kaggle-dataset owner/slug --db-path ./my.db

Kaggle credentials (only needed for the download path): create an API token at
https://www.kaggle.com/settings/account → "Create New Token", then either place
the downloaded kaggle.json at %USERPROFILE%\\.kaggle\\kaggle.json, or set the
KAGGLE_USERNAME and KAGGLE_KEY environment variables.
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

# Make `src` importable so we reuse the app's canonical DB path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.sql_agent.config import DB_PATH  # noqa: E402

# Default public Kaggle mirror of the BigQuery `thelook_ecommerce` dataset.
DEFAULT_KAGGLE_DATASET = "mustafakeser4/looker-ecommerce-bigquery-dataset"

# Base tables we load, keyed by the table name we create. Filenames are matched
# case-insensitively by stem, so `Orders.csv` or `orders.csv` both work.
BASE_TABLES = [
    "distribution_centers",
    "events",
    "inventory_items",
    "order_items",
    "orders",
    "products",
    "users",
]

# ── Mart definitions (ported verbatim from notebooks/03_create_marts.ipynb) ──

MART_VIEWS = {
    "mart_order_summary": """
CREATE VIEW mart_order_summary AS
SELECT
    o.order_id,
    o.user_id,
    o.status,
    o.created_at AS order_created_at,
    o.returned_at AS order_returned_at,
    o.shipped_at AS order_shipped_at,
    o.delivered_at AS order_delivered_at,
    COUNT(oi.id) AS item_count,
    SUM(oi.sale_price) AS order_revenue,
    SUM(p.cost) AS order_cost,
    SUM(oi.sale_price - p.cost) AS gross_profit,
    CASE
        WHEN SUM(oi.sale_price) > 0
        THEN ROUND(SUM(oi.sale_price - p.cost) * 1.0 / SUM(oi.sale_price), 4)
        ELSE NULL
    END AS gross_margin
FROM orders o
LEFT JOIN order_items oi ON o.order_id = oi.order_id
LEFT JOIN products p ON oi.product_id = p.id
GROUP BY
    o.order_id, o.user_id, o.status, o.created_at,
    o.returned_at, o.shipped_at, o.delivered_at;
""",
    "mart_product_sales": """
CREATE VIEW mart_product_sales AS
SELECT
    DATE(o.created_at) AS order_date,
    p.id AS product_id,
    p.name AS product_name,
    p.category,
    p.brand,
    p.department,
    COUNT(DISTINCT CASE WHEN oi.status = 'Complete' THEN oi.order_id END) AS total_orders,
    SUM(CASE WHEN oi.status = 'Complete' THEN 1 ELSE 0 END) AS units_sold,
    SUM(CASE WHEN oi.status = 'Returned' THEN 1 ELSE 0 END) AS returned_units,
    SUM(CASE WHEN oi.status = 'Complete' THEN oi.sale_price ELSE 0 END) AS gross_sales,
    SUM(CASE WHEN oi.status = 'Returned' THEN oi.sale_price ELSE 0 END) AS refunded_sales,
    SUM(CASE
        WHEN oi.status = 'Complete' THEN oi.sale_price
        WHEN oi.status = 'Returned' THEN -oi.sale_price
        ELSE 0
    END) AS net_sales,
    AVG(CASE WHEN oi.status = 'Complete' THEN oi.sale_price END) AS avg_selling_price
FROM order_items oi
JOIN orders o ON oi.order_id = o.order_id
JOIN products p ON oi.product_id = p.id
GROUP BY
    DATE(o.created_at), p.id, p.name, p.category, p.brand, p.department;
""",
    "mart_daily_sales": """
CREATE VIEW mart_daily_sales AS
SELECT
    DATE(oi.created_at) AS order_date,
    COUNT(DISTINCT oi.order_id) AS total_orders,
    COUNT(oi.id) AS units_sold,
    COUNT(DISTINCT o.user_id) AS active_customers,
    SUM(oi.sale_price) AS gross_sales,
    AVG(oi.sale_price) AS avg_item_price,
    SUM(oi.sale_price) * 1.0 / COUNT(DISTINCT oi.order_id) AS avg_order_value
FROM order_items oi
JOIN orders o ON oi.order_id = o.order_id
WHERE oi.status = 'Complete' AND oi.created_at IS NOT NULL
GROUP BY DATE(oi.created_at);
""",
    # Depends on mart_order_summary, so it must be created after it.
    "mart_user_summary": """
CREATE VIEW mart_user_summary AS
WITH first_order AS (
    SELECT user_id, MIN(created_at) AS first_order_date
    FROM orders GROUP BY user_id
),
dataset_start AS (
    SELECT MIN(created_at) AS min_dataset_date FROM orders
),
rfm_base AS (
    SELECT
        o.user_id,
        MAX(o.created_at) AS last_order_date,
        COUNT(DISTINCT o.order_id) AS frequency,
        SUM(m.order_revenue) AS monetary
    FROM orders o
    LEFT JOIN mart_order_summary m ON o.order_id = m.order_id
    GROUP BY o.user_id
)
SELECT
    u.id AS user_id,
    u.gender,
    u.age,
    u.country,
    u.traffic_source,
    f.first_order_date,
    r.last_order_date,
    CAST(julianday((SELECT MAX(created_at) FROM orders)) - julianday(r.last_order_date) AS INTEGER) AS recency,
    r.frequency,
    ROUND(r.monetary, 2) AS monetary,
    CAST(julianday((SELECT MAX(created_at) FROM orders)) - julianday(f.first_order_date) AS INTEGER) AS days_since_first_order,
    CASE
        WHEN julianday(f.first_order_date) <= julianday(d.min_dataset_date) + 30
        THEN 'New' ELSE 'Existing'
    END AS customer_type
FROM users u
LEFT JOIN first_order f ON u.id = f.user_id
LEFT JOIN rfm_base r ON u.id = r.user_id
CROSS JOIN dataset_start d;
""",
}

# Order matters: mart_user_summary reads mart_order_summary.
MART_VIEW_ORDER = [
    "mart_order_summary",
    "mart_product_sales",
    "mart_daily_sales",
    "mart_user_summary",
]


def _assign_segment(row, high_monetary_threshold):
    """RFM → segment label. Ported verbatim from notebook 03."""
    if row["frequency"] == 0:
        return "Prospect"
    elif row["customer_type"] == "New":
        return "New Customers"
    elif row["recency"] > 180 and row["monetary"] >= high_monetary_threshold:
        return "At Risk High Value"
    elif row["recency"] > 180:
        return "Inactive Users"
    elif row["frequency"] == 1:
        return "One-time Buyers"
    elif row["recency"] <= 30 and row["monetary"] >= high_monetary_threshold:
        return "High Value Active"
    else:
        return "Regular Customers"


def resolve_csv_dir(args) -> Path:
    """Return a folder containing the base-table CSVs, downloading if needed."""
    if args.csv_dir:
        csv_dir = Path(args.csv_dir).expanduser().resolve()
        if not csv_dir.is_dir():
            sys.exit(f"[error] --csv-dir not found: {csv_dir}")
        print(f"Using local CSVs: {csv_dir}")
        return csv_dir

    try:
        import kagglehub
    except ImportError:
        sys.exit(
            "[error] kagglehub is not installed.\n"
            "        Install it:  pip install kagglehub\n"
            "        Or pass CSVs you already have:  --csv-dir <folder>"
        )

    print(f"Downloading Kaggle dataset: {args.kaggle_dataset}")
    path = Path(kagglehub.dataset_download(args.kaggle_dataset))
    print(f"Downloaded to: {path}")
    return path


def find_csv(csv_dir: Path, table: str) -> Path | None:
    """Case-insensitive match of `<table>.csv` anywhere under csv_dir."""
    for p in csv_dir.rglob("*.csv"):
        if p.stem.lower() == table:
            return p
    return None


def load_base_tables(conn: sqlite3.Connection, csv_dir: Path) -> None:
    print("\nLoading base tables:")
    missing = []
    for table in BASE_TABLES:
        csv_path = find_csv(csv_dir, table)
        if csv_path is None:
            missing.append(table)
            print(f"  [MISS] {table}: no {table}.csv found under {csv_dir}")
            continue
        # chunked read/write keeps the 2.4M-row events table within memory.
        first = True
        rows = 0
        for chunk in pd.read_csv(csv_path, chunksize=100_000):
            chunk.to_sql(table, conn, if_exists="replace" if first else "append", index=False)
            first = False
            rows += len(chunk)
        print(f"  [OK]   {table}: {rows:,} rows  ({csv_path.name})")
    if missing:
        sys.exit(f"\n[error] Missing required CSV(s): {', '.join(missing)}. "
                 f"Check the dataset contents or your --csv-dir.")


def build_marts(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()

    print("\nBuilding mart views:")
    for name in MART_VIEW_ORDER:
        cursor.execute(f"DROP VIEW IF EXISTS {name};")
        cursor.execute(MART_VIEWS[name])
        print(f"  [OK]   {name} (view)")
    conn.commit()

    print("\nBuilding mart_user_segment (pandas RFM):")
    seg = pd.read_sql(
        "SELECT user_id, customer_type, recency, frequency, monetary FROM mart_user_summary",
        conn,
    )
    seg["frequency"] = seg["frequency"].fillna(0)
    seg["monetary"] = seg["monetary"].fillna(0)
    high_monetary_threshold = seg["monetary"].quantile(0.75)
    seg["customer_segment"] = seg.apply(
        lambda r: _assign_segment(r, high_monetary_threshold), axis=1
    )
    # mart_user_segment is a real table (not a view) since it depends on the
    # Python-computed 75th-percentile threshold.
    conn.execute("DROP TABLE IF EXISTS mart_user_segment;")
    seg.to_sql("mart_user_segment", conn, if_exists="replace", index=False)
    print(f"  [OK]   mart_user_segment (table): {len(seg):,} rows, "
          f"P75 monetary threshold={high_monetary_threshold:.2f}")


def verify(conn: sqlite3.Connection) -> None:
    print("\nVerification:")
    objs = pd.read_sql(
        "SELECT name, type FROM sqlite_master WHERE type IN ('table','view') ORDER BY type, name",
        conn,
    )
    print(objs.to_string(index=False))

    # A glossary-shaped sanity check: realized revenue on completed orders.
    rev = pd.read_sql(
        "SELECT ROUND(SUM(order_revenue), 2) AS complete_revenue "
        "FROM mart_order_summary WHERE status = 'Complete'",
        conn,
    )
    print(f"\nSanity: SUM(order_revenue) where status='Complete' = "
          f"{rev['complete_revenue'].iloc[0]}")


def main():
    parser = argparse.ArgumentParser(
        description="Build the TheLook analytics SQLite DB (base tables + 5 marts) "
                    "from Kaggle or local CSVs.",
        epilog="Kaggle download needs credentials: put kaggle.json in "
               "%USERPROFILE%\\.kaggle\\, or set KAGGLE_USERNAME and KAGGLE_KEY.",
    )
    parser.add_argument("--csv-dir", help="Folder with the base-table CSVs (skips Kaggle download).")
    parser.add_argument("--kaggle-dataset", default=DEFAULT_KAGGLE_DATASET,
                        help=f"Kaggle dataset slug (default: {DEFAULT_KAGGLE_DATASET}).")
    parser.add_argument("--db-path", default=str(DB_PATH),
                        help="Output SQLite path (default: the app config DB path).")
    args = parser.parse_args()

    db_path = Path(args.db_path).expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Target database: {db_path}")

    csv_dir = resolve_csv_dir(args)

    conn = sqlite3.connect(db_path)
    try:
        load_base_tables(conn, csv_dir)
        build_marts(conn)
        verify(conn)
        print(f"\nDone. Database ready at {db_path}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
