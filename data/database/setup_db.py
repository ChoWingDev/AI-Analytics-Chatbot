"""
scripts/setup_db.py
-------------------
Person A — Week 1 Task
Loads TheLook eCommerce CSV files into a local SQLite database.

Usage:
    python scripts/setup_db.py

Expected CSV files in data/raw/:
    users.csv, orders.csv, order_items.csv, products.csv, events.csv
"""

import sqlite3
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
DB_PATH = Path("data/processed/thelook.db")

TABLES = ["users", "orders", "order_items", "products", "events"]


def load_table(conn: sqlite3.Connection, table_name: str) -> None:
    csv_path = RAW_DIR / f"{table_name}.csv"
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path} not found — download from BigQuery or Kaggle first")
        return

    df = pd.read_csv(csv_path)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    print(f"  [OK]   {table_name}: {len(df):,} rows loaded")


def create_indexes(conn: sqlite3.Connection) -> None:
    """Add indexes on commonly joined / filtered columns for faster SQL Agent queries."""
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_orders_user ON orders(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)",
        "CREATE INDEX IF NOT EXISTS idx_order_items_order ON order_items(order_id)",
        "CREATE INDEX IF NOT EXISTS idx_order_items_user ON order_items(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_events_user ON events(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_events_traffic ON events(traffic_source)",
    ]
    for idx in indexes:
        conn.execute(idx)
    print("  [OK]   Indexes created")


def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Setting up database at {DB_PATH}\n")

    conn = sqlite3.connect(DB_PATH)
    try:
        for table in TABLES:
            load_table(conn, table)
        create_indexes(conn)
        conn.commit()
        print(f"\nDatabase ready: {DB_PATH}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
