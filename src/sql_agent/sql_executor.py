import re
import sqlite3
import pandas as pd

# Statement types an analytics copilot must never run against the warehouse.
# (\b boundaries mean column names like `created_at` / `updated_at` are safe.)
FORBIDDEN_KEYWORDS = (
    "drop", "delete", "insert", "update", "alter", "create", "attach", "detach",
)


class SQLExecutor:
    def __init__(self, db_path):
        # check_same_thread=False: the router executes the SQL pipeline inside an
        # asyncio worker thread (parallel SQL+RAG), so the connection created here
        # must be usable from that thread. Read-only analytics queries only.
        self.conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
        )

    @staticmethod
    def sanitize(sql: str) -> str:
        """
        Turn raw LLM output into a single safe, executable statement.

        Two real failures this closes:

        1. Multi-statement output. The model frequently answers with several
           queries at once, e.g.
               -- return rate
               SELECT ... FROM mart_product_sales;
               -- average order value
               SELECT ... FROM mart_order_summary;
           `pandas.read_sql_query` silently executes only the FIRST statement.
           That is a quiet correctness bug: the extra query looks like it ran but
           never did. We split explicitly and take the first statement, rather
           than depending on that silent behaviour.

        2. Read-only enforcement. Generated SQL is untrusted input. Only
           SELECT / WITH queries may reach the database.
        """
        # Strip -- line comments and /* block */ comments so they can't hide
        # statement separators or keywords.
        sql = re.sub(r"--[^\n]*", " ", sql)
        sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)

        statements = [s.strip() for s in sql.split(";") if s.strip()]
        if not statements:
            raise ValueError("No SQL statement found in the model output.")

        statement = statements[0]
        lowered = statement.lower()

        if not re.match(r"^\s*(select|with)\b", lowered):
            raise ValueError(
                f"Only SELECT/WITH queries are allowed; got: {statement[:60]!r}"
            )

        for keyword in FORBIDDEN_KEYWORDS:
            if re.search(rf"\b{keyword}\b", lowered):
                raise ValueError(
                    f"Refusing to run a query containing '{keyword.upper()}' (read-only)."
                )

        return statement

    def execute(self, sql):
        return pd.read_sql_query(self.sanitize(sql), self.conn)
