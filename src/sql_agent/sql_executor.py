import sqlite3
import pandas as pd

class SQLExecutor:
    def __init__(self, db_path):
        # check_same_thread=False: the router executes the SQL pipeline inside an
        # asyncio worker thread (parallel SQL+RAG), so the connection created here
        # must be usable from that thread. Read-only analytics queries only.
        self.conn = sqlite3.connect(
            db_path,
            check_same_thread=False,
        )

    def execute(self, sql):
        return pd.read_sql_query(
            sql, self.conn
        )
    
if __name__ == "__main__":
    from .config import DB_PATH

    executor = SQLExecutor(DB_PATH)
    result = executor.execute("SELECT COUNT(*) FROM orders")
    print(result)
