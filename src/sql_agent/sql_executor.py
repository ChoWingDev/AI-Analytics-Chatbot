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
    executor = SQLExecutor(
    "/Users/chowingchan/Desktop/Project/AI-Analytics-Copilot/Competitive-Intelligence-Internal-Analytics-System/data/database/thelook_ecommerce.db"
    )
    result = executor.execute("SELECT COUNT(*) FROM orders")
    print(result)
