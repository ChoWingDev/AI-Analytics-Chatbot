import sqlite3
import pandas as pd

class SQLExecutor:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(
            db_path
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
