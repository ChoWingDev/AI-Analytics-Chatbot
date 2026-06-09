import os

class MockSQLGenerator:

    def generate_sql(self, question):

        sql_map = {
            "What is total revenue in 2021?":
            """
            SELECT SUM(net_sales)
            FROM mart_product_sales
            WHERE strftime('%Y', order_date)='2021'
            """,

            "Top 5 categories by revenue":
            """
            SELECT
                category,
                SUM(net_sales) AS revenue
            FROM mart_product_sales
            GROUP BY category
            ORDER BY revenue DESC
            LIMIT 5
            """
        }

        return sql_map.get(question)


if __name__ == "__main__":
    generator = MockSQLGenerator()
    sql = generator.generate_sql("What is total revenue in 2021?")
    print(sql)