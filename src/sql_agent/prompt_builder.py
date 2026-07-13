import textwrap

# Few-shot exemplars.
#
# The model standardised on (Qwen-7B) is weaker at SQL than a frontier model and
# made two specific mistakes repeatedly. Each example below exists to kill one:
#
#   1. It filtered mart_product_sales by `status` / `order_returned_at` — columns
#      that mart does not have, because it is pre-aggregated. Example 2 shows the
#      correct pre-aggregated ratio.
#   2. It answered with several SQL statements at once. Every example shows a
#      single statement, reinforced by the "exactly ONE statement" instruction.
#   3. It invented a date filter on questions that asked for an all-time figure.
#      That was caused by these very examples: when all of them filtered by date,
#      the model learned "always filter by date" and the abstract instruction not
#      to could not outweigh three concrete demonstrations. Example 4 exists to
#      show the no-period case.
#
# NOTE: none of these questions appear in data/evaluation/test_cases.json. Using a
# benchmark question as a few-shot example would leak the test set into the prompt
# and inflate the reported accuracy.
FEW_SHOT_EXAMPLES = """
Example 1 — realized revenue (order grain, must filter to completed orders):
Q: What is the total revenue in 2022?
A: SELECT ROUND(SUM(order_revenue), 2) AS revenue
   FROM mart_order_summary
   WHERE status = 'Complete'
     AND STRFTIME('%Y', order_created_at) = '2022'

Example 2 — return rate (product mart is PRE-AGGREGATED: no status column exists):
Q: What is the return rate in 2022?
A: SELECT ROUND((SUM(returned_units) * 1.0) / SUM(units_sold), 4) AS return_rate
   FROM mart_product_sales
   WHERE STRFTIME('%Y', order_date) = '2022'

Example 3 — ranking (group, order, limit):
Q: What are the top 5 product categories by revenue in 202101?
A: SELECT category, ROUND(SUM(gross_sales), 2) AS category_revenue
   FROM mart_product_sales
   WHERE STRFTIME('%Y%m', order_date) = '202101'
   GROUP BY category
   ORDER BY category_revenue DESC
   LIMIT 5

Example 4 — NO time period in the question, so NO date filter (all-time figure):
Q: How many completed orders do we have in total?
A: SELECT COUNT(DISTINCT order_id) AS total_orders
   FROM mart_order_summary
   WHERE status = 'Complete'
""".strip()


# Injected when the retriever matched NO time period.
#
# A passive "No specific time period matched." was not enough: the model kept
# copying a year out of the worked examples and silently scoping an all-time
# question to 2022 — valid SQL, wrong answer. The guardrail has to be phrased as
# a prohibition, and it is driven by the retriever's actual state rather than
# being a static instruction that is always present.
NO_TIME_PERIOD_RULE = (
    "NONE. The question does not request any time period.\n"
    "        Your SQL MUST NOT contain any date or time filter: no STRFTIME(...) "
    "comparison, no year, no date range in the WHERE clause.\n"
    "        Aggregate over ALL rows. Do not copy a date filter from the worked examples."
)


class PromptBuilder:
    def build_prompt(self, question, retrieval_result):
        metrics = retrieval_result.get("metrics", [])
        time_periods = retrieval_result.get("time_periods", {})
        schema = retrieval_result.get("schema", {})

        # 1. 構建輕量化的 Time Periods Context
        time_period_text = ""
        for key, sql_expr in time_periods.items():
            time_period_text += f"\n- {key}: {sql_expr}"

        # 2. 極致壓縮 Schema 格式：一行展示一個 Table 嘅所有欄位
        schema_text = ""
        for table, columns in schema.items():
            schema_text += f"Table {table} ({', '.join(columns)})\n"

        # 3. 構建結構化指標 Context
        metric_text = ""
        for i, metric in enumerate(metrics[:2]): # 保持最精簡，通常前兩個最相關
            business_logic = metric.get("business_logic", [])
            business_logic_text = "\n".join([f"- {rule}" for rule in business_logic])

            # 這裡動態去 Schema 拿欄位，確保 Fields 100% 不會顯示 N/A
            current_source = metric.get("preferred_source")
            available_fields = schema.get(current_source, [])
            fields_str = ", ".join(available_fields) if available_fields else "See schema context"

            metric_text += f"""
            Metric {i + 1}: {metric.get("business_name", "N/A")}
            Metric Key: {metric.get("metric", "N/A")}
            Definition: {metric.get("definition", "N/A")}
            Preferred Source: {current_source}
            Available Fields: {fields_str}
            Formula: {metric.get("sql_formula", "N/A")}
            Business Logic:
            {business_logic_text if business_logic_text else "- Follow standard database filtering rules."}
            """

        if not metric_text:
            metric_section = "No exact metric found in glossary. Rely on standard table definitions."
        else:
            metric_section = metric_text

        prompt = textwrap.dedent(f"""
        You are a SQL expert.
        Use SQLite syntax only.

        Worked Examples:
        {FEW_SHOT_EXAMPLES}

        User Question:
        {question}

        Relevant Business Metrics:
        {metric_section}

        Relevant Time Period Rules:
        {time_period_text if time_period_text else NO_TIME_PERIOD_RULE}

        Database Schema:
        {schema_text if schema_text else "No schema context retrieved."}

        Instructions:
        - Generate SQL only.
        - Return exactly ONE SQL statement. Do not answer with multiple queries.
        - If a mart is pre-aggregated, use its existing aggregate columns; never filter
          it by a column that is not listed in the schema context above.
        - DO NOT use raw tables (orders, order_items, products) if a mart table is available.
        - Only use tables and columns provided in the schema context.
        - Do not invent tables or columns.
        - Follow glossary definitions, metric formulas, and business rules exactly.
        - If business rules conflict with formulas, business rules take precedence.
        - Use the correct grain of each mart and avoid double counting when joining tables.
        - Only apply a date filter if the question explicitly asks for a time period.
          If the question asks for an overall/all-time figure, do NOT invent one.
        - When the question DOES ask for a relative period ("last 30 days"), use the latest
          available date in the selected source table as the reference date.
        - Replace {{date_field}} with the appropriate date column.  
        - Alias the final metric using the metric key (e.g., SUM(order_revenue) AS revenue).
        - SQLite Type Safety: When dividing two COUNT() or SUM() values, always CAST the numerator or columns to REAL (e.g., CAST(COUNT(...) AS REAL)) to prevent integer division from rounding down to 0.
        - Dimension Date Rule: For mart_product_sales, use 'order_date'. Ensure time format matches exactly (e.g., STRFTIME('%Y%m', order_date)).
        - Alias exactness: Follow the metric key alias rule strictly. If calculating return_rate, alias AS return_rate. If calculating aov, alias AS aov.
        """).strip()

        return prompt