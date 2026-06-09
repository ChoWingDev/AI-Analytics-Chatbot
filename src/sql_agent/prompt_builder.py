import textwrap

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

        User Question:
        {question}

        Relevant Business Metrics:
        {metric_section}

        Relevant Time Period Rules:
        {time_period_text if time_period_text else "No specific time period matched."}

        Database Schema:
        {schema_text if schema_text else "No schema context retrieved."}

        Instructions:
        - Generate SQL only.
        - DO NOT use raw tables (orders, order_items, products) if a mart table is available.
        - Only use tables and columns provided in the schema context.
        - Do not invent tables or columns.
        - Follow glossary definitions, metric formulas, and business rules exactly.
        - If business rules conflict with formulas, business rules take precedence.
        - Use the correct grain of each mart and avoid double counting when joining tables.
        - For relative date filters, use the latest available date in the selected source table as the reference date.
        - Replace {{date_field}} with the appropriate date column.  
        - Alias the final metric using the metric key (e.g., SUM(order_revenue) AS revenue).
        - SQLite Type Safety: When dividing two COUNT() or SUM() values, always CAST the numerator or columns to REAL (e.g., CAST(COUNT(...) AS REAL)) to prevent integer division from rounding down to 0.
        - Dimension Date Rule: For mart_product_sales, use 'order_date'. Ensure time format matches exactly (e.g., STRFTIME('%Y%m', order_date)).
        - Alias exactness: Follow the metric key alias rule strictly. If calculating return_rate, alias AS return_rate. If calculating aov, alias AS aov.
        """).strip()

        return prompt