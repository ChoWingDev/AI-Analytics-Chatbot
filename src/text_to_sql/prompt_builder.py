from retriever import GlossaryRetriever
import textwrap


class PromptBuilder:
    def build_prompt(self, question, retrieval_result):
        metrics = retrieval_result.get("metrics", [])
        time_periods = retrieval_result.get("time_periods", [])
        schema = retrieval_result.get("schema", {})

        time_period_text = ""
        for key, sql_expr in time_periods.items():
            time_period_text += f"\n- {key}: {sql_expr}"
            
        schema_text = ""
        for table, columns in schema.items():
            schema_text += f"\nTable: {table}\n"
            schema_text += "\n".join(
                [f"- {col}" for col in columns]
            )
            schema_text += "\n"

        for i, metric in enumerate(metrics[:3]):
            prompt += f"\nMetric {i+1}: {metric['business_name']}"
            prompt += f"\n  Formula: {metric['sql_formula']}"
            prompt += f"\n  Source: {metric['preferred_source']}"
            prompt += f"\n  Fields: {', '.join(metric['preferred_fields'])}\n"

            prompt = textwrap.dedent(f"""
            You are a SQL expert.
            Use SQLite syntax only.

            Question: {question}

            Business Metric: {metric.get("business_name", "N/A")}
            Metric SQL Alias: {metric.get("metric", "result")}
            Preferred Source: {metric.get("preferred_source", "N/A")}
            Fields: {", ".join(metric.get("preferred_fields", []))}
            Formula: {metric.get("sql_formula", "N/A")}
            Time Period: {", ".join(time_periods)}

            Database Schema:
            {schema_text}

            Generate SQL only.
            Only use the tables and columns listed above.
            Do not invent table names or column names.
            Always alias the selected metric as the Metric SQL Alias.
            For relative time periods, do not use DATE('now'). Use MAX(date column) from the preferred table instead.
            For revenue, ROI, AOV, and delivery metrics, filter completed orders using status = 'Complete' when status column exists.
            """).strip()

        else:
            prompt = textwrap.dedent(f"""
            You are a SQL expert.
            Use SQLite syntax only.

            Question: {question}

            No exact metric found in glossary.
            Time Period: {", ".join(time_periods)}

            Database Schema:
            {schema_text}

            Generate SQL only.
            Only use the tables and columns listed above.
            Do not invent table names or column names.
            """).strip()

        return prompt
    
if __name__ == "__main__":

    question = "What is total revenue in last 30 days?"

    retriever = GlossaryRetriever("/Users/chowingchan/Desktop/Project/AI-Analytics-Copilot/AI-Analytics-Copilot/config/glossary.json",
                                  "/Users/chowingchan/Desktop/Project/AI-Analytics-Copilot/Competitive-Intelligence-Internal-Analytics-System/data/database/thelook_ecommerce.db")
    retrieval_result = retriever.retrieve(question)
    builder = PromptBuilder()

    prompt = builder.build_prompt(
        question,
        retrieval_result
    )

    print(prompt)
    print(question)
    print(retrieval_result) 