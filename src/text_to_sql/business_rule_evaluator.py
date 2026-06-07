import re
import json
import pandas as pd

from pipeline import TextToSQLPipeline

def normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.lower()).strip()


def contains_any(sql: str, keywords: list[str]) -> bool:
    return any(keyword.lower() in sql for keyword in keywords)


RULES = {
    # completed / returned filters
    "uses_completed_order_filter": lambda sql: (
        "status = 'complete'" in sql
        or 'status = "complete"' in sql
        or "status='complete'" in sql
        or 'status="complete"' in sql
        or "total_orders > 0" in sql
        or "gross_sales" in sql
    ),

    "uses_returned_order_definition": lambda sql: (
        "status = 'returned'" in sql
        or 'status = "returned"' in sql
        or "returned_at" in sql
        or "order_returned_at" in sql
    ),

    "uses_return_status_or_returned_at": lambda sql: (
        "status = 'returned'" in sql
        or "returned_at" in sql
        or "order_returned_at" in sql
    ),

    # metric formulas
    "uses_revenue_metric": lambda sql: (
        "order_revenue" in sql
        or "gross_sales" in sql
        or "sale_price" in sql
    ),

    "uses_roi_formula_gross_profit_over_order_cost": lambda sql: (
        "gross_profit" in sql
        and "order_cost" in sql
        and "/" in sql
    ),

    "uses_aov_formula_revenue_over_distinct_orders": lambda sql: (
        ("order_revenue" in sql or "gross_sales" in sql or "sale_price" in sql)
        and "count" in sql
        and "distinct" in sql
    ),

    "uses_delivery_time_formula_julianday_delivered_minus_created": lambda sql: (
        "julianday" in sql
        and ("delivered_at" in sql or "order_delivered_at" in sql)
        and ("created_at" in sql or "order_created_at" in sql)
    ),

    "uses_revenue_share_formula": lambda sql: (
        "/" in sql
        and ("order_revenue" in sql or "sale_price" in sql)
    ),

    # time filters
    "uses_last_month_filter": lambda sql: (
        "start of month" in sql
        or "-1 month" in sql
        or "last_month" in sql
    ),

    "uses_last_quarter_filter": lambda sql: (
        "-3 month" in sql
        or "quarter" in sql
        or "last_quarter" in sql
    ),

    # count / aggregation
    "counts_distinct_users": lambda sql: (
        "count(distinct user_id" in sql
        or "count (distinct user_id" in sql
    ),

    "counts_distinct_orders": lambda sql: (
        "count(distinct order_id" in sql
        or "count (distinct order_id" in sql
    ),

    "uses_avg_aggregation": lambda sql: "avg(" in sql,

    # grouping / ranking
    "groups_by_product_category": lambda sql: (
        "group by" in sql and "category" in sql
    ),

    "groups_by_traffic_source": lambda sql: (
        "group by" in sql and "traffic_source" in sql
    ),

    "groups_by_product": lambda sql: (
        "group by" in sql
        and ("product_id" in sql or "product_name" in sql)
    ),

    "orders_by_aov_desc": lambda sql: (
        "order by" in sql and "aov" in sql and "desc" in sql
    ),

    "orders_by_revenue_desc": lambda sql: (
        "order by" in sql
        and ("revenue" in sql or "gross_sales" in sql or "sale_price" in sql)
        and "desc" in sql
    ),

    "limits_to_top_1": lambda sql: "limit 1" in sql,

    "limits_to_top_5": lambda sql: "limit 5" in sql,

    # source checks
    "uses_mart_order_summary_or_orders_order_items_fallback": lambda sql: (
        "mart_order_summary" in sql
        or ("orders" in sql and "order_items" in sql)
    ),

    "uses_mart_user_segment_or_orders_fallback": lambda sql: (
    "mart_user_segment" in sql
    or "mart_user_summary" in sql
    or "from orders" in sql
    or "join orders" in sql
    ),

    "defines_returning_customers_as_more_than_one_completed_order": lambda sql: (
        "completed_order_count" in sql
        or "frequency" in sql
        or ("count" in sql and "> 1" in sql)
    ),

    "uses_events_as_primary_source": lambda sql: "events" in sql,

    # customer logic
    "uses_first_completed_order_date": lambda sql: (
        "first_order_date" in sql
        or "min(" in sql
    ),

    "uses_total_completed_revenue_denominator": lambda sql: (
        "order_revenue" in sql
        and "/" in sql
    ),

    "uses_returning_customer_revenue_numerator": lambda sql: (
        "returning" in sql
        or "completed_order_count" in sql
        or "> 1" in sql
    ),

    # churn / attribution / limitation checks
    "uses_inactivity_based_churn_definition": lambda sql: (
        "last_order_date" in sql
        or "recency" in sql
        or "julianday" in sql
    ),

    "uses_default_90_day_churn_window": lambda sql: (
        "90" in sql
    ),

    "uses_completed_order_history": lambda sql: (
    "status = 'complete'" in sql
    or "completed_order_count" in sql
    or "frequency > 0" in sql
    or "frequency>0" in sql
    ),  

    "checks_email_attribution_availability": lambda sql: (
        "email" in sql
        or "traffic_source" in sql
        or "events" in sql
    ),

    "checks_organic_traffic_attribution_availability": lambda sql: (
        "organic" in sql
        or "traffic_source" in sql
        or "events" in sql
    ),

    "does_not_use_roas_or_marketing_spend_without_spend_table": lambda sql: (
        "roas" not in sql
        and "ad_spend" not in sql
        and "marketing_spend" not in sql
        and "campaign_spend" not in sql
    ),

    "does_not_treat_cancelled_or_returned_orders_as_churn": lambda sql: (
        "cancelled" not in sql
        and "returned" not in sql
    ),

    # session / conversion logic
    "uses_distinct_sessions_denominator": lambda sql: (
        "session_id" in sql
        and "distinct" in sql
    ),

    "uses_converted_sessions_numerator": lambda sql: (
        "purchase" in sql
        or "converted" in sql
        or "complete" in sql
    ),

    "does_not_calculate_from_orders_only_without_session_mapping": lambda sql: (
        "events" in sql
        or "session_id" in sql
    ),

    # delivery
    "uses_completed_or_delivered_order_filter": lambda sql: (
        "status = 'complete'" in sql
        or "delivered_at is not null" in sql
        or "order_delivered_at is not null" in sql
    ),

    "requires_delivered_at_not_null": lambda sql: (
        "delivered_at is not null" in sql
        or "order_delivered_at is not null" in sql
    ),
}

def evaluate_business_rules(question: str, generated_sql: str, required_checks: list[str]) -> dict:
    sql = normalize_sql(generated_sql)

    failed_checks = []
    unknown_checks = []

    for check in required_checks:
        rule = RULES.get(check)

        if rule is None:
            unknown_checks.append(check)
            continue

        if not rule(sql):
            failed_checks.append(check)

    return {
        "question": question,
        "passed": len(failed_checks) == 0 and len(unknown_checks) == 0,
        "failed_checks": failed_checks,
        "unknown_checks": unknown_checks,
        "generated_sql": generated_sql,
    }

def load_ground_truth(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":

    ground_truth = load_ground_truth(
        "/Users/chowingchan/Desktop/Project/AI-Analytics-Copilot/AI-Analytics-Copilot/config/ground_truth.json"
    )

    pipeline = TextToSQLPipeline(
        glossary_path="/Users/chowingchan/Desktop/Project/AI-Analytics-Copilot/AI-Analytics-Copilot/config/glossary.json",
        db_path="/Users/chowingchan/Desktop/Project/AI-Analytics-Copilot/Competitive-Intelligence-Internal-Analytics-System/data/database/thelook_ecommerce.db"
    )

    results = []

    for test_case in ground_truth["test_cases"][:1]:

        if test_case.get("category") != "sql":
            continue

        question = test_case["question"]
        required_checks = test_case.get("required_sql_checks", [])

        output = pipeline.run(question)

        print("\nQUESTION:")
        print(question)

        print("\nSQL:")
        print(output["sql"])

        print("\nCHECKS:")
        print(required_checks)

        evaluation = evaluate_business_rules(
            question=question,
            generated_sql=output["sql"],
            required_checks=required_checks
        )
        print("\nFAILED CHECKS:")
        print(evaluation["failed_checks"])

        results.append(evaluation)

    results_df = pd.DataFrame(results)

    print(results_df)

    summary = {
        "total_cases": len(results_df),
        "passed_cases": int(results_df["passed"].sum()),
        "pass_rate": float(results_df["passed"].mean())
    }

    print(summary)