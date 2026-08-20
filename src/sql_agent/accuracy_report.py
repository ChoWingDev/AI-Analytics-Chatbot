"""
End-to-End Text-to-SQL Accuracy Report

This script evaluates the actual Text-to-SQL pipeline.

For each test case:
1. Read the user question
2. Run the TextToSQLPipeline to generate SQL using the LLM
3. Execute the expected SQL
4. Execute the generated SQL
5. Compare both results
6. Save an accuracy report
"""

import json
import sqlite3
import pandas as pd

from pathlib import Path

from .config import DB_PATH, GLOSSARY_PATH, PROJECT_ROOT, TEST_CASE_PATH
from .sql_pipeline import TextToSQLPipeline
from .sql_evaluator import evaluate_sql


def load_test_cases(path):
    """
    Load test cases from JSON file.

    Each test case should contain:
    - question
    - expected_sql
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    # Load test cases
    test_cases = load_test_cases(TEST_CASE_PATH)

    # Initialize Text-to-SQL pipeline
    pipeline = TextToSQLPipeline(
        glossary_path=GLOSSARY_PATH,
        db_path=DB_PATH
    )

    # Open database connection for evaluation
    conn = sqlite3.connect(DB_PATH)

    results = []

    for case in test_cases:
        question = case["question"]

        print("\n==============================")
        print("QUESTION:")
        print(question)

        # Generate SQL using actual pipeline
        pipeline_output = pipeline.run(question)
        generated_sql = pipeline_output["sql"]

        print("\nGENERATED SQL:")
        print(generated_sql)

        # Compare expected SQL result vs generated SQL result
        evaluation = evaluate_sql(
            question=question,
            expected_sql=case["expected_sql"],
            generated_sql=generated_sql,
            conn=conn
        )

        evaluation["expected_sql"] = case["expected_sql"]
        evaluation["generated_sql"] = generated_sql

        results.append(evaluation)

    results_df = pd.DataFrame(results)

    # Calculate execution accuracy
    accuracy = results_df["passed"].mean()

    print("\n===== END-TO-END TEXT-TO-SQL ACCURACY REPORT =====")
    print(f"Total Cases: {len(results_df)}")
    print(f"Passed: {int(results_df['passed'].sum())}")
    print(f"Failed: {len(results_df) - int(results_df['passed'].sum())}")
    print(f"Execution Accuracy: {accuracy:.2%}")

    print("\n===== CASE DETAILS =====")
    print(results_df[["question", "passed", "reason"]])

    # Save detailed report
    report_path = Path(PROJECT_ROOT) / "outputs" / "e2e_sql_accuracy_report.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(report_path, index=False)

    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()