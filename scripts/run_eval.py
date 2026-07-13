"""
scripts/run_eval.py
-------------------
The single entry point for Text-to-SQL evaluation.

There were previously two disconnected, crippled eval scripts: a result-based
one capped at 5 cases, and a rule-based one capped at 1 case with hardcoded
absolute paths (so it could not run at all). This runs BOTH graders over the
same benchmark, uncapped.

Two graders, because they catch different failures:

  1. Result-based  (sql_evaluator.evaluate_sql)
     Executes the generated SQL and the ground-truth SQL and compares the
     RESULT SETS. Two different-looking queries can both be correct, so we
     grade the answer, not the spelling. This is what catches a metric that
     silently returns the wrong number.

  2. Business-rule (business_rule_evaluator.evaluate_business_rules)
     Greps the generated SQL for required patterns (e.g. "did it filter to
     completed orders?", "did it avoid filtering the product mart by status?").
     This catches SQL that happens to produce the right number by luck, or that
     violates a governance rule.

Usage:
    python scripts/run_eval.py                 # all cases
    python scripts/run_eval.py --limit 3       # smoke test (saves API calls)
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.sql_agent.config import DB_PATH, GLOSSARY_PATH, TEST_CASE_PATH  # noqa: E402
from src.sql_agent.sql_pipeline import TextToSQLPipeline  # noqa: E402
from src.sql_agent.sql_evaluator import evaluate_sql  # noqa: E402
from src.sql_agent.business_rule_evaluator import evaluate_business_rules  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "outputs"


def main():
    parser = argparse.ArgumentParser(description="Run the Text-to-SQL benchmark.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N cases (default: all).")
    args = parser.parse_args()

    cases = json.loads(Path(TEST_CASE_PATH).read_text(encoding="utf-8"))
    if args.limit:
        cases = cases[: args.limit]

    pipeline = TextToSQLPipeline(glossary_path=GLOSSARY_PATH, db_path=DB_PATH)
    conn = sqlite3.connect(DB_PATH)

    rows = []
    for i, case in enumerate(cases, 1):
        question = case["question"]
        print(f"\n[{i}/{len(cases)}] {question}")

        # An infrastructure failure (API down, quota exhausted, timeout) is NOT a
        # model-accuracy failure. If we counted them the same, a 402 from the LLM
        # provider would silently tank the reported accuracy and we'd go hunting
        # for a bug in the prompt that doesn't exist. Track them separately.
        pipeline_error = ""
        try:
            generated_sql = pipeline.run(question)["sql"]
        except Exception as e:
            generated_sql = ""
            pipeline_error = str(e)
            print(f"  PIPELINE/API ERROR (not scored): {e}")

        # Grader 1: does it return the right numbers?
        result_eval = evaluate_sql(
            question=question,
            expected_sql=case["expected_sql"],
            generated_sql=generated_sql,
            conn=conn,
        )

        # Grader 2: does it follow the governance rules?
        rule_eval = evaluate_business_rules(
            question=question,
            generated_sql=generated_sql,
            required_checks=case.get("required_sql_checks", []),
        )

        rows.append({
            "question": question,
            "metric": case.get("metric", ""),
            "pipeline_error": pipeline_error,
            "scored": not pipeline_error,
            "result_passed": bool(result_eval.get("passed")),
            "result_reason": result_eval.get("reason", ""),
            "rules_passed": bool(rule_eval.get("passed")),
            "failed_checks": ", ".join(rule_eval.get("failed_checks", [])),
            "unknown_checks": ", ".join(rule_eval.get("unknown_checks", [])),
            "generated_sql": generated_sql,
            "expected_sql": case["expected_sql"],
        })

        if pipeline_error:
            continue
        print(f"  result: {'PASS' if rows[-1]['result_passed'] else 'FAIL'}"
              f"  |  rules: {'PASS' if rows[-1]['rules_passed'] else 'FAIL'}")
        if not rows[-1]["result_passed"]:
            print(f"    why: {rows[-1]['result_reason']}")
        if rows[-1]["failed_checks"]:
            print(f"    failed checks: {rows[-1]['failed_checks']}")

    conn.close()
    df = pd.DataFrame(rows)

    # Accuracy is computed ONLY over cases the model actually answered.
    scored = df[df["scored"]] if len(df) else df
    errored = df[~df["scored"]] if len(df) else df

    exec_acc = scored["result_passed"].mean() if len(scored) else 0.0
    rule_acc = scored["rules_passed"].mean() if len(scored) else 0.0

    print("\n" + "=" * 62)
    print("TEXT-TO-SQL EVALUATION")
    print("=" * 62)
    print(f"Cases in benchmark:    {len(df)}")
    print(f"Scored (got SQL):      {len(scored)}")
    if len(errored):
        print(f"NOT scored (API/infra):{len(errored)}  <- excluded from accuracy")
    print(f"Execution accuracy:    {exec_acc:.1%}  ({int(scored['result_passed'].sum())}/{len(scored)})")
    print(f"Business-rule pass:    {rule_acc:.1%}  ({int(scored['rules_passed'].sum())}/{len(scored)})")

    if len(scored):
        print("\nPer-metric execution accuracy:")
        by_metric = scored.groupby("metric")["result_passed"].agg(["mean", "count"])
        for metric, r in by_metric.iterrows():
            print(f"  {metric:<18} {r['mean']:.0%}  ({int(r['mean'] * r['count'])}/{int(r['count'])})")

        failed = scored[~scored["result_passed"]]
        if len(failed):
            print("\nFailing cases (genuine SQL errors):")
            for _, r in failed.iterrows():
                print(f"  - {r['question']}\n      {r['result_reason']}")

    if len(errored):
        print("\nUnscored cases (infrastructure, not model quality):")
        for _, r in errored.iterrows():
            print(f"  - {r['question']}\n      {r['pipeline_error'][:100]}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    csv_path = OUTPUT_DIR / "sql_evaluation_result.csv"
    json_path = OUTPUT_DIR / "sql_evaluation_summary.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({
        "total_cases": len(df),
        "scored_cases": len(scored),
        "unscored_api_errors": len(errored),
        "execution_accuracy": float(exec_acc),
        "business_rule_pass_rate": float(rule_acc),
        "passed": int(scored["result_passed"].sum()) if len(scored) else 0,
    }, indent=2), encoding="utf-8")

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
