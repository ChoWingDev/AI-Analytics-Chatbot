import numpy as np
import pandas as pd


def normalize_df(df):
    """
    Normalize SQL result for comparison:
    - Ignore column names
    - Sort rows to avoid row order mismatch
    - Reset index
    """
    normalized = df.copy()
    normalized.columns = range(normalized.shape[1])

    try:
        normalized = normalized.sort_values(
            by=list(normalized.columns)
        ).reset_index(drop=True)
    except Exception:
        normalized = normalized.reset_index(drop=True)

    return normalized


def columns_match(expected, generated, rtol=0.01, atol=0.01):
    """
    Compare two normalized frames column by column.

    Numeric columns are compared with a tolerance: an equivalent query may
    round differently (ROUND(x, 2) versus a raw float) or accumulate a sum in
    a different order. Text columns are compared exactly, after stripping.

    The previous implementation cast the entire frame to float in one call, so
    any result containing a text column -- a category name, a date -- raised,
    was swallowed, and fell through to an exact-equality check. Correct queries
    differing only in the last decimal place were reported as mismatches.
    """
    for col in expected.columns:
        exp, gen = expected[col], generated[col]

        exp_num = pd.to_numeric(exp, errors="coerce")
        gen_num = pd.to_numeric(gen, errors="coerce")

        if exp_num.notna().all() and gen_num.notna().all():
            if not np.allclose(
                exp_num.to_numpy(dtype=float),
                gen_num.to_numpy(dtype=float),
                rtol=rtol,
                atol=atol,
            ):
                return False
        else:
            if not exp.astype(str).str.strip().equals(gen.astype(str).str.strip()):
                return False

    return True


def evaluate_sql_result(expected_df, generated_df):
    """
    Compare SQL execution results only.
    SQL statements do not need to be identical.
    """

    if expected_df.shape != generated_df.shape:
        return {
            "passed": False,
            "reason": "Shape mismatch",
            "expected_shape": expected_df.shape,
            "generated_shape": generated_df.shape,
            "expected_result": expected_df.to_dict(orient="records"),
            "generated_result": generated_df.to_dict(orient="records")
        }

    expected_norm = normalize_df(expected_df)
    generated_norm = normalize_df(generated_df)

    if columns_match(expected_norm, generated_norm):
        return {
            "passed": True,
            "reason": "Result matched"
        }

    return {
        "passed": False,
        "reason": "Result values mismatch",
        "expected_result": expected_df.to_dict(orient="records"),
        "generated_result": generated_df.to_dict(orient="records")
    }


def evaluate_sql(question, expected_sql, generated_sql, conn):
    try:
        expected_df = pd.read_sql(expected_sql, conn)
    except Exception as e:
        return {
            "question": question,
            "passed": False,
            "stage": "expected_sql_execution",
            "reason": f"Expected SQL Error: {str(e)}"
        }

    try:
        generated_df = pd.read_sql(generated_sql, conn)
    except Exception as e:
        return {
            "question": question,
            "passed": False,
            "stage": "generated_sql_execution",
            "reason": f"Expected SQL Error: {str(e)}"
        }

    comparison_result = evaluate_sql_result(expected_df, generated_df)
    comparison_result["question"] = question

    return comparison_result