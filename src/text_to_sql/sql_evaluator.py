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

    try:
        if np.allclose(
            expected_norm.to_numpy().astype(float),
            generated_norm.to_numpy().astype(float),
            rtol=0.01,
            atol=0.01
        ):
            return {
                "passed": True,
                "reason": "Result matched"
            }
    except Exception:
        pass

    if expected_norm.equals(generated_norm):
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