import pandas as pd

#Compare function
def evaluate_sql_result(expected_df, generated_df):
    if expected_df.shape != generated_df.shape:
        return {
            "passed": False,
            "reason": "Shape mismatch",
            "expected_shape": expected_df.shape,
            "generated_shape": generated_df.shape
        }

    if list(expected_df.columns) != list(generated_df.columns):
        return {
            "passed": False,
            "reason": "Column mismatch",
            "expected_columns": list(expected_df.columns),
            "generated_columns": list(generated_df.columns)
        }

    if expected_df.equals(generated_df):
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


# Main evaluator 

def evaluate_sql(
    question,
    expected_sql,
    generated_sql,
    conn
):
    try:
        expected_df = pd.read_sql(expected_sql, conn)
        
    except Exception as e:
        return{
            "question": question,
            "passed": False,
            "stage": "expected_sql_execution",
            "error": str(e)
        }
    
    try:
        generated_df = pd.read_sql(generated_sql, conn)
        
    except Exception as e:
        return{
            "question": question,
            "passed": False,
            "stage": "generated_sql_execution",
            "error":str(e)
        }
    
    comparison_result = evaluate_sql_result(
        expected_df,
        generated_df
    )
    
    comparison_result["question"] = question
    return comparison_result
        