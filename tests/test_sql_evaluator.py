"""
Tests for result comparison.

The bug these lock down: the comparison cast the whole frame to float at once,
so any result with a text column raised, was swallowed, and fell back to exact
equality. Two queries returning the same categories and revenues -- one using
ROUND(x, 2), one not -- were reported as a mismatch, which showed up as a
false failure in the accuracy report.
"""

import pandas as pd

from src.sql_agent.sql_evaluator import evaluate_sql_result


def _mixed(revenues):
    return pd.DataFrame(
        {"category": ["Outerwear & Coats", "Jeans", "Sweaters"], "total_revenue": revenues}
    )


def test_mixed_text_and_float_matches_within_tolerance():
    """The exact case that failed: ROUND(x, 2) versus the raw float."""
    expected = _mixed([4355.92, 3904.69, 2115.66])
    generated = _mixed([4355.920000076294, 3904.6899967193604, 2115.6600021123886])
    assert evaluate_sql_result(expected, generated)["passed"]


def test_mixed_text_and_float_still_catches_a_real_difference():
    expected = _mixed([4355.92, 3904.69, 2115.66])
    generated = _mixed([9999.00, 3904.69, 2115.66])
    assert not evaluate_sql_result(expected, generated)["passed"]


def test_differing_text_fails_even_when_numbers_agree():
    expected = _mixed([4355.92, 3904.69, 2115.66])
    generated = pd.DataFrame(
        {"category": ["Outerwear & Coats", "Jeans", "Swim"],
         "total_revenue": [4355.92, 3904.69, 2115.66]}
    )
    assert not evaluate_sql_result(expected, generated)["passed"]


def test_column_names_are_ignored():
    """Equivalent SQL may alias differently; only values matter."""
    expected = pd.DataFrame({"total_revenue": [1309625.53]})
    generated = pd.DataFrame({"revenue": [1309625.53]})
    assert evaluate_sql_result(expected, generated)["passed"]


def test_numeric_only_frames_use_tolerance():
    expected = pd.DataFrame({"aov": [86.33]})
    generated = pd.DataFrame({"aov": [86.32669547468538]})
    assert evaluate_sql_result(expected, generated)["passed"]


def test_shape_mismatch_is_reported_as_such():
    expected = pd.DataFrame({"a": [1, 2]})
    generated = pd.DataFrame({"a": [1, 2, 3]})
    result = evaluate_sql_result(expected, generated)
    assert not result["passed"]
    assert result["reason"] == "Shape mismatch"
