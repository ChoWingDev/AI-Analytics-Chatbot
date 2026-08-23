"""Chart selection rules — see app/charts.py. No Streamlit involved."""

import pandas as pd
import pytest

from app.charts import choose_chart, format_metric


def test_single_number_is_a_metric():
    kind, params = choose_chart(pd.DataFrame({"return_rate": [10.01]}))
    assert kind == "metric"
    assert params["value"] == 10.01
    assert params["label"] == "return_rate"


def test_single_row_with_a_label_column_uses_the_label():
    df = pd.DataFrame({"category": ["Jeans"], "revenue": [1234.5]})
    kind, params = choose_chart(df)
    assert kind == "metric"
    assert params["label"] == "Jeans"


def test_date_column_and_numeric_is_a_line():
    df = pd.DataFrame({"month": ["2022-01", "2022-02", "2022-03"],
                       "revenue": [1.0, 2.0, 3.0]})
    assert choose_chart(df) == ("line", {"x": "month", "y": "revenue"})


def test_integer_year_column_is_a_time_axis_not_a_value():
    # `year` is numeric but names a time axis; it must not be charted as a value.
    df = pd.DataFrame({"year": [2020, 2021, 2022], "aov": [50.0, 55.0, 60.0]})
    assert choose_chart(df) == ("line", {"x": "year", "y": "aov"})


def test_text_column_and_numeric_is_a_bar():
    df = pd.DataFrame({"category": ["Jeans", "Tops", "Shoes"],
                       "revenue": [3.0, 2.0, 1.0]})
    assert choose_chart(df) == ("bar", {"x": "category", "y": "revenue"})


def test_two_row_result_still_bars():
    df = pd.DataFrame({"category": ["Jeans", "Tops"], "revenue": [3.0, 2.0]})
    assert choose_chart(df)[0] == "bar"


def test_two_point_time_series_is_not_a_line():
    # Two points make a line chart that says less than the two numbers do.
    df = pd.DataFrame({"month": ["2022-01", "2022-02"], "revenue": [1.0, 2.0]})
    assert choose_chart(df)[0] != "line"


@pytest.mark.parametrize("df", [
    None,
    pd.DataFrame(),
    # too many bars to read
    pd.DataFrame({"category": [f"c{i}" for i in range(30)],
                  "revenue": list(range(30))}),
    # two value columns: which one would we plot?
    pd.DataFrame({"category": ["a", "b"], "revenue": [1, 2], "orders": [3, 4]}),
    # two text columns: which one is the axis?
    pd.DataFrame({"category": ["a", "b"], "brand": ["x", "y"], "revenue": [1, 2]}),
    # no numeric column at all
    pd.DataFrame({"category": ["a", "b"]}),
])
def test_ambiguous_shapes_fall_back_to_the_table(df):
    assert choose_chart(df) == (None, {})


# ── Metric formatting ────────────────────────────────────────────────────────

def test_a_rate_is_shown_as_a_percentage():
    # 0.1001 rendered as "0.10" contradicted a summary reading "10.01%".
    assert format_metric("return_rate", 0.1001) == "10.01%"


def test_a_rate_already_in_percent_is_left_alone():
    assert format_metric("return_rate", 10.01) == "10.01"


def test_plain_numbers_keep_their_precision():
    assert format_metric("aov", 62.4567) == "62.4567"
    assert format_metric("revenue", 1234.5) == "1,234.5"


def test_integers_are_thousands_separated():
    assert format_metric("active_users", 104700) == "104,700"
