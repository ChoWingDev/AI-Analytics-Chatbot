"""
Regression tests for SQL extraction.

The bug these lock down: generate_sql used to strip ``` markers without
extracting the statement, so a model that wrapped its query in explanation
had the whole paragraph handed to SQLite. The failure surfaced far from its
cause -- as a "no internal data available" report -- so it is worth catching
at the point of change instead.

Pure string function: no LLM, no database, no network.
"""

import pytest

from src.sql_agent.sql_generator import extract_sql


# The exact response that broke the "return rate vs industry" demo question.
PROSE_WRAPPED = """To compare your return rate to the industry, we need to \
calculate the return rate using the provided formula and business logic. \
Here is the SQL query:

```sql
SELECT
    CAST(SUM(returned_units) AS REAL) / SUM(units_sold) AS return_rate
FROM
    mart_product_sales;
```

### Explanation:
1. **Table Selection**: We use `mart_product_sales` as it contains the fields.
2. **Alias**: We alias the result as `return_rate` to match the metric key.
"""


@pytest.mark.parametrize(
    "label, response, expected",
    [
        (
            "prose before and after a fenced block",
            PROSE_WRAPPED,
            "SELECT\n    CAST(SUM(returned_units) AS REAL) / SUM(units_sold) AS return_rate\nFROM\n    mart_product_sales;",
        ),
        (
            "bare SQL with no fence (the original happy path)",
            "SELECT SUM(order_revenue) AS revenue\nFROM mart_order_summary;",
            "SELECT SUM(order_revenue) AS revenue\nFROM mart_order_summary;",
        ),
        (
            "fence with no language tag",
            "```\nSELECT 1 AS x;\n```",
            "SELECT 1 AS x;",
        ),
        (
            "sqlite language tag",
            "```sqlite\nSELECT 2 AS y\n```",
            "SELECT 2 AS y",
        ),
        (
            "CTE starting with WITH, not SELECT",
            "Here you go:\n```sql\nWITH t AS (SELECT 1 AS n) SELECT n FROM t;\n```\nHope that helps!",
            "WITH t AS (SELECT 1 AS n) SELECT n FROM t;",
        ),
        (
            "unfenced SQL followed by commentary",
            "Sure.\nSELECT 3 AS z;\nThat gives you the answer.",
            "SELECT 3 AS z;",
        ),
    ],
)
def test_extracts_the_statement(label, response, expected):
    assert extract_sql(response) == expected, label


def test_passes_through_when_there_is_no_sql():
    """
    No statement found means the executor should raise a real error, not
    silently run some fragment we guessed at.
    """
    response = "I don't have enough information to write that query."
    assert extract_sql(response) == response
