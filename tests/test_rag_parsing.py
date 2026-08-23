"""
Filename-derived metadata for the report corpus.

year and doc_type are extracted once at parse time and are what
retrieval.py filters on, so a mistake here silently narrows or widens every
filtered search. Pure string functions: no PDFs, no LLM, no network.
"""

import pytest

from src.rag.parsing import classify_doc_type, extract_year


@pytest.mark.parametrize("filename,expected", [
    ("aritzia_annual_report_2024.pdf", 2024),
    ("ecommerce_benchmarks_2021.pdf", 2021),
    ("report-2000.pdf", 2000),
    ("report-2029.pdf", 2029),
])
def test_extract_year_finds_years_in_range(filename, expected):
    assert extract_year(filename) == expected


@pytest.mark.parametrize("filename", [
    "industry_trends.pdf",          # no digits at all
    "report_1999.pdf",              # before the 20xx window
    "report_2030.pdf",              # after the 202x window
    "sku_12345.pdf",                # digits that are not a year
])
def test_extract_year_returns_none_when_there_is_no_year(filename):
    assert extract_year(filename) is None


def test_extract_year_takes_the_first_match():
    assert extract_year("aritzia_2023_vs_2024.pdf") == 2023


@pytest.mark.parametrize("filename", [
    "aritzia_annual_report_2024.pdf",
    "LULULEMON_10-K_2023.PDF",      # matching is case-insensitive
    "inditex_earnings_2022.pdf",
    "zara_2024.pdf",
])
def test_company_keywords_classify_as_company_report(filename):
    assert classify_doc_type(filename) == "company_report"


@pytest.mark.parametrize("filename", [
    "ecommerce_industry_benchmarks_2021.pdf",
    "retail_market_trends.pdf",
    "something_unlabelled.pdf",     # the default
])
def test_everything_else_is_an_industry_benchmark(filename):
    assert classify_doc_type(filename) == "industry_benchmark"
