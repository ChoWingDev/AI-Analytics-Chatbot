"""
Context formatting for the RAG prompt.

The prompt promises citations with source, page, year and doc_type. That
promise is only keepable if format_docs actually puts those fields in front of
the model, which is what these tests pin down. No LLM involved.
"""

from langchain_core.documents import Document

from src.rag.chain import format_docs


def _doc(text="Return rates average 40% in apparel.", **metadata):
    return Document(page_content=text, metadata=metadata)


def test_every_citation_field_reaches_the_prompt():
    out = format_docs([_doc(
        source="ecommerce_benchmarks_2021.pdf",
        page=12,
        year=2021,
        doc_type="industry_benchmark",
    )])
    for expected in ("ecommerce_benchmarks_2021.pdf", "12", "2021", "industry_benchmark"):
        assert expected in out
    assert "Return rates average 40% in apparel." in out


def test_missing_metadata_degrades_instead_of_raising():
    out = format_docs([_doc()])
    assert "unknown" in out and "?" in out


def test_chunks_are_separated():
    out = format_docs([_doc("first", source="a.pdf"), _doc("second", source="b.pdf")])
    assert "---" in out
    assert out.index("first") < out.index("second")


def test_no_documents_produces_empty_context():
    assert format_docs([]) == ""
