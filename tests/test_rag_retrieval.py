"""
Hybrid retrieval: rank fusion and metadata filtering.

_rrf_merge decides what the LLM actually sees, and the filter path is easy to
break in a way that silently returns the wrong document set. A stub vectorstore
stands in for Chroma, so these run with no embeddings, no database, and no
network.
"""

from langchain_core.documents import Document

from src.rag.retrieval import _rrf_merge, create_retriever


def _doc(text, **metadata):
    return Document(page_content=text, metadata=metadata)


# ── Rank fusion ──────────────────────────────────────────────────────────────

def test_a_document_both_retrievers_found_outranks_one_only_found_once():
    both = _doc("found twice")
    bm25_only = _doc("bm25 only")
    vector_only = _doc("vector only")

    merged = _rrf_merge([both, bm25_only], [both, vector_only])

    assert merged[0].page_content == "found twice"


def test_ties_favour_the_vector_list():
    # Same rank on each side; the 0.6 vector weight beats the 0.4 BM25 weight
    # because the queries are natural-language questions.
    bm25_top = _doc("keyword hit")
    vector_top = _doc("semantic hit")

    merged = _rrf_merge([bm25_top], [vector_top])

    assert [d.page_content for d in merged] == ["semantic hit", "keyword hit"]


def test_duplicates_are_merged_not_repeated():
    shared = _doc("same text")
    merged = _rrf_merge([shared], [Document(page_content="same text", metadata={})])

    assert len(merged) == 1


def test_rank_order_within_one_list_is_preserved():
    first, second, third = (_doc(f"doc {i}") for i in range(3))
    merged = _rrf_merge([first, second, third], [])

    assert [d.page_content for d in merged] == ["doc 0", "doc 1", "doc 2"]


def test_empty_inputs_give_an_empty_result():
    assert _rrf_merge([], []) == []


# ── Filtering ────────────────────────────────────────────────────────────────

CORPUS = [
    _doc("aritzia 2024 revenue", year=2024, doc_type="company_report"),
    _doc("aritzia 2021 revenue", year=2021, doc_type="company_report"),
    _doc("industry 2021 return rate", year=2021, doc_type="industry_benchmark"),
]


class _StubRetriever:
    def invoke(self, query):
        return []


class _StubVectorstore:
    """Records the search_kwargs Chroma would have received."""

    def __init__(self):
        self.search_kwargs = None

    def as_retriever(self, search_kwargs=None):
        self.search_kwargs = search_kwargs
        return _StubRetriever()


def _bm25_hits(retriever, query="revenue return rate aritzia industry"):
    """Invoke the hybrid retriever; with a stub vector side these are BM25 only."""
    return retriever.invoke(query)


def test_no_filter_searches_the_whole_corpus():
    vs = _StubVectorstore()
    retriever = create_retriever(vs, CORPUS)

    assert "filter" not in vs.search_kwargs
    assert len(_bm25_hits(retriever)) == len(CORPUS)


def test_year_filter_narrows_both_layers():
    vs = _StubVectorstore()
    retriever = create_retriever(vs, CORPUS, year=2021)

    assert vs.search_kwargs["filter"] == {"year": {"$eq": 2021}}
    assert all(d.metadata["year"] == 2021 for d in _bm25_hits(retriever))


def test_doc_type_filter_narrows_both_layers():
    vs = _StubVectorstore()
    retriever = create_retriever(vs, CORPUS, doc_type="company_report")

    assert vs.search_kwargs["filter"] == {"doc_type": {"$eq": "company_report"}}
    assert all(d.metadata["doc_type"] == "company_report" for d in _bm25_hits(retriever))


def test_year_and_doc_type_combine_with_and():
    vs = _StubVectorstore()
    retriever = create_retriever(vs, CORPUS, year=2021, doc_type="company_report")

    assert vs.search_kwargs["filter"] == {
        "$and": [{"year": {"$eq": 2021}}, {"doc_type": {"$eq": "company_report"}}]
    }
    assert len(_bm25_hits(retriever)) == 1


def test_a_filter_that_matches_nothing_falls_back_to_the_full_corpus():
    # Returning nothing at all would look like "the reports have no answer",
    # which is a different and more misleading failure than an unfiltered search.
    vs = _StubVectorstore()
    retriever = create_retriever(vs, CORPUS, year=1999)

    assert "filter" not in vs.search_kwargs
    assert len(_bm25_hits(retriever)) == len(CORPUS)
