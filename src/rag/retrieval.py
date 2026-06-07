"""
retrieval.py
────────────
Hybrid BM25 + vector retrieval with optional metadata filtering.
Nothing here knows about prompts, LLMs, or the RAG chain.
"""

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_community.retrievers import BM25Retriever

from .config import RETRIEVER_K


def _rrf_merge(bm25_docs, vector_docs, bm25_weight=0.4, vector_weight=0.6):
    """
    Reciprocal Rank Fusion (RRF) — merges two ranked doc lists into one.

    How it works:
      Every doc scores weight / (rank + 1) from each retriever.
      Scores are summed, duplicates merged, results sorted descending.

    Why BM25 + vector?
      Pure vector misses exact keyword matches ("churn rate: 72%").
      Pure BM25 misses semantic matches ("customer retention" ≈ "churn").
      Together they cover both signals.

    Weights — BM25: 0.4, vector: 0.6:
      Weighted toward semantic because queries are natural-language questions.
    """
    scores:  dict[str, float]    = {}
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(bm25_docs):
        key = doc.page_content
        scores[key]  = scores.get(key, 0) + bm25_weight / (rank + 1)
        doc_map[key] = doc

    for rank, doc in enumerate(vector_docs):
        key = doc.page_content
        scores[key]  = scores.get(key, 0) + vector_weight / (rank + 1)
        doc_map[key] = doc

    return [doc_map[k] for k in sorted(scores, key=lambda k: scores[k], reverse=True)]


def create_retriever(vectorstore, child_docs: list[Document],
                     year: int = None, doc_type: str = None,
                     k: int = RETRIEVER_K):
    """
    Build a hybrid RRF retriever, optionally scoped to a metadata subset.

    Parameters:
      year      — filter to a specific year, e.g. 2024
      doc_type  — 'company_report' or 'industry_benchmark'
      k         — results per retriever before fusion

    Filtering is applied at both layers simultaneously:
      BM25:   pre-filters the in-memory corpus to matching docs
      Chroma: applies a 'where' clause in search_kwargs

    Examples:
      create_retriever(vs, docs)                                     # all docs
      create_retriever(vs, docs, doc_type="industry_benchmark")      # benchmarks only
      create_retriever(vs, docs, year=2024, doc_type="company_report")
    """
    # Build Chroma metadata filter
    chroma_filter = {}
    if year and doc_type:
        chroma_filter = {"$and": [{"year": {"$eq": year}},
                                  {"doc_type": {"$eq": doc_type}}]}
    elif year:
        chroma_filter = {"year": {"$eq": year}}
    elif doc_type:
        chroma_filter = {"doc_type": {"$eq": doc_type}}

    # Pre-filter BM25 corpus to the same subset
    filtered = child_docs
    if year:
        filtered = [d for d in filtered if d.metadata.get("year") == year]
    if doc_type:
        filtered = [d for d in filtered if d.metadata.get("doc_type") == doc_type]

    if not filtered:
        print(f"Warning: no docs matched (year={year}, doc_type={doc_type}). "
              "Falling back to full corpus.")
        filtered      = child_docs
        chroma_filter = {}

    bm25 = BM25Retriever.from_documents(filtered)
    bm25.k = k

    search_kwargs = {"k": k}
    if chroma_filter:
        search_kwargs["filter"] = chroma_filter
    vector = vectorstore.as_retriever(search_kwargs=search_kwargs)

    def hybrid_retrieve(query: str) -> list[Document]:
        bm25_docs   = bm25.invoke(query)
        vector_docs = vector.invoke(query)
        return _rrf_merge(bm25_docs, vector_docs)

    return RunnableLambda(hybrid_retrieve)