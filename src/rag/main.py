"""
main.py
───────
Entry point. Orchestrates the pipeline:
  parse → index → retrieve → answer.

Run from the project root with: python -m src.rag.main
"""

import os
from dotenv import load_dotenv

from .parsing    import load_and_parse_pdfs, save_parsed_documents, load_parsed_documents
from .vectorstore import create_vectorstore, load_vectorstore, rebuild_child_docs
from .retrieval  import create_retriever
from .chain      import build_rag_chain
from .config     import CHROMA_DIR

load_dotenv()


def run_queries(vectorstore, child_docs):
    """
    Run queries across three retrieval modes to demonstrate the pipeline:

    A) Full corpus      — hybrid search across all documents
    B) Benchmarks only  — filtered to industry_benchmark doc_type
    C) Most recent year — filtered to the latest year in the corpus

    Modes B and C prove metadata filtering is working correctly.
    """
    most_recent_year = max(
        (d.metadata.get("year") for d in child_docs if d.metadata.get("year")),
        default=None,
    )

    test_cases = [
        # (label, question, year_filter, doc_type_filter)
        ("All docs",
         "What were the key ecommerce growth trends reported?",
         None, None),

        ("All docs",
         "What strategies are companies using to grow ecommerce sales?",
         None, None),

        ("All docs",
         "What are the main reasons customers abandon their shopping carts?",
         None, None),

        ("Industry benchmarks only",
         "What is the average ecommerce conversion rate according to industry data?",
         None, "industry_benchmark"),

        (f"Year {most_recent_year} only",
         f"What were the most significant ecommerce developments in {most_recent_year}?",
         most_recent_year, None),
    ]

    print("\n" + "=" * 60)
    print("RAG PIPELINE — Query Results")
    print("=" * 60)

    for label, question, year, doc_type in test_cases:
        retriever = create_retriever(vectorstore, child_docs,
                                     year=year, doc_type=doc_type)
        answer = build_rag_chain(retriever).invoke(question)
        print(f"\n[{label}]")
        print(f"Q: {question}")
        print(f"A: {answer}")
        print("-" * 60)


if __name__ == "__main__":
    print("Ecommerce Analytics RAG Pipeline\n")

    # ── Step 1: Parse or load documents ──────────────────────────────
    docs = load_parsed_documents()
    if docs is None:
        docs = load_and_parse_pdfs()
        save_parsed_documents(docs)

    if not docs:
        print("No documents found. Add PDFs to data/reports/ and try again.")
        raise SystemExit

    # Corpus summary — verify metadata extraction worked before querying
    years     = sorted(set(d.metadata.get("year") for d in docs if d.metadata.get("year")))
    doc_types = sorted(set(d.metadata.get("doc_type") for d in docs))
    print(f"Corpus summary")
    print(f"  Years:     {years}")
    print(f"  Doc types: {doc_types}")
    print(f"  Elements:  {len(docs)}\n")

    # ── Step 2: Build or load vector store ───────────────────────────
    if os.path.exists(CHROMA_DIR):
        vectorstore = load_vectorstore()
        child_docs  = rebuild_child_docs(docs)  # BM25 is in-memory, rebuilt each run
    else:
        vectorstore, child_docs = create_vectorstore(docs)

    # ── Step 3: Query ─────────────────────────────────────────────────
    run_queries(vectorstore, child_docs)