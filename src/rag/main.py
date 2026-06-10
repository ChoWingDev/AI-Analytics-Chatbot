"""
main.py
───────
Entry point. Orchestrates the full pipeline.
Run from the project root with: python -m src.rag.main
"""

import os
from dotenv import load_dotenv

from .parsing      import load_and_parse_pdfs, save_parsed_documents, load_parsed_documents
from .vectorstore  import create_vectorstore, load_vectorstore, rebuild_child_docs
from .retrieval    import create_retriever
from .chain        import build_rag_chain
from .conversation import run_scripted_conversation
from .config       import CHROMA_DIR

load_dotenv()


def run_queries(vectorstore, child_docs):
    """
    5 questions across 3 retrieval modes.
    """
    most_recent_year = max(
        (d.metadata.get("year") for d in child_docs if d.metadata.get("year")),
        default=None,
    )

    test_cases = [
        # (label, question, year_filter, doc_type_filter)
        ("All docs",
         "What are the main reasons customers abandon their shopping carts?",
         None, None),

        (f"Year {most_recent_year} only",
         "What was Aritzia eCommerce net revenue in fiscal 2025 and how did it grow?",
         most_recent_year, "company_report"),

        ("All docs",
         "What ecommerce platform or technology investments did companies make?",
         None, None),

        ("Company reports only",
         "What cybersecurity data privacy or technology risks does Zara or Lululemon mention?",
         None, "company_report"),

        (f"Year {most_recent_year} only",
         f"What were the most significant ecommerce developments reported in {most_recent_year}?",
         most_recent_year, None),
    ]

    print("\n" + "=" * 60)
    print("Hybrid Search + Metadata Filtering")
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

    years     = sorted(set(d.metadata.get("year") for d in docs if d.metadata.get("year")))
    doc_types = sorted(set(d.metadata.get("doc_type") for d in docs))
    print(f"Corpus summary")
    print(f"  Years:     {years}")
    print(f"  Doc types: {doc_types}")
    print(f"  Elements:  {len(docs)}\n")

    # ── Step 2: Build or load vector store ───────────────────────────
    if os.path.exists(CHROMA_DIR):
        vectorstore = load_vectorstore()
        child_docs  = rebuild_child_docs(docs)
    else:
        vectorstore, child_docs = create_vectorstore(docs)

    # ── Step 3: RAG demo ───────────────────────────────────────────
    run_queries(vectorstore, child_docs)

    # ── Step 4: Conversation demo ───────────────────────────────────────────
    run_scripted_conversation(vectorstore, child_docs, session_id="conversation demo")