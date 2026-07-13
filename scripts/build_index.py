"""
scripts/build_index.py
----------------------
Build or update the RAG index (parsed_docs.pkl + the Chroma vector store) from
whatever PDFs are in data/reports/.

This is the RAG counterpart to scripts/build_db.py: one command that makes the
retrieval corpus reproducible from the source documents.

Why it exists: the parsed-document cache was trusted blindly. If the pickle
existed it was reused, regardless of the folder contents, so a newly added PDF
was silently never indexed and retrieval simply could not see it — with no error.
Two reports (Aritzia 2023, global-e-commerce-trend) sat unindexed this way.

By default this is INCREMENTAL: it parses only the PDFs that are not already in
the cache and adds them to the existing vector store, rather than re-embedding
the whole corpus (~105k chunks).

Usage:
    python scripts/build_index.py              # index any new PDFs
    python scripts/build_index.py --rebuild    # wipe and rebuild everything
    python scripts/build_index.py --check      # report drift, change nothing
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.parsing import (  # noqa: E402
    load_parsed_documents,
    load_and_parse_pdfs,
    save_parsed_documents,
    find_unindexed_pdfs,
    pdfs_on_disk,
    indexed_sources,
    parse_pdf,
)
from src.rag.vectorstore import (  # noqa: E402
    create_vectorstore,
    load_vectorstore,
    _make_child_docs,
)
from src.rag.config import CHROMA_DIR, REPORTS_FOLDER, PARSED_DOCS_FILE  # noqa: E402

from langchain_core.documents import Document  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Build or update the RAG index.")
    ap.add_argument("--rebuild", action="store_true",
                    help="Wipe the cache and vector store and re-index every PDF.")
    ap.add_argument("--check", action="store_true",
                    help="Report drift only; make no changes.")
    args = ap.parse_args()

    on_disk = pdfs_on_disk(REPORTS_FOLDER)
    if not on_disk:
        sys.exit(f"[error] No PDFs found in {REPORTS_FOLDER}")

    cached = load_parsed_documents() or []
    unindexed = find_unindexed_pdfs(cached, REPORTS_FOLDER)

    print(f"\nPDFs in {REPORTS_FOLDER}: {len(on_disk)}")
    print(f"Already indexed:          {len(indexed_sources(cached))}")
    print(f"Not yet indexed:          {len(unindexed)}")
    for f in sorted(unindexed):
        print(f"  - {f}")

    if args.check:
        print("\n--check: no changes made.")
        return 1 if unindexed else 0

    # ---- full rebuild -------------------------------------------------------
    if args.rebuild:
        print("\nFull rebuild: wiping cache and vector store.")
        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)
        if os.path.exists(PARSED_DOCS_FILE):
            os.remove(PARSED_DOCS_FILE)
        docs = load_and_parse_pdfs()
        save_parsed_documents(docs)
        create_vectorstore(docs)
        print(f"\nRebuilt index over {len(on_disk)} PDFs ({len(docs)} elements).")
        return 0

    # ---- incremental --------------------------------------------------------
    if not unindexed:
        print("\nIndex is up to date. Nothing to do.")
        return 0

    new_docs: list[Document] = []
    for name in sorted(unindexed):
        new_docs.extend(parse_pdf(name))

    if not new_docs:
        print("\nNo text extracted from the new PDFs.")
        return 0

    # Append to the document cache.
    save_parsed_documents(cached + new_docs)

    # Add ONLY the new chunks to the existing store (re-embedding the whole corpus
    # would be wasteful; the old vectors are still valid).
    if os.path.exists(CHROMA_DIR):
        vs = load_vectorstore()
        children = _make_child_docs(new_docs)
        print(f"Adding {len(children)} new chunks to the existing vector store...")
        vs.add_documents(children)
    else:
        create_vectorstore(cached + new_docs)

    print(f"\nIndexed {len(unindexed)} new PDF(s), {len(new_docs)} elements.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
