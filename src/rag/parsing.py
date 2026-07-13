"""
parsing.py
──────────
PDF parsing, document serialization (save/load pickle).
Nothing here knows about embeddings, retrievers, or LLMs.
"""

import os
import re
import pickle

from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document

from .config import (
    REPORTS_FOLDER,
    PARSED_DOCS_FILE,
    COMPANY_KEYWORDS,
)


def extract_year(filename: str) -> int | None:
    """
    Pull a 4-digit year (2000–2029) from a filename.
    e.g. 'aritzia_annual_report_2024.pdf' → 2024

    Why: year is the most useful metadata filter for this project.
    A PM asking 'how did we perform last year?' should only hit 2024 docs.
    """
    match = re.search(r"(20[0-2]\d)", filename)
    return int(match.group(1)) if match else None


def classify_doc_type(filename: str) -> str:
    """
    Classify a PDF as 'company_report' or 'industry_benchmark' from its filename.
    Defaults to 'industry_benchmark' if no company keyword matches.

    Why: lets retrieval target the right document set.
    Company questions hit annual reports; market sizing questions hit benchmarks.
    """
    lower = filename.lower()
    if any(kw in lower for kw in COMPANY_KEYWORDS):
        return "company_report"
    return "industry_benchmark"


def _pypdf_fallback(filepath: str, filename: str,
                    year: int | None, doc_type: str) -> list[Document]:
    """
    Page-level extraction with pypdf.

    Why this exists: `unstructured`'s fast (pdfminer) path returns ZERO elements
    on some PDFs while raising no exception at all — the parse "succeeds" and
    silently contributes nothing. Two of the twelve reports (Aritzia 2023 and
    global-e-commerce-trend) were invisible to retrieval for exactly this reason,
    even though both have a perfectly good text layer.

    Coarser than element-level parsing (one Document per page), but a whole page
    of real text beats a silently dropped document.
    """
    from pypdf import PdfReader

    docs: list[Document] = []
    reader = PdfReader(filepath)
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(Document(
                page_content=text,
                metadata={
                    "source":       filename,
                    "page":         page_number,
                    "element_type": "PdfPage",
                    "industry":     "ecommerce",
                    "year":         year,
                    "doc_type":     doc_type,
                },
            ))
    return docs


def parse_pdf(filename: str, folder: str = REPORTS_FOLDER) -> list[Document]:
    """
    Parse one PDF into tagged Documents, falling back to pypdf if unstructured
    extracts nothing. Returns [] only if the file genuinely has no text.
    """
    filepath = os.path.join(folder, filename)
    year     = extract_year(filename)
    doc_type = classify_doc_type(filename)
    # ASCII only: the Windows console encodes stdout as cp1252 when redirected,
    # and a non-ASCII character here crashes the whole parse with UnicodeEncodeError.
    print(f"Parsing: {filename}  ->  year={year}, type={doc_type}")

    elements = partition_pdf(
        filename=filepath,
        strategy="fast",
        infer_table_structure=True,
        extract_image_block_types=["Image", "Table"],
    )

    docs = [
        Document(
            page_content=element.text.strip(),
            metadata={
                "source":       filename,
                "page":         getattr(element.metadata, "page_number", None),
                "element_type": element.category,
                "industry":     "ecommerce",
                "year":         year,
                "doc_type":     doc_type,
            },
        )
        for element in elements
        if element.text and element.text.strip()
    ]

    if not docs:
        print(f"  [fallback] unstructured extracted 0 elements from {filename}; using pypdf")
        docs = _pypdf_fallback(filepath, filename, year, doc_type)
        if not docs:
            print(f"  [WARN] {filename} yielded no text at all - it may be a scanned "
                  f"PDF needing OCR. It will NOT be searchable.")

    print(f"  -> {len(docs)} documents")
    return docs


def load_and_parse_pdfs(folder: str = REPORTS_FOLDER) -> list[Document]:
    """
    Parse all PDFs in folder.

    Each text element becomes a Document with metadata:
      source, page, element_type, industry, year, doc_type.

    year and doc_type are extracted from the filename at parse time
    so retrieval.py can filter on them without re-parsing.
    """
    all_docs: list[Document] = []
    pdf_files = [f for f in os.listdir(folder) if f.lower().endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF files to process.")

    for filename in pdf_files:
        all_docs.extend(parse_pdf(filename, folder))

    print(f"Total elements extracted: {len(all_docs)}")
    return all_docs


def save_parsed_documents(documents: list[Document],
                          filename: str = PARSED_DOCS_FILE) -> None:
    """Persist parsed documents so PDFs don't need re-parsing on every run."""
    os.makedirs("data", exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(documents, f)
    print(f"Saved {len(documents)} documents to {filename}")


def pdfs_on_disk(folder: str = REPORTS_FOLDER) -> set[str]:
    """Filenames of every PDF currently in the reports folder."""
    if not os.path.isdir(folder):
        return set()
    return {f for f in os.listdir(folder) if f.lower().endswith(".pdf")}


def indexed_sources(documents: list[Document]) -> set[str]:
    """Filenames the cached corpus actually contains."""
    return {d.metadata.get("source") for d in documents if d.metadata.get("source")}


def find_unindexed_pdfs(documents: list[Document],
                        folder: str = REPORTS_FOLDER) -> set[str]:
    """PDFs sitting in the folder that were never parsed into the cache."""
    return pdfs_on_disk(folder) - indexed_sources(documents)


def load_parsed_documents(filename: str = PARSED_DOCS_FILE,
                          folder: str = REPORTS_FOLDER) -> list[Document] | None:
    """
    Load previously persisted documents. Returns None if the cache doesn't exist.

    Also warns about corpus drift. The cache used to be trusted blindly: if the
    pickle existed it was returned, no matter what was in the reports folder. Drop
    a new PDF in and it was silently never indexed — the RAG simply could not see
    it, and there was no error to tell you. (This is exactly what happened:
    global-e-commerce-trend.pdf sat unindexed while questions about ecommerce
    trends retrieved nothing useful.)
    """
    if not os.path.exists(filename):
        print("No saved documents found. Will parse PDFs.")
        return None

    with open(filename, "rb") as f:
        docs = pickle.load(f)
    print(f"Loaded {len(docs)} documents from {filename}")

    unindexed = find_unindexed_pdfs(docs, folder)
    if unindexed:
        print(
            f"\n  [STALE INDEX] {len(unindexed)} PDF(s) in {folder} have never been "
            f"indexed and are invisible to retrieval:"
        )
        for name in sorted(unindexed):
            print(f"    - {name}")
        print("  Run: python scripts/build_index.py\n")

    return docs
