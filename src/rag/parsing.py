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

from config import (
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


def load_and_parse_pdfs(folder: str = REPORTS_FOLDER) -> list[Document]:
    """
    Parse all PDFs in folder using unstructured's fast strategy.
    Each text element becomes a Document with metadata:
      source, page, element_type, industry, year, doc_type.

    year and doc_type are extracted from the filename at parse time
    so retrieval.py can filter on them without re-parsing.
    """
    all_docs = []
    pdf_files = [f for f in os.listdir(folder) if f.lower().endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF files to process.")

    for filename in pdf_files:
        filepath = os.path.join(folder, filename)
        year     = extract_year(filename)
        doc_type = classify_doc_type(filename)
        print(f"Parsing: {filename}  →  year={year}, type={doc_type}")

        elements = partition_pdf(
            filename=filepath,
            strategy="fast",
            infer_table_structure=True,
            extract_image_block_types=["Image", "Table"],
        )

        for element in elements:
            if element.text and element.text.strip():
                all_docs.append(Document(
                    page_content=element.text.strip(),
                    metadata={
                        "source":       filename,
                        "page":         getattr(element.metadata, "page_number", None),
                        "element_type": element.category,
                        "industry":     "ecommerce",
                        "year":         year,
                        "doc_type":     doc_type,
                    },
                ))

    print(f"Total elements extracted: {len(all_docs)}")
    return all_docs


def save_parsed_documents(documents: list[Document],
                          filename: str = PARSED_DOCS_FILE) -> None:
    """Persist parsed documents so PDFs don't need re-parsing on every run."""
    os.makedirs("data", exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(documents, f)
    print(f"Saved {len(documents)} documents to {filename}")


def load_parsed_documents(filename: str = PARSED_DOCS_FILE) -> list[Document] | None:
    """Load previously persisted documents. Returns None if file doesn't exist."""
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            docs = pickle.load(f)
        print(f"Loaded {len(docs)} documents from {filename}")
        return docs
    print("No saved documents found. Will parse PDFs.")
    return None
