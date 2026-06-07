"""
vectorstore.py
──────────────
Vector store creation, loading, and child-doc rebuilding for BM25.
Knows about chunking and embeddings — nothing about retrievers or LLMs.
"""

import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    CHROMA_DIR,
    CHROMA_COLLECTION,
    EMBEDDING_MODEL,
    PARENT_CHUNK_SIZE,
    PARENT_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    CHILD_CHUNK_OVERLAP,
)


def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def _make_child_docs(documents: list[Document]) -> list[Document]:
    """
    Two-level parent → child chunking.

    Parent chunks (1500 tokens, 200 overlap):
      Preserve section-level context so the LLM has enough surrounding
      information to interpret a data point correctly.

    Child chunks (400 tokens, 50 overlap):
      Smaller units for high-recall retrieval. Fine-grained enough that
      embeddings capture one idea per chunk without dilution.

    All parent metadata (year, doc_type, source, page) flows into children
    so metadata filtering works at retrieval time without re-parsing.
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=PARENT_CHUNK_OVERLAP
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP
    )

    parent_docs = parent_splitter.split_documents(documents)
    print(f"Created {len(parent_docs)} parent chunks")

    child_docs = []
    for idx, parent in enumerate(parent_docs):
        for child in child_splitter.split_documents([parent]):
            meta = dict(parent.metadata) if parent.metadata else {}
            meta.update({"parent_index": idx, "chunk_type": "child"})
            child_docs.append(Document(page_content=child.page_content, metadata=meta))

    print(f"Created {len(child_docs)} child chunks")
    return child_docs


def create_vectorstore(documents: list[Document]) -> tuple:
    """
    Build Chroma vector store from documents.

    Returns (vectorstore, child_docs).
    child_docs is returned alongside vectorstore because BM25 (in-memory)
    needs the raw document list — it can't be loaded from disk like Chroma.
    """
    child_docs  = _make_child_docs(documents)
    embeddings  = _get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=child_docs,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION,
        persist_directory=CHROMA_DIR,
    )
    print(f"Vector store saved to {CHROMA_DIR}")
    return vectorstore, child_docs


def load_vectorstore() -> Chroma:
    """
    Load existing Chroma vector store from disk.
    Skips re-embedding — embeddings are persisted in CHROMA_DIR.
    """
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=_get_embeddings(),
        persist_directory=CHROMA_DIR,
    )
    print(f"Loaded vector store from {CHROMA_DIR}")
    return vectorstore


def rebuild_child_docs(documents: list[Document]) -> list[Document]:
    """
    Rebuild child chunks from parsed documents.
    Called after load_vectorstore() to give BM25 the same corpus Chroma has.
    BM25 is in-memory only and must be rebuilt each run.
    """
    return _make_child_docs(documents)
