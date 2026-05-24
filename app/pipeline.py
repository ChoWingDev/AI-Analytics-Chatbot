"""
src/rag_pipeline/pipeline.py
-----------------------------
Person B — Week 1–2
Advanced RAG pipeline with hybrid search (BM25 + Vector) for industry reports.
"""

from pathlib import Path
from typing import List

from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

REPORTS_DIR = Path("data/raw/reports")
CHROMA_DIR = Path("data/processed/chroma_db")

SYSTEM_PROMPT = """You are a senior market research analyst.
Answer questions using ONLY the provided industry reports.
Always cite the source document and page number.
If the information is not in the documents, say: "This information is not available in the current reports."
Never fabricate statistics or benchmark numbers.
"""


def load_pdfs(reports_dir: Path = REPORTS_DIR) -> List[Document]:
    """Load all PDF files from the reports directory."""
    docs = []
    pdf_files = list(reports_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"[WARN] No PDFs found in {reports_dir}. Add industry reports to get started.")
        return docs

    for pdf_path in pdf_files:
        print(f"  Loading: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        # Tag each page with metadata for filtering
        for page in pages:
            page.metadata["source_file"] = pdf_path.name
        docs.extend(pages)

    print(f"  Loaded {len(docs)} pages from {len(pdf_files)} reports")
    return docs


def build_hybrid_retriever(docs: List[Document]) -> EnsembleRetriever:
    """
    Hybrid search: BM25 (keyword) + Vector (semantic).
    BM25 handles exact metric names like 'churn rate' well.
    Vector handles paraphrased or conceptual queries.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Small chunks for retrieval precision
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    # Large chunks for context richness (stored in parent store)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)

    vectorstore = Chroma(
        collection_name="industry_reports",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    store = InMemoryStore()

    parent_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    parent_retriever.add_documents(docs)

    # BM25 on small chunks
    small_chunks = child_splitter.split_documents(docs)
    bm25_retriever = BM25Retriever.from_documents(small_chunks)
    bm25_retriever.k = 5

    # Ensemble: 40% keyword weight, 60% semantic weight
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, parent_retriever],
        weights=[0.4, 0.6],
    )
    return hybrid_retriever


def format_docs_with_citations(docs: List[Document]) -> str:
    """Format retrieved docs with source citations."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[Source {i}: {source}, p.{page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


class RAGPipeline:
    def __init__(self):
        self.retriever = None
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)

    def build(self, reports_dir: Path = REPORTS_DIR):
        docs = load_pdfs(reports_dir)
        if docs:
            self.retriever = build_hybrid_retriever(docs)
        return self

    async def arun(self, question: str) -> dict:
        if self.retriever is None:
            return {"source": "rag", "result": "No industry reports loaded yet. Please add PDFs to data/raw/reports/"}

        retrieved_docs = self.retriever.invoke(question)
        context = format_docs_with_citations(retrieved_docs)

        messages = [
            ("system", SYSTEM_PROMPT),
            ("human", f"Context from industry reports:\n\n{context}\n\nQuestion: {question}"),
        ]
        response = await self.llm.ainvoke(messages)
        return {"source": "rag", "result": response.content}
