"""
config.py
─────────
All constants in one place. Change values here — nothing else needs editing.
"""

# Paths
REPORTS_FOLDER   = "data/reports"
PARSED_DOCS_FILE = "data/parsed_docs.pkl"
CHROMA_DIR       = "./chroma_db"

# Chroma
CHROMA_COLLECTION = "ecommerce_reports"

# Models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# General-purpose model: routing, RAG answers, and the merged PM report (prose + JSON).
LLM_MODEL_ID     = "Qwen/Qwen2.5-7B-Instruct"

# SQL specialist. Same provider (HuggingFace), so the "one provider" story holds —
# we just use the right model for the job. The general model repeatedly invented
# columns and emitted multiple statements; a code-tuned model is the cheapest
# lever on Text-to-SQL accuracy.
SQL_MODEL_ID     = "Qwen/Qwen2.5-Coder-7B-Instruct"

# Retrieval
RETRIEVER_K = 5

# Chunking
PARENT_CHUNK_SIZE    = 1500
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE     = 400
CHILD_CHUNK_OVERLAP  = 50

# PDF classification — used by parsing.py to tag doc_type metadata.
# Add company names here as you add more annual reports to the corpus.
COMPANY_KEYWORDS = [
    "annual", "10-k", "10k", "earnings",
    "lululemon", "aritzia", "zara", "inditex",
]
BENCHMARK_KEYWORDS = [
    "benchmark", "industry", "trend", "market", "logistics",
    "census", "jpmorgan", "avalara", "european", "global",
]
