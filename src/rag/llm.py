"""
llm.py
──────
Backwards-compatible re-export.

The HF LLM wrapper now lives in src/llm.py so the SQL agent, RAG pipeline,
and router share one provider (Tier-1: HuggingFace everywhere). RAG modules
still `from .llm import get_llm` unchanged.
"""

from src.llm import HFInferenceLLM, get_llm

__all__ = ["HFInferenceLLM", "get_llm"]
