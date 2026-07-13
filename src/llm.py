"""
src/llm.py
──────────
Shared LLM + embedding providers for the whole copilot.

Standardized on Hugging Face (Tier-1 decision) so the Text-to-SQL agent, the
RAG pipeline, and the router all run end-to-end on a single HF_TOKEN:

  - Chat / generation: Qwen via huggingface_hub InferenceClient
  - Embeddings:        sentence-transformers MiniLM, run locally (no token)

Swap the provider here once and every module follows.
"""

import os
import time
from typing import Optional, List

from langchain_core.language_models.llms import LLM
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient

# Single source of truth for model ids lives in the RAG config.
from src.rag.config import LLM_MODEL_ID, SQL_MODEL_ID, EMBEDDING_MODEL

# HF's free inference tier rate-limits bursts (it returned 402/429 mid-benchmark).
# That is a transient infrastructure failure, not a model failure, so retry it
# rather than letting it masquerade as a wrong answer.
_RETRYABLE = ("402", "429", "503", "504", "timeout", "overloaded")
_MAX_ATTEMPTS = 4


class HFInferenceLLM(LLM):
    """
    Thin LangChain-compatible wrapper around huggingface_hub.InferenceClient.

    Why not HuggingFaceEndpoint?
    HF's free inference API routes models to third-party providers
    (novita, featherless-ai, ...) that inconsistently reject task types.
    InferenceClient.chat_completion() bypasses that routing and calls the
    model's chat endpoint directly — stable across all providers.
    """
    model_id:       str   = LLM_MODEL_ID
    hf_token:       str   = ""
    max_new_tokens: int   = 512
    temperature:    float = 0.1

    @property
    def _llm_type(self) -> str:
        return "hf_inference_client"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        client = InferenceClient(token=self.hf_token)

        last_error: Optional[Exception] = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                response = client.chat_completion(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                )
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                if not any(code in str(e).lower() for code in _RETRYABLE):
                    raise  # a real error (bad model id, bad token) — fail fast
                if attempt < _MAX_ATTEMPTS - 1:
                    backoff = 2 ** attempt * 3  # 3s, 6s, 12s
                    print(f"[llm] transient error, retrying in {backoff}s "
                          f"(attempt {attempt + 2}/{_MAX_ATTEMPTS})")
                    time.sleep(backoff)

        raise RuntimeError(
            f"HF inference failed after {_MAX_ATTEMPTS} attempts: {last_error}"
        )


def get_llm(max_new_tokens: int = 512, temperature: float = 0.1,
            model_id: str = LLM_MODEL_ID) -> HFInferenceLLM:
    """
    Ready-to-use HF chat LLM. Reads HF_TOKEN from the environment.

    Pass model_id to pick a specialist (e.g. SQL_MODEL_ID) — same provider,
    different model for the job.
    """
    return HFInferenceLLM(
        model_id=model_id,
        hf_token=os.getenv("HF_TOKEN", ""),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


def get_sql_llm(max_new_tokens: int = 512, temperature: float = 0.0) -> HFInferenceLLM:
    """Code-tuned model for Text-to-SQL generation."""
    return get_llm(max_new_tokens=max_new_tokens, temperature=temperature,
                   model_id=SQL_MODEL_ID)


def get_embeddings() -> HuggingFaceEmbeddings:
    """Local MiniLM embeddings — no API token required, shared by SQL + RAG."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
