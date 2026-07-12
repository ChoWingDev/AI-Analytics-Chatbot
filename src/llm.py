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
from typing import Optional, List

from langchain_core.language_models.llms import LLM
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient

# Single source of truth for model ids lives in the RAG config.
from src.rag.config import LLM_MODEL_ID, EMBEDDING_MODEL


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
        client   = InferenceClient(token=self.hf_token)
        response = client.chat_completion(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content


def get_llm(max_new_tokens: int = 512, temperature: float = 0.1) -> HFInferenceLLM:
    """Ready-to-use HF chat LLM. Reads HF_TOKEN from the environment."""
    return HFInferenceLLM(
        model_id=LLM_MODEL_ID,
        hf_token=os.getenv("HF_TOKEN", ""),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


def get_embeddings() -> HuggingFaceEmbeddings:
    """Local MiniLM embeddings — no API token required, shared by SQL + RAG."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
