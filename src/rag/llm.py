"""
llm.py
──────
LLM wrapper. Swap the model or provider here without touching anything else.
"""

import os
from typing import Optional, List

from langchain_core.language_models.llms import LLM
from huggingface_hub import InferenceClient

from .config import LLM_MODEL_ID


class HFInferenceLLM(LLM):
    """
    Thin LangChain-compatible wrapper around huggingface_hub.InferenceClient.

    Why not use HuggingFaceEndpoint?
    HF's free inference API routes models to third-party providers
    (novita, featherless-ai, etc.) that inconsistently reject task types.
    InferenceClient.chat_completion() bypasses that routing and calls
    the model's chat endpoint directly — stable across all providers.
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


def get_llm() -> HFInferenceLLM:
    """Convenience function — returns a ready-to-use LLM instance."""
    return HFInferenceLLM(
        model_id=LLM_MODEL_ID,
        hf_token=os.getenv("HF_TOKEN", ""),
    )
