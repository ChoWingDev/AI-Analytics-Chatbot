"""
src/router.py
─────────────
Router + parallel execution + merge/insights layer.

This is the Tier-1 integration layer that unifies the two previously
independent pipelines into one copilot:

    question
        │
        ▼
    classify_route            ← LLM decides sql_only / rag_only / both
        │
        ├── Text-to-SQL  (src.sql_agent)      internal TheLook KPIs
        └── Advanced RAG (src.rag)            industry + company reports
        │      (run concurrently when both are needed)
        ▼
    merge_and_generate_report ← LLM fuses both into a PM report (JSON)

Everything runs on one HuggingFace provider (src.llm). A single shared
session (src.rag.memory) gives the router, the RAG chain, and the merge
step conversation history so follow-up questions resolve correctly.
"""

import asyncio
import json
import os
import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel

from src.llm import get_llm

# SQL side
from src.sql_agent.sql_pipeline import TextToSQLPipeline

# RAG side
from src.rag.parsing import (
    load_parsed_documents,
    load_and_parse_pdfs,
    save_parsed_documents,
)
from src.rag.vectorstore import (
    create_vectorstore,
    load_vectorstore,
    rebuild_child_docs,
)
from src.rag.retrieval import create_retriever
from src.rag.chain import build_rag_chain
from src.rag.memory import create_session
from src.rag.config import CHROMA_DIR


class RouteDecision(str, Enum):
    SQL_ONLY = "sql_only"
    RAG_ONLY = "rag_only"
    BOTH = "both"


class PMReport(BaseModel):
    summary: str
    comparison_table: list[dict]  # [{"metric","your_value","industry_avg","status"}]
    action_items: list[str]
    data_sources: list[str]


ROUTER_PROMPT = """You are a routing assistant for a Product Manager's data tool.
Classify the user's question into exactly one category:

- sql_only: internal company data only (our revenue, AOV, churn, return rate,
  active users, conversion — numbers from our own database).
- rag_only: external knowledge only (industry benchmarks, market trends,
  competitor/company reports such as Aritzia or Zara annual reports).
- both: needs internal data AND an industry/competitor comparison
  (e.g. "how does our return rate compare to the industry?").
{history}
Question: {question}

Respond with ONLY one of: sql_only, rag_only, both
"""

INSIGHTS_PROMPT = """You are a senior ecommerce data analyst preparing a report for a Product Manager.
{history}
Company data (from our internal database):
{sql_result}

Industry / competitor data (from research and annual reports):
{rag_result}

Original question: {question}

Return ONLY valid JSON (no markdown, no commentary) in exactly this shape:
{{
  "summary": "One-sentence executive summary of the key finding",
  "comparison_table": [
    {{"metric": "Metric name", "your_value": "value with unit", "industry_avg": "benchmark with unit", "status": "Above average / Below average / On par / N/A"}}
  ],
  "action_items": ["recommendation 1", "recommendation 2", "recommendation 3"],
  "data_sources": ["Internal TheLook DB", "Industry & company reports"]
}}

CRITICAL — never fabricate data:
- Use ONLY numbers that appear literally in the sections above.
- If a section contains an error (e.g. starts with "[SQL error]" or "[RAG error]"),
  says there is no data, returns no rows, or has no usable numbers, set that
  side's value to "N/A" and status to "N/A". Do NOT invent a number.
- ONLY when the internal company data is missing or errored: say so plainly in the
  summary and make the action items about restoring/checking that data. If the
  company data IS present, never suggest "restoring" it — give real business actions.
- Be internally consistent: if you cite an industry figure in the summary, put that
  same figure in industry_avg. If industry_avg is "N/A", do not compare against a
  number in the summary. Set status to "N/A" whenever either side is "N/A".
"""


def _is_missing(value) -> bool:
    """True when a comparison cell carries no usable number."""
    if value is None:
        return True
    v = str(value).strip().lower()
    return v in ("", "n/a", "na", "none", "unknown", "not available", "not provided")


def _enforce_consistency(report: "PMReport") -> "PMReport":
    """
    Enforce the comparison invariant in CODE, not in the prompt.

    The merge prompt asks the model to set status to "N/A" when either side of a
    comparison is missing. It does not reliably obey: it emitted
    `industry_avg: "N/A"` alongside `status: "Below average"` — claiming a
    comparison it had no benchmark for.

    A prompt is a suggestion; this is a guarantee. If either side is missing,
    there is no comparison to report, so the status is N/A. Full stop.
    """
    for row in report.comparison_table:
        if _is_missing(row.get("your_value")) or _is_missing(row.get("industry_avg")):
            row["status"] = "N/A"
    return report


def _extract_json(text: str) -> str:
    """Pull the first {...} block out of an LLM response, tolerating code fences."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


class RouterAgent:
    """
    Orchestrates SQL + RAG for a single conversation.

    Build once (loads the DB glossary index and the RAG vector store), then
    call `run(question)` per turn. Session history is shared across the router,
    the RAG chain, and the merge step.
    """

    def __init__(self, glossary_path: str, db_path: str,
                 session_id: str = "cli", history_k: int = 5):
        # One provider, two temperatures: crisp for routing, a little room for the report.
        self.router_llm = get_llm(max_new_tokens=16, temperature=0.0)
        self.insights_llm = get_llm(max_new_tokens=1024, temperature=0.1)

        # SQL pipeline (glossary retrieval → prompt → HF SQL → SQLite execution)
        self.sql_pipeline = TextToSQLPipeline(glossary_path=glossary_path, db_path=db_path)

        # RAG components (hybrid BM25 + vector over the report corpus)
        self.vectorstore = None
        self.child_docs = []
        self._build_rag()

        # Shared conversation memory (in-RAM window + SQLite persistence)
        self.memory, self.store, self.session_id = create_session(session_id, k=history_k)

    # ── RAG setup ────────────────────────────────────────────────────────────
    def _build_rag(self) -> None:
        docs = load_parsed_documents()
        if docs is None:
            docs = load_and_parse_pdfs()
            if docs:
                save_parsed_documents(docs)

        if not docs:
            print("[Router] No report documents found — RAG branch will be unavailable.")
            return

        if os.path.exists(CHROMA_DIR):
            self.vectorstore = load_vectorstore()
            self.child_docs = rebuild_child_docs(docs)
        else:
            self.vectorstore, self.child_docs = create_vectorstore(docs)

    # ── Branch runners (sync; called via asyncio.to_thread) ──────────────────
    def _run_sql(self, question: str) -> str:
        try:
            out = self.sql_pipeline.run(question)
            df = out["result"]
            if df is None or df.empty:
                return "Query returned no rows."
            return df.to_string(index=False)
        except Exception as e:  # bad SQL, missing table, etc.
            return f"[SQL error] {e}"

    def _run_rag(self, question: str) -> str:
        if self.vectorstore is None:
            return "No industry/company reports are loaded."
        try:
            retriever = create_retriever(self.vectorstore, self.child_docs)
            chain = build_rag_chain(retriever, chat_history=self.memory.format_history())
            return chain.invoke(question)
        except Exception as e:
            return f"[RAG error] {e}"

    async def _run_parallel(self, question: str) -> tuple[str, str]:
        """Run SQL and RAG concurrently. Wall time ≈ max(sql, rag), not the sum."""
        return await asyncio.gather(
            asyncio.to_thread(self._run_sql, question),
            asyncio.to_thread(self._run_rag, question),
        )

    # ── Routing + merge ──────────────────────────────────────────────────────
    def classify(self, question: str) -> RouteDecision:
        history = self.memory.format_history()
        hist_block = f"\nRecent conversation:\n{history}\n" if history else ""
        raw = self.router_llm.invoke(
            ROUTER_PROMPT.format(history=hist_block, question=question)
        ).strip().lower()

        for decision in (RouteDecision.BOTH, RouteDecision.RAG_ONLY, RouteDecision.SQL_ONLY):
            if decision.value in raw:
                return decision
        return RouteDecision.BOTH  # default when the model is unclear

    def merge(self, question: str,
              sql_result: Optional[str], rag_result: Optional[str]) -> PMReport:
        history = self.memory.format_history()
        hist_block = f"Recent conversation:\n{history}\n" if history else ""
        prompt = INSIGHTS_PROMPT.format(
            history=hist_block,
            sql_result=sql_result or "No internal data retrieved.",
            rag_result=rag_result or "No industry data retrieved.",
            question=question,
        )
        raw = self.insights_llm.invoke(prompt)

        try:
            data = json.loads(_extract_json(raw))
            return _enforce_consistency(PMReport(**data))
        except Exception:
            # Graceful fallback if the model didn't return clean JSON.
            sources = []
            if sql_result:
                sources.append("Internal TheLook DB")
            if rag_result:
                sources.append("Industry & company reports")
            return PMReport(
                summary=raw.strip()[:300],
                comparison_table=[],
                action_items=["Report generation hit a formatting issue — please retry."],
                data_sources=sources or ["Internal DB", "Reports"],
            )

    def _remember(self, question: str, report: PMReport) -> None:
        """Persist the turn so follow-ups have context (memory + SQLite)."""
        self.memory.add_turn(question, report.summary)
        self.store.save_turn(self.session_id, question, report.summary)

    async def run(self, question: str) -> PMReport:
        route = self.classify(question)
        print(f"[Router] Decision: {route.value}")

        sql_result: Optional[str] = None
        rag_result: Optional[str] = None

        if route == RouteDecision.SQL_ONLY:
            sql_result = await asyncio.to_thread(self._run_sql, question)
        elif route == RouteDecision.RAG_ONLY:
            rag_result = await asyncio.to_thread(self._run_rag, question)
        else:  # BOTH
            sql_result, rag_result = await self._run_parallel(question)

        report = self.merge(question, sql_result, rag_result)
        self._remember(question, report)
        return report


if __name__ == "__main__":
    # Minimal smoke test. For the full CLI use: python -m app.copilot
    from dotenv import load_dotenv
    from src.sql_agent.config import GLOSSARY_PATH, DB_PATH

    load_dotenv()
    agent = RouterAgent(glossary_path=GLOSSARY_PATH, db_path=DB_PATH, session_id="router-smoke")
    report = asyncio.run(agent.run("How does our return rate compare to the industry?"))
    print(report.model_dump_json(indent=2))
