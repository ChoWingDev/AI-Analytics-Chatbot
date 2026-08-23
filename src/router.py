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
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from langchain_core.runnables import RunnableLambda
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
    """
    Parsed contract for the merge step's JSON output. Deliberately narrow:
    `PMReport(**data)` is built straight from the LLM response, so this must
    contain only what the model is asked to produce. Evidence (DataFrames,
    retrieved documents) travels alongside it in CopilotTurn instead.
    """
    summary: str
    comparison_table: list[dict]  # [{"metric","your_value","industry_avg","status"}]
    action_items: list[str]
    data_sources: list[str]


@dataclass
class SqlOutcome:
    """Result of the SQL branch. `.text` is what the merge prompt consumes."""
    text: str
    df: Any = None            # pandas DataFrame, kept for charting
    sql: Optional[str] = None
    error: Optional[str] = None


@dataclass
class RagOutcome:
    """Result of the RAG branch. `.text` is what the merge prompt consumes."""
    text: str
    docs: list = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class CopilotTurn:
    """One answered question: the report plus the evidence behind it."""
    report: PMReport
    route: RouteDecision
    sql: Optional[str] = None
    sql_df: Any = None
    rag_docs: list = field(default_factory=list)
    sql_error: Optional[str] = None
    rag_error: Optional[str] = None


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
- If the internal company data is unavailable, say so plainly in the summary
  instead of guessing, and make the action items about restoring/checking that data.
"""


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
        self._retriever = None  # unfiltered hybrid retriever, built once (see _get_retriever)
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

    def _get_retriever(self):
        """
        The unfiltered hybrid retriever, built once and reused.

        BM25Retriever.from_documents re-indexes the whole 104k-chunk corpus on
        every call (~0.9s). No caller passes year/doc_type filters, so one
        instance serves every question.
        """
        if self._retriever is None:
            self._retriever = create_retriever(self.vectorstore, self.child_docs)
        return self._retriever

    # ── Sessions ───────────────────────────────────────────────
    def use_session(self, session_id: str, history_k: int = 5) -> None:
        """
        Point this agent at a different conversation.

        Lets one expensively-built agent (vector store + glossary index) serve
        several sessions. NOTE: the agent holds a single `self.memory`, so
        concurrent sessions would interleave — this is safe for one user at a
        time, not for a multi-user deployment.
        """
        self.memory, self.store, self.session_id = create_session(session_id, k=history_k)

    # ── Branch runners (sync; called via asyncio.to_thread) ──────────────────
    def _run_sql(self, question: str) -> SqlOutcome:
        try:
            out = self.sql_pipeline.run(question)
            df = out["result"]
            sql = out.get("sql")
            if df is None or df.empty:
                return SqlOutcome(text="Query returned no rows.", df=df, sql=sql)
            # `.text` keeps the exact string the merge prompt used to receive;
            # the DataFrame rides along for the UI to chart.
            return SqlOutcome(text=df.to_string(index=False), df=df, sql=sql)
        except Exception as e:  # bad SQL, missing table, etc.
            msg = f"[SQL error] {e}"
            return SqlOutcome(text=msg, error=msg)

    def _run_rag(self, question: str) -> RagOutcome:
        if self.vectorstore is None:
            return RagOutcome(text="No industry/company reports are loaded.")
        try:
            docs = self._get_retriever().invoke(question)
            # Retrieve once, then feed the same documents to the chain. A plain
            # passthrough slots into build_rag_chain's `retriever | format_docs`
            # composition, so src/rag/chain.py needs no change.
            chain = build_rag_chain(
                RunnableLambda(lambda _: docs),
                chat_history=self.memory.format_history(),
            )
            return RagOutcome(text=chain.invoke(question), docs=docs)
        except Exception as e:
            msg = f"[RAG error] {e}"
            return RagOutcome(text=msg, error=msg)

    async def _run_parallel(self, question: str) -> tuple[SqlOutcome, RagOutcome]:
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
            return PMReport(**data)
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

    async def run(self, question: str,
                  route: Optional[RouteDecision] = None) -> CopilotTurn:
        """
        Answer one question.

        `route` lets a caller that already classified (the dashboard, which
        labels its spinner with the decision) reuse it instead of paying a
        second routing call.
        """
        if route is None:
            route = self.classify(question)
        print(f"[Router] Decision: {route.value}")

        sql: Optional[SqlOutcome] = None
        rag: Optional[RagOutcome] = None

        if route == RouteDecision.SQL_ONLY:
            sql = await asyncio.to_thread(self._run_sql, question)
        elif route == RouteDecision.RAG_ONLY:
            rag = await asyncio.to_thread(self._run_rag, question)
        else:  # BOTH
            sql, rag = await self._run_parallel(question)

        report = self.merge(
            question,
            sql.text if sql else None,
            rag.text if rag else None,
        )
        self._remember(question, report)
        return CopilotTurn(
            report=report,
            route=route,
            sql=sql.sql if sql else None,
            sql_df=sql.df if sql else None,
            rag_docs=rag.docs if rag else [],
            sql_error=sql.error if sql else None,
            rag_error=rag.error if rag else None,
        )


if __name__ == "__main__":
    # Minimal smoke test. For the full CLI use: python -m app.copilot
    from dotenv import load_dotenv
    from src.sql_agent.config import GLOSSARY_PATH, DB_PATH

    load_dotenv()
    agent = RouterAgent(glossary_path=GLOSSARY_PATH, db_path=DB_PATH, session_id="router-smoke")
    turn = asyncio.run(agent.run("How does our return rate compare to the industry?"))
    print(turn.report.model_dump_json(indent=2))
