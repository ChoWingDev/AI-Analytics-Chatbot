"""
app/dashboard.py
────────────────
Streamlit chat UI for the copilot.

Run from the project root:
    streamlit run app/dashboard.py

Each answer shows the PM report plus the evidence behind it: the generated
SQL, the result rows, a chart when the result shape supports one, and the
retrieved report chunks with their page numbers.

KNOWN LIMITATION — single user. The agent is cached once per server process
(building it costs ~10s: pickle load, re-chunk, BM25 index, Chroma, embedding
model) and it holds one `self.memory`. `use_session()` swaps the conversation
per browser tab, but two people answering questions at the same time would
interleave on that shared memory. Fine for a local demo, wrong for a
deployment.
"""

import asyncio
import sys
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = str(Path(__file__).resolve().parent)

# `streamlit run app/dashboard.py` puts app/ at the front of sys.path, where
# app/app.py shadows the `app` package and `from app.charts import ...` fails
# with "'app' is not a package". Drop that entry and lead with the project root.
sys.path[:] = [p for p in sys.path if p not in ("", APP_DIR)]
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()  # HF_TOKEN

from app.charts import choose_chart, format_metric
from src.rag.clarification import is_vague
from src.router import RouterAgent, RouteDecision
from src.sql_agent.config import GLOSSARY_PATH, DB_PATH

ROUTE_LABEL = {
    RouteDecision.SQL_ONLY: "Querying the internal database…",
    RouteDecision.RAG_ONLY: "Searching the industry and company reports…",
    RouteDecision.BOTH: "Querying the database and searching the reports…",
}


@st.cache_resource(show_spinner="Loading the copilot (glossary index + report vector store)…")
def get_agent() -> RouterAgent:
    """One agent per server process — the ~10s build is paid once, not per question."""
    return RouterAgent(glossary_path=GLOSSARY_PATH, db_path=DB_PATH, session_id="dashboard")


def new_session(clear_previous: bool = False) -> None:
    """Start a fresh conversation, optionally deleting the one being replaced."""
    agent = get_agent()
    if clear_previous and "session_id" in st.session_state:
        agent.store.delete_session(st.session_state.session_id)
    st.session_state.session_id = f"dash-{uuid.uuid4().hex[:12]}"
    st.session_state.turns = []
    agent.use_session(st.session_state.session_id)


def render_chart(df) -> None:
    """Draw the result if its shape supports a chart; otherwise draw nothing."""
    kind, params = choose_chart(df)
    if kind == "metric":
        st.metric(params["label"], format_metric(params["label"], params["value"]))
    elif kind == "line":
        st.line_chart(df.set_index(params["x"])[params["y"]])
    elif kind == "bar":
        st.bar_chart(df.set_index(params["x"])[params["y"]])


def render_answer(entry: dict) -> None:
    """Render one stored answer: report, chart, then collapsed evidence."""
    # Branch failures are shown, not swallowed. The merge step turns a failed
    # branch into a quiet "N/A", which reads like a real answer when it is not.
    for err in (entry.get("sql_error"), entry.get("rag_error")):
        if err:
            st.warning(err)

    if entry.get("fatal"):
        st.error(entry["fatal"])
        return

    if entry.get("clarification"):
        st.info(entry["clarification"])
        return

    report = entry["report"]
    st.markdown(report.summary)

    if report.comparison_table:
        st.dataframe(pd.DataFrame(report.comparison_table), hide_index=True)

    df = entry.get("sql_df")
    if df is not None and not df.empty:
        render_chart(df)

    if report.action_items:
        st.markdown("**Action items**")
        for item in report.action_items:
            st.markdown(f"- {item}")

    if report.data_sources:
        st.caption("Sources: " + ", ".join(report.data_sources))

    if entry.get("sql"):
        with st.expander("Generated SQL"):
            st.code(entry["sql"], language="sql")

    if df is not None and not df.empty:
        with st.expander("Result data"):
            st.dataframe(df, hide_index=True)

    docs = entry.get("rag_docs") or []
    if docs:
        with st.expander(f"Retrieved sources ({len(docs)})"):
            st.dataframe(
                pd.DataFrame([
                    {
                        "source": d.metadata.get("source", "unknown"),
                        "page": d.metadata.get("page", "?"),
                        "year": d.metadata.get("year", "unknown"),
                        "doc_type": d.metadata.get("doc_type", "unknown"),
                    }
                    for d in docs
                ]),
                hide_index=True,
            )


def answer(agent: RouterAgent, question: str) -> dict:
    """
    Route and answer one question, or ask for clarification instead.

    Routing and the merge step call the LLM provider outside the branch-level
    error handling in router.py, so a provider outage (an expired token, a
    depleted credit balance) would otherwise surface as a Streamlit traceback
    covering the whole page. Report it as a message in the chat instead --
    with the exception type, since not every failure here is the provider.
    """
    try:
        # Heuristics only. The LLM stage classifies specific questions such as
        # "What is our average order value in 2022?" as vague, which blocks the
        # pipeline on exactly the questions it should run.
        vague, clarification = is_vague(
            question, agent.memory.format_history(), use_llm=False
        )
        if vague:
            # Cheaper and more useful than running an 8-13s pipeline on "what about risks?".
            return {"clarification": clarification}

        route = agent.classify(question)
        with st.spinner(ROUTE_LABEL[route]):
            turn = asyncio.run(agent.run(question, route=route))
    except Exception as e:
        return {"fatal": f"{type(e).__name__}: {e}"}

    return {
        "report": turn.report,
        "route": turn.route.value,
        "sql": turn.sql,
        "sql_df": turn.sql_df,
        "rag_docs": turn.rag_docs,
        "sql_error": turn.sql_error,
        "rag_error": turn.rag_error,
    }


def main() -> None:
    st.set_page_config(page_title="AI Analytics Copilot", page_icon="📊", layout="wide")
    st.title("AI Analytics Copilot")
    st.caption(
        "Ask about internal KPIs, industry reports, or how the two compare. "
        "Every answer shows the SQL and the report pages behind it."
    )

    agent = get_agent()
    if "session_id" not in st.session_state:
        new_session()

    with st.sidebar:
        st.header("Session")
        st.caption(f"`{st.session_state.session_id}`")
        if st.button("New chat", use_container_width=True):
            new_session(clear_previous=True)
            st.rerun()
        st.divider()
        st.caption(
            "The benchmark this project reports (5/5) measures whether the "
            "pipeline implements the glossary, not whether the glossary is "
            "business-correct."
        )

    # Replay the conversation — Streamlit reruns the whole script per interaction.
    for entry in st.session_state.turns:
        with st.chat_message("user"):
            st.markdown(entry["question"])
        with st.chat_message("assistant"):
            render_answer(entry)

    question = st.chat_input("What can I help you with today?")
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)

    entry = {"question": question, **answer(agent, question)}
    st.session_state.turns.append(entry)

    with st.chat_message("assistant"):
        render_answer(entry)


main()
