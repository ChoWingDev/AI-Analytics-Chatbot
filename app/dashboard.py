"""
app/dashboard.py
----------------
Streamlit UI for the AI Analytics Copilot.

Thin presentation layer only: it reuses the existing RouterAgent (src/router.py)
and renders the PMReport it already returns. No business logic lives here -- the
same code path the CLI (app/copilot.py) uses, so the UI cannot drift from it.

Run from the project root:
    streamlit run app/dashboard.py
"""

import asyncio
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.router import RouterAgent, PMReport  # noqa: E402
from src.sql_agent.config import GLOSSARY_PATH, DB_PATH  # noqa: E402

st.set_page_config(page_title="AI Analytics Copilot", page_icon="📊", layout="wide")

ROUTE_BADGE = {
    "sql_only": ("Internal data", "#2f6db0"),
    "rag_only": ("Industry reports", "#b25a2a"),
    "both": ("Internal + industry", "#6b46a8"),
}


@st.cache_resource(show_spinner="Loading copilot (vector store + glossary index)...")
def get_agent() -> RouterAgent:
    """
    Built once per Streamlit process.

    Construction loads a ~180MB Chroma store and the glossary embedding index and
    takes ~16s. cache_resource keeps a single instance alive across reruns --
    without it, Streamlit would rebuild it on every keystroke.
    """
    return RouterAgent(glossary_path=GLOSSARY_PATH, db_path=DB_PATH, session_id="dashboard")


def render_report(report: PMReport, route: str | None) -> None:
    if route and route in ROUTE_BADGE:
        label, colour = ROUTE_BADGE[route]
        st.markdown(
            f"<span style='background:{colour};color:#fff;padding:2px 10px;"
            f"border-radius:12px;font-size:0.75rem;font-weight:600;'>{label}</span>",
            unsafe_allow_html=True,
        )

    st.markdown(f"**{report.summary}**")

    if report.comparison_table:
        df = pd.DataFrame(report.comparison_table)
        # Present the columns a PM actually reads, in a sane order.
        preferred = [c for c in ("metric", "your_value", "industry_avg", "status") if c in df.columns]
        other = [c for c in df.columns if c not in preferred]
        st.dataframe(df[preferred + other], use_container_width=True, hide_index=True)

    if report.action_items:
        st.markdown("**Recommended actions**")
        for i, item in enumerate(report.action_items, 1):
            st.markdown(f"{i}. {item}")

    if report.data_sources:
        st.caption("Sources: " + ", ".join(report.data_sources))


agent = get_agent()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Session")

    sessions = agent.list_sessions()
    options = ["dashboard"] + [s for s in sessions if s != "dashboard"]
    picked = st.selectbox("Conversation", options, index=0)

    if picked != agent.session_id:
        agent.set_session(picked)
        # Replay persisted turns so the UI matches the memory the model sees.
        st.session_state.messages = [
            m
            for turn in agent.store.load_session(picked)
            for m in ({"role": "user", "content": turn.human},
                      {"role": "assistant", "content": turn.ai, "report": None})
        ]
        st.rerun()

    if st.button("New conversation"):
        import uuid
        agent.set_session(f"chat-{uuid.uuid4().hex[:6]}")
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        "Ask about internal KPIs (revenue, AOV, return rate), industry reports, "
        "or how the two compare."
    )
    st.caption(f"Turns in memory: {agent.memory.turn_count}")

# ---------------------------------------------------------------- chat
st.title("AI Analytics Copilot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("report") is not None:
            render_report(msg["report"], msg.get("route"))
        else:
            st.markdown(msg["content"])

if question := st.chat_input("How does our return rate compare to the industry?"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Routing, querying, and merging..."):
            try:
                report = asyncio.run(agent.run(question))
                route = agent.last_route.value if agent.last_route else None
                render_report(report, route)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": report.summary,
                    "report": report,
                    "route": route,
                })
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.session_state.messages.append({
                    "role": "assistant", "content": f"Error: {e}", "report": None,
                })
