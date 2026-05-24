"""
src/memory/session.py
---------------------
Person B — Week 3
Session memory: keeps conversation context across multi-turn PM questions.
Persists to SQLite so sessions survive server restarts.
"""

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from typing import Optional

SESSION_DB = "data/processed/sessions.db"

VAGUE_KEYWORDS = [
    "表現", "怎樣", "好不好", "performance", "how are we doing",
    "how is it", "what about", "compare", "versus",
]


def get_session_history(session_id: str) -> SQLChatMessageHistory:
    """Get or create a persistent chat history for a session."""
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string=f"sqlite:///{SESSION_DB}",
    )


def add_turn(session_id: str, user_message: str, ai_response: str) -> None:
    """Append a user + AI turn to the session history."""
    history = get_session_history(session_id)
    history.add_message(HumanMessage(content=user_message))
    history.add_message(AIMessage(content=ai_response))


def get_recent_context(session_id: str, n_turns: int = 5) -> str:
    """
    Return the last n_turns of conversation as a formatted string.
    Injected into the Router prompt so the LLM understands follow-up questions.
    """
    history = get_session_history(session_id)
    messages = history.messages[-(n_turns * 2):]  # each turn = 2 messages

    if not messages:
        return ""

    lines = ["## Conversation History (most recent first)"]
    for msg in reversed(messages):
        role = "PM" if isinstance(msg, HumanMessage) else "Assistant"
        lines.append(f"{role}: {msg.content[:300]}")  # truncate long AI responses

    return "\n".join(lines)


def needs_clarification(question: str) -> Optional[str]:
    """
    Detect vague questions and return a clarification prompt.
    Returns None if the question is specific enough.
    """
    lower = question.lower()
    if any(kw in lower for kw in VAGUE_KEYWORDS) and len(question.split()) < 6:
        return (
            "Could you be more specific? For example:\n"
            "- Which campaign? (email / organic / search / social)\n"
            "- Which metric? (churn rate / ROI / revenue / conversion rate)\n"
            "- Which time period? (last month / last quarter / last 30 days)"
        )
    return None


def clear_session(session_id: str) -> None:
    """Clear all history for a session (e.g. user clicks 'New Chat')."""
    history = get_session_history(session_id)
    history.clear()
