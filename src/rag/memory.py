"""
memory.py
─────────
Conversation memory with two layers:

  1. In-memory window  — keeps the last k turns in RAM for fast access
                         during a session. 

  2. SQLite persistence — writes every turn to disk so history survives
                          restarts and can be replayed across sessions.
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone


# ── Data structure ────────────────────────────────────────────────────────────

@dataclass
class Turn:
    """One round of conversation: a human question and the assistant's answer."""
    human: str
    ai:    str


# ── In-memory window memory ───────────────────────────────────────────────────

class ConversationMemory:
    """
    Sliding window over the last k conversation turns.

    How it's used:
    Before each RAG call, format_history() produces a plain-text block
    of recent turns that gets injected into the prompt as {chat_history}.
    The LLM can then resolve pronouns and follow-ups correctly —
    e.g. "How does that compare?" → the LLM knows "that" = last answer.
    """

    def __init__(self, k: int = 5):
        self.k: int             = k
        self._turns: list[Turn] = []

    def add_turn(self, human: str, ai: str) -> None:
        """Add one turn. If buffer is full, drop the oldest turn (FIFO)."""
        self._turns.append(Turn(human=human, ai=ai))
        if len(self._turns) > self.k:
            self._turns.pop(0)

    def format_history(self) -> str:
        """
        Render recent turns as plain text for prompt injection.

        Example output:
          Human: What was Aritzia's revenue?
          Assistant: Aritzia's eCommerce net revenue was $951M in 2025...

          Human: How does that compare to 2024?
          Assistant: In 2024 it was $774M, so growth was 23%...
        """
        if not self._turns:
            return ""
        lines = []
        for turn in self._turns:
            lines.append(f"Human: {turn.human}")
            lines.append(f"Assistant: {turn.ai}")
        return "\n\n".join(lines)

    def clear(self) -> None:
        """Reset memory — call this to start a new session."""
        self._turns = []

    @property
    def turn_count(self) -> int:
        return len(self._turns)


# ── SQLite persistence ────────────────────────────────────────────────────────

class SessionStore:
    """
    Persists conversation history to SQLite.

    Schema:
      sessions(session_id, human, ai, timestamp)
    One row per turn. session_id groups turns belonging to the same
    conversation.
    """

    def __init__(self, db_path: str = "sessions.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT    NOT NULL,
                    human      TEXT    NOT NULL,
                    ai         TEXT    NOT NULL,
                    timestamp  TEXT    NOT NULL
                )
            """)

    def save_turn(self, session_id: str, human: str, ai: str) -> None:
        """Write one turn to the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO sessions (session_id, human, ai, timestamp) VALUES (?, ?, ?, ?)",
                (session_id, human, ai, datetime.now(timezone.utc).isoformat()),
            )

    def load_session(self, session_id: str) -> list[Turn]:
        """Load all turns for a session, oldest first."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT human, ai FROM sessions WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        return [Turn(human=row[0], ai=row[1]) for row in rows]

    def list_sessions(self) -> list[str]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT DISTINCT session_id FROM sessions ORDER BY session_id"
            ).fetchall()
        return [row[0] for row in rows]

    def delete_session(self, session_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))


# ── Factory ───────────────────────────────────────────────────────────────────

def create_session(session_id: str, k: int = 5, db_path: str = "sessions.db"):
    """
    Create a (memory, store) pair for a session.

    If the session already exists in SQLite, the last k turns are loaded
    into the in-memory window so the conversation resumes naturally.

    Returns (memory, store, session_id).
    """
    memory = ConversationMemory(k=k)
    store  = SessionStore(db_path=db_path)

    past_turns = store.load_session(session_id)
    if past_turns:
        print(f"Resuming session '{session_id}' — {len(past_turns)} past turns found.")
        for turn in past_turns[-k:]:
            memory.add_turn(turn.human, turn.ai)
    else:
        print(f"Starting new session '{session_id}'.")

    return memory, store, session_id